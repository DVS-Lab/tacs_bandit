"""
Analysis utilities for Two-Armed Bandit Task data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from pathlib import Path


class BanditAnalyzer:
    """Analysis tools for bandit task data"""
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize analyzer with data
        
        Parameters:
        -----------
        data_path : str, optional
            Path to CSV file
        df : pd.DataFrame, optional
            Pre-loaded dataframe
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Must provide either data_path or df")
            
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess data for analysis"""
        # Remove no-response trials for most analyses
        self.df_valid = self.df[self.df['choice'].notna()].copy()
        
        # Add derived columns
        if 'correct' in self.df_valid.columns:
            self.df_valid['accuracy'] = self.df_valid['correct'].astype(float)
        
        # Calculate running averages
        if not self.df_valid.empty:
            window = 10
            self.df_valid['accuracy_smooth'] = (
                self.df_valid['accuracy']
                .rolling(window=window, min_periods=1)
                .mean()
            )
            self.df_valid['reward_smooth'] = (
                self.df_valid['reward']
                .rolling(window=window, min_periods=1)
                .mean()
            )
    
    def calculate_summary_stats(self) -> Dict:
        """
        Calculate summary statistics
        
        Returns:
        --------
        Dict : Summary statistics
        """
        stats = {}
        
        # Basic performance
        stats['n_trials'] = len(self.df)
        stats['n_responses'] = len(self.df_valid)
        stats['response_rate'] = len(self.df_valid) / len(self.df)
        
        if not self.df_valid.empty:
            stats['mean_rt'] = self.df_valid['rt'].mean()
            stats['median_rt'] = self.df_valid['rt'].median()
            stats['accuracy'] = self.df_valid['accuracy'].mean()
            stats['reward_rate'] = self.df_valid['reward'].mean()
            
            # Switch behavior
            switches = (self.df_valid['choice'].diff() != 0).sum()
            stats['switch_rate'] = switches / (len(self.df_valid) - 1)
            
            # Win-stay lose-shift
            stats.update(self.calculate_wsls())
        
        return stats
    
    def calculate_wsls(self) -> Dict:
        """
        Calculate win-stay lose-shift statistics
        
        Returns:
        --------
        Dict : WSLS statistics
        """
        df = self.df_valid.copy()
        
        if len(df) < 2:
            return {}
        
        # Previous outcomes
        df['prev_reward'] = df['reward'].shift(1)
        df['prev_choice'] = df['choice'].shift(1)
        
        # Stay/switch behavior
        df['stayed'] = df['choice'] == df['prev_choice']
        
        # Calculate probabilities
        df_subset = df.iloc[1:]  # Skip first trial
        
        win_trials = df_subset[df_subset['prev_reward'] == 1]
        lose_trials = df_subset[df_subset['prev_reward'] == 0]
        
        stats = {}
        if len(win_trials) > 0:
            stats['p_stay_after_win'] = win_trials['stayed'].mean()
        if len(lose_trials) > 0:
            stats['p_stay_after_lose'] = lose_trials['stayed'].mean()
            
        return stats
    
    def plot_learning_curve(self, 
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None):
        """
        Plot learning curves
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy over time
        ax = axes[0, 0]
        ax.plot(self.df_valid['trial_num'], self.df_valid['accuracy'], 
                alpha=0.3, label='Raw')
        ax.plot(self.df_valid['trial_num'], self.df_valid['accuracy_smooth'], 
                linewidth=2, label='Smoothed')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trial')
        ax.set_ylabel('P(Correct)')
        ax.set_title('Accuracy')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Reward rate over time
        ax = axes[0, 1]
        ax.plot(self.df_valid['trial_num'], self.df_valid['reward'], 
                alpha=0.3, label='Raw')
        ax.plot(self.df_valid['trial_num'], self.df_valid['reward_smooth'], 
                linewidth=2, label='Smoothed')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trial')
        ax.set_ylabel('P(Reward)')
        ax.set_title('Reward Rate')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # RT distribution
        ax = axes[1, 0]
        ax.hist(self.df_valid['rt'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=self.df_valid['rt'].median(), color='r', 
                  linestyle='--', label=f"Median: {self.df_valid['rt'].median():.0f} ms")
        ax.set_xlabel('Reaction Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('RT Distribution')
        ax.legend()
        
        # Choice distribution
        ax = axes[1, 1]
        choice_counts = self.df_valid['choice'].value_counts()
        ax.bar(choice_counts.index, choice_counts.values, 
               color=['blue', 'red'], edgecolor='black')
        ax.set_xlabel('Choice')
        ax.set_ylabel('Count')
        ax.set_title('Choice Distribution')
        ax.set_xticks([1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def plot_reversal_analysis(self, 
                              window_size: int = 5,
                              figsize: Tuple[int, int] = (10, 6)):
        """
        Plot behavior around reversal points
        
        Parameters:
        -----------
        window_size : int
            Trials before/after reversal to analyze
        figsize : Tuple[int, int]
            Figure size
        """
        # Detect reversals
        reversals = self.df_valid['current_good'].diff() != 0
        reversal_trials = self.df_valid[reversals]['trial_num'].values[1:]  # Skip first
        
        if len(reversal_trials) == 0:
            print("No reversals detected")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Accuracy around reversals
        ax = axes[0]
        
        for rev_trial in reversal_trials:
            # Get trials around reversal
            mask = (self.df_valid['trial_num'] >= rev_trial - window_size) & \
                   (self.df_valid['trial_num'] <= rev_trial + window_size)
            
            trials = self.df_valid[mask].copy()
            if len(trials) > 0:
                trials['relative_trial'] = trials['trial_num'] - rev_trial
                ax.plot(trials['relative_trial'], trials['accuracy'], 
                       alpha=0.3, color='gray')
        
        # Average across reversals
        all_relative = []
        all_accuracy = []
        
        for rev_trial in reversal_trials:
            mask = (self.df_valid['trial_num'] >= rev_trial - window_size) & \
                   (self.df_valid['trial_num'] <= rev_trial + window_size)
            trials = self.df_valid[mask].copy()
            if len(trials) > 0:
                trials['relative_trial'] = trials['trial_num'] - rev_trial
                all_relative.extend(trials['relative_trial'].values)
                all_accuracy.extend(trials['accuracy'].values)
        
        # Calculate mean accuracy for each relative position
        df_rev = pd.DataFrame({'relative_trial': all_relative, 'accuracy': all_accuracy})
        mean_acc = df_rev.groupby('relative_trial')['accuracy'].mean()
        
        ax.plot(mean_acc.index, mean_acc.values, 'b-', linewidth=3, label='Average')
        ax.axvline(x=0, color='r', linestyle='--', label='Reversal')
        ax.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Trials from Reversal')
        ax.set_ylabel('P(Correct)')
        ax.set_title('Accuracy Around Reversals')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Switch rate around reversals
        ax = axes[1]
        
        all_switches = []
        for rev_trial in reversal_trials:
            mask = (self.df_valid['trial_num'] >= rev_trial - window_size) & \
                   (self.df_valid['trial_num'] <= rev_trial + window_size)
            trials = self.df_valid[mask].copy()
            if len(trials) > 1:
                trials['relative_trial'] = trials['trial_num'] - rev_trial
                trials['switched'] = trials['choice'].diff() != 0
                for rel_trial in trials['relative_trial'].unique():
                    if rel_trial != trials['relative_trial'].min():
                        switch_rate = trials[trials['relative_trial'] == rel_trial]['switched'].mean()
                        all_switches.append({'relative_trial': rel_trial, 'switch_rate': switch_rate})
        
        if all_switches:
            df_switch = pd.DataFrame(all_switches)
            mean_switch = df_switch.groupby('relative_trial')['switch_rate'].mean()
            
            ax.plot(mean_switch.index, mean_switch.values, 'g-', linewidth=3)
            ax.axvline(x=0, color='r', linestyle='--', label='Reversal')
            ax.set_xlabel('Trials from Reversal')
            ax.set_ylabel('P(Switch)')
            ax.set_title('Switch Rate Around Reversals')
            ax.legend()
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_blocks(self, metric: str = 'accuracy') -> pd.DataFrame:
        """
        Compare performance across blocks
        
        Parameters:
        -----------
        metric : str
            Metric to compare ('accuracy', 'reward', 'rt', 'switch_rate')
            
        Returns:
        --------
        pd.DataFrame : Block comparison statistics
        """
        blocks = []
        
        for block_num in self.df_valid['block_num'].unique():
            block_data = self.df_valid[self.df_valid['block_num'] == block_num]
            
            block_stats = {
                'block': block_num,
                'n_trials': len(block_data)
            }
            
            if metric == 'accuracy':
                block_stats['mean'] = block_data['accuracy'].mean()
                block_stats['sem'] = block_data['accuracy'].sem()
            elif metric == 'reward':
                block_stats['mean'] = block_data['reward'].mean()
                block_stats['sem'] = block_data['reward'].sem()
            elif metric == 'rt':
                block_stats['mean'] = block_data['rt'].mean()
                block_stats['sem'] = block_data['rt'].sem()
            elif metric == 'switch_rate':
                switches = (block_data['choice'].diff() != 0).sum()
                block_stats['mean'] = switches / (len(block_data) - 1)
                block_stats['sem'] = None
            
            blocks.append(block_stats)
        
        return pd.DataFrame(blocks)
    
    def export_summary(self, save_path: str):
        """
        Export summary statistics and plots
        
        Parameters:
        -----------
        save_path : str
            Directory to save outputs
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary stats
        stats = self.calculate_summary_stats()
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(save_dir / 'summary_stats.csv', index=False)
        
        # Save block comparison
        blocks_df = self.compare_blocks()
        blocks_df.to_csv(save_dir / 'block_comparison.csv', index=False)
        
        # Save plots
        self.plot_learning_curve(save_path=save_dir / 'learning_curves.png')
        self.plot_reversal_analysis()
        
        print(f"Analysis exported to {save_dir}")


def quick_analysis(data_file: str):
    """
    Quick analysis of a single data file
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
    """
    analyzer = BanditAnalyzer(data_file)
    
    # Print summary
    print("\n=== Summary Statistics ===")
    stats = analyzer.calculate_summary_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Show plots
    analyzer.plot_learning_curve()
    analyzer.plot_reversal_analysis()
    
    return analyzer


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        quick_analysis(sys.argv[1])
    else:
        print("Usage: python analysis.py <data_file.csv>")
