#!/usr/bin/env python3
"""
Two-Armed Bandit Task
Python implementation of a reinforcement learning task with reversal learning
Originally converted from MATLAB/Neurostim framework
"""

import numpy as np
import pandas as pd
from psychopy import visual, core, event, data, gui
from datetime import datetime
import json
import os
from pathlib import Path

class TwoArmedBanditTask:
    """Main task class for the two-armed bandit experiment"""
    
    def __init__(self, config):
        """Initialize the task with configuration parameters"""
        self.config = config
        self.subject_info = {}
        self.trial_data = []
        self.current_trial = 0
        self.block_num = 0
        
        # Learning parameters
        self.current_good = np.random.randint(1, 3)  # Which symbol (1 or 2) is currently better
        self.trial_in_contingency = 0
        self.contingency_trials = self._get_contingency_duration()
        
        # Initialize window and stimuli after getting subject info
        self.win = None
        self.stimuli = {}
        self.clock = core.Clock()
        
    def _get_contingency_duration(self):
        """Get the number of trials for current contingency"""
        min_trials = self.config['task']['min_trials_same_contingency']
        jitter = self.config['task']['contingency_jitter']
        return min_trials + np.random.randint(0, jitter + 1)
    
    def _create_pseudorandom_schedule(self, n_trials, win_fraction):
        """Create pseudorandom reward schedule"""
        n_wins = int(np.round(win_fraction * n_trials))
        schedule = np.zeros(n_trials)
        schedule[:n_wins] = 1
        np.random.shuffle(schedule)
        return schedule
    
    def setup_experiment(self):
        """Get subject information and setup the experiment"""
        # GUI for subject info
        exp_info = {
            'subject_id': '',
            'session': '1',
            'run': '1',
            'age': '',
            'gender': ['M', 'F', 'Other'],
            'mode': ['behavioral', 'test']
        }
        
        dlg = gui.DlgFromDict(
            exp_info,
            title='Two-Armed Bandit Task',
            order=['subject_id', 'session', 'run', 'age', 'gender', 'mode']
        )
        
        if not dlg.OK:
            core.quit()
        
        self.subject_info = exp_info
        self.subject_info['date'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create data directory
        self.data_dir = Path(self.config['paths']['data_dir']) / f"sub-{exp_info['subject_id']}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup window
        self._setup_window()
        
        # Load stimuli
        self._load_stimuli()
        
    def _setup_window(self):
        """Setup PsychoPy window"""
        if self.subject_info['mode'] == 'test':
            # Smaller window for testing
            self.win = visual.Window(
                size=[1024, 768],
                fullscr=False,
                units='height',
                color=self.config['display']['background_color']
            )
        else:
            # Full screen for actual experiment
            self.win = visual.Window(
                fullscr=True,
                units='height',
                color=self.config['display']['background_color']
            )
    
    def _load_stimuli(self):
        """Load and create all visual stimuli"""
        stim_config = self.config['stimuli']
        
        # Fixation cross
        self.stimuli['fixation'] = visual.ShapeStim(
            self.win,
            vertices='cross',
            size=stim_config['fixation_size'],
            lineColor='white',
            fillColor='white'
        )
        
        # Instruction text
        self.stimuli['instruction'] = visual.TextStim(
            self.win,
            text='',
            height=0.03,
            color='white'
        )
        
        # Slot machine images (using placeholders for now)
        # In production, load actual flower images
        colors = ['red', 'blue']
        for i, color in enumerate(colors, 1):
            self.stimuli[f'slot_{i}'] = visual.Rect(
                self.win,
                width=stim_config['slot_size'],
                height=stim_config['slot_size'],
                fillColor=color,
                lineColor='white'
            )
        
        # Position slots on left and right
        self.stimuli['slot_1'].pos = [-stim_config['slot_separation']/2, 0]
        self.stimuli['slot_2'].pos = [stim_config['slot_separation']/2, 0]
        
        # Feedback stimuli
        self.stimuli['win'] = visual.TextStim(
            self.win,
            text='$$$',
            height=0.1,
            color='green',
            bold=True
        )
        
        self.stimuli['loss'] = visual.TextStim(
            self.win,
            text='---',
            height=0.1,
            color='red',
            bold=True
        )
        
        self.stimuli['no_response'] = visual.TextStim(
            self.win,
            text='?',
            height=0.15,
            color='yellow',
            bold=True
        )
    
    def show_instructions(self):
        """Display task instructions"""
        instructions = """
        Two-Armed Bandit Task
        
        You will see two slot machines on each trial.
        Choose the left machine with '1' or the right machine with '2'.
        
        One machine is better than the other, but this can change!
        Try to win as much as possible.
        
        Press SPACE to begin.
        """
        
        self.stimuli['instruction'].text = instructions
        self.stimuli['instruction'].draw()
        self.win.flip()
        
        # Wait for space key
        event.waitKeys(keyList=['space'])
    
    def run_trial(self):
        """Run a single trial of the task"""
        trial_info = {
            'trial_num': self.current_trial,
            'block_num': self.block_num,
            'trial_in_contingency': self.trial_in_contingency,
            'current_good': self.current_good,
            'slot_1_pos': 'left',
            'slot_2_pos': 'right'
        }
        
        timing = self.config['timing']
        
        # Randomly swap positions if configured
        if np.random.random() < 0.5:
            self.stimuli['slot_1'].pos, self.stimuli['slot_2'].pos = \
                self.stimuli['slot_2'].pos, self.stimuli['slot_1'].pos
            trial_info['slot_1_pos'] = 'right'
            trial_info['slot_2_pos'] = 'left'
        
        # Show fixation
        self.stimuli['fixation'].draw()
        self.win.flip()
        core.wait(timing['fixation_duration'])
        
        # Show slot machines and get response
        response_clock = core.Clock()
        self.stimuli['slot_1'].draw()
        self.stimuli['slot_2'].draw()
        self.win.flip()
        
        keys = event.waitKeys(
            maxWait=timing['max_response_time'],
            keyList=['1', '2', 'escape'],
            timeStamped=response_clock
        )
        
        # Process response
        if keys:
            if keys[0][0] == 'escape':
                self.save_data()
                core.quit()
            
            choice = int(keys[0][0])
            rt = keys[0][1] * 1000  # Convert to ms
            
            # Highlight chosen slot
            chosen_slot = self.stimuli[f'slot_{choice}']
            chosen_slot.lineWidth = 5
            chosen_slot.draw()
            other_slot = self.stimuli[f'slot_{3-choice}']  # 3-1=2, 3-2=1
            other_slot.draw()
            self.win.flip()
            core.wait(timing['choice_highlight_duration'])
            chosen_slot.lineWidth = 1  # Reset
            
        else:
            choice = None
            rt = None
        
        # Wait period
        self.win.flip()
        wait_time = np.random.uniform(
            timing['wait_duration_min'],
            timing['wait_duration_max']
        )
        core.wait(wait_time)
        
        # Determine outcome and show feedback
        if choice is None:
            # No response
            self.stimuli['no_response'].draw()
            reward = None
            correct = None
        else:
            # Check if this was the good choice
            correct = (choice == self.current_good)
            
            # Determine reward based on schedule
            if correct:
                reward_prob = self.config['task']['win_fraction']
            else:
                reward_prob = 1 - self.config['task']['win_fraction']
            
            reward = np.random.random() < reward_prob
            
            # Show feedback
            if reward:
                self.stimuli['win'].draw()
            else:
                self.stimuli['loss'].draw()
        
        self.win.flip()
        core.wait(timing['outcome_duration'])
        
        # ITI
        self.win.flip()
        iti = timing['iti_mean'] * np.random.exponential(1)
        iti = min(iti, timing['iti_max'])  # Cap maximum ITI
        core.wait(iti)
        
        # Store trial data
        trial_info.update({
            'choice': choice,
            'rt': rt,
            'correct': correct,
            'reward': reward,
            'wait_time': wait_time * 1000,  # Convert to ms
            'iti': iti * 1000,  # Convert to ms
            'timestamp': self.clock.getTime()
        })
        self.trial_data.append(trial_info)
        
        # Update contingency
        self.trial_in_contingency += 1
        if self.trial_in_contingency >= self.contingency_trials:
            # Switch contingency
            self.current_good = 3 - self.current_good  # Switch between 1 and 2
            self.trial_in_contingency = 0
            self.contingency_trials = self._get_contingency_duration()
        
        self.current_trial += 1
        
        return choice is not None  # Return whether a response was made
    
    def run_block(self, n_trials):
        """Run a block of trials"""
        self.block_num += 1
        block_start = self.current_trial
        
        for trial in range(n_trials):
            # Check for quit key
            if event.getKeys(['escape']):
                self.save_data()
                core.quit()
            
            self.run_trial()
            
            # Optional: Brief break every N trials
            if (trial + 1) % 50 == 0 and trial < n_trials - 1:
                self.show_break_message(trial + 1, n_trials)
        
        # Show block feedback
        self.show_block_feedback(block_start)
    
    def show_break_message(self, current, total):
        """Show a brief break message"""
        message = f"Trial {current}/{total}\n\nTake a brief break.\n\nPress SPACE to continue."
        self.stimuli['instruction'].text = message
        self.stimuli['instruction'].draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
    
    def show_block_feedback(self, block_start):
        """Show feedback at the end of a block"""
        block_trials = [t for t in self.trial_data if t['trial_num'] >= block_start]
        
        # Calculate performance
        responses = [t for t in block_trials if t['choice'] is not None]
        if responses:
            correct_pct = np.mean([t['correct'] for t in responses]) * 100
            reward_pct = np.mean([t['reward'] for t in responses]) * 100
            
            message = f"""
            Block {self.block_num} Complete!
            
            Correct choices: {correct_pct:.1f}%
            Rewards earned: {reward_pct:.1f}%
            
            Press SPACE to continue.
            """
        else:
            message = "Block complete.\n\nPress SPACE to continue."
        
        self.stimuli['instruction'].text = message
        self.stimuli['instruction'].draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
    
    def save_data(self):
        """Save trial data to CSV"""
        if not self.trial_data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.trial_data)
        
        # Add subject info
        for key, value in self.subject_info.items():
            df[key] = value
        
        # Generate filename
        filename = (
            f"sub-{self.subject_info['subject_id']}_"
            f"ses-{self.subject_info['session']}_"
            f"run-{self.subject_info['run']}_"
            f"task-bandit_{self.subject_info['date']}.csv"
        )
        
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
    
    def run(self):
        """Main experiment run function"""
        try:
            # Setup
            self.setup_experiment()
            
            # Instructions
            self.show_instructions()
            
            # Run blocks
            for block in range(self.config['task']['n_blocks']):
                self.run_block(self.config['task']['trials_per_block'])
            
            # End message
            self.stimuli['instruction'].text = "Task complete!\n\nThank you for participating!"
            self.stimuli['instruction'].draw()
            self.win.flip()
            core.wait(2)
            
        finally:
            # Clean up
            self.save_data()
            self.win.close()
            core.quit()


def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'task': {
                'n_blocks': 4,
                'trials_per_block': 50,
                'min_trials_same_contingency': 25,
                'contingency_jitter': 5,
                'win_fraction': 0.75
            },
            'timing': {
                'fixation_duration': 0.5,
                'max_response_time': 2.0,
                'choice_highlight_duration': 0.5,
                'wait_duration_min': 0.5,
                'wait_duration_max': 1.0,
                'outcome_duration': 1.0,
                'iti_mean': 1.0,
                'iti_max': 5.0
            },
            'stimuli': {
                'fixation_size': 0.02,
                'slot_size': 0.15,
                'slot_separation': 0.4
            },
            'display': {
                'background_color': [0, 0, 0]
            },
            'paths': {
                'data_dir': './data',
                'stimuli_dir': './stimuli'
            }
        }


if __name__ == '__main__':
    # Load configuration
    config = load_config()
    
    # Create and run task
    task = TwoArmedBanditTask(config)
    task.run()
