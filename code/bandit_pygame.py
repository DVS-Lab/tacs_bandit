#!/usr/bin/env python3
"""
Two-Armed Bandit Task - Pygame Implementation
Matches MATLAB/Neurostim version timing and structure
"""

import pygame
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import sys
from pathlib import Path
import time


class TwoArmedBanditTask:
    """Two-armed bandit task matching MATLAB implementation"""
    
    def __init__(self, config=None):
        """Initialize the task"""
        # Load config
        if config is None:
            config = self.load_config()
        self.config = config
        
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Task variables
        self.subject_info = {}
        self.trial_data = []
        self.current_trial = 0
        self.run_number = 0
        self.current_good = np.random.randint(1, 3)
        self.trial_in_contingency = 0
        self.contingency_trials = self._get_contingency_duration()
        self.run_start_time = None
        self.experiment_start_time = None
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 100, 100)
        self.BLUE = (100, 100, 255)
        self.GREEN = (100, 255, 100)
        self.YELLOW = (255, 255, 100)
        self.GRAY = (128, 128, 128)
        
    def load_config(self):
        """Load configuration from file or use defaults matching MATLAB"""
        config_file = 'config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Ensure display section has required fields
                if 'display' in config:
                    if 'window_size' in config['display'] and isinstance(config['display']['window_size'], list):
                        config['display']['window_width'] = config['display']['window_size'][0]
                        config['display']['window_height'] = config['display']['window_size'][1]
                    elif 'window_width' not in config['display']:
                        config['display']['window_width'] = 1024
                        config['display']['window_height'] = 768
                    if 'fullscreen' not in config['display']:
                        config['display']['fullscreen'] = False
                return config
        
        # Default configuration matching MATLAB THETA_NIC mode
        return {
            'experiment': {
                'mode': 'THETA_NIC',
                'run_duration_minutes': 6,  # 6-minute runs
                'run_types': {
                    '1': 'baseline',
                    '2': 'theta_stim',  # Or sham (counterbalanced)
                    '3': 'theta_stim',
                    '4': 'post_stim',
                    '5': 'baseline',
                    '6': 'theta_stim',  # Or sham (opposite of 2-3)
                    '7': 'theta_stim',
                    '8': 'post_stim'
                }
            },
            'task': {
                'min_trials_same_contingency': 25,  # Matches MATLAB
                'contingency_jitter': 4,  # Matches MATLAB (was 5 in my version)
                'win_fraction': 0.75  # Matches MATLAB
            },
            'timing': {
                # THETA_NIC timing from MATLAB
                'fixation_duration': 0.5,
                'max_response_time': 2.0,  # 2000ms in MATLAB
                'choice_highlight_duration': 0.5,
                'wait_duration_min': 2.0,  # Fixed 2s in THETA_NIC
                'wait_duration_max': 2.0,  # No jitter in THETA_NIC
                'outcome_duration': 1.0,   # 1000ms in MATLAB
                'iti_duration': 0.25  # 250ms fixed in THETA_NIC (no jitter)
            },
            'display': {
                'window_width': 1024,
                'window_height': 768,
                'fullscreen': False
            },
            'paths': {
                'data_dir': './data'
            }
        }
    
    def _get_contingency_duration(self):
        """Get the number of trials for current contingency (matching MATLAB)"""
        min_trials = self.config['task']['min_trials_same_contingency']
        jitter = self.config['task']['contingency_jitter']
        # MATLAB: minNrTrialsWithSameContingency + randi(contingencyJitter+1)-1
        # This gives 0 to jitter inclusive
        return min_trials + np.random.randint(0, jitter + 1)
    
    def get_subject_info(self):
        """Get subject information and run number"""
        print("\n=== Two-Armed Bandit Task (THETA Protocol) ===\n")
        
        self.subject_info = {
            'subject_id': input("Subject ID: "),
            'session': input("Session number (default 1): ") or "1",
            'age': input("Age: "),
            'gender': input("Gender (M/F/Other): ")
        }
        
        # Get run number (1-8 for full protocol)
        while True:
            run_input = input("Run number (1-8): ")
            try:
                run_num = int(run_input)
                if 1 <= run_num <= 8:
                    self.run_number = run_num
                    break
                else:
                    print("Run number must be between 1 and 8")
            except ValueError:
                print("Please enter a valid number")
        
        # Determine run type and stimulation condition
        run_types = self.config['experiment']['run_types']
        self.run_type = run_types.get(str(self.run_number), 'unknown')
        
        # Counterbalancing for stimulation (even subjects get theta first, odd get sham first)
        subject_num = int(self.subject_info['subject_id']) if self.subject_info['subject_id'].isdigit() else 0
        if subject_num % 2 == 0:
            # Even: runs 2-3 are theta, 6-7 are sham
            if self.run_number in [2, 3]:
                self.stim_condition = 'theta'
            elif self.run_number in [6, 7]:
                self.stim_condition = 'sham'
            else:
                self.stim_condition = 'none'
        else:
            # Odd: runs 2-3 are sham, 6-7 are theta
            if self.run_number in [2, 3]:
                self.stim_condition = 'sham'
            elif self.run_number in [6, 7]:
                self.stim_condition = 'theta'
            else:
                self.stim_condition = 'none'
        
        self.subject_info['run'] = str(self.run_number)
        self.subject_info['run_type'] = self.run_type
        self.subject_info['stim_condition'] = self.stim_condition
        self.subject_info['date'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        print(f"\nRun {self.run_number}: {self.run_type}")
        if self.stim_condition != 'none':
            print(f"Stimulation: {self.stim_condition}")
        print(f"Duration: {self.config['experiment']['run_duration_minutes']} minutes")
        
        # Create data directory
        self.data_dir = Path(self.config['paths']['data_dir']) / f"sub-{self.subject_info['subject_id']}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def setup_display(self):
        """Setup Pygame display"""
        if self.config['display']['fullscreen']:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
        else:
            self.width = self.config['display']['window_width']
            self.height = self.config['display']['window_height']
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(f"Two-Armed Bandit Task - Run {self.run_number}")
        
        # Setup fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Calculate positions
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.slot_size = 120
        self.slot_spacing = 300
        
        # Slot positions (will be randomized each trial like MATLAB)
        self.update_slot_positions()
    
    def update_slot_positions(self):
        """Update slot positions (can be randomized each trial)"""
        # Base positions
        left_x = self.center_x - self.slot_spacing // 2
        right_x = self.center_x + self.slot_spacing // 2
        
        # Randomly assign which slot appears where (matching MATLAB jitter)
        if np.random.random() < 0.5:
            slot1_x, slot2_x = left_x, right_x
            self.slot1_side = 'left'
            self.slot2_side = 'right'
        else:
            slot1_x, slot2_x = right_x, left_x
            self.slot1_side = 'right'
            self.slot2_side = 'left'
        
        self.slot1_rect = pygame.Rect(
            slot1_x - self.slot_size // 2,
            self.center_y - self.slot_size // 2,
            self.slot_size,
            self.slot_size
        )
        
        self.slot2_rect = pygame.Rect(
            slot2_x - self.slot_size // 2,
            self.center_y - self.slot_size // 2,
            self.slot_size,
            self.slot_size
        )
    
    def show_text(self, text, y_offset=0, font=None, color=None):
        """Display text on screen"""
        if font is None:
            font = self.font_medium
        if color is None:
            color = self.WHITE
            
        lines = text.split('\n')
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, color)
            text_rect = text_surface.get_rect(center=(self.center_x, self.center_y + y_offset + i * 50))
            self.screen.blit(text_surface, text_rect)
    
    def show_fixation(self, duration):
        """Show fixation cross"""
        self.screen.fill(self.BLACK)
        # Draw cross
        pygame.draw.line(self.screen, self.WHITE, 
                        (self.center_x - 20, self.center_y),
                        (self.center_x + 20, self.center_y), 3)
        pygame.draw.line(self.screen, self.WHITE,
                        (self.center_x, self.center_y - 20),
                        (self.center_x, self.center_y + 20), 3)
        pygame.display.flip()
        time.sleep(duration)
    
    def show_slots(self, highlight=None):
        """Show the two slot machines"""
        self.screen.fill(self.BLACK)
        
        # Draw slot 1 (red/flower 1)
        color1 = self.WHITE if highlight == 1 else self.RED
        width1 = 5 if highlight == 1 else 2
        pygame.draw.rect(self.screen, color1, self.slot1_rect, width1)
        # Add "1" label
        text = self.font_small.render("1", True, self.WHITE)
        text_rect = text.get_rect(center=(self.slot1_rect.centerx, self.slot1_rect.bottom + 30))
        self.screen.blit(text, text_rect)
        
        # Draw slot 2 (blue/flower 2)
        color2 = self.WHITE if highlight == 2 else self.BLUE
        width2 = 5 if highlight == 2 else 2
        pygame.draw.rect(self.screen, color2, self.slot2_rect, width2)
        # Add "2" label
        text = self.font_small.render("2", True, self.WHITE)
        text_rect = text.get_rect(center=(self.slot2_rect.centerx, self.slot2_rect.bottom + 30))
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def get_response(self, max_time):
        """Get keyboard response"""
        start_time = time.time()
        clock = pygame.time.Clock()
        
        while time.time() - start_time < max_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'escape', time.time() - start_time
                    elif event.key == pygame.K_1 or event.key == pygame.K_KP1:
                        return 1, time.time() - start_time
                    elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                        return 2, time.time() - start_time
                    # Also accept 'a' for left and 'l' for right (like MATLAB)
                    elif event.key == pygame.K_a:  # Left option
                        if self.slot1_side == 'left':
                            return 1, time.time() - start_time
                        else:
                            return 2, time.time() - start_time
                    elif event.key == pygame.K_l:  # Right option
                        if self.slot1_side == 'right':
                            return 1, time.time() - start_time
                        else:
                            return 2, time.time() - start_time
            
            clock.tick(60)  # 60 FPS
        
        return None, None
    
    def show_feedback(self, reward, duration):
        """Show feedback (matching MATLAB style)"""
        self.screen.fill(self.BLACK)
        
        if reward is None:
            # No response - show question mark
            self.show_text("?", font=self.font_large, color=self.YELLOW)
        elif reward:
            # Win - show money symbols
            self.show_text("$$$", font=self.font_large, color=self.GREEN)
        else:
            # Loss - show loss indicator
            self.show_text("---", font=self.font_large, color=self.RED)
        
        pygame.display.flip()
        time.sleep(duration)
    
    def run_trial(self):
        """Run a single trial matching MATLAB timing"""
        # Check if we've exceeded run duration
        if self.run_start_time:
            elapsed = time.time() - self.run_start_time
            max_duration = self.config['experiment']['run_duration_minutes'] * 60
            if elapsed >= max_duration:
                return False  # Signal to end run
        
        trial_info = {
            'trial_num': self.current_trial,
            'run': self.run_number,
            'run_type': self.run_type,
            'stim_condition': self.stim_condition,
            'trial_in_contingency': self.trial_in_contingency,
            'current_good': self.current_good,
            'contingency_trials': self.contingency_trials
        }
        
        timing = self.config['timing']
        
        # Randomize slot positions for this trial (matching MATLAB)
        self.update_slot_positions()
        trial_info['slot1_position'] = self.slot1_side
        trial_info['slot2_position'] = self.slot2_side
        
        # Fixation
        self.show_fixation(timing['fixation_duration'])
        
        # Show slots and get response
        self.show_slots()
        choice, rt = self.get_response(timing['max_response_time'])
        
        if choice == 'escape':
            self.save_data()
            self.cleanup()
            sys.exit()
        
        # Process response
        if choice is not None:
            # Highlight chosen slot (matching MATLAB behavior)
            self.show_slots(highlight=choice)
            time.sleep(timing['choice_highlight_duration'])
            
            # Determine reward (matching MATLAB logic)
            correct = (choice == self.current_good)
            if correct:
                reward_prob = self.config['task']['win_fraction']
            else:
                reward_prob = 1 - self.config['task']['win_fraction']
            reward = np.random.random() < reward_prob
            
            rt = rt * 1000  # Convert to ms (matching MATLAB)
        else:
            correct = None
            reward = None
            rt = None
        
        # Wait period (fixed 2s for THETA_NIC mode)
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        wait_time = np.random.uniform(
            timing['wait_duration_min'],
            timing['wait_duration_max']
        )
        time.sleep(wait_time)
        
        # Show feedback
        self.show_feedback(reward, timing['outcome_duration'])
        
        # ITI (fixed 250ms for THETA_NIC mode)
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        iti = timing['iti_duration']  # Fixed, no jitter in THETA_NIC
        time.sleep(iti)
        
        # Store trial data
        trial_info.update({
            'choice': choice,
            'rt': rt,
            'correct': correct,
            'reward': reward if reward is not None else None,
            'wait_time': wait_time * 1000,  # Convert to ms
            'iti': iti * 1000,  # Convert to ms
            'timestamp': time.time() - self.experiment_start_time,
            'trial_start_time': time.time() - self.run_start_time
        })
        self.trial_data.append(trial_info)
        
        # Update contingency (matching MATLAB logic)
        self.trial_in_contingency += 1
        if self.trial_in_contingency >= self.contingency_trials:
            # Switch contingency
            self.current_good = 3 - self.current_good  # Switch between 1 and 2
            self.trial_in_contingency = 0
            self.contingency_trials = self._get_contingency_duration()
            print(f"  Contingency reversed at trial {self.current_trial + 1}")
        
        self.current_trial += 1
        return True  # Continue running
    
    def show_instructions(self):
        """Show task instructions"""
        self.screen.fill(self.BLACK)
        
        instructions = [
            f"Two-Armed Bandit Task - Run {self.run_number}",
            f"({self.run_type})",
            "",
            "Choose between two options using keys 1 and 2",
            "(or A for left, L for right)",
            "",
            "One option is better than the other",
            "The better option can change!",
            "Try to win as much as possible",
            "",
            f"This run will last {self.config['experiment']['run_duration_minutes']} minutes",
            "",
            "Press SPACE to begin"
        ]
        
        if self.stim_condition != 'none':
            instructions.insert(2, f"Stimulation: {self.stim_condition.upper()}")
        
        for i, line in enumerate(instructions):
            font = self.font_large if i == 0 else self.font_small
            y_offset = -250 + i * 35
            self.show_text(line, y_offset, font)
        
        pygame.display.flip()
        
        # Wait for space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        self.cleanup()
                        sys.exit()
    
    def show_run_complete(self):
        """Show run completion message with performance"""
        # Calculate performance for this run
        run_trials = [t for t in self.trial_data if t['run'] == self.run_number]
        responses = [t for t in run_trials if t['choice'] is not None]
        
        if responses:
            n_trials = len(run_trials)
            n_responses = len(responses)
            correct_pct = np.mean([t['correct'] for t in responses]) * 100
            reward_pct = np.mean([t['reward'] for t in responses]) * 100
            
            # Count reversals
            reversals = 0
            for i in range(1, len(run_trials)):
                if run_trials[i]['current_good'] != run_trials[i-1]['current_good']:
                    reversals += 1
            
            feedback = [
                f"Run {self.run_number} Complete!",
                "",
                f"Trials completed: {n_trials}",
                f"Response rate: {n_responses}/{n_trials} ({100*n_responses/n_trials:.1f}%)",
                f"Correct choices: {correct_pct:.1f}%",
                f"Rewards earned: {reward_pct:.1f}%",
                f"Contingency reversals: {reversals}",
                "",
                "Data saved successfully",
                "",
                "Press SPACE to exit"
            ]
        else:
            feedback = ["Run complete", "", "Press SPACE to exit"]
        
        self.screen.fill(self.BLACK)
        for i, line in enumerate(feedback):
            font = self.font_large if i == 0 else self.font_small
            y_offset = -200 + i * 35
            self.show_text(line, y_offset, font)
        pygame.display.flip()
        
        # Wait for space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
    
    def save_data(self):
        """Save trial data to CSV"""
        if not self.trial_data:
            return
        
        df = pd.DataFrame(self.trial_data)
        
        # Add subject info to each row
        for key, value in self.subject_info.items():
            if key not in df.columns:
                df[key] = value
        
        # Generate filename matching MATLAB convention
        filename = (
            f"sub-{self.subject_info['subject_id']}_"
            f"ses-{self.subject_info['session']}_"
            f"run-{self.subject_info['run']}_"
            f"task-bandit_{self.subject_info['date']}.csv"
        )
        
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\nData saved to: {filepath}")
    
    def cleanup(self):
        """Clean up and close"""
        self.save_data()
        pygame.quit()
    
    def run(self):
        """Main task execution for a single run"""
        try:
            # Get subject info and run number
            if not self.get_subject_info():
                return
            
            # Setup display
            self.setup_display()
            
            # Show instructions
            self.show_instructions()
            
            # Initialize timing
            self.experiment_start_time = time.time()
            self.run_start_time = time.time()
            
            # Run trials for the duration
            print(f"\nStarting Run {self.run_number} ({self.run_type})")
            print(f"Duration: {self.config['experiment']['run_duration_minutes']} minutes")
            print("Press ESC to abort\n")
            
            trial_count = 0
            while True:
                # Run trial and check if we should continue
                if not self.run_trial():
                    break  # Time limit reached
                
                trial_count += 1
                # Print progress every 10 trials
                if trial_count % 10 == 0:
                    elapsed = time.time() - self.run_start_time
                    print(f"  Trial {trial_count}, Time: {elapsed:.1f}s")
            
            # Show completion message
            print(f"\nRun {self.run_number} complete!")
            print(f"Total trials: {trial_count}")
            self.show_run_complete()
            
        finally:
            self.cleanup()


if __name__ == '__main__':
    task = TwoArmedBanditTask()
    task.run()
