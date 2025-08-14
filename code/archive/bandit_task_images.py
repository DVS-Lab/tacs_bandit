#!/usr/bin/env python3
"""
Two-Armed Bandit Task - Pygame Implementation
Matches MATLAB/Neurostim version timing and structure
Updated to use flower images for lotteries and feedback images
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
        
        # Image tracking
        self.used_flowers = set()  # Track used flower images across runs
        self.current_flowers = []  # Current run's flower images
        self.flower_images = {}    # Loaded flower images
        self.feedback_images = {}  # Loaded feedback images
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 100, 100)
        self.BLUE = (100, 100, 255)
        self.GREEN = (100, 255, 100)
        self.YELLOW = (255, 255, 100)
        self.GRAY = (128, 128, 128)
        
        # Load images
        self.load_images()
        
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
                'data_dir': './data',
                'stimuli_dir': './bandit-task/stimuli/images'
            }
        }
    
    def load_images(self):
        """Load flower and feedback images"""
        stimuli_dir = Path(self.config['paths']['stimuli_dir'])
        
        # Load flower images (001-flowers.png to 050-flowers.png)
        for i in range(1, 51):
            flower_file = stimuli_dir / f"{i:03d}-flowers.png"
            if flower_file.exists():
                try:
                    image = pygame.image.load(str(flower_file))
                    # Scale image to appropriate size (120x120 to match slot_size)
                    image = pygame.transform.scale(image, (120, 120))
                    self.flower_images[i] = image
                except pygame.error as e:
                    print(f"Warning: Could not load {flower_file}: {e}")
        
        # Load feedback images
        # Win images (001-win.png to 009-win.png)
        win_images = []
        for i in range(1, 10):
            win_file = stimuli_dir / f"{i:03d}-win.png"
            if win_file.exists():
                try:
                    image = pygame.image.load(str(win_file))
                    # Scale to appropriate feedback size
                    image = pygame.transform.scale(image, (150, 150))
                    win_images.append(image)
                except pygame.error as e:
                    print(f"Warning: Could not load {win_file}: {e}")
        
        # Loss images (001-loss.png to 009-loss.png)
        loss_images = []
        for i in range(1, 10):
            loss_file = stimuli_dir / f"{i:03d}-loss.png"
            if loss_file.exists():
                try:
                    image = pygame.image.load(str(loss_file))
                    # Scale to appropriate feedback size
                    image = pygame.transform.scale(image, (150, 150))
                    loss_images.append(image)
                except pygame.error as e:
                    print(f"Warning: Could not load {loss_file}: {e}")
        
        # Question mark image
        question_file = stimuli_dir / "question-mark.png"
        question_image = None
        if question_file.exists():
            try:
                question_image = pygame.image.load(str(question_file))
                question_image = pygame.transform.scale(question_image, (150, 150))
            except pygame.error as e:
                print(f"Warning: Could not load {question_file}: {e}")
        
        self.feedback_images = {
            'win': win_images,
            'loss': loss_images,
            'question': question_image
        }
        
        print(f"Loaded {len(self.flower_images)} flower images")
        print(f"Loaded {len(win_images)} win images, {len(loss_images)} loss images")
        if question_image:
            print("Loaded question mark image")
    
    def select_flowers_for_run(self):
        """Select two unique flower images for this run"""
        available_flowers = set(self.flower_images.keys()) - self.used_flowers
        
        if len(available_flowers) < 2:
            print("Warning: Running out of unique flowers, resetting used flowers")
            self.used_flowers = set()
            available_flowers = set(self.flower_images.keys())
        
        if len(available_flowers) < 2:
            print("Error: Not enough flower images available")
            return False
        
        # Randomly select 2 flowers
        selected = np.random.choice(list(available_flowers), size=2, replace=False)
        self.current_flowers = list(selected)
        self.used_flowers.update(selected)
        
        print(f"Selected flowers for run {self.run_number}: {self.current_flowers}")
        return True
    
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
        
        # Select flowers for this run
        return self.select_flowers_for_run()
    
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
        """Show the two flower slot machines"""
        self.screen.fill(self.BLACK)
        
        # Draw slot 1 with flower image
        if self.current_flowers[0] in self.flower_images:
            flower1 = self.flower_images[self.current_flowers[0]]
            self.screen.blit(flower1, self.slot1_rect)
            
            # Add highlight if selected
            if highlight == 1:
                pygame.draw.rect(self.screen, self.WHITE, self.slot1_rect, 5)
        
        # Draw slot 2 with flower image
        if self.current_flowers[1] in self.flower_images:
            flower2 = self.flower_images[self.current_flowers[1]]
            self.screen.blit(flower2, self.slot2_rect)
            
            # Add highlight if selected
            if highlight == 2:
                pygame.draw.rect(self.screen, self.WHITE, self.slot2_rect, 5)
        
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
        """Show feedback using images"""
        self.screen.fill(self.BLACK)
        
        feedback_rect = pygame.Rect(
            self.center_x - 75, self.center_y - 75, 150, 150
        )
        
        if reward is None:
            # No response - show question mark
            if self.feedback_images['question']:
                self.screen.blit(self.feedback_images['question'], feedback_rect)
            else:
                # Fallback to text
                self.show_text("?", font=self.font_large, color=self.YELLOW)
        elif reward:
            # Win - show random win image
            if self.feedback_images['win']:
                win_image = np.random.choice(self.feedback_images['win'])
                self.screen.blit(win_image, feedback_rect)
            else:
                # Fallback to text
                self.show_text("$$$", font=self.font_large, color=self.GREEN)
        else:
            # Loss - show random loss image
            if self.feedback_images['loss']:
                loss_image = np.random.choice(self.feedback_images['loss'])
                self.screen.blit(loss_image, feedback_rect)
            else:
                # Fallback to text
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
            'contingency_trials': self.contingency_trials,
            'flower1': self.current_flowers[0],
            'flower2': self.current_flowers[1]
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
            "Choose between two flowers using keys 1 and 2",
            "(or A for left, L for right)",
            "",
            "One flower is better than the other",
            "The better flower can change!",
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
                f"Flowers used: {self.current_flowers}",
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
