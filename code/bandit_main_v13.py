#!/usr/bin/env python3
"""
LSL-Triggered Two-Armed Bandit Task
Waits for stimulation start signal from NIC-2 before beginning task
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
import threading
import queue

# LSL import with fallback
try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("Warning: pylsl not installed. Install with: pip install pylsl")


class LSLStimulationTrigger:
    """Listens for stimulation start/stop markers from NIC-2 via LSL"""
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize LSL trigger listener
        
        Parameters:
        -----------
        test_mode : bool
            If True, simulate triggers without LSL
        """
        self.test_mode = test_mode
        self.inlet = None
        self.listening = False
        self.marker_queue = queue.Queue()
        self.listener_thread = None
        
        # Marker codes from NIC-2
        self.RAMP_UP_START = 201
        self.RAMP_DOWN_START = 202  
        self.STIMULATION_START = 203
        self.STIMULATION_STOP = 204
        
        # We trigger task start on stimulation start (203) - works for both active and sham
        self.TASK_START_MARKER = 203  # Stimulation start
        
    def connect(self) -> bool:
        """
        Connect to NIC-2 LSL marker stream
        
        Returns:
        --------
        bool : True if connected successfully
        """
        if self.test_mode:
            print("LSL: Connected to marker stream (TEST MODE)")
            return True
            
        if not LSL_AVAILABLE:
            print("LSL: pylsl not available, cannot connect to markers")
            return False
            
        try:
            print("LSL: Looking for NIC-2 marker streams...")
            streams = pylsl.resolve_streams(wait_time=5.0)
            
            if not streams:
                print("LSL: No streams found. Make sure LSL is enabled in NIC-2.")
                return False
            
            # Look for marker streams specifically
            marker_streams = [s for s in streams if s.type() == 'Markers']
            
            if not marker_streams:
                print(f"LSL: Found {len(streams)} streams but no marker streams:")
                for i, s in enumerate(streams):
                    print(f"  Stream {i}: {s.name()} (type: {s.type()})")
                print("LSL: Looking for ANY stream with markers...")
                # Try first available stream
                if streams:
                    self.inlet = pylsl.StreamInlet(streams[0])
                    print(f"LSL: Connected to stream '{streams[0].name()}' (type: {streams[0].type()})")
                else:
                    return False
            else:
                print(f"LSL: Found {len(marker_streams)} marker stream(s)")
                self.inlet = pylsl.StreamInlet(marker_streams[0])
                print(f"LSL: Connected to marker stream '{marker_streams[0].name()}'")
                
            return True
            
        except Exception as e:
            print(f"LSL: Failed to connect to marker stream: {e}")
            return False
    
    def start_listening(self):
        """Start listening for markers in background thread"""
        if self.listening:
            return
            
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        print("LSL: Started listening for stimulation markers")
    
    def stop_listening(self):
        """Stop listening for markers"""
        self.listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
    
    def _listen_loop(self):
        """Background thread loop for listening to markers"""
        while self.listening:
            try:
                if self.test_mode:
                    # In test mode, don't actually listen
                    time.sleep(0.1)
                    continue
                    
                if self.inlet:
                    # Pull marker with timeout
                    marker, timestamp = self.inlet.pull_sample(timeout=0.1)
                    if marker:
                        marker_code = int(marker[0])
                        self.marker_queue.put((marker_code, timestamp))
                        print(f"LSL: Received marker {marker_code}")
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"LSL: Error in listen loop: {e}")
                time.sleep(0.1)
    
    def wait_for_stimulation_start(self, timeout: float = None) -> bool:
        """
        Wait for stimulation ramp-up start marker (for perfect 6-min timing)
        
        Parameters:
        -----------
        timeout : float
            Timeout in seconds (None = wait forever)
            
        Returns:
        --------
        bool : True if stimulation ramp-up detected
        """
        if self.test_mode:
            print("LSL: Simulating stimulation ramp-up start (TEST MODE)")
            print("LSL: Press ENTER to simulate ramp-up start...")
            input()  # Wait for user input in test mode
            return True
            
        start_time = time.time()
        
        while True:
            try:
                # Check for markers
                marker_code, timestamp = self.marker_queue.get(timeout=0.1)
                
                if marker_code == self.TASK_START_MARKER:  # Stimulation start (203)
                    print("LSL: Stimulation START detected! Starting task now!")
                    return True
                elif marker_code == self.STIMULATION_START:
                    print("LSL: Full stimulation detected (ramp-up complete)")
                elif marker_code == self.STIMULATION_STOP:
                    print("LSL: Stimulation STOP detected")
                elif marker_code == self.RAMP_DOWN_START:
                    print("LSL: Stimulation ramp-down detected...")
                    
            except queue.Empty:
                # No markers received, check timeout
                if timeout and (time.time() - start_time) > timeout:
                    print("LSL: Timeout waiting for stimulation ramp-up")
                    return False
                    
                # Show periodic status
                if int(time.time() - start_time) % 10 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"LSL: Still waiting for stimulation ramp-up... ({elapsed}s)")
                    
            except KeyboardInterrupt:
                print("LSL: Interrupted by user")
                return False
    
    def check_for_stimulation_stop(self) -> bool:
        """
        Check if stimulation stop marker received (non-blocking)
        
        Returns:
        --------
        bool : True if stimulation stop detected
        """
        try:
            while True:
                marker_code, timestamp = self.marker_queue.get_nowait()
                if marker_code == self.STIMULATION_STOP:
                    print("LSL: Stimulation STOP detected!")
                    return True
                elif marker_code == self.RAMP_DOWN_START:
                    print("LSL: Stimulation ramp-down detected...")
        except queue.Empty:
            pass
            
        return False


class TwoArmedBanditTask:
    """LSL-triggered two-armed bandit task"""
    
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
        self.task_should_stop = False
        
        # Image tracking
        self.used_flowers = set()
        self.current_flowers = []
        self.flower_images = {}
        self.feedback_images = {}
        
        # Stimulation and LSL components
        self.nic_interface = None
        self.stimulation_manager = None
        self.lsl_trigger = None
        self.stimulation_enabled = self.config.get('stimulation', {}).get('enabled', False)
        
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
        
        # Initialize stimulation if enabled
        if self.stimulation_enabled:
            self._initialize_stimulation()
            
        # Initialize LSL trigger
        self._initialize_lsl_trigger()
        
    def load_config(self):
        """Load configuration from file or use defaults"""
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
        
        # Default configuration
        return {
            'experiment': {
                'mode': 'THETA_NIC',
                'run_duration_minutes': 6,
                'run_types': {
                    '1': 'baseline', '2': 'stimulation', '3': 'stimulation', '4': 'post_stimulation',
                    '5': 'baseline', '6': 'stimulation', '7': 'stimulation', '8': 'post_stimulation'
                }
            },
            'task': {
                'min_trials_same_contingency': 25,
                'contingency_jitter': 4,
                'win_fraction': 0.75
            },
            'timing': {
                'fixation_duration': 0.5, 'max_response_time': 2.0, 'choice_highlight_duration': 0.5,
                'wait_duration_min': 2.0, 'wait_duration_max': 2.0, 'outcome_duration': 1.0, 'iti_duration': 0.25
            },
            'stimulation': {
                'enabled': False, 'test_mode': True, 'communication_type': 'lsl_triggered',
                'command_directory': './nic_commands', 
                'protocols': {'active': 'DLPFC_Active', 'sham': 'DLPFC_Sham'}
            },
            'display': {'window_width': 1024, 'window_height': 768, 'fullscreen': False},
            'paths': {'data_dir': '../data', 'stimuli_dir': '../stimuli/images'}
        }
    
    def _initialize_stimulation(self):
        """Initialize stimulation components"""
        try:
            from local_starstim_module import LocalNICInterface, StimulationManager, NICError
            
            stim_config = self.config['stimulation']
            
            # Initialize local NIC interface (for logging and verification)
            self.nic_interface = LocalNICInterface(
                command_dir=stim_config.get('command_directory', './nic_commands'),
                test_mode=stim_config.get('test_mode', True)
            )
            
            # Initialize stimulation manager
            self.stimulation_manager = StimulationManager(self.nic_interface)
            
            print("Stimulation system initialized (LSL-triggered)")
            if stim_config.get('test_mode', True):
                print("WARNING: Running in stimulation TEST MODE")
            
        except ImportError:
            print("Warning: Stimulation module not found. Continuing without stimulation.")
            self.stimulation_enabled = False
        except Exception as e:
            print(f"Warning: Failed to initialize stimulation: {e}")
            self.stimulation_enabled = False
    
    def _initialize_lsl_trigger(self):
        """Initialize LSL trigger system"""
        stim_config = self.config.get('stimulation', {})
        test_mode = stim_config.get('test_mode', True)
        
        self.lsl_trigger = LSLStimulationTrigger(test_mode=test_mode)
        
        if self.stimulation_enabled:
            success = self.lsl_trigger.connect()
            if success:
                self.lsl_trigger.start_listening()
            else:
                print("Warning: Could not connect to LSL markers. Task will start immediately.")
    
    def load_images(self):
        """Load flower and feedback images"""
        stimuli_dir = Path(self.config['paths']['stimuli_dir'])
        
        # Load flower images
        for i in range(1, 51):
            flower_file = stimuli_dir / f"{i:03d}-flowers.png"
            if flower_file.exists():
                try:
                    image = pygame.image.load(str(flower_file))
                    image = pygame.transform.scale(image, (120, 120))
                    self.flower_images[i] = image
                except pygame.error as e:
                    print(f"Warning: Could not load {flower_file}: {e}")
        
        # Load feedback images
        win_images = []
        for i in range(1, 10):
            win_file = stimuli_dir / f"{i:03d}-win.png"
            if win_file.exists():
                try:
                    image = pygame.image.load(str(win_file))
                    image = pygame.transform.scale(image, (150, 150))
                    win_images.append(image)
                except pygame.error as e:
                    print(f"Warning: Could not load {win_file}: {e}")
        
        loss_images = []
        for i in range(1, 10):
            loss_file = stimuli_dir / f"{i:03d}-loss.png"
            if loss_file.exists():
                try:
                    image = pygame.image.load(str(loss_file))
                    image = pygame.transform.scale(image, (150, 150))
                    loss_images.append(image)
                except pygame.error as e:
                    print(f"Warning: Could not load {loss_file}: {e}")
        
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
        
        selected = np.random.choice(list(available_flowers), size=2, replace=False)
        self.current_flowers = list(selected)
        self.used_flowers.update(selected)
        
        print(f"Selected flowers for run {self.run_number}: {self.current_flowers}")
        return True
    
    def _get_contingency_duration(self):
        """Get the number of trials for current contingency"""
        min_trials = self.config['task']['min_trials_same_contingency']
        jitter = self.config['task']['contingency_jitter']
        return min_trials + np.random.randint(0, jitter + 1)
    
    def get_subject_info(self):
        """Get subject information and setup for LSL-triggered run"""
        print("\n=== LSL-Triggered Two-Armed Bandit Task ===\n")
        
        self.subject_info = {
            'subject_id': input("Subject ID: "),
            'session': input("Session number (default 1): ") or "1",
            'age': input("Age: "),
            'gender': input("Gender (M/F/Other): ")
        }
        
        # Get run number
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
        
        # Get stimulation condition from stimulation manager if available
        if self.stimulation_enabled and self.stimulation_manager:
            self.stimulation_manager.setup_counterbalancing(
                self.subject_info['subject_id'],
                self.subject_info['session']
            )
            self.stim_condition = self.stimulation_manager.get_run_condition(self.run_number)
        else:
            # Fallback logic
            subject_num = int(self.subject_info['subject_id']) if self.subject_info['subject_id'].isdigit() else 0
            if subject_num % 2 == 0:
                if self.run_number in [2, 3]:
                    self.stim_condition = 'active'
                elif self.run_number in [6, 7]:
                    self.stim_condition = 'sham'
                else:
                    self.stim_condition = 'baseline'
            else:
                if self.run_number in [2, 3]:
                    self.stim_condition = 'sham'
                elif self.run_number in [6, 7]:
                    self.stim_condition = 'active'
                else:
                    self.stim_condition = 'baseline'
        
        self.subject_info['run'] = str(self.run_number)
        self.subject_info['run_type'] = self.run_type
        self.subject_info['stim_condition'] = self.stim_condition
        self.subject_info['date'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        print(f"\nRun {self.run_number}: {self.run_type}")
        if self.stim_condition != 'baseline':
            print(f"Stimulation: {self.stim_condition}")
            if self.stimulation_enabled:
                protocol_name = self.stimulation_manager.nic.protocols[self.stim_condition]
                print(f"** RESEARCHER: Please load protocol '{protocol_name}' in NIC-2 **")
        print(f"Duration: {self.config['experiment']['run_duration_minutes']} minutes")
        
        # Create data directory
        self.data_dir = Path(self.config['paths']['data_dir']) / f"sub-{self.subject_info['subject_id']}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        return self.select_flowers_for_run()
    
    def wait_for_stimulation_trigger(self):
        """Wait for stimulation start trigger before beginning task"""
        if not self.stimulation_enabled or self.stim_condition == 'baseline':
            print("No stimulation for this run - starting task immediately")
            return True
            
        if not self.lsl_trigger:
            print("LSL trigger not available - starting task immediately")
            return True
            
        print(f"\n** READY FOR RUN {self.run_number} **")
        print(f"Protocol: {self.stimulation_manager.nic.protocols[self.stim_condition]}")
        print("Waiting for stimulation to start...")
        print("(Start the protocol in NIC-2 when ready)")
        print("Task will begin when stimulation marker (203) is received")
        
        # Wait for stimulation start signal
        success = self.lsl_trigger.wait_for_stimulation_start()
        
        if success:
            print("Stimulation started (marker 203)! Beginning bandit task NOW!")
            
            # Force window focus to ensure participant responses work
            if hasattr(pygame.display, 'get_wm_info'):
                # Bring pygame window to front and focus
                pygame.display.flip()
                pygame.event.set_grab(True)
                pygame.event.set_grab(False)
                
            # Show brief focus instruction
            self.screen.fill(self.BLACK)
            self.show_text("Task Starting!\n\nClick this window if keys don't work", 0, self.font_medium, self.YELLOW)
            pygame.display.flip()
            time.sleep(2)  # Brief pause to let experimenter see message
            
            return True
        else:
            print("No stimulation signal received. Starting task anyway.")
            return True
    
    # [Include all the existing display methods: setup_display, show_text, show_fixation, 
    #  show_slots, get_response, show_feedback, etc. - keeping them exactly the same]
    
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
        
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.slot_size = 120
        self.slot_spacing = 300
        
        self.update_slot_positions()
    
    def update_slot_positions(self):
        """Update slot positions"""
        left_x = self.center_x - self.slot_spacing // 2
        right_x = self.center_x + self.slot_spacing // 2
        
        if np.random.random() < 0.5:
            slot1_x, slot2_x = left_x, right_x
            self.slot1_side = 'left'
            self.slot2_side = 'right'
        else:
            slot1_x, slot2_x = right_x, left_x
            self.slot1_side = 'right'
            self.slot2_side = 'left'
        
        self.slot1_rect = pygame.Rect(slot1_x - self.slot_size // 2, self.center_y - self.slot_size // 2, self.slot_size, self.slot_size)
        self.slot2_rect = pygame.Rect(slot2_x - self.slot_size // 2, self.center_y - self.slot_size // 2, self.slot_size, self.slot_size)
    
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
        pygame.draw.line(self.screen, self.WHITE, (self.center_x - 20, self.center_y), (self.center_x + 20, self.center_y), 3)
        pygame.draw.line(self.screen, self.WHITE, (self.center_x, self.center_y - 20), (self.center_x, self.center_y + 20), 3)
        pygame.display.flip()
        time.sleep(duration)
    
    def show_slots(self, highlight=None):
        """Show the two flower slot machines"""
        self.screen.fill(self.BLACK)
        
        if self.current_flowers[0] in self.flower_images:
            flower1 = self.flower_images[self.current_flowers[0]]
            self.screen.blit(flower1, self.slot1_rect)
            if highlight == 1:
                pygame.draw.rect(self.screen, self.WHITE, self.slot1_rect, 5)
        
        if self.current_flowers[1] in self.flower_images:
            flower2 = self.flower_images[self.current_flowers[1]]
            self.screen.blit(flower2, self.slot2_rect)
            if highlight == 2:
                pygame.draw.rect(self.screen, self.WHITE, self.slot2_rect, 5)
        
        pygame.display.flip()
    
    def get_response(self, max_time):
        """Get keyboard response"""
        start_time = time.time()
        clock = pygame.time.Clock()
        
        while time.time() - start_time < max_time:
            # Check for stimulation stop
            if self.lsl_trigger and self.lsl_trigger.check_for_stimulation_stop():
                self.task_should_stop = True
                return 'stim_stopped', time.time() - start_time
                
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
                    elif event.key == pygame.K_a:
                        return 1 if self.slot1_side == 'left' else 2, time.time() - start_time
                    elif event.key == pygame.K_l:
                        return 1 if self.slot1_side == 'right' else 2, time.time() - start_time
            
            clock.tick(60)
        
        return None, None
    
    def show_feedback(self, reward, duration):
        """Show feedback using images"""
        self.screen.fill(self.BLACK)
        feedback_rect = pygame.Rect(self.center_x - 75, self.center_y - 75, 150, 150)
        
        if reward is None:
            if self.feedback_images['question']:
                self.screen.blit(self.feedback_images['question'], feedback_rect)
            else:
                self.show_text("?", font=self.font_large, color=self.YELLOW)
        elif reward:
            if self.feedback_images['win']:
                win_image = np.random.choice(self.feedback_images['win'])
                self.screen.blit(win_image, feedback_rect)
            else:
                self.show_text("$$$", font=self.font_large, color=self.GREEN)
        else:
            if self.feedback_images['loss']:
                loss_image = np.random.choice(self.feedback_images['loss'])
                self.screen.blit(loss_image, feedback_rect)
            else:
                self.show_text("---", font=self.font_large, color=self.RED)
        
        pygame.display.flip()
        time.sleep(duration)
    
    def run_trial(self):
        """Run a single trial"""
        # Check for task stop conditions
        if self.task_should_stop:
            return False
            
        if self.run_start_time:
            elapsed = time.time() - self.run_start_time
            max_duration = self.config['experiment']['run_duration_minutes'] * 60
            if elapsed >= max_duration:
                return False
        
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
        
        self.update_slot_positions()
        trial_info['slot1_position'] = self.slot1_side
        trial_info['slot2_position'] = self.slot2_side
        
        # Send trial start marker
        if self.stimulation_manager:
            self.stimulation_manager.send_trial_marker('trial_start', self.current_trial)
        
        self.show_fixation(timing['fixation_duration'])
        self.show_slots()
        choice, rt = self.get_response(timing['max_response_time'])
        
        if choice == 'escape':
            self.save_data()
            self.cleanup()
            sys.exit()
        elif choice == 'stim_stopped':
            print("Stimulation stopped - ending task")
            return False
        
        # Send choice marker
        if self.stimulation_manager and choice is not None:
            self.stimulation_manager.send_trial_marker('choice', self.current_trial)
        
        if choice is not None:
            self.show_slots(highlight=choice)
            time.sleep(timing['choice_highlight_duration'])
            
            correct = (choice == self.current_good)
            reward_prob = self.config['task']['win_fraction'] if correct else 1 - self.config['task']['win_fraction']
            reward = np.random.random() < reward_prob
            rt = rt * 1000
        else:
            correct = None
            reward = None
            rt = None
        
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        wait_time = np.random.uniform(timing['wait_duration_min'], timing['wait_duration_max'])
        time.sleep(wait_time)
        
        # Send feedback marker
        if self.stimulation_manager:
            if reward is None:
                self.stimulation_manager.send_trial_marker('feedback_miss', self.current_trial)
            elif reward:
                self.stimulation_manager.send_trial_marker('feedback_win', self.current_trial)
            else:
                self.stimulation_manager.send_trial_marker('feedback_loss', self.current_trial)
        
        self.show_feedback(reward, timing['outcome_duration'])
        
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        iti = timing['iti_duration']
        time.sleep(iti)
        
        trial_info.update({
            'choice': choice,
            'rt': rt,
            'correct': correct,
            'reward': reward if reward is not None else None,
            'wait_time': wait_time * 1000,
            'iti': iti * 1000,
            'timestamp': time.time() - self.experiment_start_time,
            'trial_start_time': time.time() - self.run_start_time
        })
        self.trial_data.append(trial_info)
        
        # Update contingency
        self.trial_in_contingency += 1
        if self.trial_in_contingency >= self.contingency_trials:
            self.current_good = 3 - self.current_good
            self.trial_in_contingency = 0
            self.contingency_trials = self._get_contingency_duration()
            print(f"  Contingency reversed at trial {self.current_trial + 1}")
        
        self.current_trial += 1
        return True
    
    def show_waiting_screen(self):
        """Show waiting for stimulation screen"""
        self.screen.fill(self.BLACK)
        
        if self.stim_condition == 'baseline':
            waiting_text = [
                f"Two-Armed Bandit Task - Run {self.run_number}",
                f"({self.run_type})",
                "",
                "No stimulation for this run",
                "",
                "Press SPACE to begin task"
            ]
        else:
            protocol_name = self.stimulation_manager.nic.protocols[self.stim_condition]
            waiting_text = [
                f"Two-Armed Bandit Task - Run {self.run_number}",
                f"({self.run_type} - {self.stim_condition})",
                "",
                f"Protocol: {protocol_name}",
                "",
                "Waiting for stimulation ramp-up...",
                "",
                "Start the 6-minute protocol in NIC-2 when ready",
                "(Task begins when ramp-up starts)"
            ]
        
        for i, line in enumerate(waiting_text):
            font = self.font_large if i == 0 else self.font_small
            y_offset = -150 + i * 40
            self.show_text(line, y_offset, font)
        
        pygame.display.flip()
    
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
            "Task will begin when stimulation starts"
        ]
        
        if self.stim_condition != 'baseline':
            instructions.insert(2, f"Stimulation: {self.stim_condition.upper()}")
        
        for i, line in enumerate(instructions):
            font = self.font_large if i == 0 else self.font_small
            y_offset = -250 + i * 35
            self.show_text(line, y_offset, font)
        
        pygame.display.flip()
        
        # Wait for space (only for baseline runs)
        if self.stim_condition == 'baseline':
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
        """Show run completion message"""
        run_trials = [t for t in self.trial_data if t['run'] == self.run_number]
        responses = [t for t in run_trials if t['choice'] is not None]
        
        if responses:
            n_trials = len(run_trials)
            n_responses = len(responses)
            correct_pct = np.mean([t['correct'] for t in responses]) * 100
            reward_pct = np.mean([t['reward'] for t in responses]) * 100
            
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
        
        for key, value in self.subject_info.items():
            if key not in df.columns:
                df[key] = value
        
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
        
        if self.lsl_trigger:
            self.lsl_trigger.stop_listening()
        
        if self.stimulation_enabled and self.nic_interface:
            try:
                self.nic_interface.disconnect()
            except Exception as e:
                print(f"Warning: Error disconnecting stimulation: {e}")
        
        pygame.quit()
    
    def run(self):
        """Main LSL-triggered task execution"""
        try:
            # Get subject info
            if not self.get_subject_info():
                return
            
            # Setup display
            self.setup_display()
            
            # Show instructions
            self.show_instructions()
            
            # For stimulation runs, wait for LSL trigger
            if self.stim_condition != 'baseline':
                self.show_waiting_screen()
                if not self.wait_for_stimulation_trigger():
                    print("Failed to receive stimulation trigger")
                    return
            
            # Initialize timing
            self.experiment_start_time = time.time()
            self.run_start_time = time.time()
            
            # Log stimulation markers
            if self.stimulation_manager:
                self.stimulation_manager.send_trial_marker('run_start')
            
            # Run trials
            print(f"\nStarting Run {self.run_number} ({self.run_type})")
            if self.stim_condition != 'baseline':
                print(f"Stimulation: {self.stim_condition.upper()}")
            print(f"Duration: {self.config['experiment']['run_duration_minutes']} minutes")
            print("Press ESC to abort\n")
            
            trial_count = 0
            while True:
                try:
                    if not self.run_trial():
                        break
                    
                    trial_count += 1
                    if trial_count % 10 == 0:
                        elapsed = time.time() - self.run_start_time
                        print(f"  Trial {trial_count}, Time: {elapsed:.1f}s")
                        
                except Exception as e:
                    print(f"Error during trial {trial_count}: {e}")
                    break
            
            # End markers
            if self.stimulation_manager:
                self.stimulation_manager.send_trial_marker('run_end')
            
            print(f"\nRun {self.run_number} complete!")
            print(f"Total trials: {trial_count}")
            self.show_run_complete()
            
        except KeyboardInterrupt:
            print("\nRun interrupted by user")
        finally:
            self.cleanup()


if __name__ == '__main__':
    task = TwoArmedBanditTask()
    task.run()