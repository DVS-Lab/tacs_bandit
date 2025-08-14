"""
Local File-Based Starstim Communication Module
Handles communication with Neuroelectrics NIC-2 software via shared files
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class NICError(Exception):
    """Custom exception for NIC communication errors"""
    pass


class LocalNICInterface:
    """
    Local file-based interface for NIC-2 software
    Communicates via command and status files
    """
    
    def __init__(self, command_dir: str = "./nic_commands", test_mode: bool = False):
        """
        Initialize local NIC interface
        
        Parameters:
        -----------
        command_dir : str
            Directory for command and status files
        test_mode : bool
            If True, simulate all operations without real files
        """
        self.command_dir = Path(command_dir)
        self.test_mode = test_mode
        self.connected = False
        self.current_protocol = None
        self.stimulation_active = False
        
        # Protocol mapping for counterbalancing
        self.protocols = {
            'active': 'DLPFC_Active',
            'sham': 'DLPFC_Sham'
        }
        
        # Create command directory
        if not self.test_mode:
            self.command_dir.mkdir(exist_ok=True)
            (self.command_dir / "commands").mkdir(exist_ok=True)
            (self.command_dir / "status").mkdir(exist_ok=True)
            (self.command_dir / "logs").mkdir(exist_ok=True)
        
    def connect(self) -> bool:
        """
        Connect to NIC-2 (initialize file system)
        
        Returns:
        --------
        bool : True if connection successful
        """
        if self.test_mode:
            print(f"NIC: Connected via file system (TEST MODE)")
            self.connected = True
            return True
            
        try:
            # Create connection file
            connection_file = self.command_dir / "status" / "connection.json"
            status = {
                'connected': True,
                'timestamp': datetime.now().isoformat(),
                'python_pid': os.getpid()
            }
            
            with open(connection_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"NIC: Connected via file system at {self.command_dir}")
            self.connected = True
            return True
            
        except Exception as e:
            raise NICError(f"Failed to initialize file system: {e}")
    
    def _write_command(self, command_type: str, data: Dict) -> bool:
        """
        Write command file for NIC-2
        
        Parameters:
        -----------
        command_type : str
            Type of command ('load_protocol', 'start_stim', 'stop_stim', 'marker')
        data : Dict
            Command data
            
        Returns:
        --------
        bool : True if command written successfully
        """
        if self.test_mode:
            print(f"NIC: Command written - {command_type}: {data}")
            return True
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            command_file = self.command_dir / "commands" / f"{timestamp}_{command_type}.json"
            
            command = {
                'type': command_type,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'status': 'pending'
            }
            
            with open(command_file, 'w') as f:
                json.dump(command, f, indent=2)
                
            print(f"NIC: Command file created - {command_file.name}")
            return True
            
        except Exception as e:
            raise NICError(f"Failed to write command file: {e}")
    
    def _wait_for_completion(self, command_type: str, timeout: float = 10.0) -> bool:
        """
        Wait for command completion (optional - for future automation)
        
        Parameters:
        -----------
        command_type : str
            Command type to wait for
        timeout : float
            Timeout in seconds
            
        Returns:
        --------
        bool : True if command completed
        """
        if self.test_mode:
            time.sleep(0.1)  # Simulate brief delay
            return True
            
        # For manual execution, we don't wait - just trust it was done
        # In future, could monitor status files
        return True
    
    def load_protocol(self, protocol_type: str) -> bool:
        """
        Load stimulation protocol in NIC-2
        
        Parameters:
        -----------
        protocol_type : str
            Protocol type ('active' or 'sham')
            
        Returns:
        --------
        bool : True if protocol load command issued
        """
        if protocol_type not in self.protocols:
            raise NICError(f"Unknown protocol type: {protocol_type}")
            
        protocol_name = self.protocols[protocol_type]
        
        try:
            command_data = {
                'protocol_name': protocol_name,
                'protocol_type': protocol_type,
                'instruction': f"Please load protocol '{protocol_name}' in NIC-2"
            }
            
            success = self._write_command('load_protocol', command_data)
            
            if success:
                self.current_protocol = protocol_name
                print(f"NIC: Load protocol command issued - '{protocol_name}' ({protocol_type})")
                
                if not self.test_mode:
                    print("ACTION REQUIRED: Please load this protocol in NIC-2 software")
                
            return success
            
        except Exception as e:
            raise NICError(f"Error issuing load protocol command: {e}")
    
    def start_stimulation(self) -> bool:
        """
        Start loaded stimulation protocol
        
        Returns:
        --------
        bool : True if start command issued
        """
        if not self.current_protocol:
            raise NICError("No protocol loaded")
            
        try:
            command_data = {
                'protocol_name': self.current_protocol,
                'instruction': f"Please start protocol '{self.current_protocol}' in NIC-2",
                'duration_minutes': 6,
                'notes': "6-minute protocol: 30s ramp-up + 5min stim + 30s ramp-down"
            }
            
            success = self._write_command('start_stimulation', command_data)
            
            if success:
                self.stimulation_active = True
                print(f"NIC: Start stimulation command issued - {self.current_protocol}")
                
                if not self.test_mode:
                    print("ACTION REQUIRED: Please start the protocol in NIC-2 software")
                    print("The task will continue - stimulation should run for 6 minutes")
                
            return success
            
        except Exception as e:
            raise NICError(f"Error issuing start stimulation command: {e}")
    
    def stop_stimulation(self) -> bool:
        """
        Stop running stimulation protocol
        
        Returns:
        --------
        bool : True if stop command issued
        """
        if not self.stimulation_active:
            print("NIC: No active stimulation to stop")
            return True
            
        try:
            command_data = {
                'protocol_name': self.current_protocol,
                'instruction': "Please stop current stimulation in NIC-2 (if still running)"
            }
            
            success = self._write_command('stop_stimulation', command_data)
            
            if success:
                self.stimulation_active = False
                print("NIC: Stop stimulation command issued")
                
                if not self.test_mode:
                    print("ACTION REQUIRED: Please stop stimulation in NIC-2 if still running")
                
            return success
            
        except Exception as e:
            raise NICError(f"Error issuing stop stimulation command: {e}")
    
    def send_marker(self, marker: int, label: str = "") -> bool:
        """
        Send event marker (log to file)
        
        Parameters:
        -----------
        marker : int
            Marker value (1-255)
        label : str
            Optional marker label
            
        Returns:
        --------
        bool : True if marker logged successfully
        """
        try:
            marker_data = {
                'marker_code': marker,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'instruction': f"Event marker: {marker} - {label}"
            }
            
            if self.test_mode:
                print(f"NIC: Marker logged - {marker}: {label}")
                return True
            
            # Log to markers file
            markers_file = self.command_dir / "logs" / "markers.jsonl"
            with open(markers_file, 'a') as f:
                json.dump(marker_data, f)
                f.write('\n')
                
            return True
            
        except Exception as e:
            print(f"Warning: Failed to log marker {marker}: {e}")
            return False
    
    def get_status(self) -> Dict[str, str]:
        """
        Get current status
        
        Returns:
        --------
        Dict : Status information
        """
        return {
            'status': 'READY',
            'protocol': self.current_protocol or 'NONE',
            'stimulation': 'ACTIVE' if self.stimulation_active else 'INACTIVE',
            'mode': 'TEST_MODE' if self.test_mode else 'FILE_BASED'
        }
    
    def disconnect(self):
        """Disconnect (cleanup files)"""
        if self.stimulation_active:
            try:
                self.stop_stimulation()
            except:
                pass  # Don't raise error during cleanup
                
        if self.connected:
            if self.test_mode:
                print("NIC: Disconnected (TEST MODE)")
            else:
                try:
                    # Update connection status
                    connection_file = self.command_dir / "status" / "connection.json"
                    if connection_file.exists():
                        status = {
                            'connected': False,
                            'timestamp': datetime.now().isoformat(),
                            'disconnect_reason': 'normal_shutdown'
                        }
                        with open(connection_file, 'w') as f:
                            json.dump(status, f, indent=2)
                    
                    print("NIC: Disconnected - file system cleaned up")
                except:
                    pass
                    
            self.connected = False


class StimulationManager:
    """
    Manages stimulation protocols and counterbalancing for the bandit task
    """
    
    def __init__(self, nic_interface: LocalNICInterface):
        """
        Initialize stimulation manager
        
        Parameters:
        -----------
        nic_interface : LocalNICInterface
            Local NIC interface instance
        """
        self.nic = nic_interface
        self.subject_id = None
        self.session = None
        self.counterbalancing = {}
        
    def setup_counterbalancing(self, subject_id: str, session: str = "1"):
        """
        Setup counterbalancing for subject
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        session : str
            Session number
        """
        self.subject_id = subject_id
        self.session = session
        
        # Generate counterbalancing (can be replaced with file-based system later)
        self.counterbalancing = self._generate_counterbalancing(subject_id)
        
        print(f"Counterbalancing for Subject {subject_id}:")
        for run, condition in self.counterbalancing.items():
            if condition != 'baseline':
                print(f"  Run {run}: {condition}")
    
    def _generate_counterbalancing(self, subject_id: str) -> Dict[int, str]:
        """
        Generate counterbalancing based on subject ID
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
            
        Returns:
        --------
        Dict[int, str] : Run number to condition mapping
        """
        # Convert subject ID to number for counterbalancing
        try:
            subject_num = int(subject_id)
        except ValueError:
            # If subject ID is not numeric, use hash
            subject_num = hash(subject_id) % 1000
        
        # Counterbalancing: even subjects get active first, odd get sham first
        if subject_num % 2 == 0:
            # Even: runs 2-3 are active, 6-7 are sham
            mapping = {
                1: 'baseline',
                2: 'active',
                3: 'active', 
                4: 'baseline',
                5: 'baseline',
                6: 'sham',
                7: 'sham',
                8: 'baseline'
            }
        else:
            # Odd: runs 2-3 are sham, 6-7 are active
            mapping = {
                1: 'baseline',
                2: 'sham',
                3: 'sham',
                4: 'baseline', 
                5: 'baseline',
                6: 'active',
                7: 'active',
                8: 'baseline'
            }
        
        return mapping
    
    def get_run_condition(self, run_number: int) -> str:
        """
        Get stimulation condition for a run
        
        Parameters:
        -----------
        run_number : int
            Run number (1-8)
            
        Returns:
        --------
        str : Condition ('baseline', 'active', 'sham')
        """
        return self.counterbalancing.get(run_number, 'baseline')
    
    def prepare_run(self, run_number: int) -> bool:
        """
        Prepare stimulation for a run
        
        Parameters:
        -----------
        run_number : int
            Run number (1-8)
            
        Returns:
        --------
        bool : True if preparation successful
        """
        condition = self.get_run_condition(run_number)
        
        if condition == 'baseline':
            print(f"Run {run_number}: Baseline (no stimulation)")
            return True
        
        try:
            print(f"Run {run_number}: Preparing {condition} stimulation...")
            success = self.nic.load_protocol(condition)
            if success:
                print(f"Run {run_number}: {condition.upper()} protocol command issued")
            return success
            
        except NICError as e:
            print(f"Error preparing stimulation for run {run_number}: {e}")
            return False
    
    def start_run_stimulation(self, run_number: int) -> bool:
        """
        Start stimulation for a run
        
        Parameters:
        -----------
        run_number : int
            Run number (1-8)
            
        Returns:
        --------
        bool : True if stimulation start command issued
        """
        condition = self.get_run_condition(run_number)
        
        if condition == 'baseline':
            return True  # No stimulation needed
        
        try:
            return self.nic.start_stimulation()
        except NICError as e:
            print(f"Error starting stimulation for run {run_number}: {e}")
            return False
    
    def stop_run_stimulation(self, run_number: int) -> bool:
        """
        Stop stimulation for a run
        
        Parameters:
        -----------
        run_number : int
            Run number (1-8)
            
        Returns:
        --------
        bool : True if stimulation stop command issued
        """
        condition = self.get_run_condition(run_number)
        
        if condition == 'baseline':
            return True  # No stimulation to stop
        
        try:
            return self.nic.stop_stimulation()
        except NICError as e:
            print(f"Error stopping stimulation for run {run_number}: {e}")
            return False
    
    def send_trial_marker(self, marker_type: str, trial_number: int = None) -> bool:
        """
        Send trial event marker
        
        Parameters:
        -----------
        marker_type : str
            Type of marker ('trial_start', 'choice', 'feedback', 'trial_end')
        trial_number : int
            Trial number (optional)
            
        Returns:
        --------
        bool : True if marker logged successfully
        """
        # Define marker codes
        marker_codes = {
            'trial_start': 10,
            'choice': 20,
            'feedback_win': 31,
            'feedback_loss': 32,
            'feedback_miss': 33,
            'trial_end': 40,
            'run_start': 100,
            'run_end': 200
        }
        
        marker_code = marker_codes.get(marker_type, 99)
        label = f"{marker_type}"
        if trial_number is not None:
            label += f"_trial_{trial_number}"
            
        return self.nic.send_marker(marker_code, label)
