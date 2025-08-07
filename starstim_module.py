"""
Starstim Communication Module
Handles communication with Neuroelectrics Starstim device
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class StarstimController:
    """Controller for Neuroelectrics Starstim device"""
    
    def __init__(self, config: Dict, test_mode: bool = False):
        """
        Initialize Starstim controller
        
        Parameters:
        -----------
        config : Dict
            Stimulation configuration parameters
        test_mode : bool
            If True, run in test mode without actual device connection
        """
        self.config = config
        self.test_mode = test_mode
        self.connected = False
        self.is_stimulating = False
        self.stimulation_start_time = None
        
        # Electrode configurations (matching MATLAB setup)
        self.electrode_configs = {
            'lpfc': np.array([0, 0, 0, 0, 250, 250, 250, 750]),  # µA
            'rtpj': np.array([750, 250, 250, 250, 0, 0, 0, 0]),  # µA
            'sham': np.array([0, 0, 0, 0, 0, 0, 0, 0])  # No stimulation
        }
        
        # Phase configurations for tACS
        self.phase_config = np.array([0, 180, 180, 180, 180, 180, 180, 0])  # degrees
        
    def connect(self) -> bool:
        """
        Connect to Starstim device
        
        Returns:
        --------
        bool : True if connection successful
        """
        if self.test_mode:
            print("Starstim: Running in TEST MODE (no actual stimulation)")
            self.connected = True
            return True
            
        try:
            # TODO: Implement actual Starstim connection
            # This would involve:
            # 1. Serial/TCP connection to NIC software
            # 2. Device initialization
            # 3. Safety checks
            
            warnings.warn("Starstim connection not implemented. Running in offline mode.")
            self.connected = False
            return False
            
        except Exception as e:
            print(f"Failed to connect to Starstim: {e}")
            self.connected = False
            return False
    
    def configure_stimulation(self, 
                            protocol: str,
                            frequency: Optional[float] = None,
                            duration: Optional[float] = None) -> bool:
        """
        Configure stimulation parameters
        
        Parameters:
        -----------
        protocol : str
            Stimulation protocol ('lpfc', 'rtpj', 'sham')
        frequency : float, optional
            Stimulation frequency in Hz (for tACS)
        duration : float, optional
            Stimulation duration in seconds
            
        Returns:
        --------
        bool : True if configuration successful
        """
        if protocol not in self.electrode_configs:
            print(f"Unknown protocol: {protocol}")
            return False
            
        self.current_protocol = protocol
        self.current_amplitudes = self.electrode_configs[protocol]
        
        if frequency is not None:
            self.frequency = frequency
        else:
            self.frequency = self.config.get('frequency', 6.0)  # Default 6 Hz
            
        if duration is not None:
            self.duration = duration
            
        if self.test_mode:
            print(f"Configured: {protocol}, {self.frequency} Hz")
            
        return True
    
    def start_stimulation(self, ramp_time: float = 2.0) -> bool:
        """
        Start stimulation with ramp up
        
        Parameters:
        -----------
        ramp_time : float
            Ramp up time in seconds
            
        Returns:
        --------
        bool : True if stimulation started successfully
        """
        if not self.connected and not self.test_mode:
            print("Cannot start stimulation: Not connected")
            return False
            
        if self.is_stimulating:
            print("Stimulation already active")
            return False
            
        if self.test_mode:
            print(f"Starting {self.current_protocol} stimulation (TEST MODE)")
            print(f"  Frequency: {self.frequency} Hz")
            print(f"  Amplitudes: {self.current_amplitudes} µA")
            print(f"  Ramp time: {ramp_time} s")
            
        # TODO: Send actual start command to Starstim
        
        self.is_stimulating = True
        self.stimulation_start_time = time.time()
        
        # Simulate ramp time
        time.sleep(ramp_time)
        
        return True
    
    def stop_stimulation(self, ramp_time: float = 2.0) -> bool:
        """
        Stop stimulation with ramp down
        
        Parameters:
        -----------
        ramp_time : float
            Ramp down time in seconds
            
        Returns:
        --------
        bool : True if stimulation stopped successfully
        """
        if not self.is_stimulating:
            print("No active stimulation to stop")
            return False
            
        if self.test_mode:
            duration = time.time() - self.stimulation_start_time
            print(f"Stopping stimulation (TEST MODE)")
            print(f"  Total duration: {duration:.1f} s")
            print(f"  Ramp time: {ramp_time} s")
            
        # TODO: Send actual stop command to Starstim
        
        # Simulate ramp time
        time.sleep(ramp_time)
        
        self.is_stimulating = False
        self.stimulation_start_time = None
        
        return True
    
    def send_trigger(self, trigger_value: int) -> bool:
        """
        Send trigger/marker to Starstim for EEG synchronization
        
        Parameters:
        -----------
        trigger_value : int
            Trigger value to send
            
        Returns:
        --------
        bool : True if trigger sent successfully
        """
        if not self.connected and not self.test_mode:
            return False
            
        if self.test_mode:
            print(f"Trigger sent: {trigger_value}")
            
        # TODO: Send actual trigger to Starstim
        
        return True
    
    def get_impedances(self) -> Dict[int, float]:
        """
        Get electrode impedances
        
        Returns:
        --------
        Dict[int, float] : Impedance values for each electrode
        """
        if self.test_mode:
            # Return simulated impedances
            return {i: np.random.uniform(1, 10) for i in range(8)}
            
        # TODO: Get actual impedances from Starstim
        
        return {}
    
    def disconnect(self):
        """Disconnect from Starstim device"""
        if self.is_stimulating:
            self.stop_stimulation()
            
        if self.connected:
            if self.test_mode:
                print("Disconnected from Starstim (TEST MODE)")
            else:
                # TODO: Actual disconnection
                pass
                
            self.connected = False


class StimulationProtocol:
    """Manages stimulation protocols for the experiment"""
    
    def __init__(self, controller: StarstimController):
        """
        Initialize stimulation protocol manager
        
        Parameters:
        -----------
        controller : StarstimController
            Starstim controller instance
        """
        self.controller = controller
        self.protocol_sequence = []
        self.current_block = 0
        
    def setup_blocked_design(self, 
                            blocks: List[str],
                            trials_per_block: int):
        """
        Setup blocked stimulation design
        
        Parameters:
        -----------
        blocks : List[str]
            Sequence of stimulation conditions (e.g., ['sham', 'lpfc', 'lpfc', 'sham'])
        trials_per_block : int
            Number of trials per block
        """
        self.protocol_sequence = blocks
        self.trials_per_block = trials_per_block
        self.current_block = 0
        
    def start_block(self):
        """Start stimulation for current block"""
        if self.current_block >= len(self.protocol_sequence):
            return False
            
        protocol = self.protocol_sequence[self.current_block]
        self.controller.configure_stimulation(protocol)
        self.controller.start_stimulation()
        
        return True
    
    def end_block(self):
        """End stimulation for current block"""
        self.controller.stop_stimulation()
        self.current_block += 1
    
    def get_current_protocol(self) -> str:
        """Get current stimulation protocol"""
        if self.current_block < len(self.protocol_sequence):
            return self.protocol_sequence[self.current_block]
        return 'none'


class NICSoftwareInterface:
    """
    Interface for NIC (Neuroelectrics Instrument Controller) software
    Handles communication when using NIC-controlled protocols
    """
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize NIC interface
        
        Parameters:
        -----------
        test_mode : bool
            If True, run in test mode without actual NIC connection
        """
        self.test_mode = test_mode
        self.connected = False
        self.current_protocol = None
        
    def connect(self, host: str = 'localhost', port: int = 1234) -> bool:
        """
        Connect to NIC software via TCP/IP
        
        Parameters:
        -----------
        host : str
            NIC host address
        port : int
            NIC port number
            
        Returns:
        --------
        bool : True if connection successful
        """
        if self.test_mode:
            print(f"NIC: Connected to {host}:{port} (TEST MODE)")
            self.connected = True
            return True
            
        # TODO: Implement actual TCP/IP connection to NIC
        
        warnings.warn("NIC connection not implemented")
        return False
    
    def load_protocol(self, protocol_name: str) -> bool:
        """
        Load stimulation protocol in NIC
        
        Parameters:
        -----------
        protocol_name : str
            Name of protocol file in NIC
            
        Returns:
        --------
        bool : True if protocol loaded successfully
        """
        if not self.connected:
            return False
            
        if self.test_mode:
            print(f"NIC: Loaded protocol '{protocol_name}'")
            self.current_protocol = protocol_name
            return True
            
        # TODO: Send protocol load command to NIC
        
        return False
    
    def start_protocol(self) -> bool:
        """
        Start loaded protocol
        
        Returns:
        --------
        bool : True if protocol started successfully
        """
        if not self.current_protocol:
            print("No protocol loaded")
            return False
            
        if self.test_mode:
            print(f"NIC: Started protocol '{self.current_protocol}'")
            return True
            
        # TODO: Send start command to NIC
        
        return False
    
    def stop_protocol(self) -> bool:
        """
        Stop running protocol
        
        Returns:
        --------
        bool : True if protocol stopped successfully
        """
        if self.test_mode:
            print(f"NIC: Stopped protocol")
            return True
            
        # TODO: Send stop command to NIC
        
        return False
    
    def send_marker(self, marker: int, label: str = "") -> bool:
        """
        Send event marker to NIC for EEG synchronization
        
        Parameters:
        -----------
        marker : int
            Marker value
        label : str
            Optional marker label
            
        Returns:
        --------
        bool : True if marker sent successfully
        """
        if not self.connected:
            return False
            
        if self.test_mode:
            print(f"NIC: Marker {marker} - {label}")
            return True
            
        # TODO: Send marker to NIC
        
        return False
    
    def disconnect(self):
        """Disconnect from NIC software"""
        if self.connected:
            if self.test_mode:
                print("NIC: Disconnected")
            else:
                # TODO: Close TCP/IP connection
                pass
                
            self.connected = False
