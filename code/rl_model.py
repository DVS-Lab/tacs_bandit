"""
Reinforcement Learning Models for Two-Armed Bandit Task
Implements Rescorla-Wagner, Win-Stay-Lose-Shift, and Choice Kernel models
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class RLModel:
    """Base class for reinforcement learning models"""
    
    def __init__(self, n_options: int = 2):
        self.n_options = n_options
        self.values = None
        self.choice_history = []
        self.reward_history = []
        self.value_history = []
        self.rpe_history = []
        
    def reset(self):
        """Reset model to initial state"""
        self.choice_history = []
        self.reward_history = []
        self.value_history = []
        self.rpe_history = []
        
    def get_choice_probability(self, values: np.ndarray) -> np.ndarray:
        """Get probability of choosing each option"""
        raise NotImplementedError
        
    def update(self, choice: int, reward: float):
        """Update model based on choice and reward"""
        raise NotImplementedError
        
    def simulate_choice(self, values: np.ndarray) -> int:
        """Simulate a choice based on current values"""
        probs = self.get_choice_probability(values)
        return np.random.choice(self.n_options, p=probs) + 1  # 1-indexed


class RescorlaWagner(RLModel):
    """Rescorla-Wagner model with softmax choice rule"""
    
    def __init__(self, 
                 learning_rate: float = 0.3,
                 inverse_temperature: float = 3.0,
                 initial_value: float = 0.5,
                 n_options: int = 2):
        """
        Initialize RW model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (alpha) for value updates (0-1)
        inverse_temperature : float
            Inverse temperature (beta) for softmax choice
        initial_value : float
            Initial value for all options
        n_options : int
            Number of choice options
        """
        super().__init__(n_options)
        self.learning_rate = learning_rate
        self.inverse_temperature = inverse_temperature
        self.initial_value = initial_value
        self.values = np.ones(n_options) * initial_value
        
    def reset(self):
        """Reset model to initial state"""
        super().reset()
        self.values = np.ones(self.n_options) * self.initial_value
        
    def get_choice_probability(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate choice probabilities using softmax
        
        Parameters:
        -----------
        values : np.ndarray, optional
            Values to use for calculation. If None, use current values
            
        Returns:
        --------
        np.ndarray : Probability of choosing each option
        """
        if values is None:
            values = self.values
            
        # Softmax calculation with numerical stability
        exp_values = np.exp(self.inverse_temperature * values)
        return exp_values / np.sum(exp_values)
    
    def update(self, choice: int, reward: float):
        """
        Update values based on choice and reward
        
        Parameters:
        -----------
        choice : int
            Chosen option (1-indexed)
        reward : float
            Received reward (0 or 1)
        """
        if choice is None:
            return
            
        # Convert to 0-indexed
        choice_idx = choice - 1
        
        # Calculate reward prediction error
        rpe = reward - self.values[choice_idx]
        
        # Update chosen value
        self.values[choice_idx] += self.learning_rate * rpe
        
        # Store history
        self.choice_history.append(choice)
        self.reward_history.append(reward)
        self.value_history.append(self.values.copy())
        self.rpe_history.append(rpe)
        
    def get_log_likelihood(self, choices: List[int], rewards: List[float]) -> float:
        """
        Calculate log likelihood of choices given parameters
        
        Parameters:
        -----------
        choices : List[int]
            Sequence of choices (1-indexed)
        rewards : List[float]
            Sequence of rewards
            
        Returns:
        --------
        float : Negative log likelihood (for minimization)
        """
        self.reset()
        log_lik = 0
        
        for choice, reward in zip(choices, rewards):
            if choice is not None:
                probs = self.get_choice_probability()
                log_lik += np.log(probs[choice - 1])
                self.update(choice, reward)
                
        return -log_lik  # Return negative for minimization


class WinStayLoseShift(RLModel):
    """Win-Stay-Lose-Shift model"""
    
    def __init__(self,
                 win_stay_prob: float = 0.8,
                 lose_shift_prob: float = 0.8,
                 n_options: int = 2):
        """
        Initialize WSLS model
        
        Parameters:
        -----------
        win_stay_prob : float
            Probability of staying after win
        lose_shift_prob : float
            Probability of shifting after loss
        n_options : int
            Number of choice options
        """
        super().__init__(n_options)
        self.win_stay_prob = win_stay_prob
        self.lose_shift_prob = lose_shift_prob
        self.last_choice = None
        self.last_reward = None
        
    def reset(self):
        """Reset model to initial state"""
        super().reset()
        self.last_choice = None
        self.last_reward = None
        
    def get_choice_probability(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate choice probabilities based on WSLS strategy
        
        Returns:
        --------
        np.ndarray : Probability of choosing each option
        """
        probs = np.ones(self.n_options) / self.n_options  # Default: equal probability
        
        if self.last_choice is not None and self.last_reward is not None:
            if self.last_reward > 0:
                # Win: stay with higher probability
                probs = np.ones(self.n_options) * (1 - self.win_stay_prob) / (self.n_options - 1)
                probs[self.last_choice - 1] = self.win_stay_prob
            else:
                # Loss: shift with higher probability
                probs = np.ones(self.n_options) * self.lose_shift_prob / (self.n_options - 1)
                probs[self.last_choice - 1] = 1 - self.lose_shift_prob
                
        return probs
    
    def update(self, choice: int, reward: float):
        """
        Update model based on choice and reward
        
        Parameters:
        -----------
        choice : int
            Chosen option (1-indexed)
        reward : float
            Received reward (0 or 1)
        """
        if choice is None:
            return
            
        self.last_choice = choice
        self.last_reward = reward
        
        # Store history
        self.choice_history.append(choice)
        self.reward_history.append(reward)


class ChoiceKernel(RLModel):
    """Choice Kernel model that tracks choice history bias"""
    
    def __init__(self,
                 decay_rate: float = 0.9,
                 inverse_temperature: float = 1.0,
                 n_options: int = 2):
        """
        Initialize Choice Kernel model
        
        Parameters:
        -----------
        decay_rate : float
            Decay rate for choice history (0-1)
        inverse_temperature : float
            Inverse temperature for choice bias
        n_options : int
            Number of choice options
        """
        super().__init__(n_options)
        self.decay_rate = decay_rate
        self.inverse_temperature = inverse_temperature
        self.choice_kernel = np.zeros(n_options)
        
    def reset(self):
        """Reset model to initial state"""
        super().reset()
        self.choice_kernel = np.zeros(self.n_options)
        
    def get_choice_probability(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate choice probabilities based on choice kernel
        
        Returns:
        --------
        np.ndarray : Probability of choosing each option
        """
        # Softmax with choice kernel
        exp_values = np.exp(self.inverse_temperature * self.choice_kernel)
        return exp_values / np.sum(exp_values)
    
    def update(self, choice: int, reward: float):
        """
        Update choice kernel based on choice
        
        Parameters:
        -----------
        choice : int
            Chosen option (1-indexed)
        reward : float
            Received reward (0 or 1) - not used in pure CK model
        """
        if choice is None:
            return
            
        # Decay all values
        self.choice_kernel *= self.decay_rate
        
        # Increment chosen option
        self.choice_kernel[choice - 1] += 1
        
        # Store history
        self.choice_history.append(choice)
        self.reward_history.append(reward)


class HybridRWCK(RLModel):
    """Hybrid model combining Rescorla-Wagner and Choice Kernel"""
    
    def __init__(self,
                 rw_learning_rate: float = 0.3,
                 rw_inverse_temp: float = 3.0,
                 ck_decay_rate: float = 0.9,
                 ck_inverse_temp: float = 1.0,
                 initial_value: float = 0.5,
                 n_options: int = 2):
        """
        Initialize hybrid RW+CK model
        """
        super().__init__(n_options)
        
        # RW parameters
        self.rw_learning_rate = rw_learning_rate
        self.rw_inverse_temp = rw_inverse_temp
        self.initial_value = initial_value
        self.values = np.ones(n_options) * initial_value
        
        # CK parameters
        self.ck_decay_rate = ck_decay_rate
        self.ck_inverse_temp = ck_inverse_temp
        self.choice_kernel = np.zeros(n_options)
        
    def reset(self):
        """Reset model to initial state"""
        super().reset()
        self.values = np.ones(self.n_options) * self.initial_value
        self.choice_kernel = np.zeros(self.n_options)
        
    def get_choice_probability(self, values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate choice probabilities combining RW values and choice kernel
        
        Returns:
        --------
        np.ndarray : Probability of choosing each option
        """
        if values is None:
            values = self.values
            
        # Combine RW and CK contributions
        combined = (self.rw_inverse_temp * values + 
                   self.ck_inverse_temp * self.choice_kernel)
        
        # Softmax
        exp_values = np.exp(combined)
        return exp_values / np.sum(exp_values)
    
    def update(self, choice: int, reward: float):
        """
        Update both RW values and choice kernel
        
        Parameters:
        -----------
        choice : int
            Chosen option (1-indexed)
        reward : float
            Received reward (0 or 1)
        """
        if choice is None:
            return
            
        # Convert to 0-indexed
        choice_idx = choice - 1
        
        # Update RW values
        rpe = reward - self.values[choice_idx]
        self.values[choice_idx] += self.rw_learning_rate * rpe
        
        # Update choice kernel
        self.choice_kernel *= self.ck_decay_rate
        self.choice_kernel[choice_idx] += 1
        
        # Store history
        self.choice_history.append(choice)
        self.reward_history.append(reward)
        self.value_history.append(self.values.copy())
        self.rpe_history.append(rpe)