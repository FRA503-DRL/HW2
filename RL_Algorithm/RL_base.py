import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch

class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range
        self.discretize_state_weight = discretize_state_weight

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.check_update = True

    
    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
            obs (dict): Observation dictionary containing policy states.

        Returns:
            Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
        """

        """
        Discretize the observation state using clip + bucketize.
        """

        # Number of bins from config
        pose_cart_bin = self.discretize_state_weight[0]
        pose_pole_bin = self.discretize_state_weight[1]
        vel_cart_bin = self.discretize_state_weight[2]
        vel_pole_bin = self.discretize_state_weight[3]

        # Reasonable physical bounds (based on CartPole domain knowledge)
        # Extract observations
        # cart pose range : [-4.8 , 4.8]
        # pole pose range : [-pi , pi]
        # cart vel  range : [-inf , inf]
        # pole vel range  : [-inf , inf]
        pose_cart_bound = 4.0
        pose_pole_bound = float(np.deg2rad(30.0))  # ±30 degrees
        vel_cart_bound = 15.0
        vel_pole_bound = 15.0

        # Get tensor values
        pose_cart_raw, pose_pole_raw, vel_cart_raw, vel_pole_raw = obs['policy'][0, 0], obs['policy'][0, 1], obs['policy'][0, 2], obs['policy'][0, 3]

        # Clip to bounds
        pose_cart_clip = torch.clip(pose_cart_raw, -pose_cart_bound, pose_cart_bound)
        pose_pole_clip = torch.clip(pose_pole_raw, -pose_pole_bound, pose_pole_bound)
        vel_cart_clip = torch.clip(vel_cart_raw, -vel_cart_bound, vel_cart_bound)
        vel_pole_clip = torch.clip(vel_pole_raw, -vel_pole_bound, vel_pole_bound)

        device = pose_cart_clip.device  # get same device

        # Create bin edges with linspace
        pose_cart_grid = torch.linspace(-pose_cart_bound, pose_cart_bound, pose_cart_bin, device=device)
        pose_pole_grid = torch.linspace(-pose_pole_bound, pose_pole_bound, pose_pole_bin, device=device)
        vel_cart_grid = torch.linspace(-vel_cart_bound, vel_cart_bound, vel_cart_bin, device=device)
        vel_pole_grid = torch.linspace(-vel_pole_bound, vel_pole_bound, vel_pole_bin, device=device)

        # Bucketize (convert to discrete bin index)
        pose_cart_dig = torch.bucketize(pose_cart_clip, pose_cart_grid)
        pose_pole_dig = torch.bucketize(pose_pole_clip, pose_pole_grid)
        vel_cart_dig = torch.bucketize(vel_cart_clip, vel_cart_grid)
        vel_pole_dig = torch.bucketize(vel_pole_clip, vel_pole_grid)

        return (int(pose_cart_dig), int(pose_pole_dig), int(vel_cart_dig), int(vel_pole_dig))

    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            # Exploration
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.num_of_action)

            # Exploitation with Double Q
            else:
                qa = self.qa_values[obs_dis]
                qb = self.qb_values[obs_dis]

                # Choose best action from either Q_a or Q_b
                action_from_q1 = np.argmax(qa)
                action_from_q2 = np.argmax(qb)

                best_action = action_from_q1 if qa[action_from_q1] >= qb[action_from_q2] else action_from_q2
                return int(best_action)
        
        else:
            # Epsilon-greedy for Q-Learning, SARSA, MC
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.num_of_action)
            else:
                return np.argmax(self.q_values[obs_dis])

    
    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n]
            n (int): Number of discrete actions
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here =========#
        action_min, action_max = self.action_range  # Extract the range [min, max]
    
        # Avoid division by zero if there's only one discrete action
        if self.num_of_action == 1:
            return torch.tensor(action_min, dtype=torch.float32)

        # Scale discrete action to continuous space
        continuous_action = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)
        
        return torch.tensor(continuous_action, dtype=torch.float32)
        # ======================================#s

    def get_action(self, obs) -> torch.tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)
        
        # return action_tensor, action_idx
    
        #  Ensure shape (1,1,1) → [[ [value] ]]
        return action_tensor.unsqueeze(0).unsqueeze(0), action_idx

    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.final_epsilon)

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Convert tuple keys to strings
        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
        
        # Save model parameters to a JSON file
        if self.control_type == ControlType.MONTE_CARLO:
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys
            }
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            qa_values_str_keys = {str(k): v.tolist() for k, v in self.qa_values.items()}
            qb_values_str_keys = {str(k): v.tolist() for k, v in self.qb_values.items()}
            model_params = {
                'qa_values': qa_values_str_keys,
                'qb_values': qb_values_str_keys
            }
        else:
            model_params = {
                'q_values': q_values_str_keys,
            }
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)

            if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                data_qa_values = data['qa_values']
                data_qb_values = data['qb_values']
                for state, qa_values in data_qa_values.items():
                    tuple_state = tuple(map(float, state.strip('()').split(', ')))
                    self.qa_values[tuple_state] = qa_values.copy()
                for state, qb_values in data_qb_values.items():
                    tuple_state = tuple(map(float, state.strip('()').split(', ')))
                    self.qb_values[tuple_state] = qb_values.copy()
                return self.qa_values

            data_q_values = data['q_values']
            for state, action_values in data_q_values.items():
                state = state.replace('(', '')
                state = state.replace(')', '')
                tuple_state = tuple(map(float, state.split(', ')))
                self.q_values[tuple_state] = action_values.copy()
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    self.qa_values[tuple_state] = action_values.copy()
                    self.qb_values[tuple_state] = action_values.copy()
            if self.control_type == ControlType.MONTE_CARLO:
                data_n_values = data['n_values']
                for state, n_values in data_n_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.n_values[tuple_state] = n_values.copy()

            return self.q_values

