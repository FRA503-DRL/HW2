from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
    
    def update(self, done, obs, action, reward_value):
        """
        Update Q-values using Monte Carlo.
        
        This method applies the Monte Carlo update rule using first-visit MC updates.
        """
        obs = self.discretize_state(obs)

        # 1. Store transitions during the episode
        if not done:
            self.obs_hist.append(obs)
            self.action_hist.append(action)
            self.reward_hist.append(reward_value)
            return  # Do not proceed until the episode ends

        # 2. Compute Returns (G) in reverse order
        G = 0  
        return_list = [0] * len(self.reward_hist)  # Pre-allocate space

        for i in reversed(range(len(self.reward_hist))):
            G = self.reward_hist[i] + self.discount_factor * G
            return_list[i] = G

        # 3. First-Visit MC Update
        visited_states = set()

        for t in range(len(self.obs_hist)):
            state = self.obs_hist[t]
            action = self.action_hist[t]

            if (state, action) not in visited_states:
                visited_states.add((state, action))  # Mark as visited

                # Increment visit count
                self.n_values[state][action] += 1  

                # Compute step size (alpha = 1/N)
                alpha = 1 / self.n_values[state][action]

                # Update Q-value using incremental mean formula
                self.q_values[state][action] += alpha * (return_list[t] - self.q_values[state][action])

        # 4. Reset episode history
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()