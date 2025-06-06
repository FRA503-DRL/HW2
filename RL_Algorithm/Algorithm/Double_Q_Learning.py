from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self, obs, action, reward, next_obs):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """
        obs = self.discretize_state(obs)
        next_obs = self.discretize_state(next_obs)

        if(self.check_update): # update q_a
            best_q_b = np.max(self.qb_values[next_obs]) if next_obs in self.qb_values else 0
            # Q-learning
            self.qa_values[obs][action] += self.lr * (
                reward + self.discount_factor * best_q_b - self.qa_values[obs][action]
            )

        else: # update q_b
            best_q_a = np.max(self.qa_values[next_obs]) if next_obs in self.qa_values else 0
            # Q-learning
            self.qb_values[obs][action] += self.lr * (
                reward + self.discount_factor * best_q_a - self.qb_values[obs][action]
            )
        
        # สลับการอัปเดต Q_A และ Q_B ในรอบถัดไป
        self.check_update = not self.check_update





     