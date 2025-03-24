"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import csv
import numpy as np

from omni.isaac.lab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np
import matplotlib.pyplot as plt

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,  #use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 11
    action_range = [-20, 20]  # [min, max]
    discretize_state_weight = [5, 5, 5, 5]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 100
    start_epsilon = 0.01
    epsilon_decay = 1.0  # reduce the exploration over time
    final_epsilon = 0.01
    discount = 0.5

    agent = Q_Learning(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "Q_Learning"  
    episode = 9900
    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
    agent.load_q_value(full_path, q_value_file)


    # Define CSV file for storing test results
    test_csv_filename = "Q_test_results.csv"
    with open(test_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Cumulative Reward", "Steps"])


    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in range(n_episodes):

                obs, _ = env.reset()
                done = False

                cumulative_reward = 0
                step_count = 0

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    # action_tensor = torch.tensor([[action]], dtype=torch.float32, device="cuda")  # แปลงเป็น tensor [batch_size, action_dim]
                    # next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated
                    obs = next_obs

                    cumulative_reward += reward.item()
                    step_count += 1 
                print("Terminate: ", episode)

                # Save test results to CSV for comparison
                with open(test_csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode, cumulative_reward, step_count])
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()