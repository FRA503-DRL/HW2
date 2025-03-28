import pandas as pd
import matplotlib.pyplot as plt

files = {
    # EXP1
    "Q-Learning": "Q_test_results.csv",
    "SARSA": "SARSA_test_results.csv",
    "Double Q-Learningt": "Double_Q_test_results.csv",
    "Monte Carlo": "MC_test_results.csv",

    # EXP2
    "Q-Learning_EXP2": "/Users/buzz/Documents/GitHub/HW2/output_exp2/Q_test_results.csv",

    # EXP3
    "Q-Learning_EXP3_9900": "/Users/buzz/Documents/GitHub/HW2/output_exp3/Q_test_results_9900.csv",
    "Q-Learning_EXP3_19900": "/Users/buzz/Documents/GitHub/HW2/output_exp3/Q_test_results_19900.csv"

}

dfs = {name: pd.read_csv(path) for name, path in files.items()}

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
for name, df in dfs.items():
    plt.plot(df["Episode"], df["Cumulative Reward"], label=name)
plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
for name, df in dfs.items():
    plt.plot(df["Episode"], df["Epsilon"], label=name)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
for name, df in dfs.items():
    plt.plot(df["Episode"], df["Average Q-Value"], label=name)
plt.title("Average Q-Value per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Q-Value")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
for name, df in dfs.items():
    plt.plot(df["Episode"], df["Steps"], label=name)
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle("RL Algorithm Training Comparison", fontsize=16, y=1.02)
plt.subplots_adjust(top=0.92)
plt.show()
