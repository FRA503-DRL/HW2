import pandas as pd
import matplotlib.pyplot as plt
import os

# List of model filenames
model_files = {
    "MC": "MC_training_results.csv",
    "Q-Learning": "Q_Learning_training_results.csv",
    "SARSA": "SARSA_training_results.csv",
}

# Load and combine all CSV files
df_list = []
for model_name, filename in model_files.items():
    if os.path.exists(filename):  # Check if the file exists before reading
        df = pd.read_csv(filename)
        df["Model"] = model_name  # Add a column to identify the model
        df_list.append(df)
    else:
        print(f"⚠️ Warning: {filename} not found, skipping.")

# Ensure at least one file was loaded
if df_list:
    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv("combined_training_results.csv", index=False)
    print("Training data combined and saved as 'combined_training_results.csv'.")
else:
    print("No valid training files found. Exiting.")
    exit()

# Smoothed Learning Curve for Visualization
plt.figure(figsize=(10, 5))
for model in df_all["Model"].unique():
    model_data = df_all[df_all["Model"] == model]
    model_data = model_data.sort_values("Episode")  # Ensure episode order
    model_data["Smoothed Reward"] = model_data["Cumulative Reward"].rolling(window=20, min_periods=1).mean()

    plt.plot(model_data["Episode"], model_data["Smoothed Reward"], label=model)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Performance Comparison of RL Models")
plt.legend()
plt.show()


# Compute final average reward over the last 100 episodes
final_performance = df_all.groupby("Model").apply(
    lambda x: x.tail(min(100, len(x)))["Cumulative Reward"].mean()
)

# Find the episode where reward stabilizes (moving average threshold)
stability_threshold = 10  # Change this if needed
convergence_episode = {}

for model in df_all["Model"].unique():
    model_data = df_all[df_all["Model"] == model].sort_values("Episode")
    model_data["Rolling Mean"] = model_data["Cumulative Reward"].rolling(stability_threshold, min_periods=1).mean()
    
    stable_index = (model_data["Rolling Mean"].diff().abs() < 1).idxmax()  # Find first stable episode
    convergence_episode[model] = model_data.loc[stable_index, "Episode"] if stable_index in model_data.index else "Not Converged"

# Print results
print("\n🔹 **Final Performance (Last 100 Episodes):**")
print(final_performance)

print("\n🔹 **Convergence Episode (First Stable Point):**")
print(convergence_episode)
