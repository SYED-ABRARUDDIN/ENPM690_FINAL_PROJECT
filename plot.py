import os
import pandas as pd
import matplotlib.pyplot as plt

# 1) Paths to your log files
metrics_csv   = os.path.join("model", "metrics.csv")
training_csv  = os.path.join("model", "training_data.csv")

# 2) Load each into a DataFrame
metrics   = pd.read_csv(metrics_csv,   header=None, names=["episode","avg_reward"])
training  = pd.read_csv(training_csv,  header=None, names=["episode","avg_reward"])

# 3) Plot metrics.csv
plt.figure()
plt.plot(metrics["episode"], metrics["avg_reward"])
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Training Performance (metrics.csv)")
plt.tight_layout()
plt.show()

# 4) Plot training_data.csv
plt.figure()
plt.plot(training["episode"], training["avg_reward"])
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Training Performance (training_data.csv)")
plt.tight_layout()
plt.show()
