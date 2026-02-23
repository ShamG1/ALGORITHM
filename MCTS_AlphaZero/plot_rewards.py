import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# 路径可以根据需要调整
CSV_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "episode_rewards.csv")

episodes = []
mean_rewards = []

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            ep = int(row["Episode"])
            mr = float(row["Mean_Reward"])
        except (KeyError, ValueError):
            continue
        episodes.append(ep)
        mean_rewards.append(mr)

if not episodes:
    print("No data loaded from CSV; check the path or file content.")
    exit(0)

episodes = np.array(episodes)
mean_rewards = np.array(mean_rewards)

# 计算移动平均，窗口可根据需要调整
window = 50
if len(mean_rewards) >= window:
    kernel = np.ones(window) / window
    smooth = np.convolve(mean_rewards, kernel, mode="valid")
    smooth_eps = episodes[window - 1:]
else:
    smooth = None
    smooth_eps = None

plt.figure(figsize=(10, 5))
plt.plot(episodes, mean_rewards, alpha=0.3, label="Mean Reward per Episode")
if smooth is not None:
    plt.plot(smooth_eps, smooth, color="red", linewidth=2, label=f"Moving Avg (window={window})")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Training Reward Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 可选：保存成图片
out_path = os.path.join("MCTS_AlphaZero", "checkpoints", "reward_curve.png")
plt.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

plt.show()