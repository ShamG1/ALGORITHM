import os
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 中文字体回退（按顺序尝试，系统没有时会自动跳过）
rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei',
    'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'
]
rcParams['axes.unicode_minus'] = False

# 路径可按需调整
CSV_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "episode_rewards.csv")
COMP_CSV_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "episode_reward_components.csv")
EVAL_CSV_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "eval_metrics.csv")
OUT_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "reward_curve.png")
OUT_COMPONENT_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "reward_components_curve.png")
OUT_EVAL_PATH = os.path.join("MCTS_AlphaZero", "checkpoints", "eval_metrics_curve.png")


def moving_average(x: np.ndarray, window: int):
    if x.size < window:
        return None
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def plot_reward_distribution():
    episodes = []
    mean_rewards = []
    p25_rewards = []
    p50_rewards = []
    p75_rewards = []

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["Episode"])
                mean_r = float(row["Mean_Reward"])

                # 兼容旧 CSV：没有分位数字段时，用 mean 兜底
                p25 = float(row.get("Reward_P25", mean_r))
                p50 = float(row.get("Reward_P50", mean_r))
                p75 = float(row.get("Reward_P75", mean_r))
            except (ValueError, KeyError):
                continue

            episodes.append(ep)
            mean_rewards.append(mean_r)
            p25_rewards.append(p25)
            p50_rewards.append(p50)
            p75_rewards.append(p75)

    if not episodes:
        print("CSV 无有效数据，请检查路径或文件内容。")
        return

    episodes = np.array(episodes, dtype=np.int32)
    mean_rewards = np.array(mean_rewards, dtype=np.float32)
    p25_rewards = np.array(p25_rewards, dtype=np.float32)
    p50_rewards = np.array(p50_rewards, dtype=np.float32)
    p75_rewards = np.array(p75_rewards, dtype=np.float32)

    window = 50
    smooth_mean = moving_average(mean_rewards, window)
    smooth_p50 = moving_average(p50_rewards, window)
    smooth_eps = episodes[window - 1:] if smooth_mean is not None else None

    plt.figure(figsize=(11, 6))
    plt.plot(episodes, mean_rewards, alpha=0.22, linewidth=1.0, label="Mean Reward")
    plt.plot(episodes, p50_rewards, alpha=0.35, linewidth=1.0, label="P50 Reward")
    plt.fill_between(episodes, p25_rewards, p75_rewards, alpha=0.18, label="P25~P75 区间")

    if smooth_mean is not None and smooth_eps is not None:
        plt.plot(smooth_eps, smooth_mean, linewidth=2.0, label=f"Mean 移动平均(window={window})")
    if smooth_p50 is not None and smooth_eps is not None:
        plt.plot(smooth_eps, smooth_p50, linewidth=2.0, linestyle="--", label=f"P50 移动平均(window={window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MCTS 训练奖励曲线（含分布变化）")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_PATH, dpi=150)
    print(f"图已保存: {OUT_PATH}")
    plt.show()


def plot_reward_components(top_k: int = 10):
    if not os.path.exists(COMP_CSV_PATH):
        print(f"未找到奖励分解文件: {COMP_CSV_PATH}，跳过组件分析图。")
        return

    per_comp = defaultdict(dict)
    episodes_set = set()

    with open(COMP_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["Episode"])
                comp = str(row["Component"])
                val = float(row["Value"])
            except (ValueError, KeyError):
                continue
            per_comp[comp][ep] = val
            episodes_set.add(ep)

    if not episodes_set:
        print("奖励分解 CSV 无有效数据，跳过组件分析图。")
        return

    episodes = np.array(sorted(episodes_set), dtype=np.int32)

    # 按绝对贡献均值排序，保留前 top_k 条曲线
    comp_scores = []
    for comp, mp in per_comp.items():
        vals = np.array([float(mp.get(int(ep), 0.0)) for ep in episodes], dtype=np.float32)
        score = float(np.mean(np.abs(vals)))
        comp_scores.append((score, comp, vals))
    comp_scores.sort(reverse=True, key=lambda x: x[0])

    if top_k > 0:
        comp_scores = comp_scores[:top_k]

    plt.figure(figsize=(12, 7))
    window = 50
    for _score, comp, vals in comp_scores:
        sm = moving_average(vals, window)
        if sm is not None:
            x = episodes[window - 1:]
            y = sm
        else:
            x = episodes
            y = vals
        plt.plot(x, y, linewidth=1.8, label=comp)

    plt.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Component Reward Sum per Episode")
    plt.title("MCTS 奖励分解趋势（组件级）")
    plt.grid(True, alpha=0.28)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plt.savefig(OUT_COMPONENT_PATH, dpi=150)
    print(f"组件图已保存: {OUT_COMPONENT_PATH}")
    plt.show()


def plot_eval_metrics(window: int = 1):
    if not os.path.exists(EVAL_CSV_PATH):
        print(f"未找到评估指标文件: {EVAL_CSV_PATH}，跳过评估曲线图。")
        return

    eval_eps = []
    success = []
    crash = []
    trunc = []

    with open(EVAL_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["Episode"])
                s = float(row["Success_Rate"])
                c = float(row["Crash_Rate"])
                t = float(row["Truncated_Rate"])
            except (ValueError, KeyError):
                continue
            eval_eps.append(ep)
            success.append(s)
            crash.append(c)
            trunc.append(t)

    if not eval_eps:
        print("评估指标 CSV 无有效数据，跳过评估曲线图。")
        return

    eval_eps = np.array(eval_eps, dtype=np.int32)
    success = np.array(success, dtype=np.float32)
    crash = np.array(crash, dtype=np.float32)
    trunc = np.array(trunc, dtype=np.float32)

    if window > 1:
        s_sm = moving_average(success, window)
        c_sm = moving_average(crash, window)
        t_sm = moving_average(trunc, window)
        if s_sm is not None:
            x = eval_eps[window - 1:]
            success, crash, trunc = s_sm, c_sm, t_sm
            eval_eps = x

    plt.figure(figsize=(11, 6))
    plt.plot(eval_eps, success, linewidth=2.0, label="Success Rate")
    plt.plot(eval_eps, crash, linewidth=2.0, label="Crash Rate")
    plt.plot(eval_eps, trunc, linewidth=2.0, label="Truncated Rate")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Training Episode (Eval Checkpoint)")
    plt.ylabel("Rate")
    plt.title("MCTS 评估指标曲线")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_EVAL_PATH, dpi=150)
    print(f"评估图已保存: {OUT_EVAL_PATH}")
    plt.show()


def main():
    plot_reward_distribution()
    plot_reward_components(top_k=10)
    plot_eval_metrics(window=1)


if __name__ == "__main__":
    main()
