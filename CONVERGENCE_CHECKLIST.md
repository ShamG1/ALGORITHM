# 多智能体训练收敛诊断清单（MCTS_AlphaZero / MAPPO）

> 目的：快速判断训练是否在“健康收敛”，以及出现偏差时如何定位和调参。

## 1. 评估协议（先固定，再看曲线）

- 每 `200` 个 episode 做一次评估（建议固定）。
- 评估时使用固定配置（尽量减少探索噪声）：
  - MCTS：降低/关闭 Dirichlet 噪声，使用较低温度。
  - MAPPO：使用 deterministic action（如实现支持）。
- 每次评估至少跑 `20~50` 个 episode，避免单次偶然性。

评估必须记录：
- `success_rate`
- `collision_rate`
- `truncated_rate`（超步数截断率）
- `avg_episode_length`
- `mean_reward`

---

## 2. 训练“健康区间”判定（建议阈值）

> 以下为通用经验阈值，不同场景可微调。

### A. 成功/碰撞/截断

- `success_rate`：应呈上升趋势（至少在中后期不下降）。
- `collision_rate`：应显著下降并趋稳。
- `truncated_rate`：理想是逐步下降。
  - 若长期高于 `60%`，通常意味着“拖时/保守停滞”问题。

### B. 奖励趋势

- `Mean Reward` 与 `P50 Reward` 的移动平均应整体上升或稳定。
- `P25~P75` 区间应逐步收窄（方差降低）。
- 若 `mean` 上升但 `success_rate` 不升：可能是 reward hacking（奖励设计偏差）。

### C. 损失与策略稳定

- MAPPO：
  - policy/value loss 不应长期爆炸或剧烈振荡。
  - 熵（entropy）不应过早塌缩到极低。
- MCTS：
  - `root_n` 分布应稳定在合理区间（接近模拟预算）。
  - 搜索统计不应频繁出现空/异常值。

---

## 3. 奖励组件诊断（重点）

使用 `episode_reward_components.csv` + `plot_rewards.py` 检查：

- `cpp_progress` 是否长期为主要正向贡献。
- `cpp_crash_wall` / `cpp_crash_vehicle` 是否逐步下降。
- `py_no_progress_penalty` 是否在中后期下降。
- `py_respawn_penalty` 若长期很高，说明“撞了重生”依赖严重。
- `py_cooperative_mix` / `py_cooperative_credit` 是否带来稳定改进（而不是放大波动）。

### 不健康信号

- 惩罚项（crash/no_progress）长期绝对值大于 progress 正项。
- success 项提升不明显但总奖励波动很大。
- 组件曲线长期无趋势，仅高频震荡。

---

## 4. 发现问题时的调参映射（速查）

### 问题 1：碰撞率高，成功率上不去

优先尝试：
- 降低探索噪声（MCTS `dirichlet_eps`）
- 增大碰撞惩罚（env reward）
- 增强 pairwise 协调：
  - `pairwise_coordination_enabled=true`
  - 适度提高 `pairwise_brake_scale`

### 问题 2：长期截断（跑满步数），策略拖时

优先尝试：
- 提高 `no_progress_penalty` 或收紧阈值（`no_progress_threshold`）
- 适度提高 `respawn_penalty`
- 适度降低 `max_steps` 做课程学习

### 问题 3：过于保守，不敢通过冲突区

优先尝试：
- 降低 `pairwise_brake_scale`
- 降低 `cooperative_alpha`（例如 `0.3 -> 0.2`）
- 略提升进度奖励权重（`progress_scale`）

### 问题 4：训练方差大，曲线抖动明显

优先尝试：
- 增大评估 episode 数（减少观测噪声）
- 降低学习率（MAPPO actor/critic）
- 缩小更新强度（如降低 update_epochs 或增大 batch）
- 先在更简单场景训练后再迁移

---

## 5. 课程学习建议（强烈推荐）

阶段化示例：
1. `2 agents` + 简单场景 + 较短 `max_steps`
2. `4 agents` + 同场景
3. `6 agents` + 复杂场景（roundabout/onramp/bottleneck）

每阶段通过标准：
- success_rate 达标（例如 > 70%）
- collision_rate 低且稳定
- 奖励组件中 progress 主导

达标再进下一阶段。

---

## 6. 复现与稳健性检查

即使不固定 seed，也建议至少做 `3` 次独立训练对比：
- 关注均值 + 方差，不看单次最好成绩。
- 若 3 次结论方向一致，策略可靠性更高。

---

## 7. 最小执行流程（建议每周固定跑）

1. 训练到固定里程碑（如每 5k episode）。
2. 运行评估协议并记录 5 个核心指标。
3. 运行 `plot_rewards.py` 查看：
   - reward 分布图
   - reward 组件趋势图
4. 对照本清单判定是否健康。
5. 若不健康，按“调参映射”只改 1~2 个参数再迭代。

---

## 8. 备注

- 当前项目已将主要奖励塑形下沉到环境层（DriveSimX），trainer 侧尽量不二次改奖励。
- 这有助于 MCTS 与 MAPPO 的公平对比与问题定位。
