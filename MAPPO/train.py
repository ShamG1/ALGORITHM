# --- train.py ---
# Training script for MAPPO with 6 agents

import os
import sys
import numpy as np
import torch
import csv
import time
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

CRASH_STATUSES = frozenset({'CRASH_CAR', 'CRASH_WALL'})
SUCCESS_STATUS = 'SUCCESS'

# Add parent directory to path to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from SIM_MARL instead of Scenario
from DriveSimX.core.env import ScenarioEnv
from DriveSimX.core.utils import DEFAULT_REWARD_CONFIG, OBS_DIM

# Import MAPPO
try:
    from .mappo import MAPPO
except ImportError:
    from mappo import MAPPO


def generate_ego_routes(num_agents: int, scenario_name: str):
    from DriveSimX.core.utils import ROUTE_MAP_BY_SCENARIO

    mapping = ROUTE_MAP_BY_SCENARIO.get(str(scenario_name))
    if not mapping:
        raise ValueError(f"No route mapping found for scenario_name={scenario_name!r}")

    # Flatten mapping: turn_type -> {in_id: out_id}
    all_routes = []
    for mp in mapping.values():
        for in_id, out_id in mp.items():
            start = in_id if isinstance(in_id, str) else f"IN_{in_id}"
            end = out_id if isinstance(out_id, str) else f"OUT_{out_id}"
            all_routes.append((start, end))

    if not all_routes:
        raise RuntimeError(f"No valid routes generated for scenario_name={scenario_name!r}")

    return [all_routes[i % len(all_routes)] for i in range(num_agents)]


class Trainer:
    """MAPPO Trainer for multi-agent scenario navigation."""
    
    def __init__(
        self,
        num_agents: int = 6,
        scenario_name: str = "cross_3lane",
        num_envs: int = 1,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 500,
        update_frequency: int = 2048,
        update_epochs: int = 5,
        batch_size: int = 128,
        save_frequency: int = 100,
        log_frequency: int = 10,
        device: str = 'cpu',
        use_lstm: bool = True,
        use_tcn: bool = False,
        sequence_length: int = 5,
        lr_actor: float = 1e-4,
        lr_critic: float = 2e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_team_reward: bool = True,
        cooperative_mode: bool = False,
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = True,
        max_steps_penalty_no_respawn: float = -5.0,
        respawn_penalty: float = -0.5,
        no_progress_penalty: float = -0.2,
        no_progress_window_steps: int = 30,
        no_progress_threshold: float = 0.01,
        cooperative_alpha: float = 0.3,
        cooperative_credit_coef: float = 0.3,
        pairwise_coordination_enabled: bool = True,
        pairwise_distance_threshold: float = 80.0,
        pairwise_brake_scale: float = 0.35,
        pairwise_cooldown_steps: int = 6,
        save_dir: str = 'policy/checkpoints',
        use_tqdm: bool = False,
        best_metric_window: int = 50,
        best_save_start_episode: int = 200,
        best_min_delta: float = 1e-3,
        eval_interval_episodes: int = 500,
        eval_episodes: int = 20,
    ):
        self.num_agents = num_agents
        self.scenario_name = scenario_name
        self.num_envs = num_envs
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.device = device
        self.use_team_reward = use_team_reward
        self.cooperative_mode = cooperative_mode
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.max_steps_penalty_no_respawn = float(max_steps_penalty_no_respawn)
        self.respawn_penalty = float(respawn_penalty)
        self.no_progress_penalty = float(no_progress_penalty)
        self.no_progress_window_steps = int(no_progress_window_steps)
        self.no_progress_threshold = float(no_progress_threshold)
        self.cooperative_alpha = float(np.clip(cooperative_alpha, 0.0, 1.0))
        self.cooperative_credit_coef = float(np.clip(cooperative_credit_coef, 0.0, 1.0))
        self.pairwise_coordination_enabled = bool(pairwise_coordination_enabled)
        self.pairwise_distance_threshold = float(max(1.0, pairwise_distance_threshold))
        self.pairwise_brake_scale = float(np.clip(pairwise_brake_scale, 0.0, 1.0))
        self.pairwise_cooldown_steps = int(max(1, pairwise_cooldown_steps))
        self.save_dir = save_dir
        self.use_tqdm = use_tqdm
        self.best_metric_window = int(best_metric_window)
        self.best_save_start_episode = int(best_save_start_episode)
        self.best_min_delta = float(best_min_delta)
        self.eval_interval_episodes = int(max(0, eval_interval_episodes))
        self.eval_episodes = int(max(1, eval_episodes))
        self.best_score = float('-inf')
        self.best_episode = 0
        
        os.makedirs(save_dir, exist_ok=True)
        ego_routes = generate_ego_routes(num_agents, scenario_name)
        
        # Initialize environments
        self.env_base_config = {
            'traffic_flow': False,
            'num_agents': num_agents,
            'scenario_name': scenario_name,
            'use_team_reward': use_team_reward,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'max_steps_penalty_no_respawn': float(self.max_steps_penalty_no_respawn),
            'respawn_penalty': float(self.respawn_penalty),
            'no_progress_penalty': float(self.no_progress_penalty),
            'no_progress_window_steps': int(self.no_progress_window_steps),
            'no_progress_threshold': float(self.no_progress_threshold),
            'cooperative_mode': bool(self.cooperative_mode),
            'cooperative_alpha': float(self.cooperative_alpha),
            'cooperative_credit_coef': float(self.cooperative_credit_coef),
            'pairwise_coordination_enabled': bool(self.pairwise_coordination_enabled),
            'pairwise_distance_threshold': float(self.pairwise_distance_threshold),
            'pairwise_brake_scale': float(self.pairwise_brake_scale),
            'pairwise_cooldown_steps': int(self.pairwise_cooldown_steps),
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            'ego_routes': ego_routes,
        }

        self.envs = []
        for i in range(num_envs):
            env_config = dict(self.env_base_config)
            env_config['render_mode'] = 'human' if (render and i == 0) else None
            self.envs.append(ScenarioEnv(env_config))
        self.env = self.envs[0]

        # Initialize MAPPO
        self.mappo = MAPPO(
            num_agents=num_agents,
            obs_dim=OBS_DIM,
            action_dim=2,
            hidden_dim=256,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_clip=True,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_lstm=use_lstm,
            use_tcn=use_tcn,
            sequence_length=sequence_length,
            lstm_hidden_dim=128,
            num_envs=num_envs,
            cooperative_mode=cooperative_mode,
        )
        print(f"[INFO] Critic input dim (centralized): {self.mappo.global_obs_dim} = {OBS_DIM} x {num_agents}")

        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Num agents: {num_agents}, Num Envs: {num_envs}, Device: {device}\n")
            f.write("=" * 80 + "\n")
        
        self.csv_file = os.path.join(save_dir, 'episode_rewards.csv')
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Total_Reward', 'Mean_Reward', 'Episode_Length', 'Duration_Sec',
                'Reward_Min', 'Reward_Max', 'Reward_Std', 'Reward_P25', 'Reward_P50', 'Reward_P75'
            ])

        self.components_csv_file = os.path.join(save_dir, 'episode_reward_components.csv')
        with open(self.components_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Component', 'Value'])

        self.eval_csv_file = os.path.join(save_dir, 'eval_metrics.csv')
        with open(self.eval_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Eval_Episodes', 'Reward_Mean', 'Reward_Std',
                'Length_Mean', 'Success_Rate', 'Crash_Rate', 'Truncated_Rate'
            ])
    
    def log(self, message: str):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def _to_agent_reward_array(self, rewards) -> np.ndarray:
        """Normalize env reward output into float32 array of shape (num_agents,)."""
        if isinstance(rewards, (list, np.ndarray)):
            return np.asarray(rewards, dtype=np.float32)
        return np.full(self.num_agents, float(rewards), dtype=np.float32)

    def _maybe_save_best_checkpoint(self, episode: int, episode_reward_history: list):
        """Save best model using moving-average episode mean reward."""
        if episode < self.best_save_start_episode:
            return

        window = max(1, int(self.best_metric_window))
        if len(episode_reward_history) < window:
            return

        score = float(np.mean(episode_reward_history[-window:]))
        if score <= (float(self.best_score) + float(self.best_min_delta)):
            return

        self.best_score = score
        self.best_episode = int(episode)

        best_ckpt_path = os.path.join(self.save_dir, 'best_model.pth')
        self.mappo.save(best_ckpt_path)

        best_meta = {
            'best_episode': int(self.best_episode),
            'best_score_ma_reward': float(self.best_score),
            'window': int(window),
            'min_delta': float(self.best_min_delta),
            'updated_at': str(datetime.now()),
        }
        best_meta_path = os.path.join(self.save_dir, 'best_metric.json')
        with open(best_meta_path, 'w', encoding='utf-8') as f:
            json.dump(best_meta, f, ensure_ascii=False, indent=2)

        self.log(
            f"[BEST] episode={episode} ma_reward(window={window})={score:.4f} saved={best_ckpt_path}"
        )
    
    def _run_periodic_eval(self, episode: int):
        """Run deterministic periodic evaluation (no training update)."""
        if self.eval_interval_episodes <= 0:
            return
        if episode % self.eval_interval_episodes != 0:
            return

        eval_success = 0
        eval_crash = 0
        eval_trunc = 0
        eval_rewards = []
        eval_lengths = []

        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_len = 0
            ep_reward = np.zeros(self.num_agents, dtype=np.float32)

            # reset seq states for deterministic eval
            self.mappo.reset_env_states(0)

            while (not done) and ep_len < self.max_steps_per_episode:
                actions, _ = self.mappo.select_actions(np.expand_dims(obs, axis=0), deterministic=True)
                n_obs, rewards, terminated, truncated, info = self.env.step(actions[0])
                rewards = self._to_agent_reward_array(rewards)
                ep_reward += rewards
                ep_len += 1
                done = bool(terminated or truncated)
                obs = n_obs

                if done:
                    statuses = list(info.get('status', [])) if isinstance(info, dict) else []
                    if any(s == SUCCESS_STATUS for s in statuses):
                        eval_success += 1
                    if any(s in CRASH_STATUSES for s in statuses):
                        eval_crash += 1
                    if bool(truncated):
                        eval_trunc += 1

            eval_rewards.append(float(np.mean(ep_reward)))
            eval_lengths.append(int(ep_len))

        n = max(1, self.eval_episodes)
        reward_mean = float(np.mean(eval_rewards))
        reward_std = float(np.std(eval_rewards))
        length_mean = float(np.mean(eval_lengths))
        success_rate = float(eval_success / n)
        crash_rate = float(eval_crash / n)
        trunc_rate = float(eval_trunc / n)

        with open(self.eval_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                int(episode), int(self.eval_episodes), reward_mean, reward_std,
                length_mean, success_rate, crash_rate, trunc_rate
            ])

        self.log(
            f"[EVAL] ep={episode} episodes={self.eval_episodes} "
            f"reward_mean={reward_mean:.2f} reward_std={reward_std:.2f} "
            f"len_mean={length_mean:.1f} "
            f"success_rate={success_rate:.3f} crash_rate={crash_rate:.3f} trunc_rate={trunc_rate:.3f}"
        )

    def train(self):
        """Main training loop supporting vectorized inference."""
        self.log("=" * 80)
        self.log(f"Starting MAPPO Training with {self.num_envs} environments")
        self.log("=" * 80)
        
        # Initial reset
        all_obs = []
        for env in self.envs:
            obs, _ = env.reset()
            all_obs.append(obs)
        obs = np.array(all_obs) # (num_envs, num_agents, obs_dim)
        
        if self.mappo.use_seq_model:
            self.mappo.reset_buffer()
            
        total_episodes = 0
        total_steps = 0
        last_saved_episode = 0
        episode_mean_reward_history = []
        episode_rewards = np.zeros((self.num_envs, self.num_agents))
        episode_component_sums = [defaultdict(float) for _ in range(self.num_envs)]
        episode_lengths = np.zeros(self.num_envs)
        episode_start_times = [time.time() for _ in range(self.num_envs)]
        
        pbar = None
        if self.use_tqdm:
            pbar = tqdm(total=self.max_episodes, desc="Training")

        while total_episodes < self.max_episodes:
            # 1. Batched Inference
            actions, log_probs = self.mappo.select_actions(obs)
            values = self.mappo.get_values(obs)
            
            # 2. Step Environments
            next_obs_list = []
            rewards_list = []
            dones_list = []
            
            for i in range(self.num_envs):
                n_obs, rewards, terminated, truncated, info = self.envs[i].step(actions[i])

                if self.render and i == 0:
                    self.envs[i].render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)

                rewards = self._to_agent_reward_array(rewards)

                # Accumulate per-component rewards for this env-episode.
                rc_cpp = info.get('reward_components_cpp', {}) if isinstance(info, dict) else {}
                rc_py = info.get('reward_components_py', {}) if isinstance(info, dict) else {}
                if isinstance(rc_cpp, dict):
                    for k, v in rc_cpp.items():
                        try:
                            episode_component_sums[i][f"cpp_{k}"] += float(np.sum(np.asarray(v, dtype=np.float32)))
                        except Exception:
                            pass
                if isinstance(rc_py, dict):
                    for k, v in rc_py.items():
                        if k == 'pairwise_action_adjust_applied':
                            continue
                        try:
                            episode_component_sums[i][f"py_{k}"] += float(np.sum(np.asarray(v, dtype=np.float32)))
                        except Exception:
                            pass

                # Respawn logic
                if self.respawn_enabled:
                    agent_dones = np.array(info.get('done', [False] * self.num_agents))
                    env_done = terminated or truncated or (info.get('agents_alive', 1) == 0)
                else:
                    env_done = terminated or truncated
                    agent_dones = np.array([env_done] * self.num_agents)

                episode_rewards[i] += rewards
                episode_lengths[i] += 1
                total_steps += 1
                
                if env_done:
                    total_episodes += 1
                    ep_rewards = np.asarray(episode_rewards[i], dtype=np.float32)
                    ep_total_reward = float(ep_rewards.sum())
                    ep_mean_reward = float(ep_rewards.mean())
                    ep_min_reward = float(np.min(ep_rewards))
                    ep_max_reward = float(np.max(ep_rewards))
                    ep_std_reward = float(np.std(ep_rewards))
                    ep_p25, ep_p50, ep_p75 = [float(x) for x in np.percentile(ep_rewards, [25, 50, 75])]

                    duration_sec = time.time() - episode_start_times[i]
                    with open(self.csv_file, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            total_episodes, ep_total_reward, ep_mean_reward, int(episode_lengths[i]), duration_sec,
                            ep_min_reward, ep_max_reward, ep_std_reward, ep_p25, ep_p50, ep_p75
                        ])

                    if episode_component_sums[i]:
                        with open(self.components_csv_file, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for ck, cv in sorted(episode_component_sums[i].items()):
                                writer.writerow([total_episodes, ck, float(cv)])

                    episode_mean_reward_history.append(float(ep_mean_reward))
                    self._maybe_save_best_checkpoint(total_episodes, episode_mean_reward_history)
                    
                    # Log by configured frequency (always log episode 1)
                    if total_episodes == 1 or (total_episodes % self.log_frequency == 0):
                        self.log(
                            f"Episode {total_episodes:5d} | Reward: {ep_mean_reward:7.2f} (Total: {ep_total_reward:7.2f}) | "
                            f"Length: {int(episode_lengths[i]):4d} | Time: {duration_sec:.2f}s"
                        )
                    
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({'Steps': total_steps, 'Reward': f"{ep_mean_reward:.2f}"})
                    
                    # Reset this env (LSTM history & hidden)
                    self.mappo.reset_env_states(i)
                    n_obs, _ = self.envs[i].reset()
                    episode_rewards[i] = 0
                    episode_component_sums[i].clear()
                    episode_lengths[i] = 0
                    episode_start_times[i] = time.time()

                    self._run_periodic_eval(total_episodes)
                
                next_obs_list.append(n_obs)
                rewards_list.append(rewards)
                dones_list.append(agent_dones)
            
            # 3. Store transition (batched)
            next_obs = np.array(next_obs_list)
            self.mappo.store_transition(obs, actions, np.array(rewards_list), values, log_probs, np.array(dones_list))
            
            # 4. Update
            if len(self.mappo.buffer['obs'][0]) >= self.update_frequency:
                self.mappo.update(next_obs, epochs=self.update_epochs, batch_size=self.batch_size)
                #self.log(f"Update at step {total_steps}")

            obs = next_obs
            
            if (
                total_episodes > 0
                and total_episodes % self.save_frequency == 0
                and total_episodes != last_saved_episode
            ):
                #self.mappo.save(os.path.join(self.save_dir, f"mappo_episode_{total_episodes}.pth"))
                last_saved_episode = total_episodes

        self.mappo.save(os.path.join(self.save_dir, 'mappo_final.pth'))
        for env in self.envs: env.close()


def load_config_from_yaml(yaml_path: str):
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Configuration loaded from {yaml_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load YAML config from {yaml_path}: {e}")
        sys.exit(1)
    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MAPPO Training (YAML Config Only)')
    parser.add_argument('--config', type=str, default='MAPPO/train_config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)

    device_name = config.get('net', {}).get('device', 'cpu')
    if device_name == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_name

    print(f"[INFO] Training device: {device}")
    worker_device = 'CPU' if device == 'cpu' else 'GPU'
    print(f"[INFO] Worker inference device: {worker_device} (optimal for small networks)")

    scenario_name = str(config.get('env', {}).get('scenario_name', 'cross_3lane'))
    use_team_reward = bool(config.get('reward', {}).get('use_team_reward', True))
    cooperative_mode = use_team_reward
    print(f"[INFO] Scenario: {scenario_name}")
    print(f"[INFO] MARL mode: {'cooperative' if cooperative_mode else 'independent'} (derived from reward.use_team_reward={use_team_reward})")
    use_tcn = bool(config.get('net', {}).get('use_tcn', False))
    use_lstm = bool(config.get('net', {}).get('use_lstm', True))
    sequence_length = int(config.get('net', {}).get('sequence_length', 5))

    if use_tcn and use_lstm:
        print('[WARN] Both use_tcn and use_lstm are true; using TCN backbone by priority.')

    optim_cfg = config.get('optim', {})
    rl_cfg = config.get('rl', {})

    trainer = Trainer(
        num_agents=int(config.get('env', {}).get('num_agents', 6)),
        scenario_name=scenario_name,
        num_envs=int(config.get('env', {}).get('num_envs', 1)),
        max_episodes=int(config.get('train', {}).get('max_episodes', 10000)),
        max_steps_per_episode=int(config.get('env', {}).get('max_steps', 500)),
        update_frequency=int(config.get('train', {}).get('update_frequency', 2048)),
        update_epochs=int(config.get('train', {}).get('update_epochs', 5)),
        batch_size=int(config.get('train', {}).get('batch_size', 128)),
        save_frequency=int(config.get('train', {}).get('save_frequency', 100)),
        log_frequency=int(config.get('train', {}).get('log_frequency', 10)),
        device=device,
        use_lstm=use_lstm,
        use_tcn=use_tcn,
        sequence_length=sequence_length,
        lr_actor=float(optim_cfg.get('lr_actor', 1e-4)),
        lr_critic=float(optim_cfg.get('lr_critic', 2e-4)),
        gamma=float(rl_cfg.get('gamma', 0.99)),
        gae_lambda=float(rl_cfg.get('gae_lambda', 0.95)),
        clip_epsilon=float(optim_cfg.get('clip_epsilon', 0.2)),
        entropy_coef=float(optim_cfg.get('entropy_coef', 0.005)),
        value_coef=float(optim_cfg.get('value_coef', 0.5)),
        max_grad_norm=float(optim_cfg.get('max_grad_norm', 0.5)),
        use_team_reward=use_team_reward,
        cooperative_mode=cooperative_mode,
        render=bool(config.get('render', {}).get('enabled', False)),
        respawn_enabled=bool(config.get('env', {}).get('respawn_enabled', True)),
        max_steps_penalty_no_respawn=float(config.get('env', {}).get('max_steps_penalty_no_respawn', DEFAULT_REWARD_CONFIG.get('max_steps_penalty_no_respawn', -5.0))),
        respawn_penalty=float(config.get('env', {}).get('respawn_penalty', DEFAULT_REWARD_CONFIG.get('respawn_penalty', -0.5))),
        no_progress_penalty=float(config.get('env', {}).get('no_progress_penalty', DEFAULT_REWARD_CONFIG.get('no_progress_penalty', -0.2))),
        no_progress_window_steps=int(config.get('env', {}).get('no_progress_window_steps', DEFAULT_REWARD_CONFIG.get('no_progress_window_steps', 30))),
        no_progress_threshold=float(config.get('env', {}).get('no_progress_threshold', DEFAULT_REWARD_CONFIG.get('no_progress_threshold', 0.01))),
        cooperative_alpha=float(config.get('marl', {}).get('cooperative_alpha', 0.3)),
        cooperative_credit_coef=float(config.get('marl', {}).get('cooperative_credit_coef', 0.3)),
        pairwise_coordination_enabled=bool(config.get('marl', {}).get('pairwise_coordination_enabled', True)),
        pairwise_distance_threshold=float(config.get('marl', {}).get('pairwise_distance_threshold', 80.0)),
        pairwise_brake_scale=float(config.get('marl', {}).get('pairwise_brake_scale', 0.35)),
        pairwise_cooldown_steps=int(config.get('marl', {}).get('pairwise_cooldown_steps', 6)),
        save_dir=str(config.get('train', {}).get('save_dir', 'MAPPO/checkpoints')),
        use_tqdm=bool(config.get('misc', {}).get('tqdm', True)),
        best_metric_window=int(config.get('train', {}).get('best_metric_window', 50)),
        best_save_start_episode=int(config.get('train', {}).get('best_save_start_episode', 200)),
        best_min_delta=float(config.get('train', {}).get('best_min_delta', 1e-3)),
        eval_interval_episodes=int(config.get('eval', {}).get('interval_episodes', 500)),
        eval_episodes=int(config.get('eval', {}).get('episodes', 20)),
    )

    print("Calling trainer.train()...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        #trainer.mappo.save(os.path.join(trainer.save_dir, 'mappo_interrupted.pth'))

if __name__ == '__main__':
    main()
