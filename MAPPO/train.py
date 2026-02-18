# --- train.py ---
# Training script for MAPPO with 6 agents

import os
import sys
import numpy as np
import torch
import csv
import time
from datetime import datetime
from tqdm import tqdm

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
        use_team_reward: bool = True,
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = True,
        save_dir: str = 'policy/checkpoints',
        use_tqdm: bool = False
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
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.save_dir = save_dir
        self.use_tqdm = use_tqdm
        
        os.makedirs(save_dir, exist_ok=True)
        ego_routes = generate_ego_routes(num_agents, scenario_name)
        
        # Initialize environments
        self.envs = []
        for i in range(num_envs):
            env_config = {
                'traffic_flow': False,
                'num_agents': num_agents,
                'scenario_name': scenario_name,
                'use_team_reward': use_team_reward,
                'render_mode': 'human' if (render and i == 0) else None,
                'max_steps': max_steps_per_episode,
                'respawn_enabled': respawn_enabled,
                'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
                'ego_routes': ego_routes
            }
            self.envs.append(ScenarioEnv(env_config))
        self.env = self.envs[0]

        # Initialize MAPPO
        self.mappo = MAPPO(
            num_agents=num_agents,
            obs_dim=OBS_DIM,
            action_dim=2,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_clip=True,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            use_lstm=True,
            sequence_length=5,
            lstm_hidden_dim=128,
            num_envs=num_envs
        )
        
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Num agents: {num_agents}, Num Envs: {num_envs}, Device: {device}\n")
            f.write("=" * 80 + "\n")
        
        self.csv_file = os.path.join(save_dir, 'episode_rewards.csv')
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward', 'Mean_Reward', 'Episode_Length', 'Duration_Sec'])
    
    def log(self, message: str):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
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
        
        if self.mappo.use_lstm:
            self.mappo.reset_buffer()
            
        total_episodes = 0
        total_steps = 0
        episode_rewards = np.zeros((self.num_envs, self.num_agents))
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
                
                # Respawn logic
                if self.respawn_enabled:
                    agent_dones = np.array(info.get('done', [False] * self.num_agents))
                    env_done = (info.get('agents_alive', 1) == 0) or truncated
                else:
                    env_done = terminated or truncated
                    agent_dones = np.array([env_done] * self.num_agents)
                
                episode_rewards[i] += rewards
                episode_lengths[i] += 1
                total_steps += 1
                
                if env_done:
                    total_episodes += 1
                    ep_total_reward = episode_rewards[i].sum()
                    ep_mean_reward = episode_rewards[i].mean()
                    
                    duration_sec = time.time() - episode_start_times[i]
                    with open(self.csv_file, 'a', newline='') as f:
                        csv.writer(f).writerow([total_episodes, ep_total_reward, ep_mean_reward, int(episode_lengths[i]), duration_sec])
                    
                    # Log every episode
                    self.log(f"Episode {total_episodes:5d} | Mean Reward: {ep_mean_reward:7.2f} | Length: {int(episode_lengths[i]):4d} steps | Time: {duration_sec:.2f}s | Steps: {total_steps}")
                    
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({'Steps': total_steps, 'Reward': f"{ep_mean_reward:.2f}"})
                    
                    # Reset this env (LSTM history & hidden)
                    self.mappo.reset_env_states(i)
                    n_obs, _ = self.envs[i].reset()
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_start_times[i] = time.time()
                
                next_obs_list.append(n_obs)
                rewards_list.append(rewards)
                dones_list.append(agent_dones)
            
            # 3. Store transition (batched)
            next_obs = np.array(next_obs_list)
            self.mappo.store_transition(obs, actions, np.array(rewards_list), values, log_probs, np.array(dones_list))
            
            # 4. Update
            if len(self.mappo.buffer['obs'][0]) >= self.update_frequency:
                self.mappo.update(next_obs, epochs=self.update_epochs, batch_size=self.batch_size)
                self.log(f"Update at step {total_steps}")

            obs = next_obs
            
            if total_episodes > 0 and total_episodes % self.save_frequency == 0:
                self.mappo.save(os.path.join(self.save_dir, f"mappo_episode_{total_episodes}.pth"))

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

    scenario_name = str(config.get('env', {}).get('scenario_name', 'cross_3lane'))

    trainer = Trainer(
        num_agents=int(config.get('env', {}).get('num_agents', 6)),
        num_envs=int(config.get('env', {}).get('num_envs', 1)),
        max_episodes=int(config.get('train', {}).get('max_episodes', 10000)),
        max_steps_per_episode=int(config.get('env', {}).get('max_steps', 500)),
        update_frequency=int(config.get('train', {}).get('update_frequency', 2048)),
        update_epochs=int(config.get('train', {}).get('update_epochs', 5)),
        batch_size=int(config.get('train', {}).get('batch_size', 128)),
        save_frequency=int(config.get('train', {}).get('save_frequency', 100)),
        log_frequency=int(config.get('train', {}).get('log_frequency', 10)),
        device=device,
        use_team_reward=bool(config.get('reward', {}).get('use_team_reward', True)),
        render=bool(config.get('render', {}).get('enabled', False)),
        respawn_enabled=bool(config.get('env', {}).get('respawn_enabled', True)),
        save_dir=str(config.get('train', {}).get('save_dir', 'MAPPO/checkpoints')),
        use_tqdm=bool(config.get('misc', {}).get('tqdm', True)),
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.mappo.save(os.path.join(trainer.save_dir, 'mappo_interrupted.pth'))

if __name__ == '__main__':
    main()
