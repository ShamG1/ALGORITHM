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
from SIM_MARL.envs.env import ScenarioEnv, DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
from SIM_MARL.envs.utils import DEFAULT_REWARD_CONFIG, OBS_DIM

# Import MAPPO
try:
    from .mappo import MAPPO
except ImportError:
    from mappo import MAPPO


def generate_ego_routes(num_agents: int, num_lanes: int):
    """
    Generate ego routes for agents based on default route mappings, ensuring balanced distribution.
    """
    if num_lanes == 2:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    elif num_lanes == 3:
        route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
    else:
        route_mapping = {}
        for dir_idx in range(4):
            for lane_idx in range(num_lanes):
                in_id = dir_idx * num_lanes + lane_idx + 1
                opposite_dir_idx = (dir_idx + 2) % 4
                out_id = opposite_dir_idx * num_lanes + lane_idx + 1
                route_mapping[f"IN_{in_id}"] = [f"OUT_{out_id}"]
    
    all_routes = []
    for in_id, out_ids in route_mapping.items():
        for out_id in out_ids:
            all_routes.append((in_id, out_id))
    
    routes_by_dir = {i: [] for i in range(4)}
    for route in all_routes:
        in_id = route[0]
        in_num = int(in_id.split('_')[1])
        dir_idx = (in_num - 1) // num_lanes
        routes_by_dir[dir_idx].append(route)
    
    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4
    
    for dir_idx in range(4):
        count = agents_per_dir + (1 if dir_idx < extra_agents else 0)
        if count > 0 and routes_by_dir[dir_idx]:
            for i in range(count):
                route_idx = i % len(routes_by_dir[dir_idx])
                selected_routes.append(routes_by_dir[dir_idx][route_idx])
    
    if len(selected_routes) < num_agents:
        all_remaining = []
        used = set(selected_routes)
        for dir_idx in range(4):
            remaining = [r for r in routes_by_dir[dir_idx] if r not in used]
            all_remaining.extend(remaining)
        
        remaining_needed = num_agents - len(selected_routes)
        if all_remaining:
            step = max(1, len(all_remaining) // remaining_needed) if remaining_needed > 0 else 1
            selected_routes.extend(all_remaining[::step][:remaining_needed])
    
    return selected_routes[:num_agents]


class Trainer:
    """MAPPO Trainer for multi-agent scenario navigation."""
    
    def __init__(
        self,
        num_agents: int = 6,
        num_lanes: int = 2,
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
        self.num_lanes = num_lanes
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
        ego_routes = generate_ego_routes(num_agents, num_lanes)
        
        # Initialize environments
        self.envs = []
        for i in range(num_envs):
            env_config = {
                'traffic_flow': False,
                'num_agents': num_agents,
                'num_lanes': num_lanes,
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=6)
    parser.add_argument('--num-lanes', type=int, default=2)
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--max-episodes', type=int, default=10000)
    parser.add_argument('--update-frequency', type=int, default=2048)
    parser.add_argument('--update-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-dir', type=str, default='MAPPO/checkpoints')
    parser.add_argument('--tqdm', action='store_true')
    
    args = parser.parse_args()
    
    trainer = Trainer(
        num_agents=args.num_agents,
        num_lanes=args.num_lanes,
        num_envs=args.num_envs,
        max_episodes=args.max_episodes,
        update_frequency=args.update_frequency,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        device=args.device,
        render=args.render,
        save_dir=args.save_dir,
        use_tqdm=args.tqdm
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.mappo.save(os.path.join(args.save_dir, 'mappo_interrupted.pth'))

if __name__ == '__main__':
    main()
