# --- mappo.py ---
# Multi-Agent Proximal Policy Optimization (MAPPO) Implementation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

# Import networks - handle both absolute and relative imports
try:
    from .networks import Actor, Critic
except ImportError:
    from networks import Actor, Critic


class MAPPO:
    """Multi-Agent Proximal Policy Optimization algorithm with vectorized LSTM support."""
    
    def __init__(
        self,
        num_agents: int = 6,
        obs_dim: int = 127,
        action_dim: int = 2,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        use_lstm: bool = True,
        sequence_length: int = 5,
        lstm_hidden_dim: int = 128,
        num_envs: int = 1
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_envs = num_envs
        
        # Create actor and critic networks for each agent
        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim, hidden_dim, lstm_hidden_dim, use_lstm, sequence_length).to(self.device)
            for _ in range(num_agents)
        ])
        
        self.critics = nn.ModuleList([
            Critic(obs_dim, hidden_dim, lstm_hidden_dim, use_lstm, sequence_length).to(self.device)
            for _ in range(num_agents)
        ])
        
        self.actor_optimizers = [optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=lr_critic) for c in self.critics]
        
        # Batched buffers (List of lists: [agent][time_step])
        self.reset_buffer()

    def reset_buffer(self):
        """Reset experience buffer and LSTM states for all envs."""
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'values': [[] for _ in range(self.num_agents)],
            'log_probs': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
        }
        
        if self.use_lstm:
            # actor_hidden_states[agent_idx] = (h, c) where h/c is (1, num_envs, lstm_hidden_dim)
            self.actor_hidden_states = [None for _ in range(self.num_agents)]
            self.critic_hidden_states = [None for _ in range(self.num_agents)]
            # History: [agent][env] -> deque
            self.obs_history = [[deque(maxlen=self.sequence_length) for _ in range(self.num_envs)] for _ in range(self.num_agents)]

    def reset_env_states(self, env_idx: int):
        """Reset LSTM states and history for a specific environment (called on env reset)."""
        if not self.use_lstm:
            return
            
        for i in range(self.num_agents):
            self.obs_history[i][env_idx].clear()
            # Partial reset of hidden states (zero out the specific env slice)
            for states in [self.actor_hidden_states, self.critic_hidden_states]:
                if states[i] is not None:
                    h, c = states[i]
                    h.data[:, env_idx, :].fill_(0)
                    c.data[:, env_idx, :].fill_(0)

    def _prepare_obs_history(self, agent_idx: int, obs: np.ndarray) -> torch.Tensor:
        """Update and return batched sequence observations (num_envs, seq_len, obs_dim)."""
        batch_seq = []
        for env_idx in range(self.num_envs):
            self.obs_history[agent_idx][env_idx].append(obs[env_idx])
            hist = list(self.obs_history[agent_idx][env_idx])
            # Pad with the first observation if not enough history
            while len(hist) < self.sequence_length:
                hist.insert(0, hist[0])
            batch_seq.append(np.array(hist))
        return torch.FloatTensor(np.array(batch_seq)).to(self.device)

    def select_actions(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions for all agents in all envs.
        obs: (num_envs, num_agents, obs_dim)
        Returns: actions (num_envs, num_agents, action_dim), log_probs (num_envs, num_agents, 1)
        """
        all_actions = []
        all_log_probs = []
        
        for i in range(self.num_agents):
            agent_obs = obs[:, i, :] # (num_envs, obs_dim)
            
            if self.use_lstm:
                seq_obs = self._prepare_obs_history(i, agent_obs)
                if deterministic:
                    action, new_h = self.actors[i].get_action(seq_obs, deterministic=True, hidden_state=self.actor_hidden_states[i])
                    lp = torch.zeros((self.num_envs, 1), device=self.device)
                else:
                    action, lp, new_h = self.actors[i].get_action(seq_obs, deterministic=False, hidden_state=self.actor_hidden_states[i])
                self.actor_hidden_states[i] = new_h
            else:
                obs_t = torch.FloatTensor(agent_obs).to(self.device)
                if deterministic:
                    action = self.actors[i].get_action(obs_t, deterministic=True)
                    lp = torch.zeros((self.num_envs, 1), device=self.device)
                else:
                    action, lp = self.actors[i].get_action(obs_t, deterministic=False)
            
            all_actions.append(action.detach().cpu().numpy())
            all_log_probs.append(lp.detach().cpu().numpy())
            
        return np.stack(all_actions, axis=1), np.stack(all_log_probs, axis=1)

    def get_values(self, obs: np.ndarray) -> np.ndarray:
        """Get values for all agents in all envs. Note: History already updated in select_actions."""
        all_values = []
        for i in range(self.num_agents):
            if self.use_lstm:
                # Reconstruct sequence from history (without updating it)
                batch_seq = []
                for env_idx in range(self.num_envs):
                    hist = list(self.obs_history[i][env_idx])
                    while len(hist) < self.sequence_length:
                        hist.insert(0, hist[0] if hist else obs[env_idx, i])
                    batch_seq.append(np.array(hist))
                seq_obs = torch.FloatTensor(np.array(batch_seq)).to(self.device)
                val, new_h = self.critics[i](seq_obs, hidden_state=self.critic_hidden_states[i])
                self.critic_hidden_states[i] = new_h
            else:
                obs_t = torch.FloatTensor(obs[:, i, :]).to(self.device)
                val = self.critics[i](obs_t)
            all_values.append(val.detach().cpu().numpy())
        return np.stack(all_values, axis=1) # (num_envs, num_agents, 1)

    def store_transition(self, obs, actions, rewards, values, log_probs, dones):
        """Store transitions (num_envs, num_agents, ...)."""
        for i in range(self.num_agents):
            self.buffer['obs'][i].append(obs[:, i, :])
            self.buffer['actions'][i].append(actions[:, i, :])
            self.buffer['rewards'][i].append(rewards[:, i])
            self.buffer['values'][i].append(values[:, i, 0])
            self.buffer['log_probs'][i].append(log_probs[:, i, 0])
            self.buffer['dones'][i].append(dones[:, i])

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, dones: np.ndarray):
        """GAE computation on (T, num_envs) matrices."""
        T = rewards.shape[0]
        adv = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(T)):
            nv = next_values if t == T-1 else values[t+1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            adv[t] = last_gae
        return adv, adv + values

    def update(self, next_obs: np.ndarray, epochs: int = 5, batch_size: int = 128):
        """Batched PPO update."""
        next_values = self.get_values(next_obs) # (num_envs, num_agents, 1)
        
        for i in range(self.num_agents):
            # T steps, N envs -> T*N samples
            obs = np.concatenate(self.buffer['obs'][i], axis=0) # (T*N, obs_dim)
            actions = np.concatenate(self.buffer['actions'][i], axis=0)
            old_lps = np.concatenate(self.buffer['log_probs'][i], axis=0)
            
            # (T, N) matrices for GAE
            rew_m = np.stack(self.buffer['rewards'][i], axis=0)
            val_m = np.stack(self.buffer['values'][i], axis=0)
            don_m = np.stack(self.buffer['dones'][i], axis=0)
            
            adv, ret = self.compute_gae(rew_m, val_m, next_values[:, i, 0], don_m)
            adv = adv.reshape(-1, 1)
            ret = ret.reshape(-1, 1)
            old_vals = val_m.reshape(-1, 1)
            
            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # Prepare Tensors
            if self.use_lstm:
                # Optimized Vectorized Sequence Building:
                # We flatten the (T, N) steps into (T*N) samples, but sequences must not cross env boundaries.
                # Since the buffer is built from sequential steps, we can use a sliding window if we are careful.
                # However, for simplicity and correctness with sequences, we'll rebuild sequences properly.
                seq_obs_list = []
                # Each agent's obs buffer has T entries, each is (num_envs, obs_dim)
                raw_obs_3d = np.stack(self.buffer['obs'][i], axis=0) # (T, N, obs_dim)
                for env_idx in range(self.num_envs):
                    env_obs = raw_obs_3d[:, env_idx, :] # (T, obs_dim)
                    for t in range(len(env_obs)):
                        start = max(0, t - self.sequence_length + 1)
                        seq = env_obs[start:t+1]
                        if len(seq) < self.sequence_length:
                            # Pad with the very first observation of this sequence
                            seq = np.concatenate([np.tile(seq[0], (self.sequence_length - len(seq), 1)), seq], axis=0)
                        seq_obs_list.append(seq)
                obs_tensor = torch.FloatTensor(np.array(seq_obs_list)).to(self.device)
            else:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            act_t = torch.FloatTensor(actions).to(self.device)
            lp_t = torch.FloatTensor(old_lps).to(self.device).unsqueeze(1)
            adv_t = torch.FloatTensor(adv).to(self.device)
            ret_t = torch.FloatTensor(ret).to(self.device)
            val_t = torch.FloatTensor(old_vals).to(self.device)

            indices = np.arange(len(obs))
            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, len(obs), batch_size):
                    idx = indices[start:start+batch_size]
                    
                    # Actor
                    if self.use_lstm:
                        curr_lp, ent, _ = self.actors[i].evaluate_actions(obs_tensor[idx], act_t[idx])
                    else:
                        curr_lp, ent = self.actors[i].evaluate_actions(obs_tensor[idx], act_t[idx])
                    
                    ratio = torch.exp(curr_lp - lp_t[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_t[idx]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()
                    
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
                    
                    # Critic
                    if self.use_lstm:
                        curr_v, _ = self.critics[i](obs_tensor[idx])
                    else:
                        curr_v = self.critics[i](obs_tensor[idx])
                        
                    if self.value_clip:
                        v_clipped = val_t[idx] + torch.clamp(curr_v - val_t[idx], -self.clip_epsilon, self.clip_epsilon)
                        v_loss = torch.max((curr_v - ret_t[idx]).pow(2), (v_clipped - ret_t[idx]).pow(2)).mean()
                    else:
                        v_loss = (curr_v - ret_t[idx]).pow(2).mean()
                        
                    self.critic_optimizers[i].zero_grad()
                    (self.value_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
                    self.critic_optimizers[i].step()
        
        self.reset_buffer()

    def save(self, path):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.critics[i].load_state_dict(ckpt['critics'][i])
