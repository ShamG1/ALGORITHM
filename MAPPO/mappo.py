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
        use_tcn: bool = False,
        sequence_length: int = 5,
        lstm_hidden_dim: int = 128,
        num_envs: int = 1,
        cooperative_mode: bool = False
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = obs_dim * num_agents
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
        self.use_tcn = use_tcn
        self.use_seq_model = self.use_lstm or self.use_tcn
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_envs = num_envs
        self.cooperative_mode = cooperative_mode
        # Create actor and critic networks for each agent
        self.actors = nn.ModuleList([
            Actor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                use_lstm=use_lstm,
                use_tcn=use_tcn,
                sequence_length=sequence_length,
            ).to(self.device)
            for _ in range(num_agents)
        ])
        self.critics = nn.ModuleList([
            Critic(
                obs_dim=self.global_obs_dim,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                use_lstm=use_lstm,
                use_tcn=use_tcn,
                sequence_length=sequence_length,
            ).to(self.device)
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
            'global_obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'values': [[] for _ in range(self.num_agents)],
            'log_probs': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
        }
        if self.use_seq_model:
            # actor_hidden_states[agent_idx] = (h, c) where h/c is (1, num_envs, lstm_hidden_dim)
            self.actor_hidden_states = [None for _ in range(self.num_agents)]
            self.critic_hidden_states = [None for _ in range(self.num_agents)]
            # Actor local history: [agent][env] -> deque(local_obs)
            self.obs_history = [[deque(maxlen=self.sequence_length) for _ in range(self.num_envs)] for _ in range(self.num_agents)]
            # Critic global history: [env] -> deque(global_obs)
            self.global_obs_history = [deque(maxlen=self.sequence_length) for _ in range(self.num_envs)]
    def reset_env_states(self, env_idx: int):
        """Reset LSTM states and history for a specific environment (called on env reset)."""
        if not self.use_seq_model:
            return
        self.global_obs_history[env_idx].clear()
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
            if self.use_seq_model:
                seq_obs = self._prepare_obs_history(i, agent_obs)
                if deterministic:
                    action, new_h = self.actors[i].get_action(seq_obs, deterministic=True, hidden_state=self.actor_hidden_states[i])
                    lp = torch.zeros((self.num_envs, 1), device=self.device)
                else:
                    action, lp, new_h = self.actors[i].get_action(seq_obs, deterministic=False, hidden_state=self.actor_hidden_states[i])
                if self.use_lstm:
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
    def _build_global_obs(self, obs: np.ndarray) -> np.ndarray:
        """Build centralized critic input: (num_envs, num_agents * obs_dim)."""
        return obs.reshape(self.num_envs, -1)

    def _prepare_global_obs_history(self, global_obs: np.ndarray) -> torch.Tensor:
        """Update and return batched global sequences (num_envs, seq_len, global_obs_dim)."""
        batch_seq = []
        for env_idx in range(self.num_envs):
            self.global_obs_history[env_idx].append(global_obs[env_idx])
            hist = list(self.global_obs_history[env_idx])
            while len(hist) < self.sequence_length:
                hist.insert(0, hist[0])
            batch_seq.append(np.array(hist))
        return torch.FloatTensor(np.array(batch_seq)).to(self.device)

    def get_values(self, obs: np.ndarray) -> np.ndarray:
        """Get centralized values for all agents in all envs."""
        global_obs = self._build_global_obs(obs)

        if self.use_seq_model:
            seq_global_obs = self._prepare_global_obs_history(global_obs)

        all_values = []
        for i in range(self.num_agents):
            if self.use_seq_model:
                val, new_h = self.critics[i](seq_global_obs, hidden_state=self.critic_hidden_states[i])
                if self.use_lstm:
                    self.critic_hidden_states[i] = new_h
            else:
                obs_t = torch.FloatTensor(global_obs).to(self.device)
                val = self.critics[i](obs_t)
            all_values.append(val.detach().cpu().numpy())
        return np.stack(all_values, axis=1) # (num_envs, num_agents, 1)
    def store_transition(self, obs, actions, rewards, values, log_probs, dones):
        """Store transitions (num_envs, num_agents, ...)."""
        if self.cooperative_mode:
            # Shared team reward / done for all agents in each environment
            team_rewards = rewards.mean(axis=1, keepdims=True)  # (num_envs, 1)
            team_rewards = np.repeat(team_rewards, self.num_agents, axis=1)  # (num_envs, num_agents)
            env_done = np.any(dones, axis=1, keepdims=True).astype(dones.dtype)  # (num_envs, 1)
            team_dones = np.repeat(env_done, self.num_agents, axis=1)  # (num_envs, num_agents)
        else:
            team_rewards = rewards
            team_dones = dones

        global_obs = self._build_global_obs(obs)
        for i in range(self.num_agents):
            self.buffer['obs'][i].append(obs[:, i, :])
            self.buffer['global_obs'][i].append(global_obs)
            self.buffer['actions'][i].append(actions[:, i, :])
            self.buffer['rewards'][i].append(team_rewards[:, i])
            self.buffer['values'][i].append(values[:, i, 0])
            self.buffer['log_probs'][i].append(log_probs[:, i, 0])
            self.buffer['dones'][i].append(team_dones[:, i])
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
            obs_local = np.concatenate(self.buffer['obs'][i], axis=0) # (T*N, obs_dim)
            obs_global = np.concatenate(self.buffer['global_obs'][i], axis=0) # (T*N, global_obs_dim)
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
            if self.use_seq_model:
                # Build actor local sequences
                seq_obs_local_list = []
                raw_obs_local_3d = np.stack(self.buffer['obs'][i], axis=0) # (T, N, obs_dim)
                for env_idx in range(self.num_envs):
                    env_obs = raw_obs_local_3d[:, env_idx, :] # (T, obs_dim)
                    for t in range(len(env_obs)):
                        start = max(0, t - self.sequence_length + 1)
                        seq = env_obs[start:t+1]
                        if len(seq) < self.sequence_length:
                            seq = np.concatenate([np.tile(seq[0], (self.sequence_length - len(seq), 1)), seq], axis=0)
                        seq_obs_local_list.append(seq)

                # Build critic global sequences
                seq_obs_global_list = []
                raw_obs_global_3d = np.stack(self.buffer['global_obs'][i], axis=0) # (T, N, global_obs_dim)
                for env_idx in range(self.num_envs):
                    env_obs = raw_obs_global_3d[:, env_idx, :] # (T, global_obs_dim)
                    for t in range(len(env_obs)):
                        start = max(0, t - self.sequence_length + 1)
                        seq = env_obs[start:t+1]
                        if len(seq) < self.sequence_length:
                            seq = np.concatenate([np.tile(seq[0], (self.sequence_length - len(seq), 1)), seq], axis=0)
                        seq_obs_global_list.append(seq)

                obs_tensor_actor = torch.FloatTensor(np.array(seq_obs_local_list)).to(self.device)
                obs_tensor_critic = torch.FloatTensor(np.array(seq_obs_global_list)).to(self.device)
            else:
                obs_tensor_actor = torch.FloatTensor(obs_local).to(self.device)
                obs_tensor_critic = torch.FloatTensor(obs_global).to(self.device)
            act_t = torch.FloatTensor(actions).to(self.device)
            lp_t = torch.FloatTensor(old_lps).to(self.device).unsqueeze(1)
            adv_t = torch.FloatTensor(adv).to(self.device)
            ret_t = torch.FloatTensor(ret).to(self.device)
            val_t = torch.FloatTensor(old_vals).to(self.device)
            indices = np.arange(len(obs_local))
            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, len(obs_local), batch_size):
                    idx = indices[start:start+batch_size]
                    # Actor
                    if self.use_seq_model:
                        curr_lp, ent, _ = self.actors[i].evaluate_actions(obs_tensor_actor[idx], act_t[idx])
                    else:
                        curr_lp, ent = self.actors[i].evaluate_actions(obs_tensor_actor[idx], act_t[idx])
                    ratio = torch.exp(curr_lp - lp_t[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_t[idx]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
                    # Critic
                    if self.use_seq_model:
                        curr_v, _ = self.critics[i](obs_tensor_critic[idx])
                    else:
                        curr_v = self.critics[i](obs_tensor_critic[idx])
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
