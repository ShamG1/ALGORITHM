# --- networks.py ---
# Dual Network (Policy + Value) with LSTM for MCTS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Handle both relative and absolute imports
try:
    from .utils import OBS_DIM
except ImportError:
    from utils import OBS_DIM


class DualNetwork(nn.Module):
    """
    Dual Network combining policy and value networks with shared LSTM backbone.
    Similar to AlphaZero architecture.

    CTDE extension:
    - Actor (policy) remains decentralized: input is local obs.
    - Centralized critic (optional): input is fixed-size global_state.
      This critic is intended for training only.
    """
    
    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=2,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        sequence_length=5,
        global_state_dim: int = 24,
        use_centralized_critic: bool = True,
    ):
        super(DualNetwork, self).__init__()
        
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.global_state_dim = int(global_state_dim)
        self.use_centralized_critic = bool(use_centralized_critic)
        
        # Shared input projection (+norm for stability)
        self.fc_input = nn.Linear(obs_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)
        
        if self.use_lstm:
            # Shared LSTM layer
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            # Action-conditioned transition (for MCTS branch-specific hidden propagation)
            self.action_embed = nn.Linear(action_dim, hidden_dim)
            self.lstm_step = nn.LSTMCell(hidden_dim + hidden_dim, lstm_hidden_dim)
            # Shared post-LSTM layers
            self.fc_shared = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc_shared2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln_shared = nn.LayerNorm(hidden_dim)
            self.ln_shared2 = nn.LayerNorm(hidden_dim)
        else:
            # Standard MLP layers
            self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
            self.fc_shared2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln_shared = nn.LayerNorm(hidden_dim)
            self.ln_shared2 = nn.LayerNorm(hidden_dim)
        
        # Policy head (outputs action distribution)
        self.policy_fc = nn.Linear(hidden_dim, hidden_dim)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        # Predict log_std and clamp for numerical stability
        self.policy_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Local value head (outputs state value from local features)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Centralized critic head (optional): value from global state
        if self.use_centralized_critic:
            self.global_fc1 = nn.Linear(self.global_state_dim, hidden_dim)
            self.global_ln1 = nn.LayerNorm(hidden_dim)
            self.global_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.global_ln2 = nn.LayerNorm(hidden_dim)
            self.global_value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)
    
    def _encode_obs(self, obs, hidden_state=None, return_sequence: bool = False):
        """Encode local observation(s) into a hidden feature representation."""
        if self.use_lstm:
            # Handle both sequence and single step inputs
            if len(obs.shape) == 2:
                # Single step: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
                obs = obs.unsqueeze(1)
            
            # Project input
            x = self.fc_input(obs)  # (B, T, hidden_dim)
            x = self.ln_input(x)
            x = F.relu(x)
            
            # LSTM
            if hidden_state is None:
                lstm_out, new_hidden = self.lstm(x)
            else:
                lstm_out, new_hidden = self.lstm(x, hidden_state)
            
            if return_sequence:
                # Keep per-timestep features for TBPTT training
                x = self.fc_shared(lstm_out)  # (B, T, hidden_dim)
                x = self.ln_shared(x)
                x = F.relu(x)
                x = self.fc_shared2(x)
                x = self.ln_shared2(x)
                x = F.relu(x)  # (B, T, hidden_dim)
                return x, new_hidden

            # Use last timestep output (inference / compatibility path)
            x = lstm_out[:, -1, :]  # (B, H_lstm)
            x = self.fc_shared(x)
            x = self.ln_shared(x)
            x = F.relu(x)
            x = self.fc_shared2(x)
            x = self.ln_shared2(x)
            x = F.relu(x)
            return x, new_hidden

        # Standard MLP
        x = self.fc_input(obs)
        x = self.ln_input(x)
        x = F.relu(x)
        x = self.fc_shared(x)
        x = self.ln_shared(x)
        x = F.relu(x)
        x = self.fc_shared2(x)
        x = self.ln_shared2(x)
        x = F.relu(x)
        return x, None

    def forward_actor(self, obs, hidden_state=None, return_sequence: bool = False):
        """Actor forward: returns policy (mean,std) and optional hidden state."""
        x, new_hidden = self._encode_obs(obs, hidden_state, return_sequence=return_sequence)

        policy_x = F.relu(self.policy_fc(x))
        policy_mean = torch.tanh(self.policy_mean(policy_x))
        log_std = self.policy_log_std(policy_x)
        log_std = torch.clamp(log_std, min=-5.0, max=1.0)
        policy_std = torch.exp(log_std)
        
        return policy_mean, policy_std, new_hidden

    def forward_value_local(self, obs, hidden_state=None, return_sequence: bool = False):
        """Local value head: value from local obs features (kept for MCTS / execution path)."""
        x, new_hidden = self._encode_obs(obs, hidden_state, return_sequence=return_sequence)
        value_x = F.relu(self.value_fc(x))
        value = self.value_head(value_x)
        return value, new_hidden

    def forward_value_global(self, global_state: torch.Tensor):
        """Centralized critic: value from fixed-size global_state (training only)."""
        if not self.use_centralized_critic:
            raise RuntimeError("forward_value_global called but use_centralized_critic=False")

        # Expect (B, global_state_dim)
        x = self.global_fc1(global_state)
        x = self.global_ln1(x)
        x = F.relu(x)
        x = self.global_fc2(x)
        x = self.global_ln2(x)
        x = F.relu(x)
        v = self.global_value_head(x)
        return v

    def forward(self, obs, hidden_state=None, return_sequence: bool = False):
        """Backward-compatible forward: returns (mean, std, local_value, hidden)."""
        mean, std, new_hidden = self.forward_actor(obs, hidden_state, return_sequence=return_sequence)
        value, _ = self.forward_value_local(obs, hidden_state, return_sequence=return_sequence)
        if self.use_lstm:
            return mean, std, value, new_hidden
        return mean, std, value
    
    def get_policy(self, obs, hidden_state=None):
        """Get policy distribution (mean and std)."""
        mean, std, new_hidden = self.forward_actor(obs, hidden_state, return_sequence=False)
        return mean, std, new_hidden
    
    def get_value(self, obs, hidden_state=None):
        """Get state value estimate (local value head)."""
        value, new_hidden = self.forward_value_local(obs, hidden_state, return_sequence=False)
        return value, new_hidden
    
    def sample_action(self, obs, hidden_state=None, deterministic=False):
        """Sample action from policy distribution."""
        policy_mean, policy_std, new_hidden = self.forward_actor(obs, hidden_state, return_sequence=False)
        
        if deterministic:
            action = policy_mean
            log_prob = torch.zeros(policy_mean.shape[0], 1).to(policy_mean.device)
        else:
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            action = torch.clamp(action, -1.0, 1.0)
        
        if self.use_lstm:
            return action, log_prob, new_hidden
            return action, log_prob

    def next_hidden(self, obs, action, hidden_state=None):
        """Compute action-conditioned next LSTM hidden state."""
        if not self.use_lstm:
            raise RuntimeError("next_hidden() requires use_lstm=True")

        if obs.dim() == 3:
            obs_step = obs[:, -1, :]
        else:
            obs_step = obs

        x_obs = self.fc_input(obs_step)
        x_obs = self.ln_input(x_obs)
        x_obs = F.relu(x_obs)

        x_act = self.action_embed(action)
        x_act = F.relu(x_act)

        x = torch.cat([x_obs, x_act], dim=-1)

        B = x.shape[0]
        H = self.lstm.hidden_size

        if hidden_state is None:
            h0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
            if h.dim() == 3:
                h0 = h[0]
                c0 = c[0]
            else:
                h0 = h
                c0 = c

        h1, c1 = self.lstm_step(x, (h0, c0))
        return (h1.unsqueeze(0), c1.unsqueeze(0))
