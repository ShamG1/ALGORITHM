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
    """
    
    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=2,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        sequence_length=5
    ):
        super(DualNetwork, self).__init__()
        
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        
        # Shared input projection (+norm for stability)
        self.fc_input = nn.Linear(obs_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)
        
        if self.use_lstm:
            # Shared LSTM layer
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
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
        
        # Value head (outputs state value)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
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
    
    def forward(self, obs, hidden_state=None, return_sequence: bool = False):
        """
        Forward pass through dual network.
        
        Args:
            obs: Observation tensor
                - If use_lstm=False: (batch_size, obs_dim)
                - If use_lstm=True: (batch_size, sequence_length, obs_dim) or (batch_size, obs_dim) for single step
            hidden_state: LSTM hidden state tuple (h, c) if use_lstm=True
        
        Returns:
            policy_mean: Action mean (batch_size, action_dim)
            policy_std: Action std (batch_size, action_dim)
            value: State value (batch_size, 1)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_lstm:
            # Handle both sequence and single step inputs
            if len(obs.shape) == 2:
                # Single step: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
                obs = obs.unsqueeze(1)
            
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]
            
            # Project input
            x = self.fc_input(obs)  # (batch_size, seq_len, hidden_dim)
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
            else:
                # Use last timestep output (inference / compatibility path)
                x = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
                x = self.fc_shared(x)
                x = self.ln_shared(x)
                x = F.relu(x)
                x = self.fc_shared2(x)
                x = self.ln_shared2(x)
                x = F.relu(x)
        else:
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
            new_hidden = None
        
        # Policy head
        policy_x = F.relu(self.policy_fc(x))
        policy_mean = torch.tanh(self.policy_mean(policy_x))  # Mean in [-1, 1]
        # log_std clamp for stability
        log_std = self.policy_log_std(policy_x)
        log_std = torch.clamp(log_std, min=-5.0, max=1.0)  # std in ~[0.0067, 2.718]
        policy_std = torch.exp(log_std)
        
        # Value head
        value_x = F.relu(self.value_fc(x))
        value = self.value_head(value_x)
        
        if self.use_lstm:
            return policy_mean, policy_std, value, new_hidden
        else:
            return policy_mean, policy_std, value
    
    def get_policy(self, obs, hidden_state=None):
        """Get policy distribution (mean and std)."""
        if self.use_lstm:
            policy_mean, policy_std, _, new_hidden = self.forward(obs, hidden_state)
            return policy_mean, policy_std, new_hidden
        else:
            policy_mean, policy_std, _ = self.forward(obs)
            return policy_mean, policy_std
    
    def get_value(self, obs, hidden_state=None):
        """Get state value estimate."""
        if self.use_lstm:
            _, _, value, new_hidden = self.forward(obs, hidden_state)
            return value, new_hidden
        else:
            _, _, value = self.forward(obs)
            return value
    
    def sample_action(self, obs, hidden_state=None, deterministic=False):
        """
        Sample action from policy distribution.
        
        Returns:
            action: Sampled action (batch_size, action_dim)
            log_prob: Log probability of action (batch_size, 1)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_lstm:
            policy_mean, policy_std, _, new_hidden = self.forward(obs, hidden_state)
        else:
            policy_mean, policy_std, _ = self.forward(obs)
            new_hidden = None
        
        
        if deterministic:
            action = policy_mean
            log_prob = torch.zeros(policy_mean.shape[0], 1).to(policy_mean.device)
        else:
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # Clip action to [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
        
        if self.use_lstm:
            return action, log_prob, new_hidden
        else:
            return action, log_prob
