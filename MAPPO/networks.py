# --- networks.py ---
# Actor-Critic Networks for MAPPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DriveSimX.core.utils import OBS_DIM


class TCNBlock(nn.Module):
    """Causal temporal convolution block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.ln = nn.LayerNorm(out_channels)
        self.res = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = self.res(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        x = F.relu(self.ln(x.transpose(1, 2)).transpose(1, 2))
        return x + residual


class Actor(nn.Module):
    """Actor network (policy network) for MAPPO with TCN/LSTM/MLP backbone."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=2,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        use_tcn=False,
        sequence_length=5,
    ):
        super(Actor, self).__init__()

        self.use_lstm = use_lstm
        self.use_tcn = use_tcn
        self.sequence_length = sequence_length

        # Input projection
        self.fc_input = nn.Linear(obs_dim, hidden_dim)

        if self.use_tcn:
            self.tcn_blocks = nn.ModuleList([
                TCNBlock(hidden_dim, lstm_hidden_dim, kernel_size=3, dilation=1),
                TCNBlock(lstm_hidden_dim, lstm_hidden_dim, kernel_size=3, dilation=2),
            ])
            self.ln = nn.LayerNorm(lstm_hidden_dim)
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        elif self.use_lstm:
            # LSTM layer
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            # Layer normalization after LSTM for stability
            self.ln = nn.LayerNorm(lstm_hidden_dim)
            # Post-LSTM layers
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Standard MLP layers
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Output: mean and std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

        # Initialize std_head bias to ensure reasonable initial exploration
        nn.init.constant_(self.std_head.bias, 0.5)

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

    def forward(self, obs, hidden_state=None):
        """
        Forward pass through actor network.

        Args:
            obs: Observation tensor
                - If use_lstm/use_tcn=False: (batch_size, obs_dim)
                - If use_lstm/use_tcn=True: (batch_size, sequence_length, obs_dim) or (batch_size, obs_dim)
            hidden_state: LSTM hidden state tuple (h, c) if use_lstm=True

        Returns:
            mean: Action mean (batch_size, action_dim)
            std: Action std (batch_size, action_dim)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_tcn:
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(1)
            x = self.fc_input(obs)  # (B, T, H)
            x = x.transpose(1, 2)   # (B, H, T)
            for block in self.tcn_blocks:
                x = block(x)
            x = x[:, :, -1]  # take last timestep
            x = self.ln(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None
        elif self.use_lstm:
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(1)

            x = F.relu(self.fc_input(obs))

            if hidden_state is None:
                lstm_out, new_hidden = self.lstm(x)
            else:
                lstm_out, new_hidden = self.lstm(x, hidden_state)

            x = lstm_out[:, -1, :]
            x = self.ln(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        else:
            x = F.relu(self.fc_input(obs))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None

        raw_mean = torch.tanh(self.mean_head(x))
        # Action layout: [throttle, steering]
        # Enforce non-negative throttle to disable reverse/brake-like negative acceleration.
        throttle_mean = (raw_mean[..., :1] + 1.0) * 0.5   # [-1,1] -> [0,1]
        steering_mean = raw_mean[..., 1:2]                # keep [-1,1]
        mean = torch.cat([throttle_mean, steering_mean], dim=-1)
        std = F.softplus(self.std_head(x)) + 1e-5

        if self.use_lstm or self.use_tcn:
            return mean, std, new_hidden
        return mean, std

    def get_action(self, obs, deterministic=False, hidden_state=None):
        """Sample action from policy distribution."""
        if self.use_lstm or self.use_tcn:
            mean, std, new_hidden = self.forward(obs, hidden_state)
        else:
            mean, std = self.forward(obs)
            new_hidden = None

        if deterministic:
            if self.use_lstm or self.use_tcn:
                return mean, new_hidden
            return mean

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        action = torch.clamp(action, -1.0, 1.0)
        # throttle in [0,1], steering in [-1,1]
        action[..., :1] = torch.clamp(action[..., :1], 0.0, 1.0)

        if self.use_lstm or self.use_tcn:
            return action, log_prob, new_hidden
        return action, log_prob

    def evaluate_actions(self, obs, actions, hidden_state=None):
        """Evaluate actions and return log probs and entropy."""
        if self.use_lstm or self.use_tcn:
            mean, std, new_hidden = self.forward(obs, hidden_state)
        else:
            mean, std = self.forward(obs)
            new_hidden = None

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        if self.use_lstm or self.use_tcn:
            return log_probs, entropy, new_hidden
        return log_probs, entropy


class Critic(nn.Module):
    """Critic network (value network) for MAPPO with TCN/LSTM/MLP backbone."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        use_tcn=False,
        sequence_length=5,
    ):
        super(Critic, self).__init__()

        self.use_lstm = use_lstm
        self.use_tcn = use_tcn
        self.sequence_length = sequence_length

        # Input projection
        self.fc_input = nn.Linear(obs_dim, hidden_dim)

        if self.use_tcn:
            self.tcn_blocks = nn.ModuleList([
                TCNBlock(hidden_dim, lstm_hidden_dim, kernel_size=3, dilation=1),
                TCNBlock(lstm_hidden_dim, lstm_hidden_dim, kernel_size=3, dilation=2),
            ])
            self.ln = nn.LayerNorm(lstm_hidden_dim)
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        elif self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            self.ln = nn.LayerNorm(lstm_hidden_dim)
            self.fc2 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, 1)

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
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)

    def forward(self, obs, hidden_state=None):
        """
        Forward pass through critic network.

        Args:
            obs: Observation tensor
                - If use_lstm/use_tcn=False: (batch_size, obs_dim)
                - If use_lstm/use_tcn=True: (batch_size, sequence_length, obs_dim) or (batch_size, obs_dim)
            hidden_state: LSTM hidden state tuple (h, c) if use_lstm=True

        Returns:
            value: State value (batch_size, 1)
            hidden_state: Updated LSTM hidden state (if use_lstm=True)
        """
        if self.use_tcn:
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(1)
            x = self.fc_input(obs)
            x = x.transpose(1, 2)
            for block in self.tcn_blocks:
                x = block(x)
            x = x[:, :, -1]
            x = self.ln(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None
        elif self.use_lstm:
            if len(obs.shape) == 2:
                obs = obs.unsqueeze(1)

            x = F.relu(self.fc_input(obs))

            if hidden_state is None:
                lstm_out, new_hidden = self.lstm(x)
            else:
                lstm_out, new_hidden = self.lstm(x, hidden_state)

            x = lstm_out[:, -1, :]
            x = self.ln(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        else:
            x = F.relu(self.fc_input(obs))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            new_hidden = None

        value = self.fc4(x)

        if self.use_lstm or self.use_tcn:
            return value, new_hidden
        return value