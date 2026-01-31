# --- networks.py ---
# Dual Network (Policy + Value) with LSTM for MCTS
# Optimized version with bug fixes and performance improvements.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

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
        
        self.obs_dim = obs_dim  # Store for external access
        self.use_lstm = use_lstm
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim  # Store for external access
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
    
    def _encode_obs(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: bool = False,
    ):
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

    def forward_actor(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: bool = False,
    ):
        """Actor forward: returns policy (mean,std) and optional hidden state."""
        x, new_hidden = self._encode_obs(obs, hidden_state, return_sequence=return_sequence)

        policy_x = F.relu(self.policy_fc(x))
        policy_mean = torch.tanh(self.policy_mean(policy_x))
        log_std = self.policy_log_std(policy_x)
        log_std = torch.clamp(log_std, min=-5.0, max=1.0)
        policy_std = torch.exp(log_std)
        
        return policy_mean, policy_std, new_hidden

    def forward_value_local(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: bool = False,
    ):
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

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: bool = False,
    ):
        """Backward-compatible forward: returns (mean, std, local_value, hidden)."""
        mean, std, new_hidden = self.forward_actor(obs, hidden_state, return_sequence=return_sequence)
        value, _ = self.forward_value_local(obs, hidden_state, return_sequence=return_sequence)
        if self.use_lstm:
            return mean, std, value, new_hidden
        return mean, std, value, None
    
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
            log_prob = torch.zeros(policy_mean.shape[0], 1, device=policy_mean.device)
        else:
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            action = torch.clamp(action, -1.0, 1.0)
        
        # FIXED: Proper return statement structure
        if self.use_lstm:
            return action, log_prob, new_hidden
        return action, log_prob

    def next_hidden(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
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


class MCTSInferWrapper(nn.Module):
    """
    TorchScript-compatible wrapper for MCTS inference.
    
    Uses nn.Module + @torch.jit.export (modern API) so that
    torch.jit.script() correctly exports infer_policy_value and
    infer_next_hidden as callable methods after save/load.
    
    NOTE: The old torch.jit.ScriptModule + @torch.jit.script_method API
    conflicts with torch.jit.script() called in train.py, causing exported
    methods to be silently dropped. This version uses the modern API.
    
    All network layers are copied directly (not held as sub-module reference)
    to ensure TorchScript can properly trace all operations.
    """
    
    __constants__ = ['obs_dim', 'lstm_hidden_dim', 'use_lstm']
    
    def __init__(self, net: DualNetwork):
        super().__init__()
        
        # Store constants
        self.obs_dim: int = net.obs_dim
        self.lstm_hidden_dim: int = net.lstm_hidden_dim
        self.use_lstm: bool = net.use_lstm
        
        # Copy all layers directly (creates new modules with same weights)
        # This is crucial - referencing net.xxx would cause TorchScript issues
        self.fc_input = nn.Linear(net.fc_input.in_features, net.fc_input.out_features)
        self.fc_input.load_state_dict(net.fc_input.state_dict())
        
        self.ln_input = nn.LayerNorm(net.ln_input.normalized_shape)
        self.ln_input.load_state_dict(net.ln_input.state_dict())
        
        self.lstm = nn.LSTM(net.lstm.input_size, net.lstm.hidden_size, batch_first=True)
        self.lstm.load_state_dict(net.lstm.state_dict())
        
        self.fc_shared = nn.Linear(net.fc_shared.in_features, net.fc_shared.out_features)
        self.fc_shared.load_state_dict(net.fc_shared.state_dict())
        
        self.ln_shared = nn.LayerNorm(net.ln_shared.normalized_shape)
        self.ln_shared.load_state_dict(net.ln_shared.state_dict())
        
        self.fc_shared2 = nn.Linear(net.fc_shared2.in_features, net.fc_shared2.out_features)
        self.fc_shared2.load_state_dict(net.fc_shared2.state_dict())
        
        self.ln_shared2 = nn.LayerNorm(net.ln_shared2.normalized_shape)
        self.ln_shared2.load_state_dict(net.ln_shared2.state_dict())
        
        self.policy_fc = nn.Linear(net.policy_fc.in_features, net.policy_fc.out_features)
        self.policy_fc.load_state_dict(net.policy_fc.state_dict())
        
        self.policy_mean = nn.Linear(net.policy_mean.in_features, net.policy_mean.out_features)
        self.policy_mean.load_state_dict(net.policy_mean.state_dict())
        
        self.policy_log_std = nn.Linear(net.policy_log_std.in_features, net.policy_log_std.out_features)
        self.policy_log_std.load_state_dict(net.policy_log_std.state_dict())
        
        self.value_fc = nn.Linear(net.value_fc.in_features, net.value_fc.out_features)
        self.value_fc.load_state_dict(net.value_fc.state_dict())
        
        self.value_head = nn.Linear(net.value_head.in_features, net.value_head.out_features)
        self.value_head.load_state_dict(net.value_head.state_dict())
        
        self.action_embed = nn.Linear(net.action_embed.in_features, net.action_embed.out_features)
        self.action_embed.load_state_dict(net.action_embed.state_dict())
        
        self.lstm_step = nn.LSTMCell(net.lstm_step.input_size, net.lstm_step.hidden_size)
        self.lstm_step.load_state_dict(net.lstm_step.state_dict())

    def _encode_obs(
        self,
        obs: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode observation sequence and return features + new hidden state."""
        # Ensure 3D: (B, T, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        # Input projection
        x = self.fc_input(obs)
        x = self.ln_input(x)
        x = F.relu(x)
        
        # LSTM
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        # Take last timestep
        x = lstm_out[:, -1, :]
        
        # Shared layers
        x = self.fc_shared(x)
        x = self.ln_shared(x)
        x = F.relu(x)
        x = self.fc_shared2(x)
        x = self.ln_shared2(x)
        x = F.relu(x)
        
        return x, new_hidden

    @torch.jit.export
    def infer_policy_value(
        self, 
        obs_seq: torch.Tensor, 
        h: torch.Tensor, 
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer policy (mean, std) and value from observation sequence.
        
        Args:
            obs_seq: (B, T, obs_dim) observation sequence
            h: (B, H) or (1, B, H) LSTM hidden state
            c: (B, H) or (1, B, H) LSTM cell state
            
        Returns:
            mean: (B, action_dim) policy mean
            std: (B, action_dim) policy std  
            value: (B, 1) state value
        """
        # Normalize h/c to (1, B, H)
        if h.dim() == 2:
            h_in = h.unsqueeze(0)
            c_in = c.unsqueeze(0)
        else:
            h_in = h
            c_in = c
        
        hidden = (h_in, c_in)
        x, _ = self._encode_obs(obs_seq, hidden)
        
        # Policy head
        policy_x = F.relu(self.policy_fc(x))
        mean = torch.tanh(self.policy_mean(policy_x))
        log_std = self.policy_log_std(policy_x)
        log_std = torch.clamp(log_std, -5.0, 1.0)
        std = torch.exp(log_std)
        
        # Value head
        value_x = F.relu(self.value_fc(x))
        value = self.value_head(value_x)
        
        return mean, std, value

    @torch.jit.export
    def infer_next_hidden(
        self, 
        obs_seq: torch.Tensor, 
        h: torch.Tensor, 
        c: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action-conditioned next LSTM hidden state.
        
        Args:
            obs_seq: (B, T, obs_dim) observation sequence
            h: (B, H) or (1, B, H) LSTM hidden state
            c: (B, H) or (1, B, H) LSTM cell state
            action: (B, action_dim) action taken
            
        Returns:
            h_next: (1, B, H) next hidden state
            c_next: (1, B, H) next cell state
        """
        # Normalize h/c
        if h.dim() == 2:
            h_in = h.unsqueeze(0)
            c_in = c.unsqueeze(0)
        else:
            h_in = h
            c_in = c
        
        # Take last observation
        if obs_seq.dim() == 3:
            obs_step = obs_seq[:, -1, :]
        else:
            obs_step = obs_seq
        
        # Encode observation
        x_obs = self.fc_input(obs_step)
        x_obs = self.ln_input(x_obs)
        x_obs = F.relu(x_obs)
        
        # Encode action
        x_act = self.action_embed(action)
        x_act = F.relu(x_act)
        
        # Concatenate
        x = torch.cat([x_obs, x_act], dim=-1)
        
        # Get (B, H) from (1, B, H)
        h0 = h_in[0]
        c0 = c_in[0]
        
        # LSTMCell step
        h1, c1 = self.lstm_step(x, (h0, c0))
        
        # Return as (1, B, H)
        return h1.unsqueeze(0), c1.unsqueeze(0)
    
    @torch.jit.export
    def infer_expand(
        self,
        obs_seq: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused inference for MCTS node expansion: policy/value + next hidden states.
        
        This method combines infer_policy_value and infer_next_hidden into a single
        call, reducing the run_method dispatch overhead in the C++ backend.
        
        Use case: In MCTS expand_node, we need both:
        1. Policy distribution (mean, std) and value for the current node
        2. Next hidden states for each sampled action (to store in edges)
        
        Args:
            obs_seq: (B, T, obs_dim) observation sequence
                     Note: For expand_node, B = num_action_samples, and all B rows
                     contain the same obs_seq (replicated for batched next_hidden)
            h: (1, B, H) LSTM hidden state (same h replicated B times)
            c: (1, B, H) LSTM cell state (same c replicated B times)
            actions: (B, 2) sampled actions for computing next hidden states
            
        Returns:
            mean: (1, 2) policy mean (only first sample needed, all identical)
            std: (1, 2) policy std
            value: (1, 1) state value
            h_next: (B, H) next hidden states for each action
            c_next: (B, H) next cell states for each action
        """
        # Normalize h/c to (1, B, H)
        if h.dim() == 2:
            h_in = h.unsqueeze(0)
            c_in = c.unsqueeze(0)
        else:
            h_in = h
            c_in = c
        
        B = obs_seq.size(0)
        
        # ========== Part 1: Policy and Value (only need first sample) ==========
        # Take first sample for policy/value (all obs_seq rows are identical)
        obs_first = obs_seq[0:1]  # (1, T, obs_dim)
        h_first = h_in[:, 0:1, :]  # (1, 1, H)
        c_first = c_in[:, 0:1, :]  # (1, 1, H)
        
        hidden_first = (h_first, c_first)
        x, _ = self._encode_obs(obs_first, hidden_first)
        
        # Policy head
        policy_x = F.relu(self.policy_fc(x))
        mean = torch.tanh(self.policy_mean(policy_x))  # (1, 2)
        log_std = self.policy_log_std(policy_x)
        log_std = torch.clamp(log_std, -5.0, 1.0)
        std = torch.exp(log_std)  # (1, 2)
        
        # Value head
        value_x = F.relu(self.value_fc(x))
        value = self.value_head(value_x)  # (1, 1)
        
        # ========== Part 2: Next Hidden States (batched for all actions) ==========
        # Take last observation for all samples
        if obs_seq.dim() == 3:
            obs_step = obs_seq[:, -1, :]  # (B, obs_dim)
        else:
            obs_step = obs_seq  # (B, obs_dim)
        
        # Encode observations
        x_obs = self.fc_input(obs_step)
        x_obs = self.ln_input(x_obs)
        x_obs = F.relu(x_obs)  # (B, embed_dim)
        
        # Encode actions
        x_act = self.action_embed(actions)
        x_act = F.relu(x_act)  # (B, embed_dim)
        
        # Concatenate
        x_combined = torch.cat([x_obs, x_act], dim=-1)  # (B, embed_dim * 2)
        
        # Get (B, H) from (1, B, H)
        h0 = h_in[0]  # (B, H)
        c0 = c_in[0]  # (B, H)
        
        # LSTMCell step (batched)
        h_next, c_next = self.lstm_step(x_combined, (h0, c0))  # both (B, H)
        
        return mean, std, value, h_next, c_next

    @torch.jit.export
    def infer_batch_policy_value(
        self,
        obs_seq_batch: torch.Tensor,
        h_batch: torch.Tensor,
        c_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched policy/value inference for multiple agents.
        
        This is a thin wrapper around infer_policy_value that makes the batching
        semantics explicit. Useful for rollout where we want to infer for all
        agents in a single call.
        
        Args:
            obs_seq_batch: (B, T, obs_dim) stacked observation sequences
            h_batch: (1, B, H) stacked hidden states (or zeros for non-ego)
            c_batch: (1, B, H) stacked cell states
            
        Returns:
            mean: (B, 2) policy means for all agents
            std: (B, 2) policy stds
            value: (B, 1) state values
        """
        return self.infer_policy_value(obs_seq_batch, h_batch, c_batch)
    
    def forward(
        self, 
        obs_seq: torch.Tensor, 
        h: torch.Tensor, 
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass (alias for infer_policy_value)."""
        return self.infer_policy_value(obs_seq, h, c)