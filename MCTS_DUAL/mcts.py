# --- mcts.py ---
# This file is now a thin wrapper for the C++ MCTS implementation.
# Optimized version with bug fixes and performance improvements.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import math
import os

# When running as a package (python -m C_MCTS.train) we can use relative import.
# When running as a script (python C_MCTS/train.py) relative import fails, so fallback to absolute.
try:
    from DriveSimX.core.cpp import cpp_backend  # type: ignore
except Exception:
    try:
        from . import cpp_backend  # type: ignore
    except Exception:
        try:
            import cpp_backend  # type: ignore
        except Exception:
            cpp_backend = None

HAS_SIM_MARL = bool(cpp_backend) and hasattr(cpp_backend, "mcts_search")


class TensorCache:
    """Pre-allocated tensor cache to avoid repeated allocations."""
    
    __slots__ = ('device', 'obs_buffer', 'h_buffer', 'c_buffer', 'initialized', 
                 'obs_dim', 'seq_len', 'hidden_dim', 'batch_size')
    
    def __init__(self, obs_dim: int, seq_len: int, hidden_dim: int, device: torch.device, batch_size: int = 32):
        self.device = device
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.initialized = False
        
        self.obs_buffer: Optional[torch.Tensor] = None
        self.h_buffer: Optional[torch.Tensor] = None
        self.c_buffer: Optional[torch.Tensor] = None
    
    def ensure_initialized(self):
        if self.initialized:
            return
        
        # Pre-allocate buffers
        self.obs_buffer = torch.zeros(self.batch_size, self.obs_dim, dtype=torch.float32)
        self.h_buffer = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float32)
        self.c_buffer = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float32)
        
        # Pin memory for faster CPU->GPU transfer
        if self.device.type != 'cpu':
            self.obs_buffer = self.obs_buffer.pin_memory()
            self.h_buffer = self.h_buffer.pin_memory()
            self.c_buffer = self.c_buffer.pin_memory()
        
        self.initialized = True
    
    def get_obs_tensor(self, obs_batch: List[List[float]]) -> torch.Tensor:
        """Convert observation batch to tensor using pre-allocated buffer."""
        self.ensure_initialized()
        batch_size = len(obs_batch)
        
        if batch_size > self.batch_size:
            # Fallback for larger batches
            return torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        
        # Copy into pre-allocated buffer
        obs_np = np.asarray(obs_batch, dtype=np.float32)
        self.obs_buffer[:batch_size].copy_(torch.from_numpy(obs_np))
        return self.obs_buffer[:batch_size].to(self.device, non_blocking=True)


class MCTS:
    """
    Wrapper for the C++ MCTS implementation.
    This class is now a thin client that calls the C++ backend.
    Optimized version with tensor caching and reduced Python overhead.
    """

    def __init__(
        self,
        network,
        action_space: np.ndarray,  # Kept for API compatibility
        num_simulations: int = 50,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cpu',
        rollout_depth: int = 3,
        env_factory: Optional[Callable] = None,  # Kept for API compatibility
        all_networks: Optional[List] = None,  # Kept for API compatibility
        agent_id: int = 0,
        num_action_samples: int = 5,
        use_SIM_MARL: bool = True,  # Kept for API compatibility
        ts_model_path: Optional[str] = None
    ):
        if not HAS_SIM_MARL:
            raise RuntimeError("C++ MCTS backend is not available. Please build it first.")

        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = torch.device(device)
        self.rollout_depth = rollout_depth
        self.agent_id = agent_id  # Still needed for context
        self.num_action_samples = num_action_samples
        self.ts_model_path = ts_model_path

        # Compatibility stats for training script
        self._rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'total_env_steps': 0,
        }
        
        # Pre-compute and cache these values
        self._lstm_hidden_dim = int(getattr(self.network, 'lstm_hidden_dim', 128))
        self._use_lstm = bool(getattr(self.network, 'use_lstm', False))
        self._use_tcn = bool(getattr(self.network, 'use_tcn', False))
        self._seq_len = int(getattr(self.network, 'sequence_length', 5))
        
        # Initialize tensor cache for faster inference
        obs_dim = int(getattr(self.network, 'obs_dim', 127))
        self._tensor_cache = TensorCache(
            obs_dim=obs_dim,
            seq_len=self._seq_len,
            hidden_dim=self._lstm_hidden_dim,
            device=self.device
        )
        
        # Debug flags
        self._printed_cpp_debug = False
        self._printed_cpp_debug_stats = False

    def get_rollout_stats(self) -> Dict[str, int]:
        # Keep train.py compatible; C++ backend currently doesn't expose these counters.
        return dict(self._rollout_stats)

    def reset_rollout_stats(self) -> None:
        self._rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'total_env_steps': 0,
        }

    def search(
        self,
        root_obs: np.ndarray,
        env: any,
        obs_history: Optional[List] = None,
        hidden_state=None,
        env_state: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:

        if not hasattr(env, 'env') or not hasattr(env.env, 'get_state'):
            raise TypeError(
                "C++ MCTS requires a FastScenarioEnv wrapper with a C++ backend (env.env) that has get_state/set_state methods."
            )

        cpp_env = env.env
        root_state_cpp = cpp_env.get_state()

        # One-time debug: help verify which C++ path is taken and what stats are returned.
        debug_cpp = os.environ.get("MCTS_DEBUG_CPP", "0") == "1"

        lstm_hidden_dim = self._lstm_hidden_dim
        use_lstm = self._use_lstm
        use_tcn = self._use_tcn
        seq_len = self._seq_len
        obs_dim = root_obs.size

        # Construct observation sequence for TCN/LSTM
        # This ensures the search phase also sees the temporal context
        root_obs_seq_tensor = None
        if use_lstm or use_tcn:
            hist = []
            if obs_history:
                hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]
            else:
                # Fallback: if no history provided, use root_obs repeated
                hist = [root_obs.reshape(-1)]
            
            # Non-uniform sampling (seq_len==5): [t, t-1, t-3, t-7, t-15]
            # Falls back to uniform/padded history if insufficient length.
            if seq_len == 5 and use_tcn:
                idx = [-1, -2, -4, -8, -16]
                sampled = []
                for i in idx:
                    if -i <= len(hist):
                        sampled.append(hist[i])
                    else:
                        sampled.append(hist[0])
                sampled = sampled[::-1]  # oldest -> newest
                # Delta features: concat [obs, obs - prev_obs]
                feat = []
                prev = sampled[0]
                for x in sampled:
                    dx = x - prev
                    feat.append(np.concatenate([x, dx], axis=0))
                    prev = x
                hist = feat
                
                # IMPORTANT: Update root_obs to be the latest feature vector (254 dim)
                # This ensures C++ backend and initial inference use the correct dimension.
                root_obs = hist[-1]
                obs_dim = int(root_obs.size)
            else:
                # Ensure history length matches seq_len (uniform)
                if len(hist) < seq_len:
                    pad = [hist[0]] * (seq_len - len(hist))
                    hist = pad + hist
                else:
                    hist = hist[-seq_len:]

            root_obs_seq_tensor = torch.as_tensor(np.stack(hist), device=self.device).unsqueeze(0) # (1, T, D)

        @torch.inference_mode()
        def infer_policy_value(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not obs_batch:
                empty = np.array([])
                return empty, empty, empty

            # Case 1: TCN - infer_fn receives sequences (T*D) or single-step (D)
            if use_tcn:
                obs_np = np.asarray(obs_batch, dtype=np.float32)
                
                # Raw dimension from environment (without delta)
                # If we're using delta features, obs_dim is 2*raw_dim
                is_delta = (seq_len == 5 and obs_dim > 150) # Heuristic to detect delta mode
                raw_obs_dim = obs_dim // 2 if is_delta else obs_dim

                # Handle raw observation from C++ (127) by padding delta=0
                if obs_np.ndim == 2 and obs_np.shape[1] == raw_obs_dim and is_delta:
                    obs_np = np.concatenate([obs_np, np.zeros_like(obs_np)], axis=1)

                # Dynamic adaptation: if C++ only sends single step (D), pad it to (T, D)
                if obs_np.ndim == 2 and obs_np.shape[1] == obs_dim:
                    # Upgrade single frame to sequence using root context as prefix
                    # Shape: (B, D) -> (B, T, D)
                    obs_step = torch.as_tensor(obs_np, device=self.device).unsqueeze(1) # (B, 1, D)
                    if root_obs_seq_tensor is not None:
                        # Use root sequence's first T-1 frames as prefix
                        prefix = root_obs_seq_tensor[:, :seq_len-1, :].expand(obs_np.shape[0], -1, -1)
                        obs_tensor = torch.cat([prefix, obs_step], dim=1)
                    else:
                        # Fallback: repeat single frame
                        obs_tensor = obs_step.expand(-1, seq_len, -1)
                elif obs_np.ndim == 2 and obs_np.shape[1] == seq_len * obs_dim:
                    # Correct sequence batch from mcts_search_seq
                    obs_seq = obs_np.reshape(obs_np.shape[0], seq_len, obs_dim)
                    obs_tensor = torch.as_tensor(obs_seq, device=self.device)
                else:
                    raise RuntimeError(f"TCN infer got unexpected obs shape {obs_np.shape}. Expected {obs_dim} (delta={is_delta}, raw={raw_obs_dim}) or {seq_len*obs_dim}")

                means, stds, values, _ = self.network(obs_tensor)
                return (
                    means.cpu().numpy(),
                    stds.cpu().numpy(),
                    values.view(-1).cpu().numpy(),
                )

            # Case 2: LSTM
            if use_lstm:
                obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
                h_np = np.asarray(h, dtype=np.float32).reshape(-1)
                c_np = np.asarray(c, dtype=np.float32).reshape(-1)

                expected_h = lstm_hidden_dim
                if h_np.size != expected_h or c_np.size != expected_h:
                    h_np = np.zeros((expected_h,), dtype=np.float32)
                    c_np = np.zeros((expected_h,), dtype=np.float32)
                
                h_np = np.tile(h_np[None, :], (obs_tensor.shape[0], 1))
                c_np = np.tile(c_np[None, :], (obs_tensor.shape[0], 1))

                h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], expected_h)
                c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], expected_h)

                means, stds, values, _ = self.network(obs_tensor, (h_t, c_t))
                return (
                    means.cpu().numpy(),
                    stds.cpu().numpy(),
                    values.view(-1).cpu().numpy(),
                )

            # Case 3: Standard MLP
            obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
            means, stds, values, _ = self.network(obs_tensor)
            return (
                means.cpu().numpy(),
                stds.cpu().numpy(),
                values.view(-1).cpu().numpy(),
            )

        # Root hidden state (1,1,H) in training code -> flatten to (H,)
        if hidden_state is not None and isinstance(hidden_state, tuple) and len(hidden_state) == 2:
            h0 = np.asarray(hidden_state[0], dtype=np.float32).reshape(-1)
            c0 = np.asarray(hidden_state[1], dtype=np.float32).reshape(-1)
        else:
            h0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)
            c0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)

        @torch.inference_mode()
        def infer_next_hidden(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray,
            action_batch: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            # This callback is only used by the recurrent (LSTM) C++ MCTS backend.
            if not use_lstm:
                raise RuntimeError("infer_next_hidden called but network.use_lstm=False")

            if not obs_batch:
                empty = np.array([])
                return empty, empty

            obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
            act_tensor = torch.as_tensor(np.asarray(action_batch, dtype=np.float32), device=self.device)

            # h/c expected as (B,H) or flat; convert to (1,B,H)
            h_np = np.asarray(h, dtype=np.float32)
            c_np = np.asarray(c, dtype=np.float32)
            if h_np.ndim == 1:
                h_np = np.tile(h_np[None, :], (obs_tensor.shape[0], 1))
                c_np = np.tile(c_np[None, :], (obs_tensor.shape[0], 1))

            h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], -1)
            c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], -1)

            hn, cn = self.network.next_hidden(obs_tensor, act_tensor, (h_t, c_t))

            # Return (B,H)
            return hn.squeeze(0).cpu().numpy(), cn.squeeze(0).cpu().numpy()

        # Prefer TorchScript recurrent C++ MCTS when available.
        ts_path = self.ts_model_path
        has_ts = bool(ts_path) and isinstance(ts_path, str) and os.path.isfile(ts_path)

        if debug_cpp and not self._printed_cpp_debug:
            try:
                print(
                    "[MCTS_DEBUG_CPP] use_lstm=", use_lstm,
                    "has_ts=", has_ts,
                    "ts_path=", ts_path,
                    "has_mcts_search_lstm_torchscript=", hasattr(cpp_backend, 'mcts_search_lstm_torchscript'),
                    "has_mcts_search_lstm=", hasattr(cpp_backend, 'mcts_search_lstm'),
                    "has_mcts_search=", hasattr(cpp_backend, 'mcts_search'),
                    flush=True,
                )
            except Exception:
                pass
            self._printed_cpp_debug = True

        # Generate random seed once per search
        seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))

        # Flag to track if we should fall back to callback-based MCTS
        use_torchscript_path = False
        
        if use_lstm and has_ts and hasattr(cpp_backend, 'mcts_search_lstm_torchscript'):
            # Try TorchScript path first, but be ready to fall back
            use_torchscript_path = True
            
            if obs_history is None or len(obs_history) == 0:
                # Cannot use TorchScript without obs_history, fall back
                use_torchscript_path = False

        if use_torchscript_path:
            try:
                seq_len = self._seq_len
                obs_dim = int(root_obs.size)

                # Build flattened (T*obs_dim) sequence. Pad left with first frame if history is short.
                hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]
                if len(hist) < seq_len:
                    pad = [hist[0]] * (seq_len - len(hist))
                    hist = pad + hist
                else:
                    hist = hist[-seq_len:]

                root_obs_seq = np.concatenate(hist, axis=0).astype(np.float32)

                action, stats = cpp_backend.mcts_search_lstm_torchscript(
                    cpp_env,
                    root_state_cpp,
                    root_obs_seq.tolist(),
                    seq_len,
                    obs_dim,
                    ts_path,
                    h0.tolist(),
                    c0.tolist(),
                    lstm_hidden_dim,
                    self.agent_id,
                    num_simulations=self.num_simulations,
                    num_action_samples=self.num_action_samples,
                    rollout_depth=self.rollout_depth,
                    c_puct=self.c_puct,
                    temperature=self.temperature,
                    gamma=0.99,
                    dirichlet_alpha=0.3,
                    dirichlet_eps=0.25,
                    seed=seed
                )
            except RuntimeError as e:
                # TorchScript model format mismatch or other error
                error_str = str(e)
                import sys
                
                # Print detailed error for diagnosis
                sys.stderr.write(
                    f"\n[MCTS] TorchScript RuntimeError: {error_str}\n"
                )
                
                if "infer_policy_value" in error_str or "infer_next_hidden" in error_str or "not defined" in error_str:
                    # Method signature mismatch - delete model for regeneration
                    sys.stderr.write(
                        f"[MCTS] Incompatible TorchScript model at {ts_path}. "
                        f"Deleting for regeneration. Using Python callbacks this run.\n"
                    )
                    try:
                        os.remove(ts_path)
                    except Exception:
                        pass
                    # Fall back to callback path for this run only
                    use_torchscript_path = False
                else:
                    raise

        if not use_torchscript_path:
            # Use Python callback path (either LSTM or non-LSTM)
            if use_lstm and hasattr(cpp_backend, 'mcts_search_lstm'):
                # Callback-based recurrent search
                action, stats = cpp_backend.mcts_search_lstm(
                    cpp_env,
                    root_state_cpp,
                    root_obs.tolist(),
                    infer_policy_value,
                    infer_next_hidden,
                    h0.tolist(),
                    c0.tolist(),
                    lstm_hidden_dim,
                    self.agent_id,
                    num_simulations=self.num_simulations,
                    num_action_samples=self.num_action_samples,
                    rollout_depth=self.rollout_depth,
                    c_puct=self.c_puct,
                    temperature=self.temperature,
                    gamma=0.99,
                    dirichlet_alpha=0.3,
                    dirichlet_eps=0.25,
                    seed=seed
                )
            else:
                # TCN sequence-based C++ MCTS (tree maintains obs history per node)
                if use_tcn and hasattr(cpp_backend, 'mcts_search_seq'):
                    if obs_history is None or len(obs_history) == 0:
                        hist = [np.asarray(root_obs, dtype=np.float32).reshape(-1)]
                    else:
                        hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]

                    obs_dim = int(root_obs.size)
                    if len(hist) < self._seq_len:
                        pad = [hist[0]] * (self._seq_len - len(hist))
                        hist = pad + hist
                    else:
                        hist = hist[-self._seq_len:]

                    root_obs_seq_flat = np.concatenate(hist, axis=0).astype(np.float32)

                    action, stats = cpp_backend.mcts_search_seq(
                        cpp_env,
                        root_state_cpp,
                        root_obs_seq_flat.tolist(),
                        self._seq_len,
                        obs_dim,
                        infer_policy_value,
                        self.agent_id,
                        num_simulations=self.num_simulations,
                        num_action_samples=self.num_action_samples,
                        rollout_depth=self.rollout_depth,
                        c_puct=self.c_puct,
                        temperature=self.temperature,
                        gamma=0.99,
                        dirichlet_alpha=0.3,
                        dirichlet_eps=0.25,
                        seed=seed,
                    )
                else:
                    # Fallback: non-LSTM C++ MCTS (single-step obs)
                    action, stats = cpp_backend.mcts_search(
                        cpp_env,
                        root_state_cpp,
                        root_obs.tolist(),
                        infer_policy_value,
                        self.agent_id,
                        num_simulations=self.num_simulations,
                        num_action_samples=self.num_action_samples,
                        rollout_depth=self.rollout_depth,
                        c_puct=self.c_puct,
                        temperature=self.temperature,
                        gamma=0.99,
                        dirichlet_alpha=0.3,
                        dirichlet_eps=0.25,
                        seed=seed
                    )

        # Rough compatibility counters: 1 search == 1 rollout batch.
        self._rollout_stats['total_rollouts'] += 1
        # We don't know success rate/env steps from C++ side yet.

        if debug_cpp and not self._printed_cpp_debug_stats:
            try:
                keys = list(stats.keys()) if isinstance(stats, dict) else None
                print("[MCTS_DEBUG_CPP] returned_stats_keys=", keys, flush=True)
                if isinstance(stats, dict) and ("profile" in stats):
                    print("[MCTS_DEBUG_CPP] profile=", stats.get("profile"), flush=True)
            except Exception:
                pass
            self._printed_cpp_debug_stats = True

        return np.array(action, dtype=np.float32), stats