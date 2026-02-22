# --- mcts.py ---
# This file is now a thin wrapper for the C++ MCTS implementation.
# Optimized version with bug fixes and performance improvements.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import math
import os

import sys

# Add C++ build directory to path to ensure DRIVESIMX_ENV can be loaded
cpp_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DriveSimX/core/cpp/build")
if cpp_build_dir not in sys.path:
    sys.path.insert(0, cpp_build_dir)

try:
    import DRIVESIMX_ENV as cpp_backend  # type: ignore
    if cpp_backend:
        print(f"[DEBUG] Loaded cpp_backend from: {getattr(cpp_backend, '__file__', 'unknown')}")
        print(f"[DEBUG] cpp_backend has mcts_search_to_shm: {hasattr(cpp_backend, 'mcts_search_to_shm')}")
except ImportError:
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

HAS_SIM_MARL = bool(cpp_backend) and (hasattr(cpp_backend, "mcts_search") or hasattr(cpp_backend, "mcts_search_lstm_torchscript_compact"))


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
        env_state: Optional[Dict] = None,
        shm_ptrs: Optional[Tuple[int, int]] = None
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
        root_obs_seq_tensor = None
        if use_lstm or use_tcn:
            hist = []
            if obs_history:
                hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]
            else:
                hist = [root_obs.reshape(-1)]
            
            if seq_len == 5 and use_tcn:
                idx = [-1, -2, -4, -8, -16]
                sampled = []
                for i in idx:
                    if -i <= len(hist):
                        sampled.append(hist[i])
                    else:
                        sampled.append(hist[0])
                sampled = sampled[::-1]  # oldest -> newest
                feat = []
                prev = sampled[0]
                for x in sampled:
                    dx = x - prev
                    feat.append(np.concatenate([x, dx], axis=0))
                    prev = x
                hist = feat
                root_obs = hist[-1]
                obs_dim = int(root_obs.size)
            else:
                if len(hist) < seq_len:
                    pad = [hist[0]] * (seq_len - len(hist))
                    hist = pad + hist
                else:
                    hist = hist[-seq_len:]

            root_obs_seq_tensor = torch.as_tensor(np.stack(hist), device=self.device).unsqueeze(0) # (1, T, D)

        # Root hidden state
        if hidden_state is not None and isinstance(hidden_state, tuple) and len(hidden_state) == 2:
            h0 = np.asarray(hidden_state[0], dtype=np.float32).reshape(-1)
            c0 = np.asarray(hidden_state[1], dtype=np.float32).reshape(-1)
        else:
            h0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)
            c0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)

        # Generate random seed once per search
        seed = int(np.random.randint(0, 2**31 - 1, dtype=np.int64))

        # --- OPTIMIZED SHM PATH ---
        if shm_ptrs is not None and use_lstm and hasattr(cpp_backend, 'mcts_search_to_shm'):
            action_ptr, stats_ptr = shm_ptrs
            
            # Prepare root_obs_seq for C++
            hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]
            if len(hist) < seq_len:
                pad = [hist[0]] * (seq_len - len(hist))
                hist = pad + hist
            else:
                hist = hist[-seq_len:]
            root_obs_seq_flat = np.concatenate(hist, axis=0).astype(np.float32)

            cpp_backend.mcts_search_to_shm(
                cpp_env,
                root_state_cpp,
                root_obs_seq_flat.tolist(),
                seq_len,
                obs_dim,
                self.ts_model_path,
                h0.tolist(),
                c0.tolist(),
                lstm_hidden_dim,
                self.agent_id,
                self.num_simulations,
                self.num_action_samples,
                self.rollout_depth,
                self.c_puct,
                self.temperature,
                0.99, # gamma
                0.3,  # dirichlet_alpha
                0.25, # dirichlet_eps
                seed,
                action_ptr,
                stats_ptr
            )
            # In SHM mode, the worker doesn't need to return action/stats to Python
            return np.zeros(2, dtype=np.float32), {}

        # --- LEGACY CALLBACK/TS PATHS ---
        ts_path = self.ts_model_path
        has_ts = bool(ts_path) and isinstance(ts_path, str) and os.path.isfile(ts_path)

        @torch.inference_mode()
        def infer_policy_value(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not obs_batch:
                empty = np.array([])
                return empty, empty, empty

            # Case 1: TCN
            if use_tcn:
                obs_np = np.asarray(obs_batch, dtype=np.float32)
                is_delta = (seq_len == 5 and obs_dim > 150)
                raw_obs_dim = obs_dim // 2 if is_delta else obs_dim

                if obs_np.ndim == 2 and obs_np.shape[1] == raw_obs_dim and is_delta:
                    obs_np = np.concatenate([obs_np, np.zeros_like(obs_np)], axis=1)

                if obs_np.ndim == 2 and obs_np.shape[1] == seq_len * raw_obs_dim:
                    obs_reshaped = obs_np.reshape(-1, seq_len, raw_obs_dim)
                    if is_delta:
                        obs_delta = np.zeros_like(obs_reshaped)
                        obs_delta[:, 1:, :] = obs_reshaped[:, 1:, :] - obs_reshaped[:, :-1, :]
                        obs_np_final = np.concatenate([obs_reshaped, obs_delta], axis=-1)
                        obs_tensor = torch.as_tensor(obs_np_final, device=self.device)
                    else:
                        obs_tensor = torch.as_tensor(obs_reshaped, device=self.device)
                elif obs_np.ndim == 2 and obs_np.shape[1] == obs_dim:
                    obs_step = torch.as_tensor(obs_np, device=self.device).unsqueeze(1)
                    if root_obs_seq_tensor is not None:
                        prefix = root_obs_seq_tensor[:, :seq_len-1, :].expand(obs_np.shape[0], -1, -1)
                        obs_tensor = torch.cat([prefix, obs_step], dim=1)
                    else:
                        obs_tensor = obs_step.expand(-1, seq_len, -1)
                elif obs_np.ndim == 2 and obs_np.shape[1] == seq_len * obs_dim:
                    obs_seq = obs_np.reshape(obs_np.shape[0], seq_len, obs_dim)
                    obs_tensor = torch.as_tensor(obs_seq, device=self.device)
                else:
                    raise RuntimeError(f"TCN infer unexpected shape {obs_np.shape}")

                means, stds, values, _ = self.network(obs_tensor)
                return (means.cpu().numpy(), stds.cpu().numpy(), values.view(-1).cpu().numpy())

            # Case 2: LSTM
            if use_lstm:
                obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
                h_np = np.tile(np.asarray(h, dtype=np.float32).reshape(-1)[None, :], (obs_tensor.shape[0], 1))
                c_np = np.tile(np.asarray(c, dtype=np.float32).reshape(-1)[None, :], (obs_tensor.shape[0], 1))
                h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], -1)
                c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], -1)
                means, stds, values, _ = self.network(obs_tensor, (h_t, c_t))
                return (means.cpu().numpy(), stds.cpu().numpy(), values.view(-1).cpu().numpy())

            # Case 3: MLP
            obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
            means, stds, values, _ = self.network(obs_tensor)
            return (means.cpu().numpy(), stds.cpu().numpy(), values.view(-1).cpu().numpy())

        @torch.inference_mode()
        def infer_next_hidden(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray,
            action_batch: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            obs_tensor = self._tensor_cache.get_obs_tensor(obs_batch)
            act_tensor = torch.as_tensor(np.asarray(action_batch, dtype=np.float32), device=self.device)
            h_np = np.tile(np.asarray(h, dtype=np.float32).reshape(-1)[None, :], (obs_tensor.shape[0], 1))
            c_np = np.tile(np.asarray(c, dtype=np.float32).reshape(-1)[None, :], (obs_tensor.shape[0], 1))
            h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], -1)
            c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], -1)
            hn, cn = self.network.next_hidden(obs_tensor, act_tensor, (h_t, c_t))
            return hn.squeeze(0).cpu().numpy(), cn.squeeze(0).cpu().numpy()

        use_torchscript_path = use_lstm and has_ts and hasattr(cpp_backend, 'mcts_search_lstm_torchscript')
        if use_torchscript_path:
            try:
                hist = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history]
                if len(hist) < seq_len:
                    pad = [hist[0]] * (seq_len - len(hist))
                    hist = pad + hist
                else:
                    hist = hist[-seq_len:]
                root_obs_seq = np.concatenate(hist, axis=0).astype(np.float32)

                if hasattr(cpp_backend, 'mcts_search_lstm_torchscript_compact'):
                    res_compact = cpp_backend.mcts_search_lstm_torchscript_compact(
                        cpp_env, root_state_cpp, root_obs_seq.tolist(), seq_len, obs_dim, ts_path,
                        h0.tolist(), c0.tolist(), lstm_hidden_dim, self.agent_id,
                        self.num_simulations, self.num_action_samples, self.rollout_depth,
                        self.c_puct, self.temperature, 0.99, 0.3, 0.25, seed
                    )
                    action_arr, actions_k2, visits_k, root_v, root_n, h_next, c_next = res_compact
                    action = action_arr.flatten()
                    edge_stats = [{"action": tuple(actions_k2[k]), "n": int(visits_k[k])} for k in range(len(visits_k))]
                    stats = {"root_v": float(root_v), "root_n": int(root_n), "edges": edge_stats, "h_next": h_next, "c_next": c_next}
                else:
                    action, stats = cpp_backend.mcts_search_lstm_torchscript(
                        cpp_env, root_state_cpp, root_obs_seq.tolist(), seq_len, obs_dim, ts_path,
                        h0.tolist(), c0.tolist(), lstm_hidden_dim, self.agent_id,
                        self.num_simulations, self.num_action_samples, self.rollout_depth,
                        self.c_puct, self.temperature, 0.99, 0.3, 0.25, seed
                    )
            except RuntimeError as e:
                if any(x in str(e) for x in ["infer_policy_value", "infer_next_hidden"]):
                    use_torchscript_path = False
                else: raise

        if not use_torchscript_path:
            if use_lstm and hasattr(cpp_backend, 'mcts_search_lstm'):
                action, stats = cpp_backend.mcts_search_lstm(
                    cpp_env, root_state_cpp, root_obs.tolist(), infer_policy_value, infer_next_hidden,
                    h0.tolist(), c0.tolist(), lstm_hidden_dim, self.agent_id,
                    self.num_simulations, self.num_action_samples, self.rollout_depth,
                    self.c_puct, self.temperature, 0.99, 0.3, 0.25, seed
                )
            elif use_tcn and hasattr(cpp_backend, 'mcts_search_seq'):
                raw_obs_dim = int(getattr(self.network, 'obs_dim', 127))
                if seq_len == 5:
                    raw_obs_dim //= 2

                hist = [
                    np.asarray(x, dtype=np.float32).reshape(-1)[:raw_obs_dim]
                    for x in (obs_history or [root_obs])
                ]
                if len(hist) < seq_len:
                    hist = [hist[0]] * (seq_len - len(hist)) + hist
                else:
                    hist = hist[-seq_len:]

                root_obs_seq_flat = np.concatenate(hist, axis=0).astype(np.float32)
                action, stats = cpp_backend.mcts_search_seq(
                    cpp_env, root_state_cpp, root_obs_seq_flat.tolist(), seq_len, raw_obs_dim,
                    infer_policy_value, self.agent_id, self.num_simulations, self.num_action_samples,
                    self.rollout_depth, self.c_puct, self.temperature, 0.99, 0.3, 0.25, seed
                )
            else:
                action, stats = cpp_backend.mcts_search(
                    cpp_env, root_state_cpp, root_obs.tolist(), infer_policy_value, self.agent_id,
                    self.num_simulations, self.num_action_samples, self.rollout_depth,
                    self.c_puct, self.temperature, 0.99, 0.3, 0.25, seed
                )

        self._rollout_stats['total_rollouts'] += 1
        return np.array(action, dtype=np.float32), stats