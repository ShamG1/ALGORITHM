# --- mcts.py ---
# This file is now a thin wrapper for the C++ MCTS implementation.
# Optimized version with bug fixes and performance improvements.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import os

import sys

# Add C++ build directory to path to ensure DRIVESIMX_ENV can be loaded
cpp_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DriveSimX/core/cpp/build")
if cpp_build_dir not in sys.path:
    sys.path.insert(0, cpp_build_dir)

try:
    import DRIVESIMX_ENV as cpp_backend  # type: ignore
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

HAS_SIM_MARL = bool(cpp_backend) and (
    hasattr(cpp_backend, "mcts_search_to_shm")
    or hasattr(cpp_backend, "mcts_search_seq_to_shm")
    or hasattr(cpp_backend, "mcts_search_lstm_torchscript")
)


def prepare_obs_sequence(obs_history: List[np.ndarray], seq_len: int, use_tcn: bool) -> np.ndarray:
    """
    Unified logic for preparing observation sequences for TCN/LSTM.
    Supports non-uniform sampling and delta features for TCN (seq_len=5).
    """
    hist_list = list(obs_history)
    
    # Non-uniform sampling [t, t-1, t-3, t-7, t-15] with delta features
    if use_tcn and seq_len == 5:
        idx = [-1, -2, -4, -8, -16]
        sampled = []
        for i in idx:
            if -i <= len(hist_list):
                sampled.append(hist_list[i])
            else:
                sampled.append(hist_list[0])
        sampled = sampled[::-1] # oldest -> newest
        
        # Delta features: concatenate current obs with (current - previous)
        feat = []
        prev = sampled[0]
        for x in sampled:
            dx = x - prev
            feat.append(np.concatenate([x, dx], axis=0))
            prev = x
        return np.stack(feat)
    else:
        # Standard uniform sampling
        if len(hist_list) < seq_len:
            pad = [hist_list[0]] * (seq_len - len(hist_list))
            hist_list = pad + hist_list
        else:
            hist_list = hist_list[-seq_len:]
        return np.stack(hist_list)


class MCTS:
    """
    Wrapper for the C++ MCTS implementation.
    This class is now a thin client that calls the C++ backend.
    Optimized version with tensor caching and reduced Python overhead.
    """

    def __init__(
        self,
        network,
        num_simulations: int = 50,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cpu',
        rollout_depth: int = 3,
        agent_id: int = 0,
        num_action_samples: int = 5,
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
        self.agent_id = agent_id
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
        shm_ptrs: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Dict]:

        if not hasattr(env, 'env') or not hasattr(env.env, 'get_state'):
            raise TypeError(
                "C++ MCTS requires a FastScenarioEnv wrapper with a C++ backend (env.env) that has get_state/set_state methods."
            )

        cpp_env = env.env
        root_state_cpp = cpp_env.get_state()

        lstm_hidden_dim = self._lstm_hidden_dim
        use_lstm = self._use_lstm
        use_tcn = self._use_tcn
        seq_len = self._seq_len
        obs_dim = root_obs.size

        # Construct observation sequence for TCN/LSTM
        if use_lstm or use_tcn:
            obs_seq_np = prepare_obs_sequence(obs_history if obs_history else [root_obs], seq_len, use_tcn)
            root_obs = obs_seq_np[-1]
            obs_dim = int(root_obs.size)

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
            
            # Use the unified obs_seq_np
            root_obs_seq_flat = obs_seq_np.astype(np.float32).flatten()

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
                float(getattr(self, 'dirichlet_alpha', 0.3)),  # dirichlet_alpha
                float(getattr(self, 'dirichlet_eps', 0.25)), # dirichlet_eps
                seed,
                action_ptr,
                stats_ptr
            )
            # In SHM mode, the worker doesn't need to return action/stats to Python
            return np.zeros(2, dtype=np.float32), {}

        # --- LEGACY NON-SHM PATHS REMOVED: training now relies on SHM+TS only ---
        raise RuntimeError("Non-SHM MCTS path is disabled; training should use SHM pointers (shm_ptrs).")
