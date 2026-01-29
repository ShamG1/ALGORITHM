# --- mcts.py ---
# This file is now a thin wrapper for the C++ MCTS implementation.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import math

# When running as a package (python -m C_MCTS.train) we can use relative import.
# When running as a script (python C_MCTS/train.py) relative import fails, so fallback to absolute.
try:
    from . import cpp_backend  # type: ignore
except Exception:
    try:
        import cpp_backend  # type: ignore
    except Exception:
        cpp_backend = None

HAS_CPP_MCTS = bool(cpp_backend) and hasattr(cpp_backend, "mcts_search")


class MCTS:
    """
    Wrapper for the C++ MCTS implementation.
    This class is now a thin client that calls the C++ backend.
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
        use_cpp_mcts: bool = True  # Kept for API compatibility
    ):
        if not HAS_CPP_MCTS:
            raise RuntimeError("C++ MCTS backend is not available. Please build it first.")

        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = torch.device(device)
        self.rollout_depth = rollout_depth
        self.agent_id = agent_id  # Still needed for context
        self.num_action_samples = num_action_samples

        # Compatibility stats for training script
        self._rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'total_env_steps': 0,
        }

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
        obs_history: Optional[List] = None,  # Not used by C++ MCTS, kept for API
        hidden_state=None,  # Not used by C++ MCTS, kept for API
        env_state: Optional[Dict] = None  # Not used by C++ MCTS, kept for API
    ) -> Tuple[np.ndarray, Dict]:

        if not hasattr(env, 'env') or not hasattr(env.env, 'get_state'):
            raise TypeError(
                "C++ MCTS requires a FastIntersectionEnv wrapper with a C++ backend (env.env) that has get_state/set_state methods."
            )

        cpp_env = env.env
        root_state_cpp = cpp_env.get_state()

        lstm_hidden_dim = int(getattr(self.network, 'lstm_hidden_dim', 128))

        def infer_policy_value(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not obs_batch:
                empty = np.array([])
                return empty, empty, empty

            # Non-recurrent policy/value: ignore incoming h/c.
            if not getattr(self.network, 'use_lstm', False):
                obs_tensor = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
                with torch.inference_mode():
                    means, stds, values = self.network(obs_tensor)
                return (
                    means.cpu().numpy(),
                    stds.cpu().numpy(),
                    values.view(-1).cpu().numpy(),
                )

            obs_tensor = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)

            # h/c expected as (B,H) or flat; convert to (1,B,H) for PyTorch LSTM.
            # Be robust to bad shapes coming from the C++ backend stats.
            h_np = np.asarray(h, dtype=np.float32).reshape(-1)
            c_np = np.asarray(c, dtype=np.float32).reshape(-1)

            expected_h = int(lstm_hidden_dim)
            if h_np.size != expected_h or c_np.size != expected_h:
                h_np = np.zeros((expected_h,), dtype=np.float32)
                c_np = np.zeros((expected_h,), dtype=np.float32)

            # Expand to batch
                h_np = np.tile(h_np[None, :], (obs_tensor.shape[0], 1))
                c_np = np.tile(c_np[None, :], (obs_tensor.shape[0], 1))

            h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], expected_h)
            c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], expected_h)

            # DualNetwork supports batched forward for single-step inputs: (B, obs_dim)
            # with LSTM hidden state shaped (1, B, H). This avoids a Python loop per node.
            with torch.inference_mode():
                # DualNetwork expects hidden_state for batched input shaped (1, B, H).
                means, stds, values, _ = self.network(obs_tensor, (h_t, c_t))

            # means/stds: (B, A), values: (B, 1)
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

        def infer_next_hidden(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray,
            action_batch: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            # This callback is only used by the recurrent (LSTM) C++ MCTS backend.
            if not getattr(self.network, 'use_lstm', False):
                raise RuntimeError("infer_next_hidden called but network.use_lstm=False")

            if not obs_batch:
                empty = np.array([])
                return empty, empty

            obs_tensor = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
            act_tensor = torch.as_tensor(np.asarray(action_batch, dtype=np.float32), device=self.device)

            # h/c expected as (B,H) or flat; convert to (1,B,H)
            h_np = np.asarray(h, dtype=np.float32)
            c_np = np.asarray(c, dtype=np.float32)
            if h_np.ndim == 1:
                h_np = np.tile(h_np[None, :], (obs_tensor.shape[0], 1))
                c_np = np.tile(c_np[None, :], (obs_tensor.shape[0], 1))

            h_t = torch.as_tensor(h_np, device=self.device).view(1, obs_tensor.shape[0], -1)
            c_t = torch.as_tensor(c_np, device=self.device).view(1, obs_tensor.shape[0], -1)

            with torch.inference_mode():
                hn, cn = self.network.next_hidden(obs_tensor, act_tensor, (h_t, c_t))

            # Return (B,H)
            return hn.squeeze(0).cpu().numpy(), cn.squeeze(0).cpu().numpy()

        # Prefer recurrent C++ MCTS only when the network uses LSTM.
        if getattr(self.network, 'use_lstm', False) and hasattr(cpp_backend, 'mcts_search_lstm'):
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
                seed=int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
            )
        else:
            # Fallback: non-LSTM C++ MCTS
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
                seed=int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
            )

        # Rough compatibility counters: 1 search == 1 rollout batch.
        self._rollout_stats['total_rollouts'] += 1
        # We don't know success rate/env steps from C++ side yet.

        return np.array(action, dtype=np.float32), stats
