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

        def infer_fn_lstm(
            obs_batch: List[List[float]],
            h: np.ndarray,
            c: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if not obs_batch:
                empty = np.array([])
                return empty, empty, empty, empty, empty

            obs_tensor = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)

            # h/c expected as flat (H,). Convert to (1,1,H) as required by nn.LSTM.
            h_t = torch.as_tensor(np.asarray(h, dtype=np.float32), device=self.device).view(1, 1, -1)
            c_t = torch.as_tensor(np.asarray(c, dtype=np.float32), device=self.device).view(1, 1, -1)

            means_list = []
            stds_list = []
            values_list = []
            hn_list = []
            cn_list = []

            with torch.no_grad():
                # Evaluate each item with its own recurrent state (tree nodes).
                for i in range(obs_tensor.shape[0]):
                    x = obs_tensor[i].view(1, -1)  # (1, obs_dim)
                    m, s, v, (hn, cn) = self.network(x, (h_t, c_t))
                    means_list.append(m.squeeze(0))
                    stds_list.append(s.squeeze(0))
                    values_list.append(v.view(-1)[0])
                    hn_list.append(hn.squeeze(0).squeeze(0))
                    cn_list.append(cn.squeeze(0).squeeze(0))

            means = torch.stack(means_list, dim=0)
            stds = torch.stack(stds_list, dim=0)
            values = torch.stack(values_list, dim=0)
            hn = torch.stack(hn_list, dim=0)  # (B,H)
            cn = torch.stack(cn_list, dim=0)  # (B,H)

            return (
                means.cpu().numpy(),
                stds.cpu().numpy(),
                values.cpu().numpy(),
                hn.cpu().numpy(),
                cn.cpu().numpy()
            )

        # Root hidden state (1,1,H) in training code -> flatten to (H,)
        if hidden_state is not None and isinstance(hidden_state, tuple) and len(hidden_state) == 2:
            h0 = np.asarray(hidden_state[0], dtype=np.float32).reshape(-1)
            c0 = np.asarray(hidden_state[1], dtype=np.float32).reshape(-1)
        else:
            h0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)
            c0 = np.zeros((lstm_hidden_dim,), dtype=np.float32)

        # Prefer recurrent C++ MCTS if available
        if hasattr(cpp_backend, 'mcts_search_lstm'):
            action, stats = cpp_backend.mcts_search_lstm(
                cpp_env,
                root_state_cpp,
                root_obs.tolist(),
                infer_fn_lstm,
                h0.tolist(),
                c0.tolist(),
                lstm_hidden_dim,
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
            action, stats = cpp_backend.mcts_search(
                cpp_env,
                root_state_cpp,
                root_obs.tolist(),
                infer_fn_lstm,
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
