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

        def infer_fn(obs_batch: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not obs_batch:
                return np.array([]), np.array([]), np.array([])

            obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)
            with torch.no_grad():
                # Note: hidden_state is not batched; C++ MCTS does not handle recurrent state yet.
                means, stds, values, _ = self.network(obs_tensor, None)
            return means.cpu().numpy(), stds.cpu().numpy(), values.cpu().numpy()

        action, stats = cpp_backend.mcts_search(
            cpp_env,
            root_state_cpp,
            root_obs.tolist(),
            infer_fn,
            num_simulations=self.num_simulations,
            num_action_samples=self.num_action_samples,
            rollout_depth=self.rollout_depth,
            c_puct=self.c_puct,
            temperature=self.temperature,
            gamma=0.99,  # Standard gamma, can be parameterized if needed
            dirichlet_alpha=0.3,  # Common value
            dirichlet_eps=0.25,  # Common value
            seed=int(np.random.randint(0, 2**31 - 1, dtype=np.int64))
        )

        # Rough compatibility counters: 1 search == 1 rollout batch.
        self._rollout_stats['total_rollouts'] += 1
        # We don't know success rate/env steps from C++ side yet.

        return np.array(action, dtype=np.float32), stats
