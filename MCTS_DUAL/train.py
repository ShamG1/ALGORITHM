# --- train.py ---
# Multi-Agent MCTS Training Script

import os
import sys

# Suppress pygame messages in worker processes (set before any pygame imports)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Avoid over-subscription when using multi-process MCTS: keep BLAS/OMP + torch single-threaded.
# (These must be set before heavy numeric libraries fully initialize.)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import torch
import torch.optim as optim

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from datetime import datetime
from time import time
from collections import deque
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
from typing import Optional
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle both relative and absolute imports
try:
    from .env import IntersectionEnv
    from .utils import DEFAULT_REWARD_CONFIG, OBS_DIM
except ImportError:
    from env import IntersectionEnv
    from utils import DEFAULT_REWARD_CONFIG, OBS_DIM


def set_global_seeds(seed: int, deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Note: may reduce performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

# define global cache variables
_WORKER_CACHE = {
    "network": None,
    "mcts": None,
    "env": None,
    "env_config": None,
}

# Shared-memory weight broadcast (to avoid per-step pickling of state_dict)
# These globals are initialized in the main process and inherited by worker processes.
_SHARED_STATE = {
    "tensors": None,   # Ordered list[torch.Tensor] on CPU, shared_memory_()
    "version": None,   # multiprocessing.Value("Q")
    "lock": None       # multiprocessing.Lock
}

def _init_worker_shared_state(shared_tensors, version, lock, env_config=None):
    global _SHARED_STATE, _WORKER_CACHE
    _SHARED_STATE["tensors"] = shared_tensors
    _SHARED_STATE["version"] = version
    _SHARED_STATE["lock"] = lock
    if env_config is not None:
        _WORKER_CACHE["env_config"] = env_config

def _search_agent_wrapper(args):
    """
    Wrapper function for process pool MCTS search.
    Must be at module level to be picklable.
    Uses global cache to persist objects across calls within the same process.
    """
    # suppress pygame initialization messages
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

    agent_id, obs_data, hidden_state_data, config = args
    
    # reference global variables
    global _WORKER_CACHE
    
    try:
        # Deterministic per-call seed (optional)
        base_seed = config.get("base_seed", None)
        if base_seed is not None:
            episode_i = int(config.get("episode", 0))
            step_i = int(config.get("step", 0))
            # Mix in identifiers to avoid identical streams across agents/steps.
            call_seed = int(base_seed) + episode_i * 100000 + step_i * 100 + int(agent_id)
            import random as _random
            _random.seed(call_seed)
            np.random.seed(call_seed)
            torch.manual_seed(call_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(call_seed)

        torch.set_num_threads(1)
        device = torch.device(config['device'])
        
        # 1. lazy load network (only initialize objects when called for the first time)
        if _WORKER_CACHE["network"] is None:
            # Import strictly inside function/worker to avoid circular imports or pickling issues
            # Handle both relative and absolute imports for worker process
            try:
                from .dual_net import DualNetwork
                from .utils import OBS_DIM
            except ImportError:
                try:
                    from dual_net import DualNetwork
                    from utils import OBS_DIM
                except ImportError:
                    # Fallback: import from MCTS_DUAL package
                    import sys
                    mcts_dual_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if mcts_dual_path not in sys.path:
                        sys.path.insert(0, mcts_dual_path)
                    from MCTS_DUAL.dual_net import DualNetwork
                    from MCTS_DUAL.utils import OBS_DIM
            
            network = DualNetwork(
                obs_dim=OBS_DIM,
                action_dim=2,
                hidden_dim=config['hidden_dim'],
                lstm_hidden_dim=config['lstm_hidden_dim'],
                use_lstm=config['use_lstm'],
                sequence_length=config['sequence_length']
            ).to(device)
            network.eval()
            _WORKER_CACHE["network"] = network
        
        # Load latest weights from shared memory if configured; fallback to state_dict path.
        if _SHARED_STATE.get("tensors") is not None and _SHARED_STATE.get("version") is not None:
            last_ver = _WORKER_CACHE.get("_last_weight_version", -1)
            cur_ver = int(_SHARED_STATE["version"].value)
            if cur_ver != last_ver:
                # Ensure consistent snapshot while we copy from shared tensors.
                lock = _SHARED_STATE.get("lock")
                if lock is None:
                    # Best-effort without lock
                    state = _WORKER_CACHE["network"].state_dict()
                    for p, src in zip(state.values(), _SHARED_STATE["tensors"]):
                        p.copy_(src)
                else:
                    with lock:
                        state = _WORKER_CACHE["network"].state_dict()
                        for p, src in zip(state.values(), _SHARED_STATE["tensors"]):
                            p.copy_(src)
                _WORKER_CACHE["_last_weight_version"] = cur_ver
        else:
            # Legacy: always load the latest weights (slow due to pickling)
            _WORKER_CACHE["network"].load_state_dict(config['network_state_dict'])

        # 2. lazy load environment (for MCTS rollouts)
        if _WORKER_CACHE["env"] is None:
            # Handle both relative and absolute imports for worker process
            try:
                from .train import generate_ego_routes
                from .env import IntersectionEnv
            except ImportError:
                try:
                    from train import generate_ego_routes
                    from env import IntersectionEnv
                except ImportError:
                    # Fallback: import from MCTS_DUAL package
                    import sys
                    mcts_dual_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if mcts_dual_path not in sys.path:
                        sys.path.insert(0, mcts_dual_path)
                    from MCTS_DUAL.train import generate_ego_routes
                    from MCTS_DUAL.env import IntersectionEnv
                    from MCTS_DUAL.utils import DEFAULT_REWARD_CONFIG
            env_cfg = config.get('env_config', None)
            if env_cfg is None:
                env_cfg = _WORKER_CACHE.get('env_config') or {}
            num_agents_cfg = env_cfg.get('num_agents', 1)
            num_lanes_cfg = env_cfg.get('num_lanes', 3)
            ego_routes_copy = env_cfg.get('ego_routes')
            if ego_routes_copy is None:
                ego_routes_copy = generate_ego_routes(num_agents_cfg, num_lanes_cfg)
            else:
                # ensure worker copy is independent
                ego_routes_copy = list(ego_routes_copy)
            # Import DEFAULT_REWARD_CONFIG if not already imported
            try:
                DEFAULT_REWARD_CONFIG
            except NameError:
                try:
                    from .utils import DEFAULT_REWARD_CONFIG
                except ImportError:
                    try:
                        from utils import DEFAULT_REWARD_CONFIG
                    except ImportError:
                        import sys
                        mcts_dual_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if mcts_dual_path not in sys.path:
                            sys.path.insert(0, mcts_dual_path)
                        from MCTS_DUAL.utils import DEFAULT_REWARD_CONFIG
            reward_cfg = env_cfg.get('reward_config', DEFAULT_REWARD_CONFIG.get('reward_config', {}))
            # create FastIntersectionEnv (C++ backend) for C++ MCTS
            _WORKER_CACHE["env"] = IntersectionEnv({
                'traffic_flow': False,
                'traffic_density': 0,
                'num_agents': num_agents_cfg,
                'num_lanes': num_lanes_cfg,
                'render_mode': None,
                'max_steps': env_cfg.get('max_steps', 2000),
                'respawn_enabled': env_cfg.get('respawn_enabled', True),
                'reward_config': reward_cfg,
                'ego_routes': ego_routes_copy,
            })

        # 3. lazy load MCTS instance
        if _WORKER_CACHE["mcts"] is None:
            from mcts import MCTS
            
            # build shared network list
            env_cfg_for_mcts = config.get('env_config', None)
            if env_cfg_for_mcts is None:
                env_cfg_for_mcts = _WORKER_CACHE.get('env_config') or {}
            num_agents = int(env_cfg_for_mcts.get('num_agents', 1))
            shared_network_list = [_WORKER_CACHE["network"] for _ in range(num_agents)]
            
            mcts = MCTS(
                network=_WORKER_CACHE["network"],
                action_space=None,
                num_simulations=config['num_simulations'],
                c_puct=config['c_puct'],
                temperature=config['temperature'],
                device=device,
                rollout_depth=config['rollout_depth'],
                env_factory=None, 
                all_networks=shared_network_list, 
                agent_id=agent_id,
                num_action_samples=config['num_action_samples']
            )
            
            # inject the cached env into MCTS's internal cache
            mcts._env_cache = _WORKER_CACHE["env"]
            # avoid assertion error
            mcts.env_factory = lambda: None 
            
            _WORKER_CACHE["mcts"] = mcts
        
        # get the reused MCTS instance
        mcts = _WORKER_CACHE["mcts"]
        
        # update ID (because the same Worker process may serve different Agents)
        mcts.agent_id = agent_id 
        
        # Convert hidden state back to tensor if needed (only when using LSTM)
        hidden_state_tensor = None
        if config.get('use_lstm', True) and hidden_state_data is not None:
            if isinstance(hidden_state_data, tuple):
                hidden_state_tensor = tuple(torch.as_tensor(h, device=device) for h in hidden_state_data)
            else:
                hidden_state_tensor = torch.as_tensor(hidden_state_data, device=device)
        
        # Perform search
        # Use positional args to avoid keyword mismatches across different MCTS wrappers
        with torch.inference_mode():
            action, search_stats = mcts.search(
                obs_data,
                _WORKER_CACHE["env"],
                None,
                hidden_state_tensor,
                None
            )

        
        # Ensure action is numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = action.flatten()
        
        # Return updated hidden state if provided by C++ LSTM-MCTS
        hidden_out = None
        try:
            if config.get('use_lstm', True) and isinstance(search_stats, dict) and ("h_next" in search_stats) and ("c_next" in search_stats):
                hn = search_stats["h_next"]
                cn = search_stats["c_next"]
                # Convert to torch tensors shaped (1,1,H)
                hn_t = torch.as_tensor(np.asarray(hn, dtype=np.float32), device=device).view(1, 1, -1)
                cn_t = torch.as_tensor(np.asarray(cn, dtype=np.float32), device=device).view(1, 1, -1)
                hidden_out = (hn_t, cn_t)
        except Exception:
            hidden_out = None

        return agent_id, action, hidden_out
    except Exception as e:
        import sys
        sys.stdout.write(f"[Process {agent_id}] Error: {e}\n")
        sys.stdout.flush()
        return agent_id, np.zeros(2), None

try:
    from .dual_net import DualNetwork
    from .mcts import MCTS
except ImportError:
    from dual_net import DualNetwork
    from mcts import MCTS

def generate_ego_routes(num_agents: int, num_lanes: int):
    """Generate routes for agents based on num_agents and num_lanes."""
    # Import route mappings from env
    from env import DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
    
    if num_lanes == 2:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    elif num_lanes == 3:
        route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
    else:
        # Fallback to 2 lanes
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    
    # Get all available routes
    all_routes = []
    for start, ends in route_mapping.items():
        for end in ends:
            all_routes.append((start, end))
    
    # Select routes for agents (balanced distribution, ensuring uniqueness)
    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4
    
    # Track used routes to avoid duplicates
    used_routes = set()
    
    for i in range(4):
        count = agents_per_dir + (1 if i < extra_agents else 0)
        # Select routes for this direction
        # For 3 lanes: direction 0 uses IN_1, IN_2, IN_3; direction 1 uses IN_4, IN_5, IN_6; etc.
        if num_lanes == 3:
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        else:  # 2 lanes
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        
        if len(dir_routes) == 0:
            # Fallback: use all routes
            dir_routes = all_routes
        
        # Filter out already used routes
        available_routes = [r for r in dir_routes if r not in used_routes]
        if len(available_routes) == 0:
            # If all routes in this direction are used, allow reuse but try to avoid exact duplicates
            available_routes = dir_routes
        
        # Select routes for agents in this direction
        dir_route_idx = 0
        for _ in range(count):
            if available_routes:
                # Cycle through available routes
                route = available_routes[dir_route_idx % len(available_routes)]
                selected_routes.append(route)
                used_routes.add(route)
                dir_route_idx += 1
            elif dir_routes:
                # Fallback: use any route from this direction
                route = dir_routes[dir_route_idx % len(dir_routes)]
                selected_routes.append(route)
                dir_route_idx += 1
    
    # If we need more routes, use remaining available routes
    remaining_routes = [r for r in all_routes if r not in used_routes]
    while len(selected_routes) < num_agents:
        if remaining_routes:
            route = remaining_routes.pop(0)
            selected_routes.append(route)
            used_routes.add(route)
        else:
            # If all routes are used, cycle through all routes
            route = all_routes[(len(selected_routes) - len(used_routes)) % len(all_routes)]
            selected_routes.append(route)
    
    return selected_routes[:num_agents]


class MCTSTrainer:
    """Multi-Agent MCTS Trainer."""
    
    def close(self):
        """Release resources like process pools and environments."""
        if self._process_pool is not None:
            try:
                # cancel_futures is supported on Python 3.9+
                self._process_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                # Older Python: no cancel_futures
                self._process_pool.shutdown(wait=True)
            except Exception:
                pass
            finally:
                self._process_pool = None

        # Best-effort env cleanup (pygame windows, etc.)
        try:
            if hasattr(self, 'env') and hasattr(self.env, 'close'):
                self.env.close()
        except Exception:
            pass

    def __del__(self):
        # Best-effort cleanup; not guaranteed to run.
        try:
            self.close()
        except Exception:
            pass

    def __init__(
        self,
        num_agents: int = 6,
        num_lanes: int = 3,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 2000,
        mcts_simulations: int = 50,
        rollout_depth: int = 3,
        num_action_samples: int = 5,
        save_frequency: int = 100,
        log_frequency: int = 10,
        device: str = 'cpu',
        use_team_reward: bool = True,
        use_lstm: bool = True,
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = True,  # Default enabled for autonomous driving
        save_dir: str = 'C/checkpoints',
        parallel_mcts: bool = True,  # Enable parallel MCTS for multiple agents
        max_workers: int = None,  # Number of processes for parallel MCTS (None = num_agents)
        use_tqdm: bool = False,  # Enable/disable tqdm progress bar
        seed: Optional[int] = None,
        deterministic: bool = False
    ):
        """
        Initialize MCTS Trainer.
        
        Args:
            num_agents: Number of agents
            num_lanes: Number of lanes per direction
            max_episodes: Maximum training episodes
            mcts_simulations: Number of MCTS simulations per step
            save_frequency: Episodes before saving checkpoint
            log_frequency: Episodes before logging stats
            device: Device to use ('cpu' or 'cuda')
            use_team_reward: Whether to use team reward
            render: Whether to render environment
            show_lane_ids: Whether to show lane IDs in render
            show_lidar: Whether to show lidar in render
            respawn_enabled: Whether to enable respawn
            save_dir: Directory to save checkpoints
        """
        self.num_agents = num_agents
        self.num_lanes = num_lanes
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.mcts_simulations = mcts_simulations
        self.rollout_depth = rollout_depth
        self.num_action_samples = num_action_samples
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.device = torch.device(device)
        self.use_team_reward = use_team_reward
        self.use_lstm = use_lstm
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.parallel_mcts = parallel_mcts
        self.max_workers = max_workers if max_workers is not None else num_agents
        self.use_tqdm = use_tqdm and tqdm is not None
        self._process_pool = None  # Initialize process pool to None
        self.seed = seed

        # Shared-memory weight broadcast optimization:
        # only copy weights to shared memory when they've actually changed (after optimizer.step()).
        self._weights_dirty = True
        self.deterministic = deterministic
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # Generate routes
        ego_routes = generate_ego_routes(num_agents, num_lanes)
        
        # Store routes for debugging
        self.ego_routes = ego_routes
        
        # Debug: Check for duplicate routes
        route_counts = {}
        for route in ego_routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        duplicates = {r: c for r, c in route_counts.items() if c > 1}
        if duplicates:
            print(f"WARNING: Found duplicate routes: {duplicates}")
            print(f"All routes: {ego_routes}")
        
        # Initialize environment
        self.env = IntersectionEnv({
            'traffic_flow': False,
            'num_agents': num_agents,
            'num_lanes': num_lanes,
            'use_team_reward': use_team_reward,
            'render_mode': 'human' if render else None,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            'ego_routes': ego_routes
        })
        
        # Shared network for all agents (homogeneous policy/value)
        self.network = DualNetwork(
            obs_dim=OBS_DIM,
            action_dim=2,
            hidden_dim=256,
            lstm_hidden_dim=128,
            use_lstm=self.use_lstm,
            sequence_length=5
        ).to(self.device)

        # Keep a compatibility list so existing code that expects per-agent networks still works.
        # IMPORTANT: Per-agent LSTM hidden states are tracked separately in self.hidden_states.
        self.networks = [self.network for _ in range(num_agents)]
        
        # Create environment factory function for MCTS rollouts
        # Note: With process pool, each process will create its own environment copy
        # No need for locks since processes don't share memory
        # Base env config reused for copies (Python or C++)
        self._base_env_config = {
            'traffic_flow': False,
            'num_agents': num_agents,
            'num_lanes': num_lanes,
            'use_team_reward': use_team_reward,
            'render_mode': None,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
        }

        def create_env_copy():
            """Factory: prefer C++ backend env when available for speed."""
            # Fresh routes per copy to avoid shared mutable state
            ego_routes_copy = generate_ego_routes(num_agents, num_lanes)
            cfg = dict(self._base_env_config)
            cfg['ego_routes'] = ego_routes_copy
            try:
                from cpp_backend import cpp_backend  # local import to avoid early import cost
                if cpp_backend.has_cpp_backend():
                    # Use high-performance C++ env with full config
                    from env import IntersectionEnv
                    return IntersectionEnv(cfg)
            except Exception:
                # Fallback to Python env below
                pass
            # --- Python fallback fast_mode ---
            try:
                cfg['fast_mode'] = True
                from env import IntersectionEnv
                return IntersectionEnv(cfg)
            except Exception as e:
                import sys, traceback
                sys.stdout.write(f"Error creating env copy: {e}\n")
                traceback.print_exc()
                raise
        
        # Initialize MCTS for each agent with real environment rollouts
        # Note: Rollouts run in background without rendering
        self.mcts_instances = [
            MCTS(
                network=self.network,
                action_space=None,  # Continuous actions
                num_simulations=mcts_simulations,
                c_puct=1.0,
                temperature=1.0,
                device=device,
                rollout_depth=self.rollout_depth,
                num_action_samples=self.num_action_samples,
                env_factory=create_env_copy,
                all_networks=self.networks,  # Compatibility list; same shared network repeated
                agent_id=i  # Agent ID
            )
            for i in range(num_agents)
        ]
        
        # Enable rollout debugging for first few episodes
        self.enable_rollout_debug = False  # Set to True to enable detailed rollout logs
        self.enable_debug = False  # Set to True to enable all debug output
        
        # Process pool for parallel MCTS (created on first use)
        self._process_pool = None
        
        # Single optimizer for the shared network
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Observation history for LSTM
        self.obs_history = [deque(maxlen=5) for _ in range(num_agents)] if self.use_lstm else None
        self.hidden_states = [None for _ in range(num_agents)] if self.use_lstm else None
        # TBPTT settings (sequence training)
        self.unroll_len = 16
        
        # Training statistics
        self.stats = {
            'episode': 0,
            'total_steps': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'crash_count': 0,
        }
        
        # Log file
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Num agents: {num_agents}\n")
            f.write(f"Num lanes: {num_lanes}\n")
            f.write(f"Use team reward: {use_team_reward}\n")
            f.write(f"Respawn enabled: {respawn_enabled}\n")
            f.write(f"MCTS simulations: {mcts_simulations}\n")
            f.write("Generated routes:\n")
            for i, route in enumerate(ego_routes):
                f.write(f"  Agent {i}: {route[0]} -> {route[1]}\n")
            f.write("=" * 80 + "\n")
        
        # CSV file for episode rewards
        self.csv_file = os.path.join(save_dir, 'episode_rewards.csv')
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward', 'Mean_Reward', 'Episode_Length'])
    
    def log(self, message: str):
        """Log message to file and console."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def train(self):
        """Main training loop."""
        self.log("=" * 80)
        self.log("Starting Multi-Agent MCTS Training")
        self.log("=" * 80)

        try:

            if self.seed is not None:
                self.log(f"[Seed] seed={self.seed} deterministic={self.deterministic}")
            
            episode = 0
            step_count = 0
            
            while episode < self.max_episodes:
                episode_start_time = time()
                episode += 1
                self.stats['episode'] = episode
                
                try:
                    # Reset environment
                    obs, info = self.env.reset()
                except Exception as e:
                    self.log(f"Error resetting environment at episode {episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Debug: Check initial agent positions for first few episodes
                if self.enable_debug and episode <= 3:
                    if hasattr(self.env, 'agents'):
                        print(f"\n[Episode {episode}] Initial agent positions:")
                        positions = {}
                        for i, agent in enumerate(self.env.agents):
                            pos_key = (round(agent.pos_x, 1), round(agent.pos_y, 1))
                            if pos_key not in positions:
                                positions[pos_key] = []
                            positions[pos_key].append(i)
                            route_info = self.ego_routes[i] if i < len(self.ego_routes) else 'N/A'
                            print(f"  Agent {i}: pos=({agent.pos_x:.1f}, {agent.pos_y:.1f}), heading={agent.heading:.2f}, route={route_info}")
                        # Check for overlapping positions
                        overlaps = {pos: agents for pos, agents in positions.items() if len(agents) > 1}
                        if overlaps:
                            print(f"  WARNING: Agents with overlapping positions: {overlaps}")
                
                # Reset observation history and hidden states
                if self.use_lstm:
                    for i in range(self.num_agents):
                        self.obs_history[i].clear()
                        self.hidden_states[i] = None
                        # Initialize history with first observation
                        self.obs_history[i].append(obs[i])
                
                # Reset buffer at start of episode
                if hasattr(self, 'buffer'):
                    self.buffer = {
                        'obs': [[] for _ in range(self.num_agents)],
                        'next_obs': [[] for _ in range(self.num_agents)],
                        'actions': [[] for _ in range(self.num_agents)],
                        'rewards': [[] for _ in range(self.num_agents)],
                        'dones': [[] for _ in range(self.num_agents)],
                    }
                
                episode_reward = np.zeros(self.num_agents)
                episode_length = 0
                done = False
                
                # Create progress bar for this episode (optional)
                pbar = None
                if getattr(self, 'use_tqdm', True) and tqdm is not None:
                    # Use miniters=1 and mininterval=0.1 for more frequent updates
                    # Use smaller smoothing (0.05) to show more recent rate
                    pbar = tqdm(
                        total=self.max_steps_per_episode,
                        desc=f"Episode {episode}",
                        unit="step",
                        leave=False,  # Don't leave progress bar after completion
                        ncols=100,  # Width of progress bar
                        miniters=1,  # Update every iteration
                        mininterval=0.1,  # Update at least every 0.1 seconds
                        smoothing=0.05,  # Smaller smoothing for more recent rate (was 0.1)
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    )
                
                # Episode loop
                while not done and episode_length < self.max_steps_per_episode:
                    # Render current state before MCTS search (so user can see what's happening)
                    if self.render:
                        try:
                            self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                            # Process events to keep window responsive
                            import pygame
                            if pygame.get_init():
                                pygame.event.pump()
                        except Exception:
                            pass
                    
                    # Select actions using MCTS for each agent
                    # Prepare environment state for rollouts (shared by all agents)
                    # Only include serializable data (no pygame objects)
                    env_state = {
                        'step_count': getattr(self.env, 'step_count', 0),
                        'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,  # Current observations for all agents
                        # Extract only serializable agent data (no pygame surfaces)
                        'agent_states': []
                    }
                    if hasattr(self.env, 'agents') and self.env.agents:
                        for agent in self.env.agents:
                            # Extract only basic attributes (no pygame objects)
                            # Use getattr with safe defaults to avoid AttributeError
                            agent_state = {
                                'pos_x': getattr(agent, 'pos_x', 0.0),
                                'pos_y': getattr(agent, 'pos_y', 0.0),
                                'heading': getattr(agent, 'heading', 0.0),
                                'speed': getattr(agent, 'speed', 0.0),  # Car uses 'speed', not 'velocity'
                                'target_lane': getattr(agent, 'target_lane', None),
                                'route': getattr(agent, 'route', None)
                            }
                            env_state['agent_states'].append(agent_state)
                    
                    # Update observation history for all agents
                    if self.use_lstm:
                        for i in range(self.num_agents):
                            if episode_length > 0:
                                self.obs_history[i].append(obs[i])
                    
                    # Enable debug output for first episode to diagnose hanging issue
                    for i in range(self.num_agents):
                        # Enable debug for first episode only
                        self.mcts_instances[i].debug_rollout = (episode == 1 and episode_length == 0)
                    
                    # Handle pygame events before MCTS search
                    if self.render:
                        try:
                            import pygame
                            if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        return
                        except Exception:
                            pass
                    
                    # Perform MCTS search: parallel or sequential
                    try:
                        if self.parallel_mcts and self.num_agents > 1:
                            # Parallel MCTS: all agents search simultaneously
                            actions = self._parallel_mcts_search(obs, env_state, episode=episode, step=episode_length)

                            # NOTE: In parallel MCTS, worker processes return updated LSTM states via
                            # search_stats['h_next'/'c_next'] and we already write them back in
                            # _parallel_mcts_search(). Avoid doing an extra per-agent forward here (CPU bottleneck).
                        else:
                            # Sequential MCTS: one agent at a time (original behavior)
                            actions = self._sequential_mcts_search(obs, env_state)
                        
                        # MCTS search completed (output disabled - only episode-level logs are shown)
                    except Exception as e:
                        self.log(f"Error in MCTS search at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Use zero actions as fallback
                        actions = np.zeros((self.num_agents, 2))
                    
                    # Debug: print actions for first few episodes
                    if self.enable_debug and episode <= 10 and episode_length == 0:
                        print(f"\n[Episode {episode}, Step 0] Actions selected:")
                        for i, act in enumerate(actions):
                            print(f"  Agent {i}: {act} (shape: {act.shape}, dtype: {act.dtype})")
                        # Check if actions are valid
                        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                            print(f"  WARNING: Invalid actions detected (NaN or Inf)!")
                        if np.any(np.abs(actions) > 1.0):
                            print(f"  WARNING: Actions out of range [-1, 1]!")
                            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                    
                    # Step environment
                    try:
                        # Environment step (output disabled - only episode-level logs are shown)
                        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                        # Environment step completed (output disabled - only episode-level logs are shown)
                    except Exception as e:
                        self.log(f"Error stepping environment at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                    
                    # Handle reward format
                    if isinstance(rewards, (list, np.ndarray)):
                        rewards = np.array(rewards)
                    else:
                        rewards = np.array([rewards] * self.num_agents)
                    
                    done = terminated or truncated
                    
                    # Debug: check why episode ended (moved after done is set)
                    if self.enable_debug and done and episode_length == 0 and episode <= 10:
                        collisions = info.get('collisions', {})
                        print(f"\n[Episode {episode}] Ended at step 0!")
                        print(f"  Terminated: {terminated}, Truncated: {truncated}")
                        print(f"  Rewards: {rewards}")
                        print(f"  Mean reward: {rewards.mean():.2f}")
                        print(f"  Collisions: {collisions}")
                        if hasattr(self.env, 'agents'):
                            for i, agent in enumerate(self.env.agents):
                                agent_id = id(agent)
                                if agent_id in collisions:
                                    print(f"  Agent {i} (id={agent_id}): {collisions[agent_id]}")
                        # Check agent positions
                        if hasattr(self.env, 'agents'):
                            print(f"  Agent positions:")
                            for i, agent in enumerate(self.env.agents):
                                print(f"    Agent {i}: pos=({agent.pos_x:.1f}, {agent.pos_y:.1f}), heading={agent.heading:.2f}, speed={agent.speed:.2f}")
                    
                    # Store transitions and update in batches
                    self._update_networks(obs, actions, rewards, next_obs, done)
                    
                    episode_reward += rewards
                    episode_length += 1
                    step_count += 1
                    
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                        # Update progress bar description with current reward
                        avg_reward = episode_reward.mean()
                        pbar.set_postfix({
                            'reward': f'{avg_reward:.2f}',
                            'done': done
                        })
                    
                    obs = next_obs
                    
                    # Render if enabled (after action execution)
                    if self.render:
                        self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                        # Add small delay to make rendering visible and keep window responsive
                        from time import sleep
                        sleep(0.1)  # 100ms delay to make rendering visible
                        # Also handle events to keep window responsive
                        try:
                            import pygame
                            if pygame.get_init():
                                pygame.event.pump()
                        except:
                            pass
                
                # Close progress bar
                if pbar is not None:
                    pbar.close()
                
                # Final update if buffer has remaining data
                if hasattr(self, 'buffer') and len(self.buffer['obs'][0]) > 0:
                    self._batch_update_networks(obs, done)
                
                # Update statistics
                self.stats['total_steps'] = step_count
                self.stats['episode_rewards'].append(episode_reward.mean())
                self.stats['episode_lengths'].append(episode_length)
                
                # Output episode summary (every episode)
                total_reward = episode_reward.sum()
                mean_reward = episode_reward.mean()
                collisions = info.get('collisions', {})
                has_success = any(status == 'SUCCESS' for status in collisions.values())
                has_crash = any(status in ['CRASH_CAR', 'CRASH_WALL', 'CRASH_LINE'] for status in collisions.values())
                
                # Get rollout statistics
                rollout_info = ""
                total_rollouts = sum(mcts.get_rollout_stats()['total_rollouts'] for mcts in self.mcts_instances)
                total_env_steps = sum(mcts.get_rollout_stats()['total_env_steps'] for mcts in self.mcts_instances)
                successful_rollouts = sum(mcts.get_rollout_stats()['successful_rollouts'] for mcts in self.mcts_instances)
                if total_rollouts > 0:
                    rollout_info = f" | Rollouts: {total_rollouts}({successful_rollouts}✓) | EnvSteps: {total_env_steps}"
                
                # Calculate episode duration
                episode_duration = time() - episode_start_time
                
                # Print episode summary with duration
                status_str = ""
                if has_success:
                    status_str = " [SUCCESS]"
                elif has_crash:
                    status_str = " [CRASH]"
                
                print(
                    f"Episode {episode:5d} | "
                    f"Reward: {mean_reward:7.2f} (Total: {total_reward:7.2f}) | "
                    f"Length: {episode_length:5d} | "
                    f"Time: {episode_duration:.1f}s{status_str}{rollout_info}"
                )
                
                # Write episode rewards to CSV
                with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, total_reward, mean_reward, episode_length])
                
                # Update success/crash counters
                if has_success:
                    self.stats['success_count'] += 1
                if has_crash:
                    self.stats['crash_count'] += 1
                
                # Reset rollout stats after each episode (for accurate per-episode reporting)
                for mcts in self.mcts_instances:
                    mcts.reset_rollout_stats()
                
                # Debug: print detailed info for early episodes
                if self.enable_debug and episode <= 10:
                    print(f"[Episode {episode}] Detailed Summary:")
                    print(f"  Episode length: {episode_length}")
                    print(f"  Total reward: {total_reward:.2f}")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Reward per agent: {episode_reward}")
                    if collisions:
                        print(f"  Collisions: {collisions}")
                
                # Reset counters (success/crash are tracked per log_frequency in other versions;
                # here we print per-episode already, so keep counters reset every episode)
                self.stats['success_count'] = 0
                self.stats['crash_count'] = 0
                
                # Save checkpoint
                if episode % self.save_frequency == 0:
                    checkpoint_path = os.path.join(
                        self.save_dir, f"mcts_episode_{episode}.pth"
                    )
                    self.save_checkpoint(checkpoint_path, episode)
                    self.log(f"Checkpoint saved: {checkpoint_path}")
            
            self.log("Training completed!")
        finally:
            # Ensure subprocess workers are torn down to avoid lingering python processes
            self.close()
    
    def _parallel_mcts_search(self, obs, env_state, episode: int = 0, step: int = 0):
        """
        Perform MCTS search for all agents in parallel using process pool.
        
        Args:
            obs: Current observations for all agents
            env_state: Environment state for rollouts
            
        Returns:
            actions: Array of actions for all agents
        """
        # Create process pool if not exists
        if self._process_pool is None:
            # Initialize shared-memory weights for per-step freshness without pickling state_dict.
            if not hasattr(self, "_shared_weight_tensors"):
                self._shared_weight_tensors = None
                self._shared_weight_version = None
                self._shared_weight_lock = None

            if self._shared_weight_tensors is None:
                self._shared_weight_tensors = []
                # Create shared CPU buffers matching state_dict order
                with torch.no_grad():
                    for v in self.network.state_dict().values():
                        t = v.detach().to("cpu").clone()
                        t.share_memory_()
                        self._shared_weight_tensors.append(t)
                self._shared_weight_version = mp.Value('Q', 0)
                self._shared_weight_lock = mp.Lock()

            # Worker processes will inherit these objects (default 'fork' on Linux)
            init_env_config = {
                'num_agents': self.num_agents,
                'num_lanes': self.num_lanes,
                'use_team_reward': self.use_team_reward,
                'max_steps': self.max_steps_per_episode,
                'respawn_enabled': self.respawn_enabled,
                'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
                'ego_routes': self.ego_routes,
            }

            self._process_pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker_shared_state,
                initargs=(self._shared_weight_tensors, self._shared_weight_version, self._shared_weight_lock, init_env_config)
            )
        
        # Submit searches (output disabled - only episode-level logs are shown)
        
        # Note: For process pool, we need to pass serializable data
        # Networks and MCTS instances cannot be passed directly, so we'll need to
        # recreate them in each process using the module-level wrapper function
        # Prepare arguments for each agent (must be serializable)
        # Broadcast latest weights to shared memory only when they've changed.
        if (
            getattr(self, "_weights_dirty", True)
            and hasattr(self, "_shared_weight_tensors")
            and self._shared_weight_tensors is not None
        ):
            with self._shared_weight_lock:
                state = self.network.state_dict()
                for src, dst in zip(state.values(), self._shared_weight_tensors):
                    dst.copy_(src.detach().to("cpu"))
                self._shared_weight_version.value += 1
            self._weights_dirty = False

        search_args = []
        for i in range(self.num_agents):
            config = {
                'hidden_dim': 256,
                'lstm_hidden_dim': 128,
                'use_lstm': self.use_lstm,
                'sequence_length': 5,
                'device': str(self.device),
                'num_simulations': self.mcts_simulations,
                'c_puct': 1.0,
                'temperature': 1.0,
                'rollout_depth': self.rollout_depth,
                'num_action_samples': self.num_action_samples,
                # With shared-memory broadcast, workers read weights directly; no need to send state_dict.
                'network_state_dict': None,
                'base_seed': self.seed,
                'episode': int(episode),
                'step': int(step),
                # env_config is now provided once via the process initializer to reduce per-step pickling.
                'env_config': None
            }
            
            # Convert hidden state to CPU and numpy (only when using LSTM)
            hidden_state_cpu = None
            if self.use_lstm and self.hidden_states[i] is not None:
                if isinstance(self.hidden_states[i], tuple):
                    hidden_state_cpu = tuple(h.cpu().numpy() for h in self.hidden_states[i])
                else:
                    hidden_state_cpu = self.hidden_states[i].cpu().numpy()
            
            search_args.append((
                i,
                obs[i].copy(),
                hidden_state_cpu,
                config
            ))
        
        # Submit all agent searches to process pool
        # Use module-level function for process pool (must be picklable)
        futures = {self._process_pool.submit(_search_agent_wrapper, args): i for i, args in enumerate(search_args)}
        
        # Collect results as they complete
        actions = [None] * self.num_agents
        completed = 0
        errors = []
        start_time = datetime.now()
        
        try:
            # Wait for results (output disabled - only episode-level logs are shown)
            for future in as_completed(futures, timeout=600):  # 10 minute total timeout
                completed += 1
                
                try:
                    i, action, hidden_out = future.result(timeout=120)  # 2 minute timeout per result
                    actions[i] = action
                    if self.use_lstm and hidden_out is not None:
                        self.hidden_states[i] = hidden_out
                except Exception as e:
                    agent_id = futures[future]
                    error_msg = f"Error getting result for agent {agent_id}: {e}"
                    errors.append(error_msg)
                    self.log(error_msg)
                    import traceback
                    self.log(traceback.format_exc())
                    # Use zero action as fallback
                    actions[agent_id] = np.zeros(2)
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.log(f"Fatal error in as_completed after {elapsed:.1f}s: {e}")
            import traceback
            self.log(traceback.format_exc())
            # Fill remaining actions with zeros
            for i in range(self.num_agents):
                if actions[i] is None:
                    self.log(f"Warning: Agent {i} action is None, using zero action")
                    actions[i] = np.zeros(2)
        
        # Check if all actions are collected
        for i in range(self.num_agents):
            if actions[i] is None:
                self.log(f"Warning: Agent {i} action is None, using zero action")
                actions[i] = np.zeros(2)
        
        if errors:
            self.log(f"Total errors during parallel MCTS: {len(errors)}")
        
        return np.array(actions)
    
    def _batched_mcts_search(self, obs, env_state):
        """
        Perform MCTS search for all agents with batched rollouts.
        Collects rollout requests from all agents and executes them in batches.
        
        Args:
            obs: Current observations for all agents
            env_state: Environment state for rollouts
            
        Returns:
            actions: Array of actions for all agents
        """
        # Initialize a shared rollout queue for batch processing
        # This allows all agents to submit rollout requests that can be executed together
        actions = []
        
        # For now, execute sequentially but with shared environment state
        # This is a stepping stone - we can optimize further by batching rollout execution
        for i in range(self.num_agents):
            # Handle pygame events during MCTS search to keep window responsive
            if self.render:
                try:
                    import pygame
                    if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return np.array(actions + [np.zeros(2)] * (self.num_agents - len(actions)))
                except Exception:
                    pass
            
            action, search_stats = self.mcts_instances[i].search(
                root_state=obs[i],
                obs_history=list(self.obs_history[i]) if self.use_lstm else None,
                hidden_state=self.hidden_states[i] if self.use_lstm else None,
                env=self.env, # Pass env for C++ MCTS
                env_state=env_state
            )
            
            # Ensure action is numpy array with correct shape
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if action.ndim == 0:
                action = np.array([action])
            action = action.flatten()
            
            # Update hidden state after MCTS search
            if self.use_lstm:
                obs_seq = np.array(list(self.obs_history[i]))
                if len(obs_seq) < 5:
                    obs_seq = np.array([obs_seq[0]] * (5 - len(obs_seq)) + list(obs_seq))
                obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
                _, _, _, self.hidden_states[i] = self.network(obs_tensor, self.hidden_states[i])
            
            # Add exploration noise in early training
            if self.stats['episode'] < 1000:
                noise_scale = 0.3 * (1.0 - self.stats['episode'] / 1000.0)
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        
        return np.array(actions)
    
    def _update_networks(self, obs, actions, rewards, next_obs, done):
        """
        Update networks using experience buffer.
        Store transitions and update in batches for stability.
        """
        # Store transitions in buffer (per-step for TBPTT)
        for i in range(self.num_agents):
            # Store transition (we'll update in batches)
            if not hasattr(self, 'buffer'):
                self.buffer = {
                    'obs': [[] for _ in range(self.num_agents)],
                    'next_obs': [[] for _ in range(self.num_agents)],
                    'actions': [[] for _ in range(self.num_agents)],
                    'rewards': [[] for _ in range(self.num_agents)],
                    'dones': [[] for _ in range(self.num_agents)],
                }

            self.buffer['obs'][i].append(np.asarray(obs[i], dtype=np.float32))
            self.buffer['next_obs'][i].append(np.asarray(next_obs[i], dtype=np.float32))
            self.buffer['actions'][i].append(actions[i])
            self.buffer['rewards'][i].append(rewards[i])
            self.buffer['dones'][i].append(done)
        
        # Update networks when buffer reaches threshold 
        buffer_size = len(self.buffer['obs'][0])
        if buffer_size >= 64:  # Update every 64 steps
            self._batch_update_networks(next_obs, done)
    
    def _batch_update_networks(self, next_obs, done):
        """Batch update networks using stored transitions."""
        # Non-LSTM mode: the current update code path is TBPTT/LSTM-specific.
        # For A/B speed comparisons we can safely skip updates.
        if not getattr(self.network, 'use_lstm', False):
            # Clear buffer
            self.buffer = {
                'obs': [[] for _ in range(self.num_agents)],
                'next_obs': [[] for _ in range(self.num_agents)],
                'actions': [[] for _ in range(self.num_agents)],
                'rewards': [[] for _ in range(self.num_agents)],
                'dones': [[] for _ in range(self.num_agents)],
            }
            return

        gamma = 0.99  # Discount factor
        unroll = int(getattr(self, 'unroll_len', 16))
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_steps = 0

        self.network.train()
        self.optimizer.zero_grad()

        # TBPTT over each agent trajectory chunk
        for i in range(self.num_agents):
            T = len(self.buffer['obs'][i])
            if T == 0:
                continue

            obs_arr = np.asarray(self.buffer['obs'][i], dtype=np.float32)        # (T, obs_dim)
            next_obs_arr = np.asarray(self.buffer['next_obs'][i], dtype=np.float32)  # (T, obs_dim)
            act_arr = np.asarray(self.buffer['actions'][i], dtype=np.float32)    # (T, action_dim)
            rew_arr = np.asarray(self.buffer['rewards'][i], dtype=np.float32)    # (T,)
            done_arr = np.asarray(self.buffer['dones'][i], dtype=np.float32)     # (T,)

            h = None
            # iterate chunks
            for start in range(0, T, unroll):
                end = min(start + unroll, T)
                L = end - start
                if L <= 0:
                    continue

                obs_chunk = torch.from_numpy(obs_arr[start:end]).unsqueeze(0).to(self.device)      # (1, L, obs_dim)
                act_chunk = torch.from_numpy(act_arr[start:end]).unsqueeze(0).to(self.device)      # (1, L, action_dim)
                rew_chunk = torch.from_numpy(rew_arr[start:end]).to(self.device)                   # (L,)
                done_chunk = torch.from_numpy(done_arr[start:end]).to(self.device)                 # (L,)

                # Forward over sequence
                mean, std, value, h_out = self.network(obs_chunk, h, return_sequence=True)
                # mean/std: (1,L,A), value: (1,L,1)
                mean = mean.squeeze(0)
                std = std.squeeze(0)
                value = value.squeeze(0).squeeze(-1)  # (L,)

                # Bootstrap for returns from next_obs of last step (no recurrent state for bootstrap to keep it simple)
                with torch.no_grad():
                    next_obs_last = torch.from_numpy(next_obs_arr[end - 1]).view(1, 1, -1).to(self.device)
                    _, _, v_next, _ = self.network(next_obs_last, None, return_sequence=False)
                    v_next = v_next.view(-1)[0]

                returns = torch.zeros_like(rew_chunk)
                running = v_next
                for t in reversed(range(L)):
                    running = rew_chunk[t] + gamma * running * (1.0 - done_chunk[t])
                    returns[t] = running

                adv = returns - value.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_chunk.squeeze(0)).sum(dim=-1)  # (L,)

                policy_loss = -(logp * adv).mean()
                value_loss = torch.nn.functional.mse_loss(value, returns)

                (value_loss + 0.5 * policy_loss).backward()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_steps += L

                # TBPTT: carry hidden state but detach graph
                h = None
                if isinstance(h_out, tuple):
                    h = (h_out[0].detach(), h_out[1].detach())
                elif h_out is not None:
                    h = h_out.detach()

                # Reset hidden if any done happened within chunk (approx)
                if float(done_chunk.max().item()) > 0.0:
                    h = None

        if total_steps > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            # Mark weights dirty so the next parallel MCTS step broadcasts updated weights.
            self._weights_dirty = True
        
        # Clear buffer
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'next_obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
        }
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint and export TorchScript model (policy_net_jit.pt) for C++ inference."""
        checkpoint = {
            'episode': episode,
            'shared': True,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'seed': self.seed,
            'deterministic': self.deterministic,
        }
        torch.save(checkpoint, path)

        # --- TorchScript export (for C++ rollout) ---
        try:
            self.network.eval()
            seq_len = getattr(self.network, 'sequence_length', 5)
            dummy_obs = torch.randn(1, seq_len, OBS_DIM).to(self.device)
            if getattr(self.network, 'use_lstm', False):
                hidden_dim = getattr(self.network, 'lstm_hidden_dim', 128)
                h0 = (
                    torch.zeros(1, 1, hidden_dim).to(self.device),
                    torch.zeros(1, 1, hidden_dim).to(self.device)
                )
                traced = torch.jit.trace(self.network, (dummy_obs, h0))
            else:
                traced = torch.jit.trace(self.network, dummy_obs)

            # Save model on CPU to avoid GPU device dependency in C++
            traced.to(torch.device('cpu'))
            torchscript_path = os.path.join(os.path.dirname(path), 'policy_net_jit.pt')
            traced.save(torchscript_path)
        except Exception as e:
            # Non-fatal – continue training even if export fails
            self.log(f"[TorchScript export] Warning: {e}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint.get('shared', False):
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Backward compatibility: older checkpoints saved per-agent networks
            if 'networks_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['networks_state_dict'][0])
            if 'optimizers_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizers_state_dict'][0])
        
        self.stats = checkpoint['stats']
        self.log(f"Checkpoint loaded: {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent MCTS Training')
    parser.add_argument('--num-agents', type=int, default=6, help='Number of agents')
    parser.add_argument('--num-lanes', type=int, default=2, help='Number of lanes per direction')
    parser.add_argument('--max-episodes', type=int, default=100000, help='Max training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--mcts-simulations', type=int, default=5, help='Number of MCTS simulations per step (default: 25)')
    parser.add_argument('--rollout-depth', type=int, default=3, help='Number of steps to rollout in environment (default: 3)')
    parser.add_argument('--num-action-samples', type=int, default=3, help='Number of actions to sample per node expansion (default: 3)')
    parser.add_argument('--save-frequency', type=int, default=100, help='Episodes before saving checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--no-team-reward', action='store_false', dest='use_team_reward', default=True, help='Disable team reward (enabled by default for multi-agent mode)')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--show-lane-ids', action='store_true', help='Show lane IDs in render')
    parser.add_argument('--show-lidar', action='store_true', help='Show lidar in render')
    parser.add_argument('--no-respawn', action='store_true', help='Disable respawn (respawn is enabled by default)')
    parser.add_argument('--save-dir', type=str, default='MCTS_DUAL/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--no-parallel-mcts', action='store_true', help='Disable parallel MCTS (use sequential)')
    parser.add_argument('--max-workers', type=int, default=6, help='Max processes for parallel MCTS (default: num_agents)')
    parser.add_argument('--tqdm', action='store_true', help='Enable tqdm progress bar')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic torch algorithms (slower)')
    parser.add_argument('--no-lstm', action='store_true', help='Disable LSTM in policy/value network and use non-recurrent C++ MCTS backend')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"CUDA available, using CUDA")
            print(f"{torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    else:
        device = args.device

    # Set global seeds (optional)
    if args.seed is not None:
        set_global_seeds(args.seed, deterministic=args.deterministic)
    
    # Create trainer
    trainer = MCTSTrainer(
        num_agents=args.num_agents,
        num_lanes=args.num_lanes,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        mcts_simulations=args.mcts_simulations,
        rollout_depth=args.rollout_depth,
        num_action_samples=args.num_action_samples,
        save_frequency=args.save_frequency,
        device=device,
        use_team_reward=args.use_team_reward,
        use_lstm=not args.no_lstm,
        render=args.render,
        show_lane_ids=args.show_lane_ids,
        show_lidar=args.show_lidar,
        respawn_enabled=not args.no_respawn,  # Enabled by default
        save_dir=args.save_dir,
        parallel_mcts=not args.no_parallel_mcts,  # Enable by default
        max_workers=args.max_workers,
        use_tqdm=args.tqdm,
        seed=args.seed,
        deterministic=args.deterministic
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Start training
    try:
        print("Calling trainer.train()...")
        trainer.train()
        print("Training completed normally.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final checkpoint
        final_checkpoint = os.path.join(args.save_dir, 'mcts_interrupted.pth')
        trainer.save_checkpoint(final_checkpoint, trainer.stats['episode'])
        print(f"Final checkpoint saved: {final_checkpoint}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        # Save final checkpoint if possible
        try:
            final_checkpoint = os.path.join(args.save_dir, 'mcts_error.pth')
            trainer.save_checkpoint(final_checkpoint, trainer.stats.get('episode', 0))
            print(f"Error checkpoint saved: {final_checkpoint}")
        except Exception:
            pass
    finally:
        # Always shutdown worker processes
        try:
            trainer.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()