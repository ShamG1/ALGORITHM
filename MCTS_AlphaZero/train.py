# --- train.py ---
# Multi-Agent MCTS Training Script (Heavily Optimized Version)
# Optimizations:
# - Event-based synchronization instead of busy-wait polling
# - Weight sync only at episode boundaries
# - Pre-allocated tensor buffers
# - Workers use CPU for inference (faster for small networks)

import os
import sys
import time
import struct
import queue

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

# Suppress torch.compile errors in forked processes
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True

# Thread settings:
# - `set_num_threads` is safe and can be applied multiple times.
# - `set_num_interop_threads` MUST be called before any parallel work starts;
#   calling it too late can abort the process.
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
from datetime import datetime
# NOTE: don't shadow the `time` module; use `time.time()`
# from time import time
from collections import deque
import csv
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Dict, List, Any
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DriveSimX.core.env import ScenarioEnv
from DriveSimX.core.utils import DEFAULT_REWARD_CONFIG, OBS_DIM

# Local training components
try:
    from .dual_net import DualNetwork
    from .mcts import MCTS, prepare_obs_sequence
except ImportError:
    from dual_net import DualNetwork
    from mcts import MCTS, prepare_obs_sequence

# ============================================================================
# Optimized Shared Memory Buffer with Event-Based Synchronization
# ============================================================================

class SharedMemoryBuffer:
    """
    Optimized shared memory buffer with Event-based synchronization.
    Eliminates busy-wait polling for significant performance improvement.
    Includes space for MCTS stats (AlphaZero training).
    """
    def __init__(self, num_agents: int, obs_dim: int, lstm_hidden_dim: int, action_dim: int = 2, num_action_samples: int = 5):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_samples = num_action_samples
        
        # Calculate buffer size
        # Stats layout per agent (Scheme A):
        # [root_v (1), root_n (1), h_next (H), c_next (H), actions_k (K*2), visits_k (K)]
        self.lstm_hidden_dim = lstm_hidden_dim
        self.stats_per_agent = 2 + 2 * self.lstm_hidden_dim + 3 * num_action_samples
        
        self.obs_size = num_agents * obs_dim * 4  # float32
        self.action_size = num_agents * action_dim * 4
        self.stats_size = num_agents * self.stats_per_agent * 4
        self.token_size = 8
        
        self.total_size = self.obs_size + self.action_size + self.stats_size + self.token_size
        
        # Create shared memory
        self._shm = shared_memory.SharedMemory(create=True, size=self.total_size)
        
        # Calculate offsets
        self._obs_offset = 0
        self._action_offset = self.obs_size
        self._stats_offset = self.obs_size + self.action_size
        self._token_offset = self.obs_size + self.action_size + self.stats_size
        
        # Event-based synchronization (replaces busy-wait polling)
        self._ready_events = [mp.Event() for _ in range(num_agents)]
        self._done_events = [mp.Event() for _ in range(num_agents)]
        self._all_done_event = mp.Event()
        
        # Pre-allocate numpy arrays for faster read/write
        self._obs_array = np.zeros((num_agents, obs_dim), dtype=np.float32)
        self._action_array = np.zeros((num_agents, action_dim), dtype=np.float32)
    
    @property
    def name(self):
        return self._shm.name
    
    @property
    def ready_events(self):
        return self._ready_events
    
    @property
    def done_events(self):
        return self._done_events
    
    def write_all_observations(self, obs_array: np.ndarray):
        """Write all observations to shared memory."""
        buf = self._shm.buf
        obs_bytes = np.ascontiguousarray(obs_array, dtype=np.float32).tobytes()
        buf[self._obs_offset:self._obs_offset + len(obs_bytes)] = obs_bytes

    def write_token(self, token: int):
        """Write current step token to shared memory."""
        buf = self._shm.buf
        token_bytes = struct.pack('q', token)
        buf[self._token_offset:self._token_offset + 8] = token_bytes
    
    def read_all_actions(self) -> np.ndarray:
        """Read all actions from shared memory."""
        buf = self._shm.buf
        action_bytes = bytes(buf[self._action_offset:self._action_offset + self.action_size])
        return np.frombuffer(action_bytes, dtype=np.float32).reshape(self.num_agents, self.action_dim).copy()

    def read_all_stats(self) -> List[Optional[Dict]]:
        """Read all MCTS stats from shared memory."""
        buf = self._shm.buf
        stats_bytes = bytes(buf[self._stats_offset:self._stats_offset + self.stats_size])
        stats_flat = np.frombuffer(stats_bytes, dtype=np.float32).reshape(self.num_agents, self.stats_per_agent)
        
        all_stats = []
        for i in range(self.num_agents):
            s = stats_flat[i]
            root_v = float(s[0])
            root_n = int(s[1])
            
            # If root_n is 0, it means MCTS didn't run or failed
            if root_n <= 0:
                all_stats.append(None)
                continue
                
            K = self.num_action_samples
            H = self.lstm_hidden_dim
            
            # Extract h_next and c_next
            h_next = s[2:2+H].copy()
            c_next = s[2+H:2+2*H].copy()
            
            # Extract actions and visits
            actions_k = s[2+2*H:2+2*H+K*2].reshape(K, 2)
            visits_k = s[2+2*H+K*2:2+2*H+K*3]
            
            edges = []
            for k in range(K):
                if visits_k[k] > 0:
                    edges.append({
                        "action": tuple(actions_k[k]),
                        "n": int(visits_k[k])
                    })
            
            all_stats.append({
                "root_v": root_v,
                "root_n": root_n,
                "edges": edges,
                "h_next": h_next,
                "c_next": c_next,
                "is_shm": True
            })
        return all_stats
    
    def signal_all_ready(self):
        """Signal all workers to start processing."""
        self._all_done_event.clear()
        for e in self._done_events:
            e.clear()
        for e in self._ready_events:
            e.set()
    
    def wait_all_done(self, timeout: float = 60.0) -> bool:
        """Wait for all workers to complete. Returns True if successful."""
        return self._all_done_event.wait(timeout)
    
    def check_and_signal_all_done(self):
        """Check if all workers are done and signal the main event."""
        if all(e.is_set() for e in self._done_events):
            self._all_done_event.set()
    
    def close(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class SharedMemoryBufferClient:
    """Optimized shared memory buffer client for worker processes."""
    
    def __init__(self, shm_name: str, num_agents: int, agent_id: int, obs_dim: int, lstm_hidden_dim: int, action_dim: int = 2, num_action_samples: int = 5,
                 ready_event: mp.Event = None, done_event: mp.Event = None,
                 all_done_event: mp.Event = None, done_events_list: List[mp.Event] = None):
        self.num_agents = num_agents
        self._agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_samples = num_action_samples
        # We now use a dynamic hidden dim to avoid hardcoding risks.
        self.lstm_hidden_dim = lstm_hidden_dim
        self.stats_per_agent = 2 + 2 * self.lstm_hidden_dim + 3 * num_action_samples
        
        self.obs_size = num_agents * obs_dim * 4
        self.action_size = num_agents * action_dim * 4
        self.stats_size = num_agents * self.stats_per_agent * 4
        
        self._obs_offset = 0
        self._action_offset = self.obs_size
        self._stats_offset = self.obs_size + self.action_size
        self._token_offset = self.obs_size + self.action_size + self.stats_size
        
        self._shm = shared_memory.SharedMemory(name=shm_name)
        
        # Event references
        self._ready_event = ready_event
        self._done_event = done_event
        self._all_done_event = all_done_event
        self._done_events_list = done_events_list
    
    def read_observation(self, agent_id: int) -> np.ndarray:
        """Read observation for a specific agent."""
        buf = self._shm.buf
        offset = self._obs_offset + agent_id * self.obs_dim * 4
        obs_bytes = bytes(buf[offset:offset + self.obs_dim * 4])
        return np.frombuffer(obs_bytes, dtype=np.float32).copy()

    def read_token(self) -> int:
        """Read current step token from shared memory."""
        buf = self._shm.buf
        token_bytes = bytes(buf[self._token_offset:self._token_offset + 8])
        return struct.unpack('q', token_bytes)[0]
    
    @property
    def action_ptr_addr(self) -> int:
        """Get raw address of action buffer for this agent in SHM."""
        import ctypes
        # Get base address of buffer
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        # Offset for this specific agent
        return addr + self._action_offset + self._agent_id * self.action_dim * 4

    @property
    def stats_ptr_addr(self) -> int:
        """Get raw address of stats buffer for this agent in SHM."""
        import ctypes
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        return addr + self._stats_offset + self._agent_id * self.stats_per_agent * 4
    
    def write_action_and_stats(self, agent_id: int, action: np.ndarray, stats: Optional[Dict]):
        """Write action and AlphaZero stats for a specific agent to shared memory."""
        # Use numpy views for robust shared memory access
        # This avoids "memoryview assignment: lvalue and rvalue have different structures" errors
        
        # 1. Write Action
        action_offset = self._action_offset + agent_id * self.action_dim * 4
        # Create a float32 view of the specific action slot
        action_view = np.frombuffer(self._shm.buf, dtype=np.float32, count=self.action_dim, offset=action_offset)
        action_view[:] = action.flatten()
        
        # 2. Write Stats
        stats_offset = self._stats_offset + agent_id * self.stats_per_agent * 4
        stats_view = np.frombuffer(self._shm.buf, dtype=np.float32, count=self.stats_per_agent, offset=stats_offset)
        
        if stats is not None:
            s_arr = np.zeros(self.stats_per_agent, dtype=np.float32)
            s_arr[0] = float(stats.get("root_v", 0.0))
            s_arr[1] = float(stats.get("root_n", 0))
            
            K = self.num_action_samples
            H = self.lstm_hidden_dim
            
            # Scheme A: [root_v, root_n, h_next(H), c_next(H), actions_k(K*2), visits_k(K)]
            
            # Fill h_next/c_next
            if "h_next" in stats:
                h = np.asarray(stats["h_next"], dtype=np.float32).reshape(-1)
                s_arr[2 : 2+min(len(h), H)] = h[:H]
            if "c_next" in stats:
                c = np.asarray(stats["c_next"], dtype=np.float32).reshape(-1)
                s_arr[2+H : 2+H+min(len(c), H)] = c[:H]
                
            # Fill edges
            if "edges" in stats:
                edges = stats["edges"]
                for k in range(min(len(edges), K)):
                    e = edges[k]
                    if isinstance(e, dict):
                        act = e.get("action", (0.0, 0.0))
                        n = e.get("n", 0)
                        
                        s_arr[2 + 2*H + k*2] = float(act[0])
                        s_arr[2 + 2*H + k*2 + 1] = float(act[1])
                        s_arr[2 + 2*H + K*2 + k] = float(n)
            
            stats_view[:] = s_arr
        else:
            stats_view.fill(0)
    
    def wait_ready(self, timeout: float = 1.0) -> bool:
        """Wait for ready signal."""
        if self._ready_event is None:
            return False
        result = self._ready_event.wait(timeout)
        if result:
            self._ready_event.clear()
        return result
    
    def signal_done(self):
        """Signal that this worker is done."""
        if self._done_event is not None:
            self._done_event.set()
        # Check if all workers are done
        if self._done_events_list is not None and self._all_done_event is not None:
            if all(e.is_set() for e in self._done_events_list):
                self._all_done_event.set()
    
    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass


# ============================================================================
# Worker Tensor Cache for Reduced Allocations
# ============================================================================


class WorkerTensorCache:

    """Pre-allocated tensor cache for worker processes."""
    
    def __init__(self, obs_dim: int, seq_len: int, hidden_dim: int, device: str):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Pre-allocate buffers
        self.obs_buffer = torch.zeros(1, seq_len, obs_dim, dtype=torch.float32)
        self.h_buffer = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
        self.c_buffer = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
        
        # Pin memory for faster CPU->GPU transfer
        if device != 'cpu':
            self.obs_buffer = self.obs_buffer.pin_memory()
            self.h_buffer = self.h_buffer.pin_memory()
            self.c_buffer = self.c_buffer.pin_memory()
    
    def prepare_obs_sequence(self, obs_history: deque, use_tcn: bool = False) -> torch.Tensor:
        """Prepare observation sequence tensor from history (unified with main prepare_obs_sequence())."""
        seq = prepare_obs_sequence(list(obs_history), self.seq_len, use_tcn=use_tcn)
        processed_hist = list(seq)
        
        # Ensure buffer is the right size (it might need re-allocation if dim changed)
        current_dim = processed_hist[0].size
        if self.obs_buffer.shape[2] != current_dim:
            self.obs_buffer = torch.zeros(1, self.seq_len, current_dim, dtype=torch.float32)
            if self.device != 'cpu':
                self.obs_buffer = self.obs_buffer.pin_memory()

        # Copy into buffer
        for i, obs in enumerate(processed_hist):
            self.obs_buffer[0, i] = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        
        return self.obs_buffer.to(self.device, non_blocking=True)


# ============================================================================
# Global Configuration and Utility Functions
# ============================================================================

def set_global_seeds(seed: int, deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# Worker process global cache
_WORKER_CACHE: Dict[str, Any] = {
    "network": None,
    "mcts": None,
    "env": None,
    "env_config": None,
    "tensor_cache": None,
}

# Shared weight state (inter-process shared)
_SHARED_STATE: Dict[str, Any] = {
    "tensors": None,
    "version": None,
    "lock": None
}


def _init_worker_shared_state(shared_tensors, version, lock, env_config=None):
    global _SHARED_STATE, _WORKER_CACHE
    _SHARED_STATE["tensors"] = shared_tensors
    _SHARED_STATE["version"] = version
    _SHARED_STATE["lock"] = lock
    if env_config is not None:
        _WORKER_CACHE["env_config"] = env_config


# ============================================================================
# Optimized Worker Loop with Event-Based Synchronization
# ============================================================================

def _pinned_worker_loop_shm_optimized(
    agent_id: int, 
    shm_name: str, 
    num_agents: int, 
    obs_dim: int,
    num_action_samples: int, # Pass consistency
    control_queue: mp.Queue, 
    out_stats_queue: mp.Queue, # Added for stats backchannel
    stop_event: mp.Event,
    ready_event: mp.Event,
    done_event: mp.Event,
    all_done_event: mp.Event,
    done_events_list: List[mp.Event]
):
    """
    Optimized pinned worker with Event-based synchronization.
    Eliminates busy-wait polling for better performance.
    
    NOTE: Workers ALWAYS use CPU to avoid CUDA fork issues.
    For small FC+LSTM networks, CPU inference is often faster than GPU
    due to kernel launch overhead for single-sample inference.
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    torch.set_num_threads(1)
    
    # Disable torch.compile in workers (incompatible with fork + CUDA)
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.suppress_errors = True
    
    global _WORKER_CACHE, _SHARED_STATE
    _WORKER_CACHE["agent_id"] = agent_id
    _WORKER_CACHE["hidden"] = None
    _WORKER_CACHE["obs_history"] = None
    
    config = None
    # Wait for initial config if not provided
    while config is None and not stop_event.is_set():
        try:
            msg = control_queue.get(timeout=0.1)
            if msg[0] == 'CONFIG':
                config = msg[1]
            elif msg[0] == 'CLOSE':
                return
        except queue.Empty:
            continue

    # Connect to shared memory with event references
    shm_client = SharedMemoryBufferClient(
        shm_name, num_agents, agent_id, obs_dim,
        lstm_hidden_dim=int(config.get('lstm_hidden_dim', 128)),
        num_action_samples=num_action_samples,
        ready_event=ready_event,
        done_event=done_event,
        all_done_event=all_done_event,
        done_events_list=done_events_list
    )
    
    tensor_cache = None
    current_token = -1

    while not stop_event.is_set():
        # Wait for ready signal (blocking with timeout)
        if not shm_client.wait_ready(timeout=0.5):
            continue
        
        # CRITICAL: Check control commands AFTER ready signal to ensure token is picked up
        # before starting MCTS search for this specific step.
        try:
            while True:
                try:
                    msg = control_queue.get_nowait()
                    if msg[0] == 'CLOSE':
                        shm_client.close()
                        return
                    elif msg[0] == 'RESET':
                        _WORKER_CACHE["hidden"] = None
                        oh = _WORKER_CACHE.get("obs_history")
                        if oh is not None:
                            try:
                                oh.clear()
                            except Exception:
                                _WORKER_CACHE["obs_history"] = None
                        current_token = -1 # Reset token on new episode
                    elif msg[0] == 'CONFIG':
                        config = msg[1]
                    elif msg[0] == 'TOKEN':
                        current_token = int(msg[1])
                    elif msg[0] == 'SYNC_WEIGHTS':
                        # Force weight sync
                        _WORKER_CACHE["_last_weight_version"] = -1
                except queue.Empty:
                    break
        except Exception:
            pass
        
        if config is None:
            shm_client.signal_done()
            continue
        
        try:
            obs_data = shm_client.read_observation(agent_id)
            
            # Maintain observation history
            seq_len = int(config.get('sequence_length', 5))
            if _WORKER_CACHE.get("obs_history") is None:
                _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)
            else:
                try:
                    if getattr(_WORKER_CACHE["obs_history"], "maxlen", None) != seq_len:
                        _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)
                except Exception:
                    _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)
            
            _WORKER_CACHE["obs_history"].append(obs_data)
            
            # --- PREPARE OBSERVATION SEQUENCE (HANDLES TCN DELTA FEATURES) ---
            use_tcn = config.get('use_tcn', False)
            seq_len = config.get('sequence_length', 5)
            obs_seq_np = prepare_obs_sequence(list(_WORKER_CACHE["obs_history"]), seq_len, use_tcn=use_tcn)
            
            # CRITICAL: Workers ALWAYS use CPU to avoid CUDA fork issues
            # For small networks (FC+LSTM), CPU is often faster for single-sample inference
            worker_device = torch.device('cpu')
            
            if _WORKER_CACHE.get("network") is None:
                from dual_net import DualNetwork
                from DriveSimX.core.utils import OBS_DIM as _OBS_DIM
                
                # Use 2 * OBS_DIM if using TCN with delta features (seq_len 5 mode)
                effective_obs_dim = _OBS_DIM * 2 if (use_tcn and seq_len == 5) else _OBS_DIM
                
                net = DualNetwork(
                    obs_dim=effective_obs_dim, action_dim=2,
                    hidden_dim=config['hidden_dim'],
                    lstm_hidden_dim=config['lstm_hidden_dim'],
                    use_lstm=config.get('use_lstm', True),
                    use_tcn=use_tcn,
                    sequence_length=seq_len
                ).to(worker_device)
                net.eval()
                
                # NOTE: Do NOT use torch.compile in workers
                # 1. Incompatible with fork-based multiprocessing + CUDA
                # 2. For small networks, compilation overhead > benefit
                
                _WORKER_CACHE["network"] = net
                _WORKER_CACHE["worker_device"] = worker_device
                
                # Initialize tensor cache
                tensor_cache = WorkerTensorCache(
                    obs_dim=effective_obs_dim,
                    seq_len=seq_len,
                    hidden_dim=config['lstm_hidden_dim'],
                    device='cpu'
                )
                _WORKER_CACHE["tensor_cache"] = tensor_cache
            
            if _WORKER_CACHE.get("env") is None:
                from DriveSimX.core.env import ScenarioEnv as _ScenarioEnv
                
                env_cfg = _WORKER_CACHE.get('env_config') or {}
                _scenario_name = env_cfg.get('scenario_name', "cross_3lane")
                
                _WORKER_CACHE["env"] = _ScenarioEnv({
                    'scenario_name': _scenario_name,
                    'traffic_flow': False,
                    'num_agents': env_cfg.get('num_agents', 1),
                    'render_mode': None,
                    'max_steps': env_cfg.get('max_steps', 2000),
                    'respawn_enabled': env_cfg.get('respawn_enabled', True),
                    'reward_config': env_cfg.get('reward_config', {}),
                    'ego_routes': list(env_cfg.get('ego_routes', [])),
                })
            
            if _WORKER_CACHE.get("mcts") is None:
                try:
                    from .mcts import MCTS
                except ImportError:
                    from mcts import MCTS
                env_cfg = _WORKER_CACHE.get('env_config') or {}
                _num_agents = int(env_cfg.get('num_agents', 1))
                mcts = MCTS(
                    network=_WORKER_CACHE["network"],
                    num_simulations=config['num_simulations'],
                    c_puct=config['c_puct'],
                    temperature=config['temperature'],
                    device='cpu',  # Workers always use CPU
                    rollout_depth=config['rollout_depth'],
                    agent_id=agent_id,
                    num_action_samples=config['num_action_samples'],
                    ts_model_path=config.get('ts_model_path'),
                )
                mcts._env_cache = _WORKER_CACHE["env"]
                _WORKER_CACHE["mcts"] = mcts
            
            # Sync weights only when version changes
            if _SHARED_STATE.get("version") is not None:
                last_ver = _WORKER_CACHE.get("_last_weight_version", -1)
                cur_ver = int(_SHARED_STATE["version"].value)
                if cur_ver != last_ver:
                    with _SHARED_STATE["lock"]:
                        state = _WORKER_CACHE["network"].state_dict()
                        for p, src in zip(state.values(), _SHARED_STATE["tensors"]):
                            p.copy_(src)
                    _WORKER_CACHE["_last_weight_version"] = cur_ver
            
            mcts = _WORKER_CACHE["mcts"]
            mcts.agent_id = agent_id

            use_lstm_flag = bool(config.get('use_lstm', True))
            use_tcn_flag = bool(config.get('use_tcn', False))

            hidden_state_tensor = _WORKER_CACHE.get('hidden') if use_lstm_flag else None

            with torch.inference_mode():
                # Provide SHM pointers to the search function
                # This will bypass Python queues and write directly to SHM
                shm_ptrs = (shm_client.action_ptr_addr, shm_client.stats_ptr_addr)

                if use_lstm_flag:
                    # LSTM path: use unified Python wrapper, which calls C++ mcts_search_to_shm
                    action, search_stats = mcts.search(
                        obs_data,
                        _WORKER_CACHE["env"],
                        list(_WORKER_CACHE["obs_history"]) if (use_lstm_flag or use_tcn_flag) else None,
                        hidden_state_tensor,
                        shm_ptrs=shm_ptrs
                    )
                else:
                    # TCN / MLP path: call C++ TorchScript+SHM implementation directly
                    from DriveSimX.core import cpp_backend as _cpp_backend

                    env_cpp = _WORKER_CACHE["env"].env
                    root_state_cpp = env_cpp.get_state()

                    # Use the already prepared obs_seq_np (handles TCN delta features/dim doubling)
                    root_obs_seq_flat = obs_seq_np.astype(np.float32).flatten()
                    obs_dim_net = int(root_obs_seq_flat.size // seq_len)

                    _cpp_backend.mcts_search_tcn_torchscript_seq_to_shm(
                        env_cpp,
                        root_state_cpp,
                        root_obs_seq_flat.tolist(),
                        seq_len,
                        obs_dim_net,
                        config.get('ts_model_path'),
                        int(config['lstm_hidden_dim']),
                        int(agent_id),
                        int(config['num_simulations']),
                        int(config['num_action_samples']),
                        int(config['rollout_depth']),
                        float(config['c_puct']),
                        float(config['temperature']),
                        0.99,
                        0.3,
                        0.25,
                        int(config.get('base_seed', 0)),
                        shm_client.action_ptr_addr,
                        shm_client.stats_ptr_addr,
                    )

                    # In pure SHM mode, action/stats are consumed from SHM by the main process.
                    action = np.zeros(2, dtype=np.float32)
                    search_stats = {}

            # CRITICAL FIX: If C++ didn't write to SHM (fallback path), write manually
            if search_stats:
                # This means we entered Python fallback in mcts.py despite shm_ptrs being provided
                shm_client.write_action_and_stats(agent_id, action, search_stats)
            
            # --- Update hidden state ---
            # In SHM mode, search_stats is {} (Python overhead reduced).
            # We must read h_next/c_next from SHM stats buffer to update worker hidden state.
            if use_lstm_flag:
                lstm_hidden_dim = int(config['lstm_hidden_dim'])
                
                if shm_ptrs is not None:
                    # SHM Path: Read directly from stats buffer
                    # Layout: [root_v(1), root_n(1), h_next(H), c_next(H), ...]
                    import ctypes
                    stats_ptr = ctypes.cast(shm_client.stats_ptr_addr, ctypes.POINTER(ctypes.c_float))
                    # h_next starts at offset 2, c_next starts at offset 2 + H
                    h_np = np.fromiter(stats_ptr[2 : 2 + lstm_hidden_dim], dtype=np.float32, count=lstm_hidden_dim)
                    c_np = np.fromiter(stats_ptr[2 + lstm_hidden_dim : 2 + 2 * lstm_hidden_dim], dtype=np.float32, count=lstm_hidden_dim)
                elif isinstance(search_stats, dict) and ("h_next" in search_stats):
                    # Legacy/Fallback Path: Read from dict
                    h_np = np.asarray(search_stats.get("h_next"), dtype=np.float32).reshape(-1)
                    c_np = np.asarray(search_stats.get("c_next"), dtype=np.float32).reshape(-1)
                else:
                    h_np = c_np = None

                if h_np is not None and h_np.size == lstm_hidden_dim:
                    hn = torch.as_tensor(h_np).view(1, 1, lstm_hidden_dim)
                    cn = torch.as_tensor(c_np).view(1, 1, lstm_hidden_dim)
                    _WORKER_CACHE["hidden"] = (hn, cn)
                else:
                    # Reset or keep zeros if not found/mismatch
                    if _WORKER_CACHE.get("hidden") is None:
                        _WORKER_CACHE["hidden"] = (
                            torch.zeros(1, 1, lstm_hidden_dim),
                            torch.zeros(1, 1, lstm_hidden_dim)
                        )
            
            # NOTE: In SHM path, C++ already wrote the action and stats to SHM.
            # We skip the legacy queue-based backchannel for stats if shm_ptrs was used.

            if out_stats_queue is not None:
                try:
                    # Only send to queue if we're not using the direct SHM path (fallback)
                    if search_stats:
                        out_stats_queue.put((current_token, agent_id, search_stats), block=False)
                except queue.Full:
                    pass
        
        except Exception as e:
            import traceback
            sys.stdout.write(f"[PinnedWorker {agent_id}] Error: {e}\n{traceback.format_exc()}\n")
            sys.stdout.flush()
            # Clear SHM action/stats on error
            shm_client.write_action_and_stats(agent_id, np.zeros(2, dtype=np.float32), None)
            if out_stats_queue is not None:
                try:
                    out_stats_queue.put((current_token, agent_id, None), block=False)
                except Exception:
                    pass
        
        shm_client.signal_done()
    
    shm_client.close()


# ============================================================================
# Generation of ego routes
# ============================================================================

def generate_ego_routes(num_agents: int, scenario_name: Optional[str] = None):
 
    from DriveSimX.core.utils import ROUTE_MAP_BY_SCENARIO

    if scenario_name is None:
        scenario_name = "cross_3lane"

    mapping = ROUTE_MAP_BY_SCENARIO.get(str(scenario_name))
    if not mapping:
        # fallback：尽量按车道数选一个存在的场景
        fallback = "cross_3lane"
        mapping = ROUTE_MAP_BY_SCENARIO.get(fallback)

    if not mapping:
        raise ValueError(
            f"No route mapping found for scenario_name={scenario_name!r}. "
            f"Available: {sorted(ROUTE_MAP_BY_SCENARIO.keys())}"
        )

    # Flatten mapping: turn_type -> {in_idx: out_idx}
    # Support both numeric lane indices (e.g. 2 -> 8) and already-prefixed lane ids
    # (e.g. "IN_RAMP_1" -> "OUT_2").
    all_routes = []
    for mp in mapping.values():
        for in_id, out_id in mp.items():
            start = in_id if isinstance(in_id, str) else f"IN_{in_id}"
            end = out_id if isinstance(out_id, str) else f"OUT_{out_id}"
            all_routes.append((start, end))

    if not all_routes:
        raise RuntimeError(f"Empty route mapping for scenario_name={scenario_name!r}")

    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4

    used_routes = set()
    import re
    # 自动解析 num_lanes
    m = re.findall(r"(?:^|_)(\d+)lane(?:$|_)", str(scenario_name))
    num_lanes = int(m[-1]) if m else 2
    # Try to balance by approach direction blocks: indices are grouped by direction in build_lane_layout
    for i in range(4):
        count = agents_per_dir + (1 if i < extra_agents else 0)
        start_idx = i * int(num_lanes) + 1
        dir_routes = [
            r
            for r in all_routes
            if any(r[0].startswith(f"IN_{j}") for j in range(start_idx, start_idx + int(num_lanes)))
        ]

        if not dir_routes:
            dir_routes = all_routes

        available_routes = [r for r in dir_routes if r not in used_routes] or dir_routes

        for k in range(count):
            route = available_routes[k % len(available_routes)]
            selected_routes.append(route)
            used_routes.add(route)

    # Fill remaining if num_agents > distinct dir-balanced picks
    remaining_routes = [r for r in all_routes if r not in used_routes]
    while len(selected_routes) < num_agents:
        if remaining_routes:
            selected_routes.append(remaining_routes.pop(0))
        else:
            selected_routes.append(all_routes[len(selected_routes) % len(all_routes)])

    return selected_routes[:num_agents]


# ============================================================================
# MCTSTrainer Class (Heavily Optimized)
# ============================================================================

class MCTSTrainer:
    """Multi-Agent MCTS Trainer with heavy performance optimizations."""
    
    def close(self):
        """Release resources including shared memory."""
        # Set stop event
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        
        # Send close commands (optimized version)
        if hasattr(self, '_control_queues') and self._control_queues is not None:
            for q in self._control_queues:
                try:
                    q.put(('CLOSE',))
                except Exception:
                    pass
        
        # Wait for worker processes to end
        if hasattr(self, '_pinned_procs') and self._pinned_procs is not None:
            for p in self._pinned_procs:
                try:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass
        
        # Clean up shared memory
        if hasattr(self, '_shm_buffer') and self._shm_buffer is not None:
            try:
                self._shm_buffer.close()
            except Exception:
                pass
            self._shm_buffer = None
        
        # Clean up environment
        try:
            if hasattr(self, 'env') and hasattr(self.env, 'close'):
                self.env.close()
        except Exception:
            pass
        
        self._pinned_procs = None
        self._pinned_in_queues = None
        self._pinned_out_queues = None
        self._control_queues = None
        self._stop_event = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __init__(
        self,
        num_agents: int = 6,
        scenario_name: str = "cross_3lane",
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
        use_tcn: bool = False,
        render: bool = False,
        show_lane_ids: bool = False,
        show_lidar: bool = False,
        respawn_enabled: bool = True,
        save_dir: str = 'C/checkpoints',
        parallel_mcts: bool = True,
        max_workers: int = None,
        use_tqdm: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        use_shm: bool = True
    ):
        self.num_agents = num_agents
        self.scenario_name = scenario_name
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
        self.use_tcn = use_tcn
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.parallel_mcts = parallel_mcts
        self.max_workers = max_workers if max_workers is not None else num_agents
        self.use_tqdm = use_tqdm and tqdm is not None
        self.use_shm = use_shm
        if not self.use_shm:
            raise ValueError("use_shm must be True")
        # LSTM hidden dim will be taken from the network after it is constructed.
        # (Set after self.network is created; do not read from env vars.)
        self.lstm_hidden_dim = None
        
        # Worker resources
        self._pinned_procs = None
        self._control_queues = None
        self._shm_buffer = None
        self._stop_event = None

        
        self.seed = seed
        self.deterministic = deterministic
        self._weights_dirty = True
        
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        ego_routes = generate_ego_routes(num_agents, scenario_name)
        self.ego_routes = ego_routes
        
        route_counts = {}
        for route in ego_routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        # duplicates = {r: c for r, c in route_counts.items() if c > 1}
        # if duplicates:
        #     print(f"WARNING: Found duplicate routes: {duplicates}")
        #     print(f"All routes: {ego_routes}")
        
        self.env = ScenarioEnv({
            'scenario_name': scenario_name,
            'traffic_flow': False,
            'num_agents': num_agents,
            'use_team_reward': use_team_reward,
            'render_mode': 'human' if render else None,
            'max_steps': max_steps_per_episode,
            'respawn_enabled': respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            'ego_routes': ego_routes
        })
        
        # Enforce mutual exclusion: TCN replaces LSTM
        if self.use_tcn:
            self.use_lstm = False

        effective_obs_dim = OBS_DIM * 2 if self.use_tcn else OBS_DIM

        self.network = DualNetwork(
            obs_dim=effective_obs_dim,
            action_dim=2,
            hidden_dim=256,
            lstm_hidden_dim=128,
            use_lstm=self.use_lstm,
            use_tcn=self.use_tcn,
            sequence_length=5
        ).to(self.device)

        # Single source of truth for SHM layout.
        self.lstm_hidden_dim = int(getattr(self.network, "lstm_hidden_dim", 128))
        
        # NOTE: torch.compile is disabled because:
        # 1. It's incompatible with fork-based multiprocessing + CUDA
        # 2. For small FC+LSTM networks, compilation overhead > benefit
        # 3. Workers use CPU anyway for better single-sample inference speed
        
        self.networks = [self.network] * num_agents
        
        self.mcts_instances = [
            MCTS(
                network=self.network,
                num_simulations=mcts_simulations,
                c_puct=1.0,
                temperature=1.0,
                device=device,
                rollout_depth=self.rollout_depth,
                agent_id=i,
                num_action_samples=self.num_action_samples,
                ts_model_path=os.path.join(self.save_dir, 'mcts_infer.pt'),
            )
            for i in range(num_agents)
        ]
        
        self.enable_rollout_debug = False
        self.enable_debug = False

        # Shared weights
        if not hasattr(self, "_shared_weight_tensors"):
            self._shared_weight_tensors = []

        if self._shared_weight_tensors is None or len(self._shared_weight_tensors) == 0:
            self._shared_weight_tensors = []
            with torch.no_grad():
                for v in self.network.state_dict().values():
                    t = v.detach().to("cpu").clone()
                    t.share_memory_()
                    self._shared_weight_tensors.append(t)

        self._shared_weight_version = mp.Value('Q', 0)
        self._shared_weight_lock = mp.Lock()
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        self.obs_history = [deque(maxlen=16) for _ in range(num_agents)] if (self.use_lstm or self.use_tcn) else None
        self.unroll_len = 16
        
        self.stats = {
            'episode': 0,
            'total_steps': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'crash_count': 0,
        }

        self.buffer = {
            'obs': [[] for _ in range(num_agents)],
            'next_obs': [[] for _ in range(num_agents)],
            'actions': [[] for _ in range(num_agents)],
            'rewards': [[] for _ in range(num_agents)],
            'dones': [[] for _ in range(num_agents)],
            'mcts_pi': [[] for _ in range(self.num_agents)],
            'mcts_v': [[] for _ in range(self.num_agents)],
            'tokens': [[] for _ in range(self.num_agents)],
        }
        
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Num agents: {num_agents}\n")
            f.write(f"Scenario name: {scenario_name}\n")
            f.write(f"Use team reward: {use_team_reward}\n")
            f.write(f"Respawn enabled: {respawn_enabled}\n")
            f.write(f"MCTS simulations: {mcts_simulations}\n")
            f.write(f"Use shared memory: {use_shm}\n")
            f.write(f"Model: {'TCN' if self.use_tcn else 'LSTM' if self.use_lstm else 'MLP'}\n")
            f.write(f"Main device: {device} (workers use CPU)\n")
            f.write("Generated routes:\n")
            for i, route in enumerate(ego_routes):
                f.write(f"  Agent {i}: {route[0]} -> {route[1]}\n")
            f.write("=" * 80 + "\n")
        
        self.csv_file = os.path.join(save_dir, 'episode_rewards.csv')
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward', 'Mean_Reward', 'Episode_Length', 'Episode_Duration_Sec'])
        
        # Generate TorchScript model immediately for optimized MCTS inference
        self._ensure_torchscript_model()
    
    def log(self, message: str):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def _start_pinned_workers_shm(self):
        """Start worker processes with optimized Event-based synchronization."""
        if self._pinned_procs is not None:
            return
        
        # 1. Start SHM buffer with stats support
        self._shm_buffer = SharedMemoryBuffer(
            self.num_agents, 
            OBS_DIM, 
            lstm_hidden_dim=self.lstm_hidden_dim, 
            num_action_samples=self.num_action_samples
        )
        self._stop_event = mp.Event()
        
        init_env_config = {
            'num_agents': self.num_agents,
            'scenario_name': self.scenario_name,
            'use_team_reward': self.use_team_reward,
            'max_steps': self.max_steps_per_episode,
            'respawn_enabled': self.respawn_enabled,
            'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
            'ego_routes': self.ego_routes,
        }
        _init_worker_shared_state(
            self._shared_weight_tensors,
            self._shared_weight_version,
            self._shared_weight_lock,
            init_env_config
        )
        
        # Control queues must be initialized for SHM workers.
        # Control queues for SHM workers
        # Control queues for SHM workers
        self._control_queues = [mp.Queue(maxsize=8) for _ in range(self.num_agents)]

        # Stats backchannel disabled in SHM mode (Scheme A): stats are read synchronously from SHM.
        self._pinned_procs = []
        
        for i in range(self.num_agents):
            p = mp.Process(
                target=_pinned_worker_loop_shm_optimized,
                args=(
                    i, 
                    self._shm_buffer.name, 
                    self.num_agents, 
                    OBS_DIM,
                    self.num_action_samples, # Pass consistency
                    self._control_queues[i],
                    None, # Disable out_stats_queue in worker
                    self._stop_event,
                    self._shm_buffer.ready_events[i],
                    self._shm_buffer.done_events[i],
                    self._shm_buffer._all_done_event,
                    self._shm_buffer.done_events  # Pass all done events for checking
                )
            )
            p.daemon = True
            p.start()
            self._pinned_procs.append(p)

    def _broadcast_weights_if_dirty(self):
        """Broadcast weights to shared memory only if changed."""
        if not getattr(self, "_weights_dirty", True):
            return
        
        with self._shared_weight_lock:
            state = self.network.state_dict()
            for src, dst in zip(state.values(), self._shared_weight_tensors):
                dst.copy_(src.detach().cpu(), non_blocking=True)
            # Ensure copy is complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._shared_weight_version.value += 1
        
        self._weights_dirty = False

    def _parallel_mcts_search_shm(self, obs, _env_state, episode: int = 0, step: int = 0):
        """Optimized parallel MCTS search with Event-based synchronization and stats backchannel."""
        if self._pinned_procs is None:
            self._start_pinned_workers_shm()
        
        # OPTIMIZATION: Only sync weights at episode start
        if step == 0:
            self._broadcast_weights_if_dirty()
        
        config = {
            'hidden_dim': 256,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'use_lstm': self.use_lstm,
            'use_tcn': self.use_tcn,
            'sequence_length': 5,
            'device': str(self.device),  # Main process device (workers ignore this, use CPU)
            'num_simulations': self.mcts_simulations,
            'c_puct': 1.0,
            'temperature': 1.0,
            'rollout_depth': self.rollout_depth,
            'num_action_samples': self.num_action_samples,
            'base_seed': int(self.seed) if self.seed is not None else 0,
            'episode': int(episode),
            'step': int(step),
            'ts_model_path': os.path.join(self.save_dir, 'mcts_infer.pt'),
        }
        
        # 1. Send config and reset only at episode start
        if step == 0:
            for q in self._control_queues:
                q.put(('RESET',))
                q.put(('CONFIG', config))
        
        # 2. Generate and broadcast step token (strict alignment)
        # TOKEN must be sent AFTER RESET/CONFIG to ensure worker has latest context
        token = (int(episode) << 20) | int(step)
        for q in self._control_queues:
            q.put(('TOKEN', token))
        
        # 3. Write observations
        obs_array = np.asarray(obs, dtype=np.float32)
        self._shm_buffer.write_all_observations(obs_array)
        
        # 4. Signal workers and wait for completion (Event-based, no polling!)
        self._shm_buffer.signal_all_ready()
        
        if not self._shm_buffer.wait_all_done(timeout=60.0):
            self.log(f"[WARNING] MCTS search timeout at episode {episode}, step {step}")
        
        # 5. Read actions and stats from SHM (primary source)
        actions = self._shm_buffer.read_all_actions()
        all_search_stats = self._shm_buffer.read_all_stats()

        # --- Debug Instrumentation ---
        if self.stats['total_steps'] % 200 == 0:
            root_ns = [s['root_n'] if s else 0 for s in all_search_stats]
            active_agents = sum(1 for n in root_ns if n > 0)
            #self.log(f"[DEBUG_SHM] Step {step} | Actions Mean: {np.mean(actions):.4f} Std: {np.std(actions):.4f} | "
            #         f"Root_N range: [{min(root_ns)}, {max(root_ns)}] | Active Agents: {active_agents}/{self.num_agents}")
            
            # If all root_n are 0, dump raw memory snippet for the first agent to see if anything is there
            if active_agents == 0:
                # Safer way to read raw buffer without ctypes pointer arithmetic issues
                raw_buf = self._shm_buffer._shm.buf[self._shm_buffer._stats_offset : self._shm_buffer._stats_offset + 32]
                raw_vals = np.frombuffer(raw_buf, dtype=np.float32).tolist()
                self.log(f"[DEBUG_RAW_SHM] First 8 floats of Agent 0 stats buffer: {raw_vals}")

        return actions, all_search_stats, token

    def _ensure_torchscript_model(self):
        """Generate and force-verify TorchScript model for optimized MCTS inference."""
        ts_model_path = os.path.join(self.save_dir, 'mcts_infer.pt')
        self.log(f"Generating TorchScript model at {ts_model_path}...")
        print(f"[INIT] Generating TorchScript model for optimized C++ MCTS...")
        
        try:
            if os.path.exists(ts_model_path):
                os.remove(ts_model_path)
            
            # 1. Identify dimensions
            use_tcn = getattr(self.network, 'use_tcn', False)
            s_len = int(getattr(self.network, 'sequence_length', 5))
            effective_obs_dim = OBS_DIM * 2 if (use_tcn and s_len == 5) else OBS_DIM
            hidden_dim = int(getattr(self.network, 'lstm_hidden_dim', 128))
            
            # 2. Export Wrapper
            try:
                from .dual_net import MCTSInferWrapper
            except ImportError:
                from dual_net import MCTSInferWrapper
            self.network.eval()
            self.network.cpu()
            wrapper = MCTSInferWrapper(self.network)
            scripted = torch.jit.script(wrapper)
            
            # 3. Comprehensive Verification
            test_obs = torch.zeros(1, s_len, effective_obs_dim)
            test_h = torch.zeros(1, 1, hidden_dim)
            test_c = torch.zeros(1, 1, hidden_dim)
            test_action = torch.zeros(1, 2)
            
            # Verify required methods
            _ = scripted.infer_policy_value(test_obs, test_h, test_c)
            _ = scripted.infer_next_hidden(test_obs, test_h, test_c, test_action)
            
            scripted.save(ts_model_path)
            self.network.to(self.device) # Restore device
            
            file_size = os.path.getsize(ts_model_path)
            self.log(f"Successfully exported TorchScript model ({file_size} bytes).")
            print(f"[SUCCESS] TorchScript path FORCED: {ts_model_path}")
            
        except Exception as e:
            import traceback
            err_msg = f"TorchScript FORCE export failed: {e}\n{traceback.format_exc()}"
            self.log(err_msg)
            print(f"[ERROR] TorchScript export failed! MCTS will be 10x slower.")
            # Restore device even on failure
            self.network.to(self.device)
    
    def _export_torchscript_model_direct(self, output_path: str):
        """Directly export TorchScript model without full checkpoint save.
        
        This is faster than save_checkpoint() for initial model generation.
        
        NOTE: Do NOT use torch.jit.freeze() or optimize_for_inference()
        These optimizations can strip @torch.jit.export methods that the C++ backend requires.
        """
        try:
            from .dual_net import MCTSInferWrapper
        except ImportError:
            from dual_net import MCTSInferWrapper
        
        original_device = next(self.network.parameters()).device
        original_training = self.network.training
        
        try:
            self.network.eval()
            self.network.cpu()
            
            # Create and script the MCTS inference wrapper
            wrapper = MCTSInferWrapper(self.network)
            wrapper.eval()
            scripted = torch.jit.script(wrapper)
            
            # Verify exported methods exist (C++ backend requires these)
            required_methods = ['infer_policy_value', 'infer_next_hidden']
            for method_name in required_methods:
                if not hasattr(scripted, method_name):
                    raise RuntimeError(f"TorchScript model missing required method: {method_name}")
            
            # Test the methods work correctly before saving
            # IMPORTANT: Use shapes that match C++ backend exactly
            # C++ passes h/c as (1, 1, H) via make_hc_tensor()
            use_tcn = getattr(self.network, 'use_tcn', False)
            seq_len = getattr(self.network, 'sequence_length', 5)
            hidden_dim = getattr(self.network, 'lstm_hidden_dim', 128)
            batch_size = 1
            
            effective_obs_dim = OBS_DIM * 2 if (use_tcn and int(seq_len) == 5) else OBS_DIM
            
            # Test with BOTH 2D and 3D h/c shapes to ensure compatibility
            test_obs = torch.zeros(batch_size, seq_len, effective_obs_dim)
            test_action = torch.zeros(batch_size, 2)
            
            # Use appropriate test hidden states based on model type
            # test_h_2d/test_c_2d unused (kept for reference)
            # test_h_2d = torch.zeros(batch_size, hidden_dim)
            # test_c_2d = torch.zeros(batch_size, hidden_dim)
            test_h_3d = torch.zeros(1, batch_size, hidden_dim)
            test_c_3d = torch.zeros(1, batch_size, hidden_dim)

            # Test 1: infer_policy_value
            mean, std, value = scripted.infer_policy_value(test_obs, test_h_3d, test_c_3d)
            self.log(f"  infer_policy_value test: mean={mean.shape}, std={std.shape}, value={value.shape}")
            
            # Test 2: infer_next_hidden
            hn, cn = scripted.infer_next_hidden(test_obs, test_h_3d, test_c_3d, test_action)
            self.log(f"  infer_next_hidden test: hn={hn.shape}, cn={cn.shape}")
            
            scripted.save(output_path)
            self.log(f"Direct TorchScript export successful (no freeze/optimize).")
            
        finally:
            # Restore network to original device and training mode
            self.network.to(original_device)
            self.network.train(original_training)
    
    def train(self):
        """Main training loop."""
        self.log("=" * 80)
        self.log("Starting Multi-Agent MCTS Training (Optimized)")
        self.log("=" * 80)

        try:
            if self.seed is not None:
                self.log(f"[Seed] seed={self.seed} deterministic={self.deterministic}")
            
            episode = 0
            step_count = 0
            
            while episode < self.max_episodes:
                episode_start_time = time.time()
                episode += 1
                self.stats['episode'] = episode
                
                try:
                    obs, info = self.env.reset()
                except Exception as e:
                    self.log(f"Error resetting environment at episode {episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
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
                        overlaps = {pos: agents for pos, agents in positions.items() if len(agents) > 1}
                        if overlaps:
                            print(f"  WARNING: Agents with overlapping positions: {overlaps}")
                
                if self.use_lstm or self.use_tcn:
                    for i in range(self.num_agents):
                        self.obs_history[i].clear()
                        self.obs_history[i].append(obs[i])
                
                episode_reward = np.zeros(self.num_agents)
                episode_length = 0
                done = False
                
                pbar = None
                if getattr(self, 'use_tqdm', True) and tqdm is not None:
                    pbar = tqdm(
                        total=self.max_steps_per_episode,
                        desc=f"Episode {episode}",
                        unit="step",
                        leave=False,
                        ncols=100,
                        miniters=1,
                        mininterval=0.1,
                        smoothing=0.05,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    )
                
                while not done and episode_length < self.max_steps_per_episode:
                    if self.render:
                        try:
                            self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                            import pygame
                            if pygame.get_init():
                                pygame.event.pump()
                        except Exception:
                            pass
                    
                    env_state = None
                    
                    if self.use_lstm or self.use_tcn:
                        for i in range(self.num_agents):
                            if episode_length > 0:
                                self.obs_history[i].append(obs[i])
                    
                    for i in range(self.num_agents):
                        self.mcts_instances[i].debug_rollout = (episode == 1 and episode_length == 0)
                    
                    if self.render:
                        try:
                            import pygame
                            if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        return
                        except Exception:
                            pass
                    
                    step_token = -1
                    try:
                        if self.parallel_mcts and self.num_agents > 1:
                            actions, all_search_stats, step_token = self._parallel_mcts_search_shm(
                                obs, env_state, episode=episode, step=episode_length
                            )
                        else:
                            actions, all_search_stats = self._sequential_mcts_search(obs, env_state)
                    except Exception as e:
                        self.log(f"Error in MCTS search at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        actions = np.zeros((self.num_agents, 2))
                        all_search_stats = None
                    
                    if self.enable_debug and episode <= 10 and episode_length == 0:
                        print(f"\n[Episode {episode}, Step 0] Actions selected:")
                        for i, act in enumerate(actions):
                            print(f"  Agent {i}: {act} (shape: {act.shape}, dtype: {act.dtype})")

                    # Execute environment step BEFORE update
                    try:
                        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                    except Exception as e:
                        self.log(f"Error stepping environment at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                    if isinstance(rewards, (list, np.ndarray)):
                        rewards = np.array(rewards)
                    else:
                        rewards = np.array([rewards] * self.num_agents)
                    
                    done = terminated or truncated
                    
                    # Update with AlphaZero search_stats
                    self._update_networks(obs, actions, rewards, next_obs, done, search_stats=all_search_stats, token=step_token)
                    
                    episode_reward += rewards
                    episode_length += 1
                    step_count += 1
                    
                    if pbar is not None:
                        pbar.update(1)
                        avg_reward = episode_reward.mean()
                        pbar.set_postfix({
                            'reward': f'{avg_reward:.2f}',
                            'done': done
                        })
                    
                    obs = next_obs
                    
                    if self.render:
                        self.env.render(show_lane_ids=self.show_lane_ids, show_lidar=self.show_lidar)
                        from time import sleep
                        sleep(0.1)
                        try:
                            import pygame
                            if pygame.get_init():
                                pygame.event.pump()
                        except:
                            pass
                
                if pbar is not None:
                    pbar.close()

                if hasattr(self, 'buffer') and len(self.buffer['obs'][0]) > 0:
                    self._batch_update_networks()
                
                self.stats['total_steps'] = step_count
                self.stats['episode_rewards'].append(episode_reward.mean())
                self.stats['episode_lengths'].append(episode_length)
                
                total_reward = episode_reward.sum()
                mean_reward = episode_reward.mean()
                collisions = info.get('collisions', {})
                has_success = any(status == 'SUCCESS' for status in collisions.values())
                has_crash = any(status in ['CRASH_CAR', 'CRASH_WALL'] for status in collisions.values())
                
                rollout_info = ""
                total_rollouts = sum(mcts.get_rollout_stats()['total_rollouts'] for mcts in self.mcts_instances)
                total_env_steps = sum(mcts.get_rollout_stats()['total_env_steps'] for mcts in self.mcts_instances)
                successful_rollouts = sum(mcts.get_rollout_stats()['successful_rollouts'] for mcts in self.mcts_instances)
                if total_rollouts > 0:
                    rollout_info = f" | Rollouts: {total_rollouts}({successful_rollouts}✓) | EnvSteps: {total_env_steps}"
                
                episode_duration = time.time() - episode_start_time
                
                status_str = ""
                if has_success:
                    status_str = " [SUCCESS]"
                elif has_crash:
                    status_str = " [CRASH]"
                
                self.log(
                    f"Episode {episode:5d} | "
                    f"Reward: {mean_reward:7.2f} (Total: {total_reward:7.2f}) | "
                    f"Length: {episode_length:5d} | "
                    f"Time: {episode_duration:.1f}s{status_str}{rollout_info}"
                )
                
                with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, total_reward, mean_reward, episode_length, episode_duration])
                
                if has_success:
                    self.stats['success_count'] += 1
                if has_crash:
                    self.stats['crash_count'] += 1
                
                for mcts in self.mcts_instances:
                    mcts.reset_rollout_stats()
                
                if self.enable_debug and episode <= 10:
                    print(f"\n[Episode {episode}] Summary:")
                    print(f"  Total reward: {total_reward:.2f}")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Reward per agent: {episode_reward}")
                    if collisions:
                        print(f"  Collisions: {collisions}")
                
                self.stats['success_count'] = 0
                self.stats['crash_count'] = 0
                
                if episode % self.save_frequency == 0:
                    checkpoint_path = os.path.join(
                        self.save_dir, f"mcts_episode_{episode}.pth"
                    )
                    self.save_checkpoint(checkpoint_path, episode)
                    self.log(f"Checkpoint saved: {checkpoint_path}")
            
            self.log("Training completed!")
        finally:
            self.close()
    
    def _sequential_mcts_search(self, obs, env_state):
        """Sequential MCTS search for each agent, collecting AlphaZero stats."""
        actions = []
        all_search_stats = []
        
        for i in range(self.num_agents):
            if self.render:
                try:
                    import pygame
                    if pygame.get_init() and hasattr(self.env, 'screen') and self.env.screen is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return np.array(actions + [np.zeros(2)] * (self.num_agents - len(actions))), []
                except Exception:
                    pass
            
            # search returns (action, stats)
            action, search_stats = self.mcts_instances[i].search(
                root_obs=obs[i],
                env=self.env,
                obs_history=list(self.obs_history[i]) if (self.use_lstm or self.use_tcn) else None,
                hidden_state=None,
                env_state=env_state
            )
            
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            action = action.flatten()
            
            actions.append(action)
            all_search_stats.append(search_stats)
        
        return np.array(actions), all_search_stats

    def _update_networks(self, obs, actions, rewards, next_obs, done, search_stats=None, token: Optional[int] = None):
        """Update networks using experience buffer, storing AlphaZero search statistics."""
        if not hasattr(self, 'buffer') or 'mcts_pi' not in self.buffer:
            self.buffer = {
                'obs': [[] for _ in range(self.num_agents)],
                'next_obs': [[] for _ in range(self.num_agents)],
                'actions': [[] for _ in range(self.num_agents)],
                'rewards': [[] for _ in range(self.num_agents)],
                'dones': [[] for _ in range(self.num_agents)],
                'mcts_pi': [[] for _ in range(self.num_agents)], # AlphaZero: visit count distribution
                'mcts_v': [[] for _ in range(self.num_agents)],  # AlphaZero: search value
                'tokens': [[] for _ in range(self.num_agents)],  # Step tokens for late stats injection
            }

        step_token_val = int(token) if token is not None else -1

        for i in range(self.num_agents):
            self.buffer['obs'][i].append(np.asarray(obs[i], dtype=np.float32))
            self.buffer['next_obs'][i].append(np.asarray(next_obs[i], dtype=np.float32))
            self.buffer['actions'][i].append(np.asarray(actions[i], dtype=np.float32))
            self.buffer['rewards'][i].append(float(rewards[i]))
            self.buffer['dones'][i].append(float(done))
            self.buffer['tokens'][i].append(step_token_val)
            
            # AlphaZero stats
            if search_stats is not None and i < len(search_stats):
                stats = search_stats[i]
                if stats is not None and "edges" in stats:
                    # pi = [ (action, visit_prob), ... ]
                    edges = stats["edges"]
                    total_n = stats.get("root_n", sum(e["n"] for e in edges))
                    if total_n > 0:
                        pi = [(np.array(e["action"]), e["n"] / total_n) for e in edges]
                    else:
                        pi = [(np.array(e["action"]), 1.0 / len(edges)) for e in edges]
                    self.buffer['mcts_pi'][i].append(pi)
                    self.buffer['mcts_v'][i].append(float(stats.get("root_v", 0.0)))
                else:
                    self.buffer['mcts_pi'][i].append(None)
                    self.buffer['mcts_v'][i].append(0.0)
            else:
                self.buffer['mcts_pi'][i].append(None)
                self.buffer['mcts_v'][i].append(0.0)

        buffer_size = len(self.buffer['obs'][0])
        # AlphaZero usually updates at the end of episodes, but we can do it periodically.
        update_every = int(getattr(self, 'update_every', 512))
        if buffer_size >= update_every:
            self._batch_update_networks()
    
    def _batch_update_networks(self):
        """Batch update networks using AlphaZero training logic (Weighted Gaussian NLL + MSE Value)."""
        if not hasattr(self, 'buffer') or len(self.buffer['obs'][0]) == 0:
            return

        gamma = 0.99
        unroll = int(getattr(self, 'unroll_len', 16))
        c_pi = 1.0  # Policy loss weight
        c_v = 1.0   # Value loss weight
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_steps = 0

        self.network.train()
        self.optimizer.zero_grad()

        for i in range(self.num_agents):
            T = len(self.buffer['obs'][i])
            if T == 0:
                continue

            obs_arr_raw = np.asarray(self.buffer['obs'][i], dtype=np.float32)
            
            # If using TCN with delta features (obs_dim doubled), build delta features for training.
            # Env provides raw obs dim (145). Network in TCN mode is built with effective_obs_dim=2*OBS_DIM.
            obs_dim_net = int(getattr(self.network, 'obs_dim', obs_arr_raw.shape[-1]))
            if self.use_tcn and obs_arr_raw.shape[-1] * 2 == obs_dim_net:
                # Delta features: concat [obs, obs - prev_obs] per timestep
                obs_arr = np.zeros((obs_arr_raw.shape[0], obs_dim_net), dtype=np.float32)
                prev = obs_arr_raw[0]
                for t in range(obs_arr_raw.shape[0]):
                    cur = obs_arr_raw[t]
                    dx = cur - prev
                    obs_arr[t] = np.concatenate([cur, dx], axis=0)
                    prev = cur
            else:
                obs_arr = obs_arr_raw
            rew_arr = np.asarray(self.buffer['rewards'][i], dtype=np.float32)
            done_arr = np.asarray(self.buffer['dones'][i], dtype=np.float32)
            pi_list = self.buffer['mcts_pi'][i] # List of [(action, prob), ...]

            # 1. Calculate Monte-Carlo Returns (z)
            returns = np.zeros(T, dtype=np.float32)
            running_return = 0.0
            # If the last transition is not 'done', bootstrap with last search value
            if not done_arr[-1]:
                running_return = self.buffer['mcts_v'][i][-1]
            
            for t in reversed(range(T)):
                running_return = rew_arr[t] + gamma * running_return * (1.0 - done_arr[t])
                returns[t] = running_return
            
            returns_torch = torch.from_numpy(returns).to(self.device)

            # 2. Training Loop with Sequence Unrolling (for LSTM/TCN)
            h = None
            for start in range(0, T, unroll):
                end = min(start + unroll, T)
                L = end - start
                
                # Prepare Observation Sequence
                # (For TCN/LSTM, DualNetwork expects (B, T, D))
                obs_chunk = torch.from_numpy(obs_arr[start:end]).unsqueeze(0).to(self.device)
                
                # Forward Pass
                # returns: mean(B,L,2), std(B,L,2), value(B,L,1), h_out
                mean, std, value, h_out = self.network(obs_chunk, h, return_sequence=True)
                
                # Flatten batch and sequence for loss calculation
                mean = mean.view(-1, 2)
                std = std.view(-1, 2)
                value = value.view(-1)
                z_chunk = returns_torch[start:end]

                # --- AlphaZero Policy Loss (Weighted Gaussian NLL) ---
                # For each step in the sequence, we have a set of MCTS sampled actions and their visit probs
                step_policy_loss = torch.tensor(0.0, device=self.device)
                valid_steps = 0
                
                for t_idx in range(L):
                    mcts_pi = pi_list[start + t_idx]
                    if mcts_pi is None: continue
                    
                    # mcts_pi is list of (action_np, prob_float)
                    # actions: (K, 2), probs: (K,)
                    actions_k = torch.stack([torch.from_numpy(a).to(self.device) for a, _p in mcts_pi])
                    probs_k = torch.tensor([p for _a, p in mcts_pi], device=self.device)
                    
                    # Current network distribution for this step
                    dist = torch.distributions.Normal(mean[t_idx], std[t_idx])
                    
                    # Log-likelihood of all MCTS sampled actions
                    # log_prob shape: (K, 2) -> sum to (K,)
                    log_probs = dist.log_prob(actions_k).sum(dim=-1)
                    
                    # Weighted NLL: - sum( pi_i * log p(a_i) )
                    step_loss = -(probs_k * log_probs).sum()
                    step_policy_loss = step_policy_loss + step_loss
                    valid_steps += 1
                
                if valid_steps > 0:
                    step_policy_loss = step_policy_loss / valid_steps

                # --- Value Loss (MSE) ---
                step_value_loss = torch.nn.functional.mse_loss(value, z_chunk)
                # Total Combined Loss
                total_loss = c_pi * step_policy_loss + c_v * step_value_loss
                total_loss.backward()

                total_policy_loss += float(step_policy_loss.detach().cpu())
                total_value_loss += float(step_value_loss.detach().cpu())
                total_steps += L

                # Update Hidden State
                if h_out is not None:
                    if isinstance(h_out, tuple):
                        h = (h_out[0].detach(), h_out[1].detach())
                    else:
                        h = h_out.detach()
                
                if done_arr[end-1]:
                    h = None

        if total_steps > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            self._weights_dirty = True
        
        # Clear buffer after update
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'next_obs': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
            'mcts_pi': [[] for _ in range(self.num_agents)],
            'mcts_v': [[] for _ in range(self.num_agents)],
            'tokens': [[] for _ in range(self.num_agents)],
        }

    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint and export optimized TorchScript model."""
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

        try:
            self.network.eval()
            s_len = int(getattr(self.network, 'sequence_length', 5))
            use_tcn = getattr(self.network, 'use_tcn', False)
            effective_obs_dim = OBS_DIM * 2 if (use_tcn and s_len == 5) else OBS_DIM

            # NOTE: We intentionally do NOT TorchScript-export the raw DualNetwork here.
            # The C++ MCTS backend expects the exported methods on MCTSInferWrapper
            # (infer_policy_value / infer_next_hidden). Scripting DualNetwork directly
            # can produce warnings in TCN/LSTM feature-gated configs and is not required.
            # If you need a standalone policy TS artifact, export the wrapper instead.

            # Prepare hidden_dim for wrapper tests
            hidden_dim = int(getattr(self.network, 'lstm_hidden_dim', 128))

            # Export MCTSInferWrapper for C++ MCTS
            # NOTE: Do NOT use torch.jit.freeze() or optimize_for_inference()
            # These optimizations can strip @torch.jit.export methods that C++ needs
            try:
                from .dual_net import MCTSInferWrapper
            except ImportError:
                from dual_net import MCTSInferWrapper
            wrapper = MCTSInferWrapper(self.network.cpu())
            wrapper.eval()
            scripted = torch.jit.script(wrapper)
            
            # Verify exported methods exist (C++ backend requires these)
            required_methods = ['infer_policy_value', 'infer_next_hidden']
            for method_name in required_methods:
                if not hasattr(scripted, method_name):
                    raise RuntimeError(f"TorchScript model missing required method: {method_name}")
            
            # Test the methods work correctly
            test_obs = torch.zeros(1, s_len, effective_obs_dim)
            test_h = torch.zeros(1, 1, hidden_dim)
            test_c = torch.zeros(1, 1, hidden_dim)
            test_action = torch.zeros(1, 2)
            
            # Test infer_policy_value
            _ = scripted.infer_policy_value(test_obs, test_h, test_c)
            # Test infer_next_hidden
            _ = scripted.infer_next_hidden(test_obs, test_h, test_c, test_action)
            
            mcts_infer_path = os.path.join(os.path.dirname(path), 'mcts_infer.pt')
            scripted.save(mcts_infer_path)
            
            # Move network back to original device
            self.network.to(self.device)
            
        except Exception as e:
            self.log(f"[TorchScript export] Warning: {e}")
            # Ensure network is back on correct device even if export fails
            self.network.to(self.device)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint.get('shared', False):
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if 'networks_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['networks_state_dict'][0])
            if 'optimizers_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizers_state_dict'][0])
        
        self.stats = checkpoint['stats']
        self._weights_dirty = True  # Mark weights as dirty after loading
        self.log(f"Checkpoint loaded: {path}")


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Only supports the YAML structure defined in train_config.yaml.
    """
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Configuration loaded from {yaml_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load YAML config from {yaml_path}: {e}")
        sys.exit(1)
    return config


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent MCTS Training (YAML Config Only)')
    parser.add_argument('--config', type=str, default='MCTS_AlphaZero/train_config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Force load from YAML
    config = load_config_from_yaml(args.config)
    
    # Extract values from nested YAML config
    device_name = config.get('net', {}).get('device', 'cpu')
    if device_name == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_name
    
    scenario_name = str(config.get('env', {}).get('scenario_name', "cross_3lane"))

    print(f"[INFO] Training device: {device}")
    print(f"[INFO] Worker inference device: CPU (optimal for small networks)")
    print(f"[INFO] Scenario: {scenario_name}")

    seed = config.get('train', {}).get('seed')
    if seed is not None:
        set_global_seeds(int(seed), deterministic=config.get('train', {}).get('deterministic', False))
    
    trainer = MCTSTrainer(
        num_agents=int(config.get('env', {}).get('num_agents', 6)),
        scenario_name=str(config.get('env', {}).get('scenario_name', "cross_3lane")),
        max_episodes=int(config.get('train', {}).get('max_episodes', 100000)),
        max_steps_per_episode=int(config.get('env', {}).get('max_steps', 500)),
        mcts_simulations=int(config.get('mcts', {}).get('simulations', 5)),
        rollout_depth=int(config.get('mcts', {}).get('rollout_depth', 5)),
        num_action_samples=int(config.get('mcts', {}).get('num_action_samples', 3)),
        save_frequency=int(config.get('train', {}).get('save_frequency', 100)),
        device=device,
        use_team_reward=bool(config.get('reward', {}).get('use_team_reward', True)),
        use_lstm=bool(config.get('net', {}).get('use_lstm', True)),
        use_tcn=bool(config.get('net', {}).get('use_tcn', False)),
        render=bool(config.get('render', {}).get('enabled', False)),
        respawn_enabled=bool(config.get('env', {}).get('respawn_enabled', True)),
        save_dir=str(config.get('train', {}).get('save_dir', 'MCTS_AlphaZero/checkpoints')),
        parallel_mcts=bool(config.get('mcts', {}).get('parallel', True)),
        max_workers=int(config.get('mcts', {}).get('max_workers', 6)),
        use_shm=bool(config.get('train', {}).get('use_shm', True)),
        use_tqdm=bool(config.get('misc', {}).get('tqdm', True))
    )
    
    load_path = config.get('train', {}).get('load_checkpoint')
    if load_path:
        trainer.load_checkpoint(load_path)
    
    try:
        print("Calling trainer.train()...")
        trainer.train()
        print("Training completed normally.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        #final_checkpoint = os.path.join(trainer.save_dir, 'mcts_interrupted.pth')
        #trainer.save_checkpoint(final_checkpoint, trainer.stats['episode'])
        #print(f"Final checkpoint saved: {final_checkpoint}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            trainer.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
