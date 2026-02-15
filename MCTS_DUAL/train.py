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
from time import time
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

from SIM_MARL.envs.env import ScenarioEnv
from SIM_MARL.envs.utils import DEFAULT_REWARD_CONFIG, OBS_DIM

# ============================================================================
# Optimized Shared Memory Buffer with Event-Based Synchronization
# ============================================================================

class SharedMemoryBuffer:
    """
    Optimized shared memory buffer with Event-based synchronization.
    Eliminates busy-wait polling for significant performance improvement.
    """
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int = 2):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Calculate buffer size
        self.obs_size = num_agents * obs_dim * 4  # float32
        self.action_size = num_agents * action_dim * 4
        
        self.total_size = self.obs_size + self.action_size
        
        # Create shared memory
        self._shm = shared_memory.SharedMemory(create=True, size=self.total_size)
        
        # Calculate offsets
        self._obs_offset = 0
        self._action_offset = self.obs_size
        
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
    
    def read_all_actions(self) -> np.ndarray:
        """Read all actions from shared memory."""
        buf = self._shm.buf
        action_bytes = bytes(buf[self._action_offset:self._action_offset + self.action_size])
        return np.frombuffer(action_bytes, dtype=np.float32).reshape(self.num_agents, self.action_dim).copy()
    
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
    
    def __init__(self, shm_name: str, num_agents: int, obs_dim: int, action_dim: int = 2,
                 ready_event: mp.Event = None, done_event: mp.Event = None,
                 all_done_event: mp.Event = None, done_events_list: List[mp.Event] = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.obs_size = num_agents * obs_dim * 4
        self.action_size = num_agents * action_dim * 4
        
        self._obs_offset = 0
        self._action_offset = self.obs_size
        
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
    
    def write_action(self, agent_id: int, action: np.ndarray):
        """Write action for a specific agent."""
        buf = self._shm.buf
        offset = self._action_offset + agent_id * self.action_dim * 4
        action_bytes = np.ascontiguousarray(action, dtype=np.float32).tobytes()
        buf[offset:offset + len(action_bytes)] = action_bytes
    
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
    
    def prepare_obs_sequence(self, obs_history: deque) -> torch.Tensor:
        """Prepare observation sequence tensor from history with non-uniform sampling and delta."""
        hist_list = list(obs_history)
        seq_len = self.seq_len
        
        # Consistent with mcts.py: Non-uniform sampling [t, t-1, t-3, t-7, t-15]
        if seq_len == 5:
            idx = [-1, -2, -4, -8, -16]
            sampled = []
            for i in idx:
                if -i <= len(hist_list):
                    sampled.append(hist_list[i])
                else:
                    sampled.append(hist_list[0])
            sampled = sampled[::-1] # oldest -> newest
            
            # Delta features
            feat = []
            prev = sampled[0]
            for x in sampled:
                dx = x - prev
                feat.append(np.concatenate([x, dx], axis=0))
                prev = x
            processed_hist = feat
        else:
            # Standard uniform sampling
            if len(hist_list) < seq_len:
                pad = [hist_list[0]] * (seq_len - len(hist_list))
                hist_list = pad + hist_list
            else:
                hist_list = hist_list[-seq_len:]
            processed_hist = hist_list
        
        # Ensure buffer is the right size (it might need re-allocation if dim changed)
        current_dim = processed_hist[0].size
        if self.obs_buffer.shape[2] != current_dim:
            self.obs_buffer = torch.zeros(1, seq_len, current_dim, dtype=torch.float32)
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
    control_queue: mp.Queue, 
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
    
    # Connect to shared memory with event references
    shm_client = SharedMemoryBufferClient(
        shm_name, num_agents, obs_dim,
        ready_event=ready_event,
        done_event=done_event,
        all_done_event=all_done_event,
        done_events_list=done_events_list
    )
    
    config = None
    tensor_cache = None
    
    while not stop_event.is_set():
        # Check control commands (non-blocking)
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
                    elif msg[0] == 'CONFIG':
                        config = msg[1]
                    elif msg[0] == 'SYNC_WEIGHTS':
                        # Force weight sync
                        _WORKER_CACHE["_last_weight_version"] = -1
                except queue.Empty:
                    break
        except Exception:
            pass
        
        # Wait for ready signal (blocking with timeout)
        if not shm_client.wait_ready(timeout=0.5):
            continue
        
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
            
            # CRITICAL: Workers ALWAYS use CPU to avoid CUDA fork issues
            # For small networks (FC+LSTM), CPU is often faster for single-sample inference
            worker_device = torch.device('cpu')
            
            if _WORKER_CACHE.get("network") is None:
                from dual_net import DualNetwork
                from SIM_MARL.envs.utils import OBS_DIM as _OBS_DIM
                
                # Use 2 * OBS_DIM if using TCN with delta features (seq_len 5 mode)
                use_tcn = config.get('use_tcn', False)
                seq_len = config.get('sequence_length', 5)
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
                try:
                    from .train import generate_ego_routes
                    from SIM_MARL.envs.env import ScenarioEnv as _ScenarioEnv
                except ImportError:
                    from train import generate_ego_routes
                    from SIM_MARL.envs.env import ScenarioEnv as _ScenarioEnv
                
                env_cfg = _WORKER_CACHE.get('env_config') or {}
                _WORKER_CACHE["env"] = _ScenarioEnv({
                    'traffic_flow': False,
                    'num_agents': env_cfg.get('num_agents', 1),
                    'num_lanes': env_cfg.get('num_lanes', 3),
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
                    action_space=None,
                    num_simulations=config['num_simulations'],
                    c_puct=config['c_puct'],
                    temperature=config['temperature'],
                    device='cpu',  # Workers always use CPU
                    rollout_depth=config['rollout_depth'],
                    env_factory=None,
                    all_networks=[_WORKER_CACHE["network"]] * _num_agents,
                    agent_id=agent_id,
                    num_action_samples=config['num_action_samples'],
                    ts_model_path=config.get('ts_model_path')
                )
                mcts._env_cache = _WORKER_CACHE["env"]
                mcts.env_factory = lambda: None
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
            
            hidden_state_tensor = _WORKER_CACHE.get('hidden') if config.get('use_lstm') else None
            
            with torch.inference_mode():
                action, search_stats = mcts.search(
                    obs_data,
                    _WORKER_CACHE["env"],
                    list(_WORKER_CACHE["obs_history"]) if (config.get('use_lstm') or config.get('use_tcn')) else None,
                    hidden_state_tensor,
                    None
                )
            
            # Update hidden state (on CPU)
            if config.get('use_lstm') and isinstance(search_stats, dict) and ("h_next" in search_stats):
                lstm_hidden_dim = int(config.get('lstm_hidden_dim', 128))
                h_np = np.asarray(search_stats.get("h_next"), dtype=np.float32).reshape(-1)
                c_np = np.asarray(search_stats.get("c_next"), dtype=np.float32).reshape(-1)
                
                if h_np.size != lstm_hidden_dim or c_np.size != lstm_hidden_dim:
                    hn = torch.zeros(1, 1, lstm_hidden_dim, dtype=torch.float32)
                    cn = torch.zeros(1, 1, lstm_hidden_dim, dtype=torch.float32)
                else:
                    hn = torch.as_tensor(h_np).view(1, 1, lstm_hidden_dim)
                    cn = torch.as_tensor(c_np).view(1, 1, lstm_hidden_dim)
                
                _WORKER_CACHE["hidden"] = (hn, cn)
            
            action_array = np.asarray(action, dtype=np.float32).flatten()
            shm_client.write_action(agent_id, action_array)
            
        except Exception as e:
            import traceback
            sys.stdout.write(f"[PinnedWorker {agent_id}] Error: {e}\n{traceback.format_exc()}\n")
            sys.stdout.flush()
            shm_client.write_action(agent_id, np.zeros(2, dtype=np.float32))
        
        shm_client.signal_done()
    
    shm_client.close()


# ============================================================================
# Legacy Worker Loop (kept for backwards compatibility)
# ============================================================================

def _pinned_worker_loop(agent_id: int, in_q: mp.Queue, out_q: mp.Queue):
    """
    Legacy pinned worker using Queue (kept for backwards compatibility).
    """
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    torch.set_num_threads(1)

    global _WORKER_CACHE
    _WORKER_CACHE["agent_id"] = agent_id
    _WORKER_CACHE["hidden"] = None
    _WORKER_CACHE["obs_history"] = None

    while True:
        try:
            msg = in_q.get()
            if not msg: 
                continue
            mtype = msg[0]

            if mtype == 'CLOSE':
                break

            if mtype == 'RESET':
                _WORKER_CACHE["hidden"] = None
                oh = _WORKER_CACHE.get("obs_history")
                if oh is not None:
                    try:
                        oh.clear()
                    except Exception:
                        _WORKER_CACHE["obs_history"] = None
                continue

            if mtype != 'STEP':
                continue

            obs_data = msg[1]
            config = msg[2]

            seq_len = int(config.get('sequence_length', 5))
            if _WORKER_CACHE.get("obs_history") is None:
                _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)
            else:
                try:
                    if getattr(_WORKER_CACHE["obs_history"], "maxlen", None) != seq_len:
                        _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)
                except Exception:
                    _WORKER_CACHE["obs_history"] = deque(maxlen=seq_len)

            _WORKER_CACHE["obs_history"].append(np.asarray(obs_data, dtype=np.float32))

            device = torch.device(config['device'])
            if _WORKER_CACHE.get("network") is None:
                from dual_net import DualNetwork
                from SIM_MARL.envs.utils import OBS_DIM as _OBS_DIM

                net = DualNetwork(
                    obs_dim=_OBS_DIM, action_dim=2,
                    hidden_dim=config['hidden_dim'], lstm_hidden_dim=config['lstm_hidden_dim'],
                    use_lstm=config.get('use_lstm', True),
                    use_tcn=config.get('use_tcn', False),
                    sequence_length=config['sequence_length']
                ).to(device)
                net.eval()
                _WORKER_CACHE["network"] = net

            if _WORKER_CACHE.get("env") is None:
                try:
                    from .train import generate_ego_routes
                    from SIM_MARL.envs.env import ScenarioEnv as _ScenarioEnv
                except ImportError:
                    from train import generate_ego_routes
                    from SIM_MARL.envs.env import ScenarioEnv as _ScenarioEnv

                env_cfg = _WORKER_CACHE.get('env_config') or {}
                _WORKER_CACHE["env"] = _ScenarioEnv({
                    'traffic_flow': False, 'num_agents': env_cfg.get('num_agents', 1),
                    'num_lanes': env_cfg.get('num_lanes', 3), 'render_mode': None,
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
                num_agents = int(env_cfg.get('num_agents', 1))
                mcts = MCTS(
                    network=_WORKER_CACHE["network"], action_space=None,
                    num_simulations=config['num_simulations'], c_puct=config['c_puct'],
                    temperature=config['temperature'], device=device,
                    rollout_depth=config['rollout_depth'], env_factory=None,
                    all_networks=[_WORKER_CACHE["network"]] * num_agents,
                    agent_id=agent_id, num_action_samples=config['num_action_samples'],
                    ts_model_path=config.get('ts_model_path')
                )
                mcts._env_cache = _WORKER_CACHE["env"]
                mcts.env_factory = lambda: None
                _WORKER_CACHE["mcts"] = mcts

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
            
            hidden_state_tensor = _WORKER_CACHE.get('hidden') if config.get('use_lstm') else None

            with torch.inference_mode():
                action, search_stats = mcts.search(
                    obs_data,
                    _WORKER_CACHE["env"],
                    list(_WORKER_CACHE["obs_history"]) if (config.get('use_lstm') or config.get('use_tcn')) else None,
                    hidden_state_tensor,
                    None
                )

            if config.get('use_lstm') and isinstance(search_stats, dict) and ("h_next" in search_stats):
                lstm_hidden_dim = int(config.get('lstm_hidden_dim', 128))
                h_np = np.asarray(search_stats.get("h_next"), dtype=np.float32).reshape(-1)
                c_np = np.asarray(search_stats.get("c_next"), dtype=np.float32).reshape(-1)

                if h_np.size != lstm_hidden_dim or c_np.size != lstm_hidden_dim:
                    hn = torch.zeros(1, 1, lstm_hidden_dim, device=device, dtype=torch.float32)
                    cn = torch.zeros(1, 1, lstm_hidden_dim, device=device, dtype=torch.float32)
                else:
                    hn = torch.as_tensor(h_np, device=device).view(1, 1, lstm_hidden_dim)
                    cn = torch.as_tensor(c_np, device=device).view(1, 1, lstm_hidden_dim)

                _WORKER_CACHE["hidden"] = (hn, cn)

            out_q.put((agent_id, np.array(action, dtype=np.float32).flatten()))

        except Exception as e:
            import traceback
            sys.stdout.write(f"[PinnedWorker {agent_id}] Error: {e}\\n{traceback.format_exc()}\\n")
            sys.stdout.flush()
            out_q.put((agent_id, np.zeros(2, dtype=np.float32)))


try:
    from .dual_net import DualNetwork, MCTSInferWrapper
    from .mcts import MCTS
except ImportError:
    from dual_net import DualNetwork, MCTSInferWrapper
    from mcts import MCTS


def generate_ego_routes(num_agents: int, num_lanes: int):
    """Generate routes for agents based on num_agents and num_lanes."""
    from SIM_MARL.envs.utils import DEFAULT_ROUTE_MAPPING_2LANES, DEFAULT_ROUTE_MAPPING_3LANES
    
    if num_lanes == 2:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    elif num_lanes == 3:
        route_mapping = DEFAULT_ROUTE_MAPPING_3LANES
    else:
        route_mapping = DEFAULT_ROUTE_MAPPING_2LANES
    
    all_routes = []
    for start, ends in route_mapping.items():
        for end in ends:
            all_routes.append((start, end))
    
    selected_routes = []
    agents_per_dir = num_agents // 4
    extra_agents = num_agents % 4
    
    used_routes = set()
    
    for i in range(4):
        count = agents_per_dir + (1 if i < extra_agents else 0)
        if num_lanes == 3:
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        else:
            start_idx = i * num_lanes + 1
            dir_routes = [r for r in all_routes 
                         if any(r[0].startswith(f'IN_{j}') for j in range(start_idx, start_idx + num_lanes))]
        
        if len(dir_routes) == 0:
            dir_routes = all_routes
        
        available_routes = [r for r in dir_routes if r not in used_routes]
        if len(available_routes) == 0:
            available_routes = dir_routes
        
        dir_route_idx = 0
        for _ in range(count):
            if available_routes:
                route = available_routes[dir_route_idx % len(available_routes)]
                selected_routes.append(route)
                used_routes.add(route)
                dir_route_idx += 1
            elif dir_routes:
                route = dir_routes[dir_route_idx % len(dir_routes)]
                selected_routes.append(route)
                dir_route_idx += 1
    
    remaining_routes = [r for r in all_routes if r not in used_routes]
    while len(selected_routes) < num_agents:
        if remaining_routes:
            route = remaining_routes.pop(0)
            selected_routes.append(route)
            used_routes.add(route)
        else:
            route = all_routes[(len(selected_routes) - len(used_routes)) % len(all_routes)]
            selected_routes.append(route)
    
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
        
        # Send close commands (legacy version)
        if hasattr(self, '_pinned_in_queues') and self._pinned_in_queues is not None:
            for q in self._pinned_in_queues:
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
        self.use_tcn = use_tcn
        self.render = render
        self.show_lane_ids = show_lane_ids
        self.show_lidar = show_lidar
        self.respawn_enabled = respawn_enabled
        self.parallel_mcts = parallel_mcts
        self.max_workers = max_workers if max_workers is not None else num_agents
        self.use_tqdm = use_tqdm and tqdm is not None
        self.use_shm = use_shm
        
        # Worker resources
        self._pinned_in_queues = None
        self._pinned_out_queues = None
        self._pinned_procs = None
        self._control_queues = None
        self._shm_buffer = None
        self._stop_event = None
        
        self.seed = seed
        self.deterministic = deterministic
        self._weights_dirty = True
        
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        ego_routes = generate_ego_routes(num_agents, num_lanes)
        self.ego_routes = ego_routes
        
        route_counts = {}
        for route in ego_routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        duplicates = {r: c for r, c in route_counts.items() if c > 1}
        if duplicates:
            print(f"WARNING: Found duplicate routes: {duplicates}")
            print(f"All routes: {ego_routes}")
        
        self.env = ScenarioEnv({
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
        
        # NOTE: torch.compile is disabled because:
        # 1. It's incompatible with fork-based multiprocessing + CUDA
        # 2. For small FC+LSTM networks, compilation overhead > benefit
        # 3. Workers use CPU anyway for better single-sample inference speed
        
        self.networks = [self.network] * num_agents
        
        def create_env_copy():
            return ScenarioEnv({
                'traffic_flow': False,
                'num_agents': num_agents,
                'num_lanes': num_lanes,
                'render_mode': None,
                'max_steps': max_steps_per_episode,
                'respawn_enabled': respawn_enabled,
                'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
                'ego_routes': ego_routes
            })
        
        self.mcts_instances = [
            MCTS(
                network=self.network,
                action_space=None,
                num_simulations=mcts_simulations,
                c_puct=1.0,
                temperature=1.0,
                device=device,
                rollout_depth=self.rollout_depth,
                num_action_samples=self.num_action_samples,
                env_factory=create_env_copy,
                all_networks=self.networks,
                agent_id=i,
                ts_model_path=os.path.join(self.save_dir, 'mcts_infer.pt')
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
        
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Num agents: {num_agents}\n")
            f.write(f"Num lanes: {num_lanes}\n")
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
        
        self._shm_buffer = SharedMemoryBuffer(self.num_agents, OBS_DIM)
        self._stop_event = mp.Event()
        
        init_env_config = {
            'num_agents': self.num_agents,
            'num_lanes': self.num_lanes,
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
        
        self._control_queues = [mp.Queue(maxsize=8) for _ in range(self.num_agents)]
        self._pinned_procs = []
        
        for i in range(self.num_agents):
            p = mp.Process(
                target=_pinned_worker_loop_shm_optimized,
                args=(
                    i, 
                    self._shm_buffer.name, 
                    self.num_agents, 
                    OBS_DIM,
                    self._control_queues[i], 
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

    def _start_pinned_workers(self):
        """Legacy worker startup using Queue."""
        if self._pinned_procs is not None:
            return

        init_env_config = {
            'num_agents': self.num_agents,
            'num_lanes': self.num_lanes,
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

        self._pinned_in_queues = [mp.Queue(maxsize=2) for _ in range(self.num_agents)]
        self._pinned_out_queues = [mp.Queue(maxsize=2) for _ in range(self.num_agents)]
        self._pinned_procs = []

        for i in range(self.num_agents):
            p = mp.Process(
                target=_pinned_worker_loop,
                args=(i, self._pinned_in_queues[i], self._pinned_out_queues[i])
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

    def _parallel_mcts_search_shm(self, obs, env_state, episode: int = 0, step: int = 0):
        """Optimized parallel MCTS search with Event-based synchronization."""
        if self._pinned_procs is None:
            self._start_pinned_workers_shm()
        
        # OPTIMIZATION: Only sync weights at episode start
        if step == 0:
            self._broadcast_weights_if_dirty()
        
        config = {
            'hidden_dim': 256,
            'lstm_hidden_dim': 128,
            'use_lstm': self.use_lstm,
            'use_tcn': self.use_tcn,
            'sequence_length': 5,
            'device': str(self.device),  # Main process device (workers ignore this, use CPU)
            'num_simulations': self.mcts_simulations,
            'c_puct': 1.0,
            'temperature': 1.0,
            'rollout_depth': self.rollout_depth,
            'num_action_samples': self.num_action_samples,
            'base_seed': self.seed,
            'episode': int(episode),
            'step': int(step),
            'ts_model_path': os.path.join(self.save_dir, 'mcts_infer.pt'),
        }
        
        # Send config and reset only at episode start
        if step == 0:
            for q in self._control_queues:
                q.put(('RESET',))
                q.put(('CONFIG', config))
        
        # Write observations
        obs_array = np.asarray(obs, dtype=np.float32)
        self._shm_buffer.write_all_observations(obs_array)
        
        # Signal workers and wait for completion (Event-based, no polling!)
        self._shm_buffer.signal_all_ready()
        
        if not self._shm_buffer.wait_all_done(timeout=60.0):
            self.log(f"[WARNING] MCTS search timeout at episode {episode}, step {step}")
        
        return self._shm_buffer.read_all_actions()

    def _parallel_mcts_search(self, obs, env_state, episode: int = 0, step: int = 0):
        """Legacy parallel MCTS search using Queue."""
        if self._pinned_procs is None:
            self._start_pinned_workers()

        # OPTIMIZATION: Only sync weights at episode start
        if step == 0:
            self._broadcast_weights_if_dirty()

        config = {
            'hidden_dim': 256,
            'lstm_hidden_dim': 128,
            'use_lstm': self.use_lstm,
            'use_tcn': self.use_tcn,
            'sequence_length': 5,
            'device': str(self.device),
            'num_simulations': self.mcts_simulations,
            'c_puct': 1.0,
            'temperature': 1.0,
            'rollout_depth': self.rollout_depth,
            'num_action_samples': self.num_action_samples,
            'base_seed': self.seed,
            'episode': int(episode),
            'step': int(step),
            'ts_model_path': os.path.join(self.save_dir, 'mcts_infer.pt'),
        }

        if step == 0:
            for q in self._pinned_in_queues:
                q.put(('RESET',))

        for i in range(self.num_agents):
            self._pinned_in_queues[i].put(('STEP', obs[i], config))

        actions = [None] * self.num_agents
        for i in range(self.num_agents):
            agent_id, action = self._pinned_out_queues[i].get()
            actions[agent_id] = action

        return np.asarray(actions, dtype=np.float32)
    
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
            test_h_2d = torch.zeros(batch_size, hidden_dim)
            test_c_2d = torch.zeros(batch_size, hidden_dim)
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
                episode_start_time = time()
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
                
                if self.use_lstm:
                    for i in range(self.num_agents):
                        self.obs_history[i].clear()
                        self.obs_history[i].append(obs[i])
                
                if hasattr(self, 'buffer'):
                    self.buffer = {
                        'obs': [[] for _ in range(self.num_agents)],
                        'next_obs': [[] for _ in range(self.num_agents)],
                        'global_state': [[] for _ in range(self.num_agents)],
                        'next_global_state': [[] for _ in range(self.num_agents)],
                        'actions': [[] for _ in range(self.num_agents)],
                        'rewards': [[] for _ in range(self.num_agents)],
                        'dones': [[] for _ in range(self.num_agents)],
                    }
                
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
                    
                    if self.use_lstm:
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
                    
                    try:
                        if self.parallel_mcts and self.num_agents > 1:
                            if self.use_shm:
                                actions = self._parallel_mcts_search_shm(obs, env_state, episode=episode, step=episode_length)
                            else:
                                actions = self._parallel_mcts_search(obs, env_state, episode=episode, step=episode_length)
                        else:
                            actions = self._sequential_mcts_search(obs, env_state)
                    except Exception as e:
                        self.log(f"Error in MCTS search at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        actions = np.zeros((self.num_agents, 2))
                    
                    if self.enable_debug and episode <= 10 and episode_length == 0:
                        print(f"\n[Episode {episode}, Step 0] Actions selected:")
                        for i, act in enumerate(actions):
                            print(f"  Agent {i}: {act} (shape: {act.shape}, dtype: {act.dtype})")
                        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                            print(f"  WARNING: Invalid actions detected (NaN or Inf)!")
                        if np.any(np.abs(actions) > 1.0):
                            print(f"  WARNING: Actions out of range [-1, 1]!")
                            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                    
                    global_states = None
                    if getattr(self.network, 'use_centralized_critic', False):
                        global_states = [self.env.env.get_global_state(i, 3) for i in range(self.num_agents)]
                    
                    try:
                        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                    except Exception as e:
                        self.log(f"Error stepping environment at episode {episode}, step {episode_length}: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                    next_global_states = None
                    if getattr(self.network, 'use_centralized_critic', False):
                        next_global_states = [self.env.env.get_global_state(i, 3) for i in range(self.num_agents)]
                    
                    if isinstance(rewards, (list, np.ndarray)):
                        rewards = np.array(rewards)
                    else:
                        rewards = np.array([rewards] * self.num_agents)
                    
                    done = terminated or truncated
                    
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
                            print(f"  Agent positions:")
                            for i, agent in enumerate(self.env.agents):
                                print(f"    Agent {i}: pos=({agent.pos_x:.1f}, {agent.pos_y:.1f}), heading={agent.heading:.2f}, speed={agent.speed:.2f}")
                    
                    self._update_networks(obs, actions, rewards, next_obs, done, global_states, next_global_states)
                    
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
                
                episode_duration = time() - episode_start_time
                
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
        """Sequential MCTS search for each agent."""
        actions = []
        
        for i in range(self.num_agents):
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
                hidden_state=None,
                env=self.env,
                env_state=env_state
            )
            
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if action.ndim == 0:
                action = np.array([action])
            action = action.flatten()
            
            if self.use_lstm:
                obs_seq = np.array(list(self.obs_history[i]))
                if len(obs_seq) < 5:
                    obs_seq = np.array([obs_seq[0]] * (5 - len(obs_seq)) + list(obs_seq))
                obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
                _, _, _, _ = self.network(obs_tensor, None)
            
            if self.stats['episode'] < 1000:
                noise_scale = 0.3 * (1.0 - self.stats['episode'] / 1000.0)
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        
        return np.array(actions)
    
    def _update_networks(self, obs, actions, rewards, next_obs, done, global_states=None, next_global_states=None):
        """Update networks using experience buffer."""
        if not hasattr(self, 'buffer'):
            self.buffer = {
                'obs': [[] for _ in range(self.num_agents)],
                'next_obs': [[] for _ in range(self.num_agents)],
                'global_state': [[] for _ in range(self.num_agents)],
                'next_global_state': [[] for _ in range(self.num_agents)],
                'actions': [[] for _ in range(self.num_agents)],
                'rewards': [[] for _ in range(self.num_agents)],
                'dones': [[] for _ in range(self.num_agents)],
            }

        for i in range(self.num_agents):
            self.buffer['obs'][i].append(np.asarray(obs[i], dtype=np.float32))
            self.buffer['next_obs'][i].append(np.asarray(next_obs[i], dtype=np.float32))
            self.buffer['actions'][i].append(np.asarray(actions[i], dtype=np.float32))
            self.buffer['rewards'][i].append(float(rewards[i]))
            self.buffer['dones'][i].append(float(done))

            if getattr(self.network, 'use_centralized_critic', False):
                if global_states is None or next_global_states is None:
                    raise RuntimeError(
                        "CTDE enabled but global_states/next_global_states not provided."
                    )
                self.buffer['global_state'][i].append(np.asarray(global_states[i], dtype=np.float32))
                self.buffer['next_global_state'][i].append(np.asarray(next_global_states[i], dtype=np.float32))
        
        buffer_size = len(self.buffer['obs'][0])
        update_every = int(getattr(self, 'update_every', 256))
        if buffer_size >= update_every:
            self._batch_update_networks()
    
    def _batch_update_networks(self):
        """Batch update networks using stored transitions."""
        if not getattr(self.network, 'use_lstm', False):
            self.buffer = {
                'obs': [[] for _ in range(self.num_agents)],
                'next_obs': [[] for _ in range(self.num_agents)],
                'global_state': [[] for _ in range(self.num_agents)],
                'next_global_state': [[] for _ in range(self.num_agents)],
                'actions': [[] for _ in range(self.num_agents)],
                'rewards': [[] for _ in range(self.num_agents)],
                'dones': [[] for _ in range(self.num_agents)],
            }
            return

        gamma = 0.99
        unroll = int(getattr(self, 'unroll_len', 16))
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_steps = 0

        self.network.train()
        self.optimizer.zero_grad()

        for i in range(self.num_agents):
            T = len(self.buffer['obs'][i])
            if T == 0:
                continue

            obs_arr = np.asarray(self.buffer['obs'][i], dtype=np.float32)
            next_obs_arr = np.asarray(self.buffer['next_obs'][i], dtype=np.float32)
            act_arr = np.asarray(self.buffer['actions'][i], dtype=np.float32)
            rew_arr = np.asarray(self.buffer['rewards'][i], dtype=np.float32)
            done_arr = np.asarray(self.buffer['dones'][i], dtype=np.float32)

            use_ctde = bool(getattr(self.network, 'use_centralized_critic', False))
            global_arr = None
            next_global_arr = None
            if use_ctde:
                if ('global_state' not in self.buffer) or ('next_global_state' not in self.buffer):
                    raise RuntimeError("CTDE enabled but buffer missing global_state fields")
                global_arr = np.asarray(self.buffer['global_state'][i], dtype=np.float32)
                next_global_arr = np.asarray(self.buffer['next_global_state'][i], dtype=np.float32)
                if len(global_arr) != T or len(next_global_arr) != T:
                    raise RuntimeError("CTDE global_state length mismatch with obs trajectory")

            h = None
            for start in range(0, T, unroll):
                end = min(start + unroll, T)
                L = end - start
                if L <= 0:
                    continue

                obs_chunk = torch.from_numpy(obs_arr[start:end]).unsqueeze(0).to(self.device)
                act_chunk = torch.from_numpy(act_arr[start:end]).unsqueeze(0).to(self.device)
                rew_chunk = torch.from_numpy(rew_arr[start:end]).to(self.device)
                done_chunk = torch.from_numpy(done_arr[start:end]).to(self.device)

                mean, std, local_value, h_out = self.network(obs_chunk, h, return_sequence=True)
                
                if use_ctde:
                    if global_arr is None:
                        raise RuntimeError("CTDE enabled but global_state not available")
                    global_chunk = torch.from_numpy(global_arr[start:end]).to(self.device)
                    value = self.network.forward_value_global(global_chunk).view(-1)
                else:
                    value = local_value.squeeze(0).squeeze(-1)

                mean = mean.squeeze(0)
                std = std.squeeze(0)

                with torch.no_grad():
                    next_obs_last = torch.from_numpy(next_obs_arr[end - 1]).view(1, 1, -1).to(self.device)
                    if use_ctde:
                        if next_global_arr is None:
                            raise RuntimeError("CTDE enabled but next_global_state not available")
                        next_global_last = torch.from_numpy(next_global_arr[end - 1]).view(1, 1, -1).to(self.device)
                        v_next = self.network.forward_value_global(next_global_last.squeeze(0)).view(-1)[0]
                    else:
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
                logp = dist.log_prob(act_chunk.squeeze(0)).sum(dim=-1)

                policy_loss = -(logp * adv).mean()
                value_loss = torch.nn.functional.mse_loss(value, returns)

                aux_local_value_loss = None
                if use_ctde:
                    local_value_flat = local_value.squeeze(0).squeeze(-1)
                    aux_local_value_loss = torch.nn.functional.mse_loss(local_value_flat, returns)
                    total_loss = value_loss + 0.5 * policy_loss + 0.2 * aux_local_value_loss
                else:
                    total_loss = value_loss + 0.5 * policy_loss

                total_loss.backward()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_steps += L

                h = None
                if isinstance(h_out, tuple):
                    h = (h_out[0].detach(), h_out[1].detach())
                elif h_out is not None:
                    h = h_out.detach()

                if float(done_chunk.max().item()) > 0.0:
                    h = None

        if total_steps > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            self._weights_dirty = True
        
        self.buffer = {
            'obs': [[] for _ in range(self.num_agents)],
            'next_obs': [[] for _ in range(self.num_agents)],
            'global_state': [[] for _ in range(self.num_agents)],
            'next_global_state': [[] for _ in range(self.num_agents)],
            'actions': [[] for _ in range(self.num_agents)],
            'rewards': [[] for _ in range(self.num_agents)],
            'dones': [[] for _ in range(self.num_agents)],
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
            dummy_obs = torch.randn(1, s_len, effective_obs_dim).to(self.device)

            # Export policy/value network (policy_net_jit.pt)
            # Use torch.jit.script when possible to support TCN (and avoid trace issues with non-Tensor outputs).
            try:
                scripted_policy = torch.jit.script(self.network.cpu().eval())
                scripted_policy.save(os.path.join(os.path.dirname(path), 'policy_net_jit.pt'))
            finally:
                # Restore device
                self.network.to(self.device)

            # Prepare hidden_dim for wrapper tests
            hidden_dim = int(getattr(self.network, 'lstm_hidden_dim', 128))

            # Export MCTSInferWrapper for C++ MCTS
            # NOTE: Do NOT use torch.jit.freeze() or optimize_for_inference()
            # These optimizations can strip @torch.jit.export methods that C++ needs
            from .mcts import MCTSInferWrapper
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
    parser.add_argument('--config', type=str, default='MCTS_DUAL/train_config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Force load from YAML
    config = load_config_from_yaml(args.config)
    
    # Extract values from nested YAML config
    device_name = config.get('net', {}).get('device', 'cpu')
    if device_name == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_name
    
    print(f"[INFO] Training device: {device}")
    print(f"[INFO] Worker inference device: CPU (optimal for small networks)")

    seed = config.get('train', {}).get('seed')
    if seed is not None:
        set_global_seeds(int(seed), deterministic=config.get('train', {}).get('deterministic', False))
    
    trainer = MCTSTrainer(
        num_agents=int(config.get('env', {}).get('num_agents', 6)),
        num_lanes=int(config.get('env', {}).get('num_lanes', 2)),
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
        show_lane_ids=bool(config.get('render', {}).get('show_lane_ids', False)),
        show_lidar=bool(config.get('render', {}).get('show_lidar', False)),
        respawn_enabled=bool(config.get('env', {}).get('respawn_enabled', True)),
        save_dir=str(config.get('train', {}).get('save_dir', 'MCTS_DUAL/checkpoints')),
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
        final_checkpoint = os.path.join(trainer.save_dir, 'mcts_interrupted.pth')
        trainer.save_checkpoint(final_checkpoint, trainer.stats['episode'])
        print(f"Final checkpoint saved: {final_checkpoint}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        try:
            final_checkpoint = os.path.join(trainer.save_dir, 'mcts_error.pth')
            trainer.save_checkpoint(final_checkpoint, trainer.stats.get('episode', 0))
            print(f"Error checkpoint saved: {final_checkpoint}")
        except Exception:
            pass
    finally:
        try:
            trainer.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()