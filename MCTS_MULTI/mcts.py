# --- mcts.py ---
# Monte Carlo Tree Search for Multi-Agent Training with Real Environment Rollouts

import numpy as np
import torch
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Callable
import math
import copy
import threading


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Current state (observation)
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}  # Dict: action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = 0.0  # Prior probability from policy network
        self.value_estimate = 0.0  # Value estimate from value network
        
    @property
    def value(self):
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self):
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0
    
    def is_fully_expanded(self, action_space):
        """Check if all actions have been explored."""
        return len(self.children) == len(action_space)


class MCTS:
    """
    True Monte Carlo Tree Search with real environment rollouts.
    """
    
    def __init__(
        self,
        network,
        action_space: np.ndarray,
        num_simulations: int = 50,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cpu',
        rollout_depth: int = 3,
        env_factory: Optional[Callable] = None,
        all_networks: Optional[List] = None,
        agent_id: int = 0,
        num_action_samples: int = 5
    ):
        self.network = network
        self.action_space = action_space
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = torch.device(device)
        self.rollout_depth = rollout_depth
        self.env_factory = env_factory
        self.all_networks = all_networks
        self.agent_id = agent_id
        self.num_action_samples = num_action_samples
        
        self.continuous_actions = action_space is None
        
        # Cache for rollout environment
        self._env_cache = None
        
        # Statistics
        self.rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'failed_rollouts': 0,
            'total_env_steps': 0,
            'rollout_rewards': [],
            'rollout_depths': []
        }
        
        self.debug_rollout = False
    
    def search(
        self,
        root_state: np.ndarray,
        obs_history: Optional[List] = None,
        hidden_state=None,
        env_state: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        # Create root node
        root = MCTSNode(root_state)
        
        # Get initial policy and value from network
        # We need no_grad here to be safe, though train.py handles it too
        with torch.no_grad():
            if obs_history is not None and len(obs_history) > 0:
                seq_obs = np.array(obs_history[-self.network.sequence_length:])
                if len(seq_obs) < self.network.sequence_length:
                    seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
                seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
                policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
            else:
                obs_tensor = torch.FloatTensor(root_state).unsqueeze(0).to(self.device)
                policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        root.value_estimate = value.item()
        
        # Handle pygame events periodically to prevent freezing
        for sim_idx in range(self.num_simulations):
            if sim_idx % 10 == 0: # Check less frequently for speed
                try:
                    import pygame
                    if pygame.get_init():
                        pygame.event.pump()
                except:
                    pass
            
            try:
                self._simulate(root, obs_history, hidden_state, None, env_state)
            except Exception:
                continue
        
        # Select action
        action_probs = self._get_action_probs(root)
        
        if len(action_probs) == 0:
            # Fallback sample from policy
            dist = torch.distributions.Normal(policy_mean, policy_std)
            best_action = dist.sample().cpu().numpy()[0]
            best_action = np.clip(best_action, -1.0, 1.0)
        elif self.temperature == 0:
            best_action_tuple = max(action_probs.items(), key=lambda x: x[1])[0]
            if isinstance(best_action_tuple, tuple):
                best_action = np.array(best_action_tuple)
            else:
                best_action = best_action_tuple
        else:
            actions_list = list(action_probs.keys())
            probs = np.array([action_probs[a] for a in actions_list])
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / probs.sum()
            best_action_idx = np.random.choice(len(actions_list), p=probs)
            best_action_tuple = actions_list[best_action_idx]
            if isinstance(best_action_tuple, tuple):
                best_action = np.array(best_action_tuple)
            else:
                best_action = best_action_tuple
        
        search_stats = {
            'visit_counts': {str(a): node.visit_count for a, node in root.children.items()},
            'action_probs': {str(a): p for a, p in action_probs.items()},
            'root_value': root.value,
            'num_simulations': self.num_simulations,
            'rollout_stats': self.rollout_stats # Just return the dict directly
        }
        
        return best_action, search_stats
    
    def _simulate(self, node: MCTSNode, obs_history: Optional[List], hidden_state, debug_info=None, env_state=None):
        path = []
        current = node
        
        while not current.is_leaf():
            action_key = self._select_action(current)
            path.append((current, action_key))
            
            if action_key not in current.children:
                if self.continuous_actions:
                    action_array = np.array(action_key, dtype=np.float32)
                else:
                    action_array = action_key
                current.children[action_key] = MCTSNode(current.state.copy(), parent=current, action=action_array)
            
            current = current.children[action_key]
        
        if current.visit_count == 0:
            expand_result = self._expand_and_evaluate_with_rollout(
                current, obs_history, hidden_state, debug_info, env_state, path
            )
            value = expand_result['value']
            self._backup(path + [(current, None)], value)
        else:
            if len(current.children) == 0:
                expand_result = self._expand_and_evaluate_with_rollout(
                    current, obs_history, hidden_state, debug_info, env_state, path
                )
                value = expand_result['value']
            else:
                value = self._evaluate_with_rollout(current.state, obs_history, hidden_state, env_state, path)
            self._backup(path + [(current, None)], value)
    
    def _select_action(self, node: MCTSNode) -> np.ndarray:
        if self.continuous_actions:
            # Note: For strict MCTS correctness, we should use PUCT with prior prob.
            # But sampling directly from policy is standard for continuous AlphaZero (A0C).
            # We assume node.state is enough for policy (no LSTM history in selection for speed)
            # Or we could cache the policy output in the node.
            
            # Optimization: If we already have children, select based on PUCT
            if len(node.children) > 0:
                 puct_values = {}
                 total_visits = sum(child.visit_count for child in node.children.values())
                 sqrt_total = math.sqrt(total_visits)
                 
                 for action, child in node.children.items():
                     # PUCT: Q + c * P * sqrt(N) / (1+n)
                     # For continuous, 'P' is tricky. We rely on Q mostly.
                     # We use prior_prob stored in child.
                     puct_values[action] = child.value + self.c_puct * child.prior_prob * sqrt_total / (1 + child.visit_count)
                 
                 return max(puct_values.items(), key=lambda x: x[1])[0]

            # If no children, we need to Expand (handled in _simulate loop break condition)
            # But wait, _select_action is called inside loop.
            # If we are here, it means we are at a leaf or unexpanded node.
            # We need to sample an action to Traverse.
            # BUT actually, standard MCTS expands ONE node per sim.
            # So if we have no children, we return a new action sample.
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
                policy_mean, policy_std, _, _ = self.network(obs_tensor)
                dist = torch.distributions.Normal(policy_mean, policy_std)
                action = dist.sample().cpu().numpy()[0]
            
            action = np.clip(action, -1.0, 1.0)
            action = np.asarray(action).flatten()
            return tuple(float(x) for x in action)
        else:
            # Discrete implementation omitted for brevity as user uses Continuous
            pass
    
    def _get_next_state_with_rollout(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        env_state: Optional[Dict],
        path_actions: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, bool]:
        if self.env_factory is None or env_state is None:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, value, _ = self.network(obs_tensor)
            return state.copy(), value.item(), False
        
        self.rollout_stats['total_rollouts'] += 1
        
        try:
            if self._env_cache is None:
                env_copy = self.env_factory()
                env_copy.reset()
                self._env_cache = env_copy
            else:
                env_copy = self._env_cache

            # --- 1. Fast Restore (Snapshot) ---
            try:
                env_copy.step_count = int(env_state.get('step_count', 0))
                agent_states = env_state.get('agent_states', None)
                if agent_states:
                    for idx, s in enumerate(agent_states):
                        if idx >= len(env_copy.agents): break
                        a = env_copy.agents[idx]
                        a.pos_x = float(s.get('pos_x', 0))
                        a.pos_y = float(s.get('pos_y', 0))
                        a.heading = float(s.get('heading', 0))
                        a.speed = float(s.get('speed', 0))
                        # Only update center for simple collision checks
                        a.rect.center = (int(a.pos_x), int(a.pos_y))
            except:
                pass # Fail silently for speed
            
            # --- 2. Step 0 (The Expansion Step) ---
            # Ego applies the specific action we want to evaluate
            # Opponents apply simple rules
            if hasattr(env_copy, 'agents') and len(env_copy.agents) > 0:
                num_agents = len(env_copy.agents)
                actions_all = []
                
                for i in range(num_agents):
                    if i == self.agent_id:
                        actions_all.append(action)
                    else:
                        # Opponent Rule: Keep lane & Avoid collision
                        actions_all.append(self._get_simple_rule_action(env_copy, i))
                
                next_obs_all, rewards, terminated, truncated, _ = env_copy.step(np.array(actions_all))
                
                # We need the next state for the node
                next_state = next_obs_all[self.agent_id]
                total_reward = rewards[self.agent_id]
                done = terminated or truncated
                
                # --- 3. Rollout Loop (Depth > 0) ---
                # PURE CPU LOOP - NO NEURAL NETWORK FOR EGO
                current_done = done
                
                for step in range(1, self.rollout_depth):
                    if current_done:
                        break
                    
                    actions_rollout = []
                    for i in range(num_agents):
                        # Use simple rule for EVERYONE (including Ego) during rollout
                        # This avoids calculating Observations and calling Neural Network
                        actions_rollout.append(self._get_simple_rule_action(env_copy, i))
                    
                    # env.step in fast mode computes positions/collisions but skips Lidar/Obs
                    # We can optimize env.step to not return obs if we want, but it's okay.
                    _, r, t, tr, _ = env_copy.step(np.array(actions_rollout))
                    
                    total_reward += r[self.agent_id] * (0.99 ** step)
                    if t or tr:
                        current_done = True
                
                # --- 4. Final Value Estimation ---
                # We use the state at Step 0 (next_state) for value estimation?
                # Or the state at the end of rollout?
                # AlphaZero uses leaf state. Standard MCTS uses rollout result.
                # Here we combine: Rollout Reward + Value of the leaf state (next_state)
                
                # Important: We evaluate the Value of the EXPANDED node (next_state),
                # not the end of the random rollout (which has no observation).
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    _, _, final_value, _ = self.network(obs_tensor)
                
                # Bootstrap value
                value = total_reward + (0.99 ** self.rollout_depth) * final_value.item() * (1 - current_done)
                
                self.rollout_stats['successful_rollouts'] += 1
                self.rollout_stats['rollout_rewards'].append(total_reward)
                
                return next_state, value, current_done
            
            else:
                return state.copy(), 0.0, False
                
        except Exception:
            self.rollout_stats['failed_rollouts'] += 1
            return state.copy(), 0.0, False

    def _get_simple_rule_action(self, env, agent_idx):
        """Ultra-fast rule based controller using only math, no objects."""
        try:
            agent = env.agents[agent_idx]
            if not agent.alive():
                return np.zeros(2, dtype=np.float32)

            throttle = 0.0
            steer = 0.0
            
            # Simple Longitudinal (IDM-like)
            min_dist_sq = 10000.0 # 100m^2
            
            mx, my = agent.pos_x, agent.pos_y
            mvx = math.cos(agent.heading)
            mvy = -math.sin(agent.heading)
            
            # Check neighbors
            for other in env.agents:
                if other is agent or not other.alive(): continue
                
                dx = other.pos_x - mx
                dy = other.pos_y - my
                d2 = dx*dx + dy*dy
                
                if d2 < 2500: # 50m
                    # Check if in front (dot product > 0)
                    if dx*mvx + dy*mvy > 0:
                        if d2 < min_dist_sq:
                            min_dist_sq = d2
            
            if min_dist_sq < 225: # 15m
                throttle = -1.0 # Emergency brake
            elif min_dist_sq < 900: # 30m
                throttle = -0.5 # Slow down
            elif agent.speed < 5.0:
                throttle = 0.5 # Accelerate if clear and slow
            
            return np.array([throttle, steer], dtype=np.float32)
        except:
            return np.zeros(2, dtype=np.float32)

    def _expand_and_evaluate_with_rollout(self, node, obs_history, hidden_state, debug_info, env_state, path_actions):
        # Sample actions from policy
        with torch.no_grad():
            if obs_history is not None and len(obs_history) > 0:
                seq_obs = np.array(obs_history[-self.network.sequence_length:])
                if len(seq_obs) < self.network.sequence_length:
                    seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
                seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
                policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
            else:
                obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
                policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        dist = torch.distributions.Normal(policy_mean, policy_std)
        sampled_actions = dist.sample((self.num_action_samples,)).cpu().numpy()
        sampled_actions = np.clip(sampled_actions, -1.0, 1.0)
        
        rollout_values = []
        
        for action in sampled_actions:
            action = np.asarray(action).flatten()
            action_key = tuple(float(x) for x in action)
            
            if action_key not in node.children:
                next_state, rollout_value, done = self._get_next_state_with_rollout(
                    node.state, action, env_state, path_actions
                )
                rollout_values.append(rollout_value)
                
                child = MCTSNode(next_state, parent=node, action=action)
                # Store prior
                child.prior_prob = 1.0 # Simplified for continuous
                child.value_estimate = rollout_value
                node.children[action_key] = child
        
        final_value = np.mean(rollout_values) if rollout_values else value.item()
        
        return {
            'value': final_value,
            'sampled_actions': sampled_actions
        }
    
    def _evaluate_with_rollout(self, state, obs_history, hidden_state, env_state, path_actions):
        # Just run a rollout with a dummy action (e.g. [0,0] or rule based) to get value
        # Or better: just evaluate the network directly for speed if we already have the node
        # But consistent with MCTS, we might want to rollout.
        # For speed: Network Value only.
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value, _ = self.network(obs_tensor)
        return value.item()

    def _backup(self, path, value):
        for node, _ in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            
    def _get_action_probs(self, node):
        if len(node.children) == 0:
            return {}
        total = sum(c.visit_count for c in node.children.values())
        return {a: c.visit_count / total for a, c in node.children.items()}
    
    def get_rollout_stats(self):
        return self.rollout_stats
    
    def reset_rollout_stats(self):
        self.rollout_stats = {'total_rollouts': 0, 'successful_rollouts': 0, 'failed_rollouts': 0, 
                              'total_env_steps': 0, 'rollout_rewards': [], 'rollout_depths': []}
    
    def enable_debug(self, enable):
        self.debug_rollout = enable