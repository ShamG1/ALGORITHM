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
        self._env_state_cache = None
        
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
        
        self.debug_info = {
            'simulations': [],
            'root_value': value.item(),
            'root_policy_mean': policy_mean.detach().cpu().numpy()[0] if isinstance(policy_mean, torch.Tensor) else policy_mean,
            'root_policy_std': policy_std.detach().cpu().numpy()[0] if isinstance(policy_std, torch.Tensor) else policy_std
        }
        
        self._env_state_cache = env_state
        
        # Handle pygame events periodically
        for sim_idx in range(self.num_simulations):
            if sim_idx % 5 == 0:
                try:
                    import pygame
                    if pygame.get_init():
                        pygame.event.pump()
                except:
                    pass
            
            sim_info = {'simulation': sim_idx + 1}
            try:
                self._simulate(root, obs_history, hidden_state, sim_info, env_state)
            except Exception as e:
                continue
            
            self.debug_info['simulations'].append(sim_info)
        
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
            'selected_action': best_action.tolist() if isinstance(best_action, np.ndarray) else best_action,
            'debug_info': getattr(self, 'debug_info', None),
            'rollout_stats': {
                'total_rollouts': self.rollout_stats['total_rollouts'],
                'successful_rollouts': self.rollout_stats['successful_rollouts'],
                'failed_rollouts': self.rollout_stats['failed_rollouts'],
                'total_env_steps': self.rollout_stats['total_env_steps'],
                'avg_rollout_reward': np.mean(self.rollout_stats['rollout_rewards']) if self.rollout_stats['rollout_rewards'] else 0.0,
                'avg_rollout_depth': np.mean(self.rollout_stats['rollout_depths']) if self.rollout_stats['rollout_depths'] else 0.0
            }
        }
        
        return best_action, search_stats
    
    def _simulate(self, node: MCTSNode, obs_history: Optional[List], hidden_state, debug_info=None, env_state=None):
        path = []
        current = node
        selected_actions = []
        
        while not current.is_leaf():
            action_key = self._select_action(current)
            path.append((current, action_key))
            selected_actions.append(np.array(action_key))
            
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
            sampled_actions = expand_result.get('sampled_actions', [])
            if debug_info is not None:
                debug_info['selected_actions'] = [a.tolist() for a in selected_actions]
                debug_info['sampled_actions'] = [a.tolist() for a in sampled_actions]
                debug_info['value'] = value
                debug_info['node_visit_count'] = current.visit_count
            self._backup(path + [(current, None)], value)
        else:
            if len(current.children) == 0:
                expand_result = self._expand_and_evaluate_with_rollout(
                    current, obs_history, hidden_state, debug_info, env_state, path
                )
                value = expand_result['value']
                sampled_actions = expand_result.get('sampled_actions', [])
            else:
                value = self._evaluate_with_rollout(current.state, obs_history, hidden_state, env_state, path)
                sampled_actions = []
            if debug_info is not None:
                debug_info['selected_actions'] = [a.tolist() for a in selected_actions]
                debug_info['sampled_actions'] = [a.tolist() for a in sampled_actions]
                debug_info['value'] = value
                debug_info['node_visit_count'] = current.visit_count
            self._backup(path + [(current, None)], value)
    
    def _select_action(self, node: MCTSNode) -> np.ndarray:
        if self.continuous_actions:
            obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, _, _ = self.network(obs_tensor)
            
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample().cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            action = np.asarray(action).flatten()
            return tuple(float(x) for x in action)
        else:
            puct_values = {}
            total_visits = sum(child.visit_count for child in node.children.values())
            
            for action, child in node.children.items():
                if child.visit_count == 0:
                    puct_values[action] = float('inf')
                else:
                    q_value = child.value
                    prior = child.prior_prob
                    puct_values[action] = q_value + self.c_puct * prior * math.sqrt(total_visits) / (1 + child.visit_count)
            
            return max(puct_values.items(), key=lambda x: x[1])[0]
    
    def _get_next_state_with_rollout(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        env_state: Optional[Dict],
        path_actions: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, bool]:
        if self.env_factory is None or env_state is None:
            if self.debug_rollout:
                print(f"[Agent {self.agent_id}] Rollout SKIPPED")
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

            # Restore snapshot
            try:
                step_in_state = int(env_state.get('step_count', 0))
            except Exception:
                step_in_state = 0
            setattr(env_copy, 'step_count', step_in_state)

            agent_states = env_state.get('agent_states', None)
            if agent_states is not None and hasattr(env_copy, 'agents'):
                for idx, s in enumerate(agent_states):
                    if idx >= len(env_copy.agents):
                        break
                    agent = env_copy.agents[idx]
                    try:
                        agent.pos_x = float(s.get('pos_x', agent.pos_x))
                        agent.pos_y = float(s.get('pos_y', agent.pos_y))
                        agent.heading = float(s.get('heading', agent.heading))
                        agent.speed = float(s.get('speed', getattr(agent, 'speed', 0.0)))
                        if hasattr(agent, 'rect'):
                            agent.rect.center = (int(agent.pos_x), int(agent.pos_y))
                    except Exception:
                        continue
            
            all_obs = env_state.get('obs', None)
            
            if hasattr(env_copy, 'agents') and len(env_copy.agents) > 0:
                num_agents = len(env_copy.agents)
                actions_all = []
                
                for i in range(num_agents):
                    if i == self.agent_id:
                        actions_all.append(action)
                    else:
                        # === OPTIMIZED: Light-weight Rule Based Opponent ===
                        # Replaces expensive plan_autonomous_action (120x loop)
                        
                        agent_obj = env_copy.agents[i]
                        simple_throttle = 0.0
                        simple_steer = 0.0
                        
                        # 1. Simple Distance Check
                        min_dist_sq = 10000.0 # 100^2
                        my_x, my_y = agent_obj.pos_x, agent_obj.pos_y
                        my_vx = math.cos(agent_obj.heading)
                        my_vy = -math.sin(agent_obj.heading)
                        
                        for other in env_copy.agents:
                            if other is agent_obj: continue
                            if not other.alive(): continue
                            
                            dx = other.pos_x - my_x
                            dy = other.pos_y - my_y
                            d2 = dx*dx + dy*dy
                            
                            if d2 < 3600: # 60m check
                                if dx * my_vx + dy * my_vy > 0: # In front
                                    if d2 < min_dist_sq:
                                        min_dist_sq = d2
                        
                        # Simple IDM logic
                        if min_dist_sq < 400: simple_throttle = -1.0
                        elif min_dist_sq < 1600: simple_throttle = -0.5
                        elif agent_obj.speed < 5.0: simple_throttle = 0.5
                        
                        actions_all.append(np.array([simple_throttle, simple_steer], dtype=np.float32))
                
                # Step environment
                next_obs_all, rewards, terminated, truncated, info = env_copy.step(np.array(actions_all))
                done = terminated or truncated
                self.rollout_stats['total_env_steps'] += 1
                
                next_state = next_obs_all[self.agent_id] if isinstance(next_obs_all, (list, np.ndarray)) else next_obs_all
                reward = rewards[self.agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards
                
                total_reward = reward
                current_done = done
                current_obs = next_state
                actual_rollout_depth = 1 
                
                # Rollout Loop
                for step in range(1, self.rollout_depth):
                    if current_done:
                        break
                    
                    actions_rollout = []
                    for i in range(num_agents):
                        if i == self.agent_id:
                            # Current agent uses policy
                            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                            policy_mean, policy_std, _, _ = self.network(obs_tensor)
                            dist = torch.distributions.Normal(policy_mean, policy_std)
                            rollout_action = dist.sample().cpu().numpy()[0]
                            rollout_action = np.clip(rollout_action, -1.0, 1.0)
                            actions_rollout.append(rollout_action)
                        else:
                            # === OPTIMIZED: Rollout Rule Based Opponent ===
                            if hasattr(env_copy, 'agents') and i < len(env_copy.agents):
                                agent_obj = env_copy.agents[i]
                                simple_throttle = 0.0
                                simple_steer = 0.0
                                min_dist_sq = 10000.0
                                my_x, my_y = agent_obj.pos_x, agent_obj.pos_y
                                my_vx = math.cos(agent_obj.heading)
                                my_vy = -math.sin(agent_obj.heading)
                                
                                for other in env_copy.agents:
                                    if other is agent_obj: continue
                                    if not other.alive(): continue
                                    dx = other.pos_x - my_x
                                    dy = other.pos_y - my_y
                                    d2 = dx*dx + dy*dy
                                    if d2 < 3600:
                                        if dx * my_vx + dy * my_vy > 0:
                                            if d2 < min_dist_sq: min_dist_sq = d2
                                
                                if min_dist_sq < 400: simple_throttle = -1.0
                                elif min_dist_sq < 1600: simple_throttle = -0.5
                                elif agent_obj.speed < 5.0: simple_throttle = 0.5
                                
                                actions_rollout.append(np.array([simple_throttle, simple_steer], dtype=np.float32))
                            else:
                                actions_rollout.append(np.zeros(2))
                    
                    next_obs_all, rewards, terminated, truncated, info = env_copy.step(np.array(actions_rollout))
                    current_done = terminated or truncated
                    self.rollout_stats['total_env_steps'] += 1
                    current_obs = next_obs_all[self.agent_id] if isinstance(next_obs_all, (list, np.ndarray)) else next_obs_all
                    step_reward = rewards[self.agent_id] if isinstance(rewards, (list, np.ndarray)) else rewards
                    total_reward += step_reward * (0.99 ** step)
                    actual_rollout_depth += 1
                
                # Final value estimate
                obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
                _, _, final_value, _ = self.network(obs_tensor)
                
                value = total_reward + (0.99 ** self.rollout_depth) * final_value.item() * (1 - current_done)
                
                self.rollout_stats['successful_rollouts'] += 1
                self.rollout_stats['rollout_rewards'].append(total_reward)
                self.rollout_stats['rollout_depths'].append(actual_rollout_depth)
                
                return current_obs, value, current_done
            else:
                next_obs, reward, terminated, truncated, _ = env_copy.step(action)
                done = terminated or truncated
                return next_obs, reward, done
                
        except Exception as e:
            self.rollout_stats['failed_rollouts'] += 1
            try:
                obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, _, value, _ = self.network(obs_tensor)
            except Exception as network_err:
                value = torch.tensor(0.0)
            return state.copy(), value.item(), False
    
    def _expand_and_evaluate_with_rollout(
        self, 
        node: MCTSNode, 
        obs_history: Optional[List], 
        hidden_state,
        debug_info=None,
        env_state=None,
        path_actions: List = None
    ):
        # Add debug logging
        import sys
        if self.debug_rollout:
            sys.stdout.write(f"[Agent {self.agent_id}] _expand_and_evaluate_with_rollout called\n")
            sys.stdout.flush()
        """
        Expand node and evaluate with real environment rollout.
        """
        # Get policy from network for action sampling
        if obs_history is not None and len(obs_history) > 0:
            seq_obs = np.array(obs_history[-self.network.sequence_length:])
            if len(seq_obs) < self.network.sequence_length:
                seq_obs = np.array([seq_obs[0]] * (self.network.sequence_length - len(seq_obs)) + list(seq_obs))
            seq_obs_tensor = torch.FloatTensor(seq_obs).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(seq_obs_tensor, hidden_state)
        else:
            obs_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
            policy_mean, policy_std, value, _ = self.network(obs_tensor, hidden_state)
        
        sampled_actions_list = []
        rollout_values = []
        
        if self.continuous_actions:
            # Sample actions from policy
            dist = torch.distributions.Normal(policy_mean, policy_std)
            sampled_actions = dist.sample((self.num_action_samples,)).cpu().numpy()
            sampled_actions = np.clip(sampled_actions, -1.0, 1.0)
            
            for action_idx, action in enumerate(sampled_actions):
                action = np.asarray(action).flatten()
                sampled_actions_list.append(action.copy())
                action_key = tuple(float(x) for x in action)
                
                if action_key not in node.children:
                    # Add debug logging - always log for first episode
                    import sys
                    # Performing rollout (output disabled)
                    
                    # Perform real rollout
                    try:
                        next_state, rollout_value, done = self._get_next_state_with_rollout(
                            node.state, action, env_state, path_actions if path_actions else []
                        )
                        rollout_values.append(rollout_value)
                        
                        # Rollout completed (output disabled)
                    except Exception as e:
                        # Silently continue on rollout errors (to reduce verbosity)
                        pass
                        # Use network value as fallback
                        rollout_values.append(value.item())
                    
                    child = MCTSNode(next_state, parent=node, action=action)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    log_prob = dist.log_prob(action_tensor).sum()
                    child.prior_prob = torch.exp(log_prob).item()
                    child.value_estimate = rollout_value
                    node.children[action_key] = child
        
        # Use average rollout value or network value
        final_value = np.mean(rollout_values) if rollout_values else value.item()
        
        return {
            'value': final_value,
            'sampled_actions': sampled_actions_list,
            'policy_mean': policy_mean.detach().cpu().numpy()[0] if isinstance(policy_mean, torch.Tensor) else policy_mean,
            'policy_std': policy_std.detach().cpu().numpy()[0] if isinstance(policy_std, torch.Tensor) else policy_std
        }
    
    def _evaluate_with_rollout(
        self, 
        state: np.ndarray, 
        obs_history: Optional[List], 
        hidden_state,
        env_state=None,
        path_actions: List = None
    ):
        """
        Evaluate state with real environment rollout.
        """
        if self.env_factory is not None and env_state is not None:
            # Perform rollout to get value estimate
            _, rollout_value, _ = self._get_next_state_with_rollout(
                state, 
                np.zeros(2),  # Dummy action for evaluation
                env_state, 
                path_actions if path_actions else []
            )
            return rollout_value
        else:
            # Fallback to network evaluation
            return self._evaluate(state, obs_history, hidden_state)
    
    def _backup(self, path: List[Tuple[MCTSNode, Optional[np.ndarray]]], value: float):
        """
        Backup value through path.
        
        Args:
            path: List of (node, action) tuples from root to leaf
            value: Value to backup
        """
        # Backup from leaf to root
        for node, _ in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            # Value decays with depth (optional)
            # value *= 0.99
    
    def _get_action_probs(self, node: MCTSNode) -> Dict:
        """
        Get action probabilities from visit counts.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        if len(node.children) == 0:
            return {}
        
        total_visits = sum(child.visit_count for child in node.children.values())
        if total_visits == 0:
            return {action: 1.0 / len(node.children) for action in node.children.keys()}
        
        action_probs = {}
        for action, child in node.children.items():
            action_probs[action] = child.visit_count / total_visits
        
        return action_probs
    
    def get_rollout_stats(self) -> Dict:
        """Get rollout statistics."""
        return {
            'total_rollouts': self.rollout_stats['total_rollouts'],
            'successful_rollouts': self.rollout_stats['successful_rollouts'],
            'failed_rollouts': self.rollout_stats['failed_rollouts'],
            'total_env_steps': self.rollout_stats['total_env_steps'],
            'success_rate': (self.rollout_stats['successful_rollouts'] / max(1, self.rollout_stats['total_rollouts'])) * 100,
            'avg_rollout_reward': np.mean(self.rollout_stats['rollout_rewards']) if self.rollout_stats['rollout_rewards'] else 0.0,
            'avg_rollout_depth': np.mean(self.rollout_stats['rollout_depths']) if self.rollout_stats['rollout_depths'] else 0.0,
            'env_steps_per_rollout': self.rollout_stats['total_env_steps'] / max(1, self.rollout_stats['total_rollouts'])
        }
    
    def reset_rollout_stats(self):
        """Reset rollout statistics."""
        self.rollout_stats = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'failed_rollouts': 0,
            'total_env_steps': 0,
            'rollout_rewards': [],
            'rollout_depths': []
        }
    
    def enable_debug(self, enable: bool = True):
        """Enable or disable debug output for rollouts."""
        self.debug_rollout = enable