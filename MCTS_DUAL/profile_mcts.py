# profile_mcts.py
# Run this to identify remaining bottlenecks
# Usage: python MCTS_DUAL/profile_mcts.py

import os
import sys
import time
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def profile_single_episode():
    """Profile a single episode to identify bottlenecks."""
    
    from MCTS_DUAL.train import MCTSTrainer
    import torch
    
    print("=" * 60)
    print("MCTS Performance Profiler")
    print("=" * 60)
    
    # Create trainer with minimal settings for profiling
    trainer = MCTSTrainer(
        num_agents=6,
        num_lanes=2,
        max_episodes=1,
        max_steps_per_episode=50,  # Short episode for profiling
        mcts_simulations=5,
        rollout_depth=5,
        num_action_samples=3,
        device='cpu',
        use_lstm=True,
        render=False,
        save_dir='MCTS_DUAL/checkpoints',
        parallel_mcts=True,
        use_shm=True,
        use_tqdm=False
    )
    
    # Reset environment
    obs, info = trainer.env.reset()
    
    # Initialize obs history
    if trainer.use_lstm:
        for i in range(trainer.num_agents):
            trainer.obs_history[i].clear()
            trainer.obs_history[i].append(obs[i])
    
    # Timing breakdown
    timings = {
        'mcts_search': [],
        'env_step': [],
        'total_step': [],
    }
    
    print(f"\nRunning {trainer.max_steps_per_episode} steps with {trainer.num_agents} agents...")
    print(f"MCTS simulations: {trainer.mcts_simulations}")
    print(f"Rollout depth: {trainer.rollout_depth}")
    print(f"Action samples: {trainer.num_action_samples}")
    print()
    
    done = False
    step = 0
    
    while not done and step < trainer.max_steps_per_episode:
        step_start = time.perf_counter()
        
        # Update obs history
        if trainer.use_lstm and step > 0:
            for i in range(trainer.num_agents):
                trainer.obs_history[i].append(obs[i])
        
        # MCTS search
        mcts_start = time.perf_counter()
        if trainer.use_shm:
            actions = trainer._parallel_mcts_search_shm(obs, None, episode=1, step=step)
        else:
            actions = trainer._parallel_mcts_search(obs, None, episode=1, step=step)
        mcts_end = time.perf_counter()
        timings['mcts_search'].append(mcts_end - mcts_start)
        
        # Environment step
        env_start = time.perf_counter()
        next_obs, rewards, terminated, truncated, info = trainer.env.step(actions)
        env_end = time.perf_counter()
        timings['env_step'].append(env_end - env_start)
        
        step_end = time.perf_counter()
        timings['total_step'].append(step_end - step_start)
        
        done = terminated or truncated
        obs = next_obs
        step += 1
        
        if step <= 5 or step % 10 == 0:
            print(f"Step {step:3d}: MCTS={timings['mcts_search'][-1]*1000:6.1f}ms, "
                  f"Env={timings['env_step'][-1]*1000:5.1f}ms, "
                  f"Total={timings['total_step'][-1]*1000:6.1f}ms")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    
    for name, values in timings.items():
        if values:
            arr = np.array(values) * 1000  # Convert to ms
            print(f"{name:15s}: mean={arr.mean():7.2f}ms, "
                  f"std={arr.std():6.2f}ms, "
                  f"min={arr.min():6.2f}ms, "
                  f"max={arr.max():6.2f}ms")
    
    # Estimate episode time
    total_time = sum(timings['total_step'])
    steps_per_episode = 500  # Typical episode length
    estimated_episode_time = (total_time / step) * steps_per_episode
    
    print(f"\nActual steps completed: {step}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Estimated time for {steps_per_episode} steps: {estimated_episode_time:.1f}s")
    
    # Breakdown analysis
    mcts_total = sum(timings['mcts_search'])
    env_total = sum(timings['env_step'])
    other_total = total_time - mcts_total - env_total
    
    print(f"\nTime breakdown:")
    print(f"  MCTS search: {mcts_total:.2f}s ({100*mcts_total/total_time:.1f}%)")
    print(f"  Env step:    {env_total:.2f}s ({100*env_total/total_time:.1f}%)")
    print(f"  Other:       {other_total:.2f}s ({100*other_total/total_time:.1f}%)")
    
    # Optimization suggestions
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUGGESTIONS")
    print("=" * 60)
    
    mcts_pct = 100 * mcts_total / total_time
    env_pct = 100 * env_total / total_time
    
    if mcts_pct > 80:
        print("→ MCTS is the main bottleneck. Consider:")
        print("  - Reduce mcts_simulations (current: 5)")
        print("  - Reduce rollout_depth (current: 5)")
        print("  - Reduce num_action_samples (current: 3)")
        print("  - Verify TorchScript model is being used (check for warnings)")
    elif env_pct > 50:
        print("→ Environment step is significant. Consider:")
        print("  - Profile C++ environment code")
        print("  - Check for unnecessary computations in step()")
    else:
        print("→ Time is distributed. Main areas to optimize:")
        if mcts_pct > 50:
            print("  - MCTS search (try reducing simulations)")
        if env_pct > 20:
            print("  - Environment step")
        print("  - IPC/synchronization overhead")
    
    # Check TorchScript status
    ts_path = os.path.join(trainer.save_dir, 'mcts_infer.pt')
    if os.path.exists(ts_path):
        print(f"\n✓ TorchScript model exists at {ts_path}")
        # Try to load and verify
        try:
            model = torch.jit.load(ts_path)
            methods = [m for m in dir(model) if not m.startswith('_')]
            if 'infer_policy_value' in methods:
                print("✓ TorchScript model has correct format (infer_policy_value found)")
            else:
                print("✗ TorchScript model missing infer_policy_value method!")
                print(f"  Available methods: {methods}")
        except Exception as e:
            print(f"✗ Failed to load TorchScript model: {e}")
    else:
        print(f"\n✗ TorchScript model NOT found at {ts_path}")
        print("  This means Python callbacks are being used (slower)")
    
    trainer.close()
    print("\nDone.")


if __name__ == '__main__':
    profile_single_episode()
