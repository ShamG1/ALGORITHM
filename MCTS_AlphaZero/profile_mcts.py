# profile_mcts.py
# Run this to identify remaining bottlenecks
# Usage: python MCTS_AlphaZero/profile_mcts.py

import os
import sys
import time
import numpy as np
from typing import List, Optional

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt_stats(name: str, values: List[float]) -> str:
    if not values:
        return f"{name:18s}: (no data)"
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return (
        f"{name:18s}: mean={arr.mean():8.2f}ms  std={arr.std():7.2f}ms  "
        f"min={arr.min():7.2f}ms  max={arr.max():7.2f}ms"
    )


def profile_single_episode(
    num_steps: int = 30,
    num_agents: int = 6,
    scenario_name: str = "roundabout_3lane",
    simulations: int = 100,
    rollout_depth: int = 25,
    num_action_samples: int = 16,
    use_tcn: bool = True,
    use_lstm: bool = False,
):
    """Profile a short run with production-like MCTS settings.

    This profiler is designed for your current SHM+Event architecture.
    It breaks down each step into:
      - SHM write obs
      - signal+wait (dominant: worker MCTS work)
      - SHM read actions/stats
      - env.step
      - update/buffer bookkeeping
      - overhead (rest)
    """

    import torch
    from MCTS_AlphaZero.train import MCTSTrainer

    print("=" * 70)
    print("MCTS AlphaZero Profiler (SHM/Event)")
    print("=" * 70)

    trainer = MCTSTrainer(
        num_agents=num_agents,
        scenario_name=scenario_name,
        max_episodes=1,
        max_steps_per_episode=num_steps,
        mcts_simulations=simulations,
        rollout_depth=rollout_depth,
        num_action_samples=num_action_samples,
        device="cpu",
        use_tcn=use_tcn,
        use_lstm=use_lstm,
        render=False,
        save_dir="MCTS_AlphaZero/checkpoints",
        parallel_mcts=True,
        use_shm=True,
        use_tqdm=False,
    )

    # Ensure TS exists (so profiling reflects the intended fast path)
    trainer._ensure_torchscript_model()

    obs, info = trainer.env.reset()
    if trainer.use_lstm or trainer.use_tcn:
        for i in range(trainer.num_agents):
            trainer.obs_history[i].clear()
            trainer.obs_history[i].append(obs[i])

    timings = {
        "total_step": [],
        "shm_write_obs": [],
        "shm_signal_wait": [],
        "shm_read_results": [],
        "env_step": [],
        "update_buf": [],
        "overhead": [],
    }

    print(
        f"Config: agents={num_agents} scenario={scenario_name} steps={num_steps} | "
        f"sims={simulations} depth={rollout_depth} K={num_action_samples} | "
        f"mode={'TCN' if use_tcn else 'LSTM' if use_lstm else 'MLP'}"
    )
    print("-" * 70)

    done = False
    step = 0

    while (not done) and step < num_steps:
        t0 = time.perf_counter()

        # keep history consistent with train loop
        if (trainer.use_lstm or trainer.use_tcn) and step > 0:
            for i in range(trainer.num_agents):
                trainer.obs_history[i].append(obs[i])

        # --- inline _parallel_mcts_search_shm for finer timing ---
        if trainer._pinned_procs is None:
            trainer._start_pinned_workers_shm()

        if step == 0:
            trainer._broadcast_weights_if_dirty()
            cfg = {
                "hidden_dim": 256,
                "lstm_hidden_dim": trainer.lstm_hidden_dim,
                "use_lstm": trainer.use_lstm,
                "use_tcn": trainer.use_tcn,
                "sequence_length": 5,
                "device": str(trainer.device),
                "num_simulations": trainer.mcts_simulations,
                "c_puct": 1.0,
                "temperature": 1.0,
                "rollout_depth": trainer.rollout_depth,
                "num_action_samples": trainer.num_action_samples,
                "base_seed": int(trainer.seed) if trainer.seed is not None else 0,
                "episode": 1,
                "step": int(step),
                "ts_model_path": os.path.join(trainer.save_dir, "mcts_infer.pt"),
            }
            for q in trainer._control_queues:
                q.put(("RESET",))
                q.put(("CONFIG", cfg))

        # write obs
        t_w0 = time.perf_counter()
        obs_array_raw = np.asarray(obs, dtype=np.float32)
        trainer._shm_buffer.write_all_observations(obs_array_raw)
        t_w1 = time.perf_counter()

        # signal + wait (dominant)
        t_sw0 = time.perf_counter()
        trainer._shm_buffer.signal_all_ready()
        ok = trainer._shm_buffer.wait_all_done(timeout=60.0)
        t_sw1 = time.perf_counter()
        if not ok:
            print(f"[WARN] wait_all_done timeout at step={step}")

        # read results
        t_r0 = time.perf_counter()
        actions = trainer._shm_buffer.read_all_actions()
        all_search_stats = trainer._shm_buffer.read_all_stats()
        t_r1 = time.perf_counter()

        # env step
        t_e0 = time.perf_counter()
        next_obs, rewards, terminated, truncated, info = trainer.env.step(actions)
        t_e1 = time.perf_counter()

        # update/buffer
        t_u0 = time.perf_counter()
        trainer._update_networks(
            obs,
            actions,
            rewards,
            next_obs,
            terminated or truncated,
            search_stats=all_search_stats,
        )
        t_u1 = time.perf_counter()

        t1 = time.perf_counter()

        timings["shm_write_obs"].append(t_w1 - t_w0)
        timings["shm_signal_wait"].append(t_sw1 - t_sw0)
        timings["shm_read_results"].append(t_r1 - t_r0)
        timings["env_step"].append(t_e1 - t_e0)
        timings["update_buf"].append(t_u1 - t_u0)
        timings["total_step"].append(t1 - t0)

        known = (t_w1 - t_w0) + (t_sw1 - t_sw0) + (t_r1 - t_r0) + (t_e1 - t_e0) + (t_u1 - t_u0)
        timings["overhead"].append((t1 - t0) - known)

        obs = next_obs
        done = terminated or truncated
        step += 1

        if step <= 5 or step % 10 == 0:
            print(
                f"Step {step:3d}: total={timings['total_step'][-1]*1000:7.1f}ms | "
                f"wait(MCTS)={timings['shm_signal_wait'][-1]*1000:7.1f}ms | "
                f"env={timings['env_step'][-1]*1000:6.1f}ms"
            )

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    for k in [
        "shm_write_obs",
        "shm_signal_wait",
        "shm_read_results",
        "env_step",
        "update_buf",
        "overhead",
        "total_step",
    ]:
        print(_fmt_stats(k, timings[k]))

    total_avg_s = float(np.mean(timings["total_step"])) if timings["total_step"] else 0.0
    if total_avg_s > 0:
        est_500 = total_avg_s * 500
        print(f"\nEstimated 500-step episode time: {est_500:.2f}s")

        mcts_avg = float(np.mean(timings["shm_signal_wait"]))
        print("\nNotes:")
        print(
            "- 'shm_signal_wait' is the dominant worker-side time: "
            "C++ MCTS + TorchScript inference + any Python-side serialization inside workers (e.g. .tolist())."
        )
        print(
            "- If 'shm_signal_wait' dominates, the next engineering win is usually removing numpy->list conversions "
            "in the C++ bindings and accepting raw arrays/buffers." 
        )

    trainer.close()
    print("\nDone.")


if __name__ == "__main__":
    profile_single_episode()
