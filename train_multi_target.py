"""
train_multi_target.py — Full multi-objective training pipeline.

Implements:
- Adaptive Weight Scheduling (Phase A → B → C)
- Multi-target training across Eterna100 subset
- DQN vs PPO comparison
- Multi-seed experiments
- Per-objective TensorBoard logging
- Rich terminal progress output

Usage:
    python train_multi_target.py --algo ppo --seed 42
    python train_multi_target.py --algo dqn --seed 42
"""

import argparse
import os
import time
import sys
import numpy as np
from collections import deque
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment import LearnaEnv
from eterna100 import get_train_structures


# ======================================================================
# Adaptive Weight Scheduler (Proposal Section 3.3)
# ======================================================================
class AdaptiveWeightScheduler:
    """
    3 Aşamalı (Phase A, B, C) öğrenme planlayıcısı.
    Agent'in önce yapısal doğruluğu, ardından GC oranını öğrenmesini sağlar.
    """

    # Phase A starting fractions of target weights.
    # β kept high (0.7) so a perfect-structure / saturated-GC solution
    # (r_struct=1, r_gc=0) still loses ~0.14 reward to a balanced one,
    # forcing the agent off the all-GC attractor before policy entropy
    # collapses. Lower fractions (0.3) let short puzzles like #8 lock in.
    PHASE_A_BETA_FRAC = 0.7
    PHASE_A_GAMMA_FRAC = 1.0
    PHASE_A_DELTA_FRAC = 0.0

    def __init__(
        self,
        total_timesteps: int,
        target_alpha=0.5,
        target_beta=0.2,
        target_gamma=0.1,
        target_delta=0.2,
    ):
        self.total = total_timesteps
        self.target_alpha = target_alpha
        self.target_beta = target_beta
        self.target_gamma = target_gamma
        self.target_delta = target_delta

        # Phase boundaries
        self.phase_a_end = int(0.30 * total_timesteps)
        self.phase_b_end = int(0.70 * total_timesteps)

        # Phase A fixed weights (used as ramp start in Phase B)
        self._a_alpha = 1.0
        self._a_beta = self.PHASE_A_BETA_FRAC * target_beta
        self._a_gamma = self.PHASE_A_GAMMA_FRAC * target_gamma
        self._a_delta = self.PHASE_A_DELTA_FRAC * target_delta

    def get_weights(self, current_step: int):
        if current_step <= self.phase_a_end:
            return self._a_alpha, self._a_beta, self._a_gamma, self._a_delta
        elif current_step <= self.phase_b_end:
            progress = (current_step - self.phase_a_end) / (
                self.phase_b_end - self.phase_a_end
            )
            alpha = self._a_alpha + progress * (self.target_alpha - self._a_alpha)
            beta = self._a_beta + progress * (self.target_beta - self._a_beta)
            gamma = self._a_gamma + progress * (self.target_gamma - self._a_gamma)
            delta = self._a_delta + progress * (self.target_delta - self._a_delta)
            return alpha, beta, gamma, delta
        else:
            return (
                self.target_alpha,
                self.target_beta,
                self.target_gamma,
                self.target_delta,
            )

    def get_phase_name(self, current_step: int) -> str:
        if current_step <= self.phase_a_end:
            return "A (struct)"
        elif current_step <= self.phase_b_end:
            return "B (ramp)"
        else:
            return "C (joint)"


# ======================================================================
# Callback: Weight updater + per-objective logger + terminal output
# ======================================================================
class TrainingCallback(BaseCallback):
    """Updates environment weights per-step, logs metrics, and prints progress."""

    def __init__(
        self,
        scheduler: AdaptiveWeightScheduler,
        target_name: str,
        total_timesteps: int,
        print_interval: int = 2000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.target_name = target_name
        self.total_timesteps = total_timesteps
        self.print_interval = print_interval
        self._episode_count = 0
        self._start_time = None
        self._last_print_step = 0

        # Rolling window for live metrics
        self._recent_r_struct = deque(maxlen=50)
        self._recent_r_gc = deque(maxlen=50)
        self._recent_success = deque(maxlen=50)
        self._recent_reward = deque(maxlen=50)
        self._best_r_struct = 0.0

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Update environment weights
        alpha, beta, gamma, delta = self.scheduler.get_weights(self.num_timesteps)
        env = self.training_env.envs[0]
        env.alpha = alpha
        env.beta = beta
        env.gamma = gamma
        env.delta = delta

        # Log weights to TensorBoard
        self.logger.record("weights/alpha", alpha)
        self.logger.record("weights/beta", beta)
        self.logger.record("weights/gamma", gamma)
        self.logger.record("weights/delta", delta)

        # Log per-objective metrics from terminal info
        infos = self.locals.get("infos", [])
        for info in infos:
            terminal = info.get("terminal_info", info)
            if "r_struct" in terminal:
                self._episode_count += 1
                r_s = terminal["r_struct"]
                r_gc = terminal["r_gc"]
                is_succ = terminal["is_success"]

                self._recent_r_struct.append(r_s)
                self._recent_r_gc.append(r_gc)
                self._recent_success.append(is_succ)
                self._recent_reward.append(
                    terminal["r_struct"] * alpha
                    + terminal["r_gc"] * beta
                    - terminal["p_homo"] * gamma
                    + terminal["r_mfe"] * delta
                )
                self._best_r_struct = max(self._best_r_struct, r_s)

                # TensorBoard
                self.logger.record("episode/r_struct", r_s)
                self.logger.record("episode/r_gc", r_gc)
                self.logger.record("episode/p_homo", terminal["p_homo"])
                self.logger.record("episode/r_mfe", terminal["r_mfe"])
                self.logger.record("episode/gc_ratio", terminal.get("gc_ratio", 0.0))
                self.logger.record("episode/is_success", is_succ)
                self.logger.record("episode/count", self._episode_count)

        # Print progress at intervals
        if self.num_timesteps - self._last_print_step >= self.print_interval:
            self._print_progress()
            self._last_print_step = self.num_timesteps

        return True

    def _print_progress(self):
        elapsed = time.time() - self._start_time if self._start_time else 0
        pct = 100.0 * self.num_timesteps / self.total_timesteps
        phase = self.scheduler.get_phase_name(self.num_timesteps)

        # ETA calculation
        if self.num_timesteps > 0:
            fps = self.num_timesteps / max(elapsed, 0.01)
            remaining = (self.total_timesteps - self.num_timesteps) / fps
            eta_str = f"{remaining:.0f}s"
        else:
            eta_str = "?"

        # Rolling metrics
        avg_struct = np.mean(self._recent_r_struct) if self._recent_r_struct else 0.0
        avg_gc = np.mean(self._recent_r_gc) if self._recent_r_gc else 0.0
        succ_rate = np.mean(self._recent_success) if self._recent_success else 0.0
        avg_rew = np.mean(self._recent_reward) if self._recent_reward else 0.0

        # Build progress bar
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"    [{bar}] {pct:5.1f}% | Phase {phase} | "
            f"R_struct={avg_struct:.3f} (best={self._best_r_struct:.3f}) | "
            f"R_gc={avg_gc:.3f} | Succ={succ_rate:.1%} | "
            f"Reward={avg_rew:.3f} | "
            f"ETA={eta_str} | Ep={self._episode_count}",
            flush=True,
        )

    def _on_training_end(self):
        elapsed = time.time() - self._start_time if self._start_time else 0
        succ_rate = np.mean(self._recent_success) if self._recent_success else 0.0
        avg_struct = np.mean(self._recent_r_struct) if self._recent_r_struct else 0.0
        print(
            f"    ✓ Done in {elapsed:.1f}s | Final: R_struct={avg_struct:.3f}, "
            f"Success={succ_rate:.1%}, Best={self._best_r_struct:.3f}, "
            f"Episodes={self._episode_count}"
        )


# ======================================================================
# Main training loop
# ======================================================================
def train_single_target(
    algo_name,
    target_id,
    target_name,
    structure,
    seed,
    total_timesteps,
    weight_config,
    log_dir,
):
    """Train one agent on one target structure."""

    target_alpha, target_beta, target_gamma, target_delta = weight_config

    env = LearnaEnv(structure, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)

    scheduler = AdaptiveWeightScheduler(
        total_timesteps,
        target_alpha=target_alpha,
        target_beta=target_beta,
        target_gamma=target_gamma,
        target_delta=target_delta,
    )

    run_name = (
        f"{algo_name}_puzzle{target_id}"
        f"_a{target_alpha}_b{target_beta}"
        f"_g{target_gamma}_d{target_delta}_seed{seed}"
    )

    if algo_name == "ppo":
        # PPO Modeli - Genişletilmiş ağ yapısı (128x128) ve sabit öğrenme oranı.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            seed=seed,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.02,                            # Keşfi teşvik eder
            policy_kwargs=dict(net_arch=[128, 128]),  # Geniş ağ yapısı
        )
    elif algo_name == "dqn":
        # DQN Modeli - Uzun süreli keşif ve büyük replay buffer.
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=5e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500,
            exploration_fraction=0.6,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.08,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    callback = TrainingCallback(
        scheduler,
        target_name,
        total_timesteps,
        print_interval=max(
            2000, total_timesteps // 20
        ),  # ~20 progress prints per target
    )

    model.learn(
        total_timesteps=total_timesteps, callback=callback, tb_log_name=run_name
    )

    save_path = os.path.join("models", run_name)
    os.makedirs("models", exist_ok=True)
    model.save(save_path)

    return save_path


def compute_timesteps(seq_len: int, min_episodes: int = 3000) -> int:
    """
    Hedefin (RNA) uzunluğuna göre eğitim adım sayısını dinamik hesaplar.
    Uzun dizilimler (len > 30) daha fazla eğitime ihtiyaç duyar.
    """
    if seq_len > 30:
        # 1.0 at len=30, 2.0 at len=60, 2.23 at len=67. Empirically puzzle #10
        # (len=45) succ rate was still climbing (10%→58%) at end of 3750 ep —
        # need more samples to converge.
        scale = 1.0 + 1.0 * (seq_len - 30) / 30.0
        episodes = int(min_episodes * scale)
    else:
        episodes = min_episodes
    return seq_len * episodes


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective RNA inverse folding training"
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Fixed timesteps per target (0 = adaptive based on length)",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=0,
        help="Minimum episodes per target (0 = auto: 3000 for PPO, 5000 for DQN)",
    )
    parser.add_argument(
        "--weight-config",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Grid search config index: 0=(0.5,0.2,0.1,0.2), "
        "1=(0.6,0.15,0.1,0.15), 2=(0.4,0.2,0.15,0.25)",
    )
    args = parser.parse_args()

    WEIGHT_CONFIGS = [
        (0.5, 0.2, 0.1, 0.2),
        (0.6, 0.15, 0.1, 0.15),
        (0.4, 0.2, 0.15, 0.25),
    ]
    weight_config = WEIGHT_CONFIGS[args.weight_config]

    log_dir = "./tensorboard_logs/"
    train_targets = get_train_structures()
    total_experiments = len(train_targets)

    # Auto-select min episodes based on algorithm
    if args.min_episodes == 0:
        min_ep = 5000 if args.algo == "dqn" else 3000
    else:
        min_ep = args.min_episodes

    # Pre-compute per-target timesteps
    adaptive = args.timesteps == 0
    target_steps = []
    for _, _, struct in train_targets:
        if adaptive:
            ts = compute_timesteps(len(struct), min_ep)
        else:
            ts = args.timesteps
        target_steps.append(ts)
    total_steps_all = sum(target_steps)

    # Header
    print("=" * 80)
    print("  RNA Inverse Folding — Multi-Objective Training Pipeline")
    print("=" * 80)
    print(f"  Algorithm    : {args.algo.upper()}")
    print(f"  Seed         : {args.seed}")
    if adaptive:
        ep_counts = [ts // len(s) for ts, (_, _, s) in zip(target_steps, train_targets)]
        print(
            f"  Timesteps    : ADAPTIVE (min {min_ep} eps, "
            f"+50% for len>30)"
        )
        print(
            f"                 Range: {min(target_steps):,} – {max(target_steps):,} steps "
            f"({min(ep_counts):,}–{max(ep_counts):,} episodes)"
        )
    else:
        print(f"  Timesteps    : {args.timesteps:,} (fixed)")
    print(f"  Total steps  : {total_steps_all:,}")
    print(f"  Targets      : {total_experiments}")
    print(
        f"  Weight config: #{args.weight_config}  "
        f"(α={weight_config[0]}, β={weight_config[1]}, "
        f"γ={weight_config[2]}, δ={weight_config[3]})"
    )
    print(f"  Log dir      : {log_dir}")
    print("=" * 80)
    print()

    overall_start = time.time()
    steps_done = 0

    for idx, ((pid, name, struct), ts) in enumerate(
        zip(train_targets, target_steps), 1
    ):
        est_episodes = ts // len(struct)
        print(f"┌─ [{idx}/{total_experiments}] Puzzle #{pid}: {name}")
        print(
            f"│  Structure: {struct[:60]}{'...' if len(struct) > 60 else ''} (len={len(struct)})"
        )
        print(f"│  Training {ts:,} timesteps (~{est_episodes:,} episodes)...")

        t0 = time.time()
        save_path = train_single_target(
            args.algo, pid, name, struct, args.seed, ts, weight_config, log_dir
        )
        elapsed = time.time() - t0
        steps_done += ts

        # Elapsed + overall progress
        overall_elapsed = time.time() - overall_start
        overall_pct = steps_done / total_steps_all * 100
        steps_remaining = total_steps_all - steps_done
        if steps_done > 0:
            speed = steps_done / overall_elapsed
            est_remaining = steps_remaining / speed
        else:
            est_remaining = 0

        print(f"│  Model saved: {save_path}")
        print(
            f"└─ Done in {elapsed:.1f}s | "
            f"Overall: {overall_pct:.0f}% | "
            f"ETA: {est_remaining / 60:.1f} min remaining"
        )
        print()

    # Final summary
    total_time = time.time() - overall_start
    print("=" * 80)
    print(f"  ALL TRAINING COMPLETE!")
    print(f"  Total time: {total_time / 60:.1f} minutes ({total_time:.0f}s)")
    print(f"  Models saved to: ./models/")
    print(f"  TensorBoard:  tensorboard --logdir {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
