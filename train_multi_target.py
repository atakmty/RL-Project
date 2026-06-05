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
import re
import time
import sys
import numpy as np
from collections import deque
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment import LearnaEnv
from eterna100 import get_train_structures, get_test_structures


# ======================================================================
# Weight configurations (alpha, beta, gamma, delta) for the grid search.
# Single source of truth -- scripts/evaluate_deterministic.py imports this
# so the saved-model filenames and the evaluator never drift apart.
# Config 0 gamma raised 0.1 -> 0.3 -> 0.4 for terminal penalty, then lowered
# to 0.2 because the new dense per-step homopolymer penalty in environment.py
# carries the main workload; the terminal penalty is now just a final check.
# ======================================================================
WEIGHT_CONFIGS = [
    (0.5, 0.2, 0.2, 0.2),
    (0.6, 0.15, 0.1, 0.15),
    (0.4, 0.2, 0.15, 0.25),
]

# Four-objective "fully solved" thresholds -- MUST match
# scripts/evaluate_deterministic.py so live training metrics and the final
# deterministic evaluation use the same definition of "solved".
GC_LOW, GC_HIGH = 0.40, 0.60
HOMO_K = 4
TAU_MFE = 0.3


def _longest_run(seq):
    """Length of the longest run of identical characters in seq."""
    return max((m.end() - m.start() for m in re.finditer(r"(.)\1*", seq)), default=0)


def episode_scores(info):
    """Per-episode four-objective gates + closeness for one terminal info dict,
    matching scripts/evaluate_deterministic.py. Returns
    (gc_ratio, max_run, r_mfe, solved4, closeness)."""
    seq = info.get("sequence", "")
    r_struct = info["r_struct"]
    r_gc = info["r_gc"]
    p_homo = info["p_homo"]
    r_mfe = info["r_mfe"]
    gc_ratio = info.get("gc_ratio", 0.0)
    max_run = _longest_run(seq)

    struct_ok = bool(info["is_success"])      # R_struct == 1 (Hamming 0)
    gc_ok = (GC_LOW <= gc_ratio <= GC_HIGH)
    homo_ok = (max_run <= HOMO_K)
    mfe_ok = (r_mfe >= TAU_MFE)
    solved4 = struct_ok and gc_ok and homo_ok and mfe_ok

    s_struct = r_struct
    s_gc = max(0.0, min(1.0, r_gc))           # r_gc can be negative for extreme GC
    s_homo = max(0.0, 1.0 - p_homo)
    s_mfe = min(1.0, r_mfe / TAU_MFE) if TAU_MFE > 0 else 1.0
    closeness = (s_struct + s_gc + s_homo + s_mfe) / 4.0
    return gc_ratio, max_run, r_mfe, solved4, closeness


# ======================================================================
# Adaptive Weight Scheduler (Proposal Section 3.3)
# ======================================================================
class AdaptiveWeightScheduler:
    """
    3 Aşamalı (Phase A, B, C) öğrenme planlayıcısı.
    Agent'in önce yapısal doğruluğu, ardından GC oranını öğrenmesini sağlar.
    """

    # Phase A starting fractions of target weights.
    # beta now FULLY active from step 0 (1.0): combined with the signed GC
    # reward (negative at the all-GC plateau, see environment.py) and a
    # shorter Phase A, this keeps the agent off the all-GC attractor before
    # policy entropy collapses. gamma also fully active so long homopolymer
    # runs are penalised from the start.
    PHASE_A_BETA_FRAC = 1.0
    PHASE_A_GAMMA_FRAC = 1.0
    PHASE_A_DELTA_FRAC = 0.0

    def __init__(
        self,
        total_timesteps: int,
        target_alpha=0.5,
        target_beta=0.2,
        target_gamma=0.1,
        target_delta=0.2,
        alpha_floor=0.0,
    ):
        self.total = total_timesteps
        self.target_alpha = target_alpha
        self.target_beta = target_beta
        self.target_gamma = target_gamma
        self.target_delta = target_delta
        # Soft floor on alpha (Future Work 6, ii): once in Phase B/C the
        # structural weight is never allowed to drop below this, so the
        # structural gradient cannot be fully overpowered by the biophysical
        # terms (the dominant DQN failure mode). 0.0 = disabled (legacy).
        self.alpha_floor = alpha_floor

        # Phase boundaries
        # Phase A shortened 0.30 -> 0.15 so the biophysical penalties (GC band,
        # homopolymer) start ramping while the policy still has high entropy,
        # rather than after it has already collapsed onto an all-GC solution.
        self.phase_a_end = int(0.15 * total_timesteps)
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
            return max(self.alpha_floor, alpha), beta, gamma, delta
        else:
            return (
                max(self.alpha_floor, self.target_alpha),
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

        # Rolling window for live metrics (four-objective view)
        self._recent_r_struct = deque(maxlen=50)
        self._recent_r_gc = deque(maxlen=50)
        self._recent_success = deque(maxlen=50)      # R_struct == 1 hit rate
        self._recent_reward = deque(maxlen=50)
        self._recent_gc_ratio = deque(maxlen=50)     # actual GC fraction
        self._recent_maxrun = deque(maxlen=50)       # longest homopolymer run
        self._recent_r_mfe = deque(maxlen=50)        # |MFE| / n
        self._recent_solved4 = deque(maxlen=50)      # all four objectives pass
        self._recent_closeness = deque(maxlen=50)    # solution closeness [0,1]
        self._best_r_struct = 0.0

        # Replay-buffer phase-boundary reset bookkeeping (DQN only)
        self._cleared_b = False
        self._cleared_c = False

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

        # --- Replay-buffer reset at phase boundaries (DQN only) ---
        # The terminal reward is weight-dependent and the curriculum overwrites
        # the weights every step, so transitions collected in an earlier phase
        # carry stale rewards. Flushing the buffer when a new phase begins keeps
        # the off-policy Q-target consistent with the current weight regime.
        # PPO is on-policy and has no replay_buffer -> skipped automatically.
        # (A fuller fix would store the four reward components and re-score them
        #  with the current weights at sample time; the boundary reset is the
        #  cheap version that removes cross-phase staleness.)
        rb = getattr(self.model, "replay_buffer", None)
        if rb is not None:
            t = self.num_timesteps
            if not self._cleared_b and t > self.scheduler.phase_a_end:
                rb.reset()
                self._cleared_b = True
                print("    [replay] buffer cleared at Phase A->B boundary", flush=True)
            if not self._cleared_c and t > self.scheduler.phase_b_end:
                rb.reset()
                self._cleared_c = True
                print("    [replay] buffer cleared at Phase B->C boundary", flush=True)

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

                # Four-objective view (same definition as the deterministic eval)
                gc_ratio, max_run, r_mfe, solved4, closeness = episode_scores(terminal)
                self._recent_gc_ratio.append(gc_ratio)
                self._recent_maxrun.append(max_run)
                self._recent_r_mfe.append(r_mfe)
                self._recent_solved4.append(1.0 if solved4 else 0.0)
                self._recent_closeness.append(closeness)

                # TensorBoard
                self.logger.record("episode/r_struct", r_s)
                self.logger.record("episode/r_gc", r_gc)
                self.logger.record("episode/p_homo", terminal["p_homo"])
                self.logger.record("episode/r_mfe", terminal["r_mfe"])
                self.logger.record("episode/gc_ratio", terminal.get("gc_ratio", 0.0))
                self.logger.record("episode/is_success", is_succ)
                self.logger.record("episode/max_run", max_run)
                self.logger.record("episode/solved4", 1.0 if solved4 else 0.0)
                self.logger.record("episode/closeness", closeness)
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

        # Rolling metrics (four-objective view; over stochastic training episodes)
        avg_struct = np.mean(self._recent_r_struct) if self._recent_r_struct else 0.0
        hit_rate = np.mean(self._recent_success) if self._recent_success else 0.0     # R_struct==1
        avg_gc = np.mean(self._recent_gc_ratio) if self._recent_gc_ratio else 0.0     # GC fraction
        avg_run = np.mean(self._recent_maxrun) if self._recent_maxrun else 0.0        # longest run
        avg_mfe = np.mean(self._recent_r_mfe) if self._recent_r_mfe else 0.0          # |MFE|/n
        solved4 = np.mean(self._recent_solved4) if self._recent_solved4 else 0.0      # all 4 objectives
        closeness = np.mean(self._recent_closeness) if self._recent_closeness else 0.0

        # Build progress bar
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"    [{bar}] {pct:5.1f}% | Phase {phase} | "
            f"Rs={avg_struct:.3f}(best={self._best_r_struct:.3f},hit={hit_rate:.0%}) | "
            f"GC={avg_gc:.2f} | run={avg_run:.1f} | MFE/nt={avg_mfe:.2f} | "
            f"Solved4={solved4:.0%} | Close={closeness:.0%} | "
            f"ETA={eta_str} | Ep={self._episode_count}",
            flush=True,
        )

    def _on_training_end(self):
        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_struct = np.mean(self._recent_r_struct) if self._recent_r_struct else 0.0
        hit_rate = np.mean(self._recent_success) if self._recent_success else 0.0
        avg_gc = np.mean(self._recent_gc_ratio) if self._recent_gc_ratio else 0.0
        avg_run = np.mean(self._recent_maxrun) if self._recent_maxrun else 0.0
        avg_mfe = np.mean(self._recent_r_mfe) if self._recent_r_mfe else 0.0
        solved4 = np.mean(self._recent_solved4) if self._recent_solved4 else 0.0
        closeness = np.mean(self._recent_closeness) if self._recent_closeness else 0.0
        print(
            f"    Done in {elapsed:.1f}s | Final: Rs={avg_struct:.3f}"
            f"(best={self._best_r_struct:.3f}, hit={hit_rate:.0%}) | "
            f"GC={avg_gc:.2f} maxrun={avg_run:.1f} MFE/nt={avg_mfe:.2f} | "
            f"Solved4={solved4:.0%} Close={closeness:.0%} | Ep={self._episode_count}"
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
    homo_step_scale=0.15,
    mfe_tau=0.3,
    gc_falloff=0.2,
    solve_bonus_scale=0.5,
    solve_jackpot=0.5,
    alpha_floor=0.0,
):
    """Train one agent on one target structure."""

    target_alpha, target_beta, target_gamma, target_delta = weight_config

    env = LearnaEnv(structure, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
                    homo_step_scale=homo_step_scale,
                    mfe_tau=mfe_tau, gc_falloff=gc_falloff,
                    solve_bonus_scale=solve_bonus_scale, solve_jackpot=solve_jackpot)

    scheduler = AdaptiveWeightScheduler(
        total_timesteps,
        target_alpha=target_alpha,
        target_beta=target_beta,
        target_gamma=target_gamma,
        target_delta=target_delta,
        alpha_floor=alpha_floor,
    )

    # homo_step_scale is part of the run name so models trained with a different
    # dense-penalty strength (e.g. the structure-focused scale=0) do NOT overwrite
    # each other. Legacy models without the _h tag are still found by the evaluator.
    run_name = (
        f"{algo_name}_puzzle{target_id}"
        f"_a{target_alpha}_b{target_beta}"
        f"_g{target_gamma}_d{target_delta}"
        f"_h{homo_step_scale}_seed{seed}"
    )

    if algo_name == "ppo":
        # PPO model -- wider [128, 128] network and a constant learning rate.
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
            ent_coef=0.05,                            # encourages exploration (0.02->0.05: prevents all-GC entropy collapse)
            policy_kwargs=dict(net_arch=[128, 128]),  # wider network
        )
    elif algo_name == "dqn":
        # DQN model -- long exploration schedule and a large replay buffer.
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
    # ---- Command-line interface (argparse) ----
    # Defines the flags this script accepts and parses them from the command
    # line into `args` (see `args = parser.parse_args()` below). EVERY default
    # here is the value used for the reported run, so
    #   python train_multi_target.py --algo ppo   (or --algo dqn)
    # reproduces the results with no extra flags; pass a flag only to override.
    parser = argparse.ArgumentParser(
        description="Multi-objective RNA inverse folding training"
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--seed", type=int, default=44,
                        help="Random seed (default 44, the reported run).")
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
        help="Grid search config index: 0=(0.5,0.2,0.2,0.2), "
        "1=(0.6,0.15,0.1,0.15), 2=(0.4,0.2,0.15,0.25)",
    )
    parser.add_argument(
        "--homo-step-scale",
        type=float,
        default=0.05,
        help="Dense per-step homopolymer penalty strength (default 0.05, the "
        "reported run); applied identically to PPO and DQN.",
    )
    parser.add_argument(
        "--mfe-tau",
        type=float,
        default=0.3,
        help="|MFE|/n threshold; the MFE reward saturates at this value so the "
        "policy is not rewarded for over-stabilising (and drifting GC out of band). "
        "Must match the evaluator's TAU_MFE.",
    )
    parser.add_argument(
        "--gc-falloff",
        type=float,
        default=0.2,
        help="GC band reward reaches its -1.0 floor at (|gc-0.5|-0.1) == gc_falloff. "
        "Smaller = steeper out-of-band penalty (default 0.2).",
    )
    parser.add_argument(
        "--solve-bonus",
        type=float,
        default=0.5,
        help="Scale of the multiplicative joint-satisfaction bonus (Change C).",
    )
    parser.add_argument(
        "--solve-jackpot",
        type=float,
        default=0.5,
        help="Discrete terminal bonus added when all four hard gates pass.",
    )
    parser.add_argument(
        "--alpha-floor",
        type=float,
        default=0.0,
        help="Soft floor on the structural weight alpha during Phase B/C so the "
        "structural gradient is not overpowered by biophysical terms (fixes the "
        "DQN structure-collapse failure mode). 0 = disabled. Try ~0.7 for DQN.",
    )
    parser.add_argument(
        "--puzzles",
        type=str,
        default="",
        help="Comma-separated puzzle IDs to train (e.g. '1,8,15,23,26,30'). "
        "Empty = all puzzles in the chosen split. Use a subset for fast "
        "reward-knob tuning before committing to the full run.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Train on the 5 HELD-OUT test puzzles (get_test_structures) instead "
        "of the 15 training puzzles. Use the SAME reward knobs as the train run "
        "(no extra tuning) so the result is a clean generalization check.",
    )
    args = parser.parse_args()

    weight_config = WEIGHT_CONFIGS[args.weight_config]

    log_dir = "./tensorboard_logs/"
    train_targets = get_test_structures() if args.test else get_train_structures()

    # Optional subset filter (keeps the original order of the chosen split).
    if args.puzzles.strip():
        wanted = {int(p) for p in args.puzzles.split(",") if p.strip()}
        train_targets = [t for t in train_targets if t[0] in wanted]
        missing = wanted - {t[0] for t in train_targets}
        if missing:
            raise SystemExit(f"--puzzles: unknown puzzle id(s) {sorted(missing)}")
        if not train_targets:
            raise SystemExit("--puzzles: no matching puzzles to train")

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
    print(f"  Split        : {'TEST (held-out 5)' if args.test else 'TRAIN (15)'}")
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
    print(f"  Homo penalty : dense step scale = {args.homo_step_scale}")
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
            args.algo, pid, name, struct, args.seed, ts, weight_config, log_dir,
            homo_step_scale=args.homo_step_scale,
            mfe_tau=args.mfe_tau, gc_falloff=args.gc_falloff,
            solve_bonus_scale=args.solve_bonus, solve_jackpot=args.solve_jackpot,
            alpha_floor=args.alpha_floor,
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
