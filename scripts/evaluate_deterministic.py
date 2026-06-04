"""
evaluate_deterministic.py -- Four-objective deterministic evaluation.

For each puzzle and each algorithm (PPO, DQN), this loads the trained
specialist policy, generates ONE canonical sequence by running the policy
with all stochasticity disabled (PPO deterministic=True, DQN argmax /
exploration_rate=0), folds that exact sequence with ViennaRNA, and scores
all FOUR design objectives on that single canonical sequence:

    R_struct  structural accuracy 1 - Hamming/n      pass: == 1.0
    GC        GC fraction inside [0.40, 0.60]         pass: in band
    Homo      longest homopolymer run                 pass: <= k (=4)
    MFE/nt    |MFE| per nucleotide                    pass: >= tau_MFE

A puzzle is declared FULLY SOLVED only when all four criteria pass on the
SAME canonical sequence.

Instead of the old stochastic "success rate", we report a continuous
Solution Closeness in [0, 1]: the mean of the four per-objective
satisfaction scores. Closeness == 1.0 corresponds exactly to a fully-solved
sequence, and lower values show how far the canonical sequence still is from
a full multi-objective solution.

Outputs a per-algorithm console table plus a CSV (default
./evaluation_results.csv) holding the canonical sequence and every objective
for both PPO and DQN -- ready to paste into the report / appendix.

Run from the project root inside the rlrna env (ViennaRNA required):
    python scripts/evaluate_deterministic.py
    python scripts/evaluate_deterministic.py --tau-mfe 0.35 --weight-config 0
"""

import os
import re
import csv
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import DQN, PPO
from environment import LearnaEnv
from eterna100 import get_train_structures
# Import the weight configs from the trainer so the model-filename tokens
# (a<alpha>_b<beta>_g<gamma>_d<delta>) always match what was trained.
from train_multi_target import WEIGHT_CONFIGS

# Multi-objective "fully solved" thresholds
GC_LOW, GC_HIGH = 0.40, 0.60   # acceptable GC-content band
HOMO_K = 4                     # max allowed homopolymer run length


def config_token(cfg):
    """Reconstruct the filename token for a weight config, e.g. 'a0.5_b0.2_g0.1_d0.2'."""
    a, b, g, d = cfg
    return f"a{a}_b{b}_g{g}_d{d}"


def longest_run(seq):
    """Length of the longest run of identical characters in seq (0 for empty)."""
    return max((m.end() - m.start() for m in re.finditer(r"(.)\1*", seq)), default=0)


def find_model(model_dir, algo, pid, cfg, seed, homo_step_scale=None):
    """Return the path to the saved specialist model, or None if it is missing.

    Tries the new filename with the homopolymer-scale tag (_h<scale>) first, then
    falls back to the legacy name (no _h tag) so previously-trained models still
    load. Models may be stored with or without a .zip suffix; SB3 resolves both.
    """
    token = config_token(cfg)
    names = []
    if homo_step_scale is not None:
        names.append(f"{algo}_puzzle{pid}_{token}_h{homo_step_scale}_seed{seed}")
    names.append(f"{algo}_puzzle{pid}_{token}_seed{seed}")  # legacy (no _h tag)
    for run_name in names:
        base = os.path.join(model_dir, run_name)
        if os.path.exists(base) or os.path.exists(base + ".zip"):
            return base
    return None


def rollout_canonical(model, env):
    """Run one fully-deterministic episode; return the terminal info dict."""
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        # deterministic=True turns OFF epsilon-greedy (DQN) and stochastic
        # sampling (PPO), so a single rollout yields the canonical sequence.
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def score_sequence(info, tau_mfe):
    """Compute the four objectives, their satisfaction scores, and hard gates
    from a terminal info dict produced by LearnaEnv."""
    seq = info["sequence"]
    n = len(seq)

    r_struct = info["r_struct"]            # 1 - Hamming/n
    hamming = round((1.0 - r_struct) * n)
    gc_frac = info["gc_ratio"]
    r_gc = info["r_gc"]                     # 1 inside band, decays linearly outside
    p_homo = info["p_homo"]                 # sum(max(0, run-k)) / n
    r_mfe = info["r_mfe"]                   # |MFE| / n
    mfe_val = info["mfe_val"]
    max_run = longest_run(seq)

    # ---- Hard gates: the four "fully solved" criteria ----
    struct_ok = bool(info["is_success"])           # Hamming == 0
    gc_ok = (GC_LOW <= gc_frac <= GC_HIGH)
    homo_ok = (max_run <= HOMO_K)
    mfe_ok = (r_mfe >= tau_mfe)
    fully_solved = struct_ok and gc_ok and homo_ok and mfe_ok

    # ---- Continuous satisfaction scores in [0, 1] ----
    # closeness == 1.0  <=>  all four scores == 1.0  <=>  fully solved.
    s_struct = r_struct
    s_gc = max(0.0, min(1.0, r_gc))   # r_gc can be negative for extreme GC
    s_homo = max(0.0, 1.0 - p_homo)
    s_mfe = min(1.0, r_mfe / tau_mfe) if tau_mfe > 0 else 1.0
    closeness = (s_struct + s_gc + s_homo + s_mfe) / 4.0

    return {
        "sequence": seq, "len": n,
        "r_struct": r_struct, "hamming": hamming, "struct_ok": struct_ok,
        "gc_frac": gc_frac, "gc_ok": gc_ok,
        "max_run": max_run, "homo_ok": homo_ok, "p_homo": p_homo,
        "mfe": mfe_val, "mfe_per_nt": r_mfe, "mfe_ok": mfe_ok,
        "closeness": closeness, "fully_solved": fully_solved,
    }


def evaluate(model_dir, algo, pid, structure, cfg, seed, tau_mfe, homo_step_scale=0.15):
    """Load the specialist model for (algo, pid, cfg, seed) and score its
    canonical sequence. Returns a result dict, or None if the model is
    missing or fails to load. (homo_step_scale only affects which file is loaded
    and the env reward, not the deterministic argmax rollout.)"""
    path = find_model(model_dir, algo, pid, cfg, seed, homo_step_scale)
    if path is None:
        return None
    env = LearnaEnv(structure, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
                    homo_step_scale=homo_step_scale)
    try:
        if algo == "dqn":
            model = DQN.load(path, env=env, custom_objects={"exploration_rate": 0.0})
        else:
            model = PPO.load(path, env=env)
    except Exception as e:
        # Surface the problem instead of silently reporting a zero score.
        print(f"  WARNING: failed to load {os.path.basename(path)}: {e}")
        return None
    info = rollout_canonical(model, env)
    return score_sequence(info, tau_mfe)


def evaluate_best_of_seeds(model_dir, algo, pid, structure, cfg, seeds, tau_mfe,
                           homo_step_scale=0.15):
    """Evaluate the (algo, puzzle) specialist across several seeds and return the
    best result, with the winning seed stored under result['seed'].

    'Best' prefers a fully-solved canonical sequence; among ties it takes the
    higher closeness. The SAME seed set is applied to every (algo, puzzle), so
    PPO and DQN stay directly comparable -- multi-seed only adds restart
    robustness symmetrically, it does not advantage either algorithm. This is
    the standard inverse-folding "solved within a restart budget" criterion.

    Returns the result dict (with extra 'seed' and 'n_solved' keys) or None.
    """
    best = None
    n_solved = 0
    for seed in seeds:
        r = evaluate(model_dir, algo, pid, structure, cfg, seed, tau_mfe, homo_step_scale)
        if r is None:
            continue
        if r["fully_solved"]:
            n_solved += 1
        if best is None or (r["fully_solved"], r["closeness"]) > (
                best["fully_solved"], best["closeness"]):
            best = dict(r)
            best["seed"] = seed
    if best is not None:
        best["n_solved"] = n_solved
    return best


def flag(b):
    return "Y" if b else "-"


def print_table(algo, rows, tau_mfe):
    print()
    print(f"  {algo.upper()} -- Four-Objective Deterministic Evaluation")
    print(f"  {'Puzzle':<7}{'Len':>4} | {'R_struct':>9} | {'GC frac':>8} {'ok':>2} | "
          f"{'MaxRun':>6} {'ok':>2} | {'MFE/nt':>7} {'ok':>2} | {'Close':>7} | "
          f"{'Seed':>5} | {'SOLVED':>6}")
    print("  " + "-" * 98)
    for pid, name, r in rows:
        if r is None:
            print(f"  P{pid:<6}{'':>4} |   (no model found)")
            continue
        print(f"  P{pid:<6}{r['len']:>4} | "
              f"{r['r_struct']:>8.3f} {flag(r['struct_ok'])} | "
              f"{r['gc_frac']:>8.2f} {flag(r['gc_ok']):>2} | "
              f"{r['max_run']:>6} {flag(r['homo_ok']):>2} | "
              f"{r['mfe_per_nt']:>7.3f} {flag(r['mfe_ok']):>2} | "
              f"{r['closeness'] * 100:>6.1f}% | "
              f"{r.get('seed', '-'):>5} | "
              f"{('YES' if r['fully_solved'] else 'no'):>6}")
    print("  " + "-" * 98)

    valid = [r for _, _, r in rows if r is not None]
    if valid:
        n = len(valid)
        solved = sum(1 for r in valid if r["fully_solved"])
        struct1 = sum(1 for r in valid if r["struct_ok"])
        mean_close = sum(r["closeness"] for r in valid) / n
        mean_struct = sum(r["r_struct"] for r in valid) / n
        print(f"  Fully solved (4/4): {solved}/{n}   |   R_struct=1: {struct1}/{n}"
              f"   |   mean Closeness: {mean_close * 100:.1f}%"
              f"   |   mean R_struct: {mean_struct:.3f}")


def write_csv(path, targets, all_rows):
    name_by_pid = {pid: name for pid, name, _ in targets}
    fields = ["puzzle_id", "name", "algo", "seed", "length", "r_struct", "hamming",
              "gc_fraction", "gc_ok", "max_run", "homo_ok", "p_homo",
              "mfe", "mfe_per_nt", "mfe_ok", "closeness", "fully_solved", "sequence"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for algo in ("ppo", "dqn"):
            for pid, name, r in all_rows[algo]:
                if r is None:
                    continue
                w.writerow([
                    pid, name_by_pid[pid], algo, r.get("seed", ""), r["len"],
                    f"{r['r_struct']:.4f}", r["hamming"],
                    f"{r['gc_frac']:.4f}", int(r["gc_ok"]),
                    r["max_run"], int(r["homo_ok"]), f"{r['p_homo']:.4f}",
                    f"{r['mfe']:.2f}", f"{r['mfe_per_nt']:.4f}", int(r["mfe_ok"]),
                    f"{r['closeness']:.4f}", int(r["fully_solved"]),
                    r["sequence"],
                ])


def main():
    ap = argparse.ArgumentParser(
        description="Four-objective deterministic evaluation of trained RNA specialists")
    ap.add_argument("--models-dir", default="./models")
    ap.add_argument("--seed", type=int, default=42,
                    help="Single seed (used only if --seeds is not given)")
    ap.add_argument("--seeds", type=str, default="",
                    help="Comma-separated seeds for best-of-seeds eval, e.g. 42,43,44. "
                         "The SAME set is applied to PPO and DQN, so the comparison "
                         "stays fair. Empty -> use --seed.")
    ap.add_argument("--weight-config", type=int, default=0, choices=[0, 1, 2],
                    help="Which trained config to evaluate "
                         "(0=balanced [report default], 1=struct-heavy, 2=thermo-focused)")
    ap.add_argument("--homo-step-scale", type=float, default=0.15,
                    help="Dense-penalty tag of the models to load (must match training; "
                         "default 0.15). Use 0 for structure-focused models.")
    ap.add_argument("--tau-mfe", type=float, default=0.3,
                    help="MFE-per-nucleotide stability threshold in kcal/mol/nt (default 0.3)")
    ap.add_argument("--csv", default="./evaluation_results.csv")
    args = ap.parse_args()

    cfg = WEIGHT_CONFIGS[args.weight_config]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]

    seed_label = ",".join(str(s) for s in seeds)
    print("=" * 98)
    print("  RNA Inverse Folding -- FOUR-OBJECTIVE DETERMINISTIC EVALUATION")
    print(f"  Config #{args.weight_config} (a={cfg[0]}, b={cfg[1]}, g={cfg[2]}, d={cfg[3]})"
          f" | seeds {seed_label} (best-of) | h={args.homo_step_scale}"
          f" | tau_MFE = {args.tau_mfe}")
    print(f"  Pass criteria: R_struct==1.0 | GC in [{GC_LOW}, {GC_HIGH}] | "
          f"max run <= {HOMO_K} | |MFE|/n >= {args.tau_mfe}")
    if len(seeds) > 1:
        print(f"  (a puzzle counts as solved if ANY seed solves it -- applied "
              f"identically to PPO and DQN)")
    print("=" * 98)

    targets = get_train_structures()
    all_rows = {"ppo": [], "dqn": []}
    for algo in ("ppo", "dqn"):
        for pid, name, structure in targets:
            r = evaluate_best_of_seeds(args.models_dir, algo, pid, structure, cfg,
                                       seeds, args.tau_mfe, args.homo_step_scale)
            all_rows[algo].append((pid, name, r))

    for algo in ("ppo", "dqn"):
        print_table(algo, all_rows[algo], args.tau_mfe)

    write_csv(args.csv, targets, all_rows)
    print(f"\n  Per-objective results + canonical sequences written to: {args.csv}")

    # Canonical sequences of fully-solved puzzles (for the report appendix)
    print("\n  Fully-solved canonical sequences:")
    any_solved = False
    for algo in ("ppo", "dqn"):
        for pid, name, r in all_rows[algo]:
            if r is not None and r["fully_solved"]:
                any_solved = True
                print(f"    [{algo.upper()}] P{pid} ({name}): {r['sequence']}")
    if not any_solved:
        print("    (none yet -- no puzzle satisfies all four criteria at this tau_MFE)")
    print("=" * 92)


if __name__ == "__main__":
    main()
