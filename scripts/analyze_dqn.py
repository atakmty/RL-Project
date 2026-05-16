"""
analyze_dqn.py — Analyze DQN training results across all 15 puzzles.
Checks both tensorboard_logs/ (new runs) and tensorboard_logs/DQN/ (old runs).

Usage:
    cd /mnt/d/RL_project
    /home/tata/miniconda/envs/rlrna/bin/python scripts/analyze_dqn.py
"""
import glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from eterna100 import ETERNA100_SELECTED

pid_to_name = {pid: name for pid, name, _ in ETERNA100_SELECTED}
pid_to_len = {pid: len(s) for pid, name, s in ETERNA100_SELECTED}


def extract_metrics(log_dir):
    """Extract key metrics from a TensorBoard log directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    def get_last_n_mean(tag, n=50):
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            return sum(vals[-n:]) / len(vals[-n:]) if vals else 0.0
        return 0.0

    def get_max(tag):
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            return max(vals) if vals else 0.0
        return 0.0

    def get_count(tag):
        if tag in tags:
            return len(ea.Scalars(tag))
        return 0

    return {
        'r_struct': get_last_n_mean("episode/r_struct", 50),
        'best': get_max("episode/r_struct"),
        'succ': get_last_n_mean("episode/is_success", 50),
        'r_gc': get_last_n_mean("episode/r_gc", 50),
        'r_mfe': get_last_n_mean("episode/r_mfe", 50),
        'n_points': get_count("episode/r_struct"),
    }


def find_dqn_logs(log_root='./tensorboard_logs'):
    """Find all DQN log directories, checking multiple possible locations."""
    candidates = [
        # New v2 runs (saved to root)
        glob.glob(os.path.join(log_root, 'dqn_puzzle*_seed42_1')),
        # Old v1 runs (saved to DQN subfolder)
        glob.glob(os.path.join(log_root, 'DQN', 'dqn_puzzle*_seed42_1')),
    ]
    
    # Pick the set with most data points (newest run)
    best_dirs = []
    best_label = ""
    for i, dirs in enumerate(candidates):
        if dirs:
            label = ["Root (v2 — optimized)", "DQN/ subfolder (v1 — original)"][i]
            # Check data points in first dir
            if dirs:
                ea = EventAccumulator(dirs[0])
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                pts = len(ea.Scalars("episode/r_struct")) if "episode/r_struct" in tags else 0
                if pts > (len(ea.Scalars("episode/r_struct")) if best_dirs else 0):
                    pass  # will compare below
            if not best_dirs or len(dirs) > len(best_dirs):
                pass  # logic below
    
    # Simpler: prefer root-level (newer) over DQN/ subfolder (older)
    root_dirs = sorted(glob.glob(os.path.join(log_root, 'dqn_puzzle*_seed42_1')))
    sub_dirs = sorted(glob.glob(os.path.join(log_root, 'DQN', 'dqn_puzzle*_seed42_1')))
    
    if root_dirs:
        return root_dirs, "tensorboard_logs/ (v2 — optimized hyperparams)"
    elif sub_dirs:
        return sub_dirs, "tensorboard_logs/DQN/ (v1 — original)"
    else:
        return [], "NOT FOUND"


puzzle_dirs, source_label = find_dqn_logs()

if not puzzle_dirs:
    print("ERROR: No DQN logs found!")
    sys.exit(1)

print("=" * 105)
print("  RNA Inverse Folding — DQN Training Results (Config 0: a=0.5, b=0.2, g=0.1, d=0.2 | Seed 42)")
print("=" * 105)
print(f"  Source: {source_label}")
print(f"  Puzzles found: {len(puzzle_dirs)}")
print()
print(f"  {'':3} {'Puzzle':<6} {'Name':<42} {'Len':>4} {'R_str':>7} {'Best':>7} {'Succ%':>6} {'R_gc':>7} {'R_mfe':>7} {'Pts':>6}")
print(f"  {'-' * 97}")

results = []
for pdir in puzzle_dirs:
    dirname = os.path.basename(pdir)
    pid = int(dirname.split('_puzzle')[1].split('_a')[0])
    name = pid_to_name.get(pid, "?")
    slen = pid_to_len.get(pid, 0)

    r = extract_metrics(pdir)
    r['pid'] = pid
    r['name'] = name
    r['len'] = slen
    results.append(r)

    if r['succ'] > 0.5:
        status = "✅"
    elif r['best'] >= 0.8:
        status = "🟡"
    else:
        status = "❌"

    print(f"  {status} P{r['pid']:<5} {r['name']:<42} {r['len']:>4} "
          f"{r['r_struct']:>7.3f} {r['best']:>7.3f} {r['succ']:>5.0%} "
          f"{r['r_gc']:>7.3f} {r['r_mfe']:>7.3f} {r['n_points']:>6}")

# Summary
solved = sum(1 for r in results if r['succ'] > 0.5)
near = sum(1 for r in results if r['best'] >= 0.8 and r['succ'] <= 0.5)
failed = len(results) - solved - near

print(f"  {'-' * 97}")
print()
print(f"  {'SUMMARY':=^80}")
print(f"  Solved (>50% success rate):  {solved}/{len(results)}")
print(f"  Near-solved (best >= 0.8):   {near}/{len(results)}")
print(f"  Failed:                      {failed}/{len(results)}")
print()

if results:
    avg_rs = sum(r['r_struct'] for r in results) / len(results)
    avg_best = sum(r['best'] for r in results) / len(results)
    avg_gc = sum(r['r_gc'] for r in results) / len(results)
    avg_mfe = sum(r['r_mfe'] for r in results) / len(results)
    best_puzzle = max(results, key=lambda r: r['r_struct'])

    print(f"  {'AGGREGATE METRICS':=^80}")
    print(f"  Average R_struct (last 50):  {avg_rs:.4f}")
    print(f"  Average Best R_struct:       {avg_best:.4f}")
    print(f"  Average R_gc:                {avg_gc:.4f}")
    print(f"  Average R_mfe:               {avg_mfe:.4f}")
    print(f"  Best performing puzzle:      P{best_puzzle['pid']} {best_puzzle['name']} (R_struct={best_puzzle['r_struct']:.3f})")

# Also check if old v1 logs exist for comparison
root_dirs = sorted(glob.glob('./tensorboard_logs/dqn_puzzle*_seed42_1'))
sub_dirs = sorted(glob.glob('./tensorboard_logs/DQN/dqn_puzzle*_seed42_1'))

if root_dirs and sub_dirs:
    print(f"\n  {'V1 vs V2 COMPARISON':=^80}")
    print(f"  {'Puzzle':<8} {'V1 R_str':>9} {'V2 R_str':>9} {'V1 Best':>9} {'V2 Best':>9} {'Change':>9}")
    print(f"  {'-' * 50}")
    
    for v1dir in sub_dirs:
        dirname = os.path.basename(v1dir)
        pid = int(dirname.split('_puzzle')[1].split('_a')[0])
        v2dir_match = [d for d in root_dirs if f'_puzzle{pid}_' in d]
        if v2dir_match:
            v1 = extract_metrics(v1dir)
            v2 = extract_metrics(v2dir_match[0])
            diff = v2['r_struct'] - v1['r_struct']
            arrow = "↑" if diff > 0.02 else "↓" if diff < -0.02 else "="
            print(f"  P{pid:<6} {v1['r_struct']:>9.3f} {v2['r_struct']:>9.3f} "
                  f"{v1['best']:>9.3f} {v2['best']:>9.3f} {diff:>+8.3f} {arrow}")

print(f"\n{'=' * 105}")
