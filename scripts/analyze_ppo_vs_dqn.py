"""
analyze_ppo_vs_dqn.py — Compare PPO (Run 3 / adaptive) vs DQN results side-by-side.

Usage:
    cd /mnt/d/RL_project
    /home/tata/miniconda/envs/rlrna/bin/python scripts/analyze_ppo_vs_dqn.py
"""
import glob, os, sys

# Add parent directory so we can import eterna100
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


def analyze_algo(base_dir, algo_prefix, suffix='1'):
    """Analyze all puzzle logs for a given algorithm."""
    pattern = os.path.join(base_dir, f'{algo_prefix}_puzzle*_seed42_{suffix}')
    puzzle_dirs = sorted(glob.glob(pattern))

    results = {}
    for pdir in puzzle_dirs:
        dirname = os.path.basename(pdir)
        pid = int(dirname.split('_puzzle')[1].split('_a')[0])
        metrics = extract_metrics(pdir)
        metrics['pid'] = pid
        metrics['name'] = pid_to_name.get(pid, "?")
        metrics['len'] = pid_to_len.get(pid, 0)
        results[pid] = metrics
    return results


# Detect directory structure
log_root = './tensorboard_logs'
ppo_base = os.path.join(log_root, 'PPO') if os.path.isdir(os.path.join(log_root, 'PPO')) else log_root
dqn_base = os.path.join(log_root, 'DQN') if os.path.isdir(os.path.join(log_root, 'DQN')) else log_root

print(f"PPO log dir: {ppo_base}")
print(f"DQN log dir: {dqn_base}")

print("=" * 120)
print("  RNA Inverse Folding — PPO vs DQN Comparison (Config 0: alpha=0.5, beta=0.2, gamma=0.1, delta=0.2 | Seed 42)")
print("=" * 120)

# Get PPO results — try Run 3 (adaptive timesteps) first, then 2, then 1
ppo_results = {}
ppo_run_label = ""
for suffix, label in [('3', 'Run 3 (adaptive)'), ('2', 'Run 2 (one-hot fix)'), ('1', 'Run 1 (50K fixed)')]:
    ppo_results = analyze_algo(ppo_base, 'ppo', suffix=suffix)
    if ppo_results:
        ppo_run_label = label
        break

# Get DQN results
dqn_results = analyze_algo(dqn_base, 'dqn', suffix='1')

print(f"\n  PPO source: {ppo_run_label} ({len(ppo_results)} puzzles)")
print(f"  DQN source: Run 1 ({len(dqn_results)} puzzles)")

all_pids = sorted(set(list(ppo_results.keys()) + list(dqn_results.keys())))

if not all_pids:
    print("\n  ERROR: No results found! Check log directory structure.")
    sys.exit(1)

print()
print(f"  {'':3} {'Puzzle':<6} {'Name':<30} {'Len':>4} |"
      f" {'PPO R_str':>9} {'PPO Best':>9} {'PPO Succ':>9} |"
      f" {'DQN R_str':>9} {'DQN Best':>9} {'DQN Succ':>9} |"
      f" {'Winner':>8}")
print(f"  {'-' * 114}")

ppo_solved = 0
dqn_solved = 0
ppo_better = 0
dqn_better = 0
tie = 0

for pid in all_pids:
    name = pid_to_name.get(pid, "?")[:30]
    slen = pid_to_len.get(pid, 0)

    ppo = ppo_results.get(pid, {})
    dqn = dqn_results.get(pid, {})

    ppo_rs = ppo.get('r_struct', 0.0)
    ppo_best = ppo.get('best', 0.0)
    ppo_succ = ppo.get('succ', 0.0)
    dqn_rs = dqn.get('r_struct', 0.0)
    dqn_best = dqn.get('best', 0.0)
    dqn_succ = dqn.get('succ', 0.0)

    if ppo_succ > 0.5:
        ppo_solved += 1
    if dqn_succ > 0.5:
        dqn_solved += 1

    # Determine winner based on final r_struct
    diff = ppo_rs - dqn_rs
    if abs(diff) < 0.02:
        winner = "  TIE"
        tie += 1
    elif diff > 0:
        winner = "  PPO ✅"
        ppo_better += 1
    else:
        winner = "  DQN ✅"
        dqn_better += 1

    # Status emoji
    if ppo_succ > 0.5 or dqn_succ > 0.5:
        status = "✅"
    elif ppo_best >= 0.8 or dqn_best >= 0.8:
        status = "🟡"
    else:
        status = "❌"

    print(f"  {status} P{pid:<5} {name:<30} {slen:>4} |"
          f" {ppo_rs:>9.3f} {ppo_best:>9.3f} {ppo_succ:>8.0%} |"
          f" {dqn_rs:>9.3f} {dqn_best:>9.3f} {dqn_succ:>8.0%} |"
          f" {winner}")

print(f"  {'-' * 114}")
print()
print(f"  {'SUMMARY':=^80}")
print(f"  PPO solved (>50% success): {ppo_solved}/{len(all_pids)}")
print(f"  DQN solved (>50% success): {dqn_solved}/{len(all_pids)}")
print(f"  PPO better (higher R_struct): {ppo_better}")
print(f"  DQN better (higher R_struct): {dqn_better}")
print(f"  Tied (diff < 0.02): {tie}")
print()

# Per-metric averages
if ppo_results and dqn_results:
    common_pids = set(ppo_results.keys()) & set(dqn_results.keys())
    if common_pids:
        ppo_avg_rs = sum(ppo_results[p]['r_struct'] for p in common_pids) / len(common_pids)
        dqn_avg_rs = sum(dqn_results[p]['r_struct'] for p in common_pids) / len(common_pids)
        ppo_avg_gc = sum(ppo_results[p]['r_gc'] for p in common_pids) / len(common_pids)
        dqn_avg_gc = sum(dqn_results[p]['r_gc'] for p in common_pids) / len(common_pids)
        ppo_avg_mfe = sum(ppo_results[p]['r_mfe'] for p in common_pids) / len(common_pids)
        dqn_avg_mfe = sum(dqn_results[p]['r_mfe'] for p in common_pids) / len(common_pids)
        ppo_avg_best = sum(ppo_results[p]['best'] for p in common_pids) / len(common_pids)
        dqn_avg_best = sum(dqn_results[p]['best'] for p in common_pids) / len(common_pids)

        print(f"  {'AVERAGE METRICS (common puzzles)':=^80}")
        print(f"  {'Metric':<25} {'PPO':>10} {'DQN':>10} {'Better':>10}")
        print(f"  {'-' * 55}")
        print(f"  {'R_struct (last 50 avg)':<25} {ppo_avg_rs:>10.4f} {dqn_avg_rs:>10.4f} {'PPO' if ppo_avg_rs > dqn_avg_rs else 'DQN':>10}")
        print(f"  {'Best R_struct (max)':<25} {ppo_avg_best:>10.4f} {dqn_avg_best:>10.4f} {'PPO' if ppo_avg_best > dqn_avg_best else 'DQN':>10}")
        print(f"  {'R_gc (avg)':<25} {ppo_avg_gc:>10.4f} {dqn_avg_gc:>10.4f} {'PPO' if ppo_avg_gc > dqn_avg_gc else 'DQN':>10}")
        print(f"  {'R_mfe (avg)':<25} {ppo_avg_mfe:>10.4f} {dqn_avg_mfe:>10.4f} {'PPO' if ppo_avg_mfe > dqn_avg_mfe else 'DQN':>10}")

print(f"\n{'=' * 120}")
