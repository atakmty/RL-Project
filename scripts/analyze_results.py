"""Analyze all training logs — compares Run 1 (50K fixed) vs Run 3 (adaptive)."""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from eterna100 import ETERNA100_SELECTED

pid_to_name = {pid: name for pid, name, _ in ETERNA100_SELECTED}
pid_to_len = {pid: len(s) for pid, name, s in ETERNA100_SELECTED}

def analyze_run(suffix):
    """Analyze all puzzle logs ending with given suffix."""
    log_base = './tensorboard_logs/'
    puzzle_dirs = sorted(glob.glob(os.path.join(log_base, f'ppo_puzzle*_seed42_{suffix}')))
    
    results = []
    for pdir in puzzle_dirs:
        dirname = os.path.basename(pdir)
        pid = int(dirname.split('_puzzle')[1].split('_a')[0])
        name = pid_to_name.get(pid, "?")
        slen = pid_to_len.get(pid, 0)

        ea = EventAccumulator(pdir)
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

        results.append({
            'pid': pid, 'name': name, 'len': slen,
            'r_struct': get_last_n_mean("episode/r_struct", 50),
            'best': get_max("episode/r_struct"),
            'succ': get_last_n_mean("episode/is_success", 50),
            'r_gc': get_last_n_mean("episode/r_gc", 50),
            'r_mfe': get_last_n_mean("episode/r_mfe", 50),
            'n_points': get_count("episode/r_struct"),
        })
    return results


def print_table(title, results):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"  {'':3} {'Puzzle':<6} {'Name':<42} {'Len':>4} {'R_str':>6} {'Best':>6} {'Succ%':>6} {'R_gc':>6} {'R_mfe':>6} {'Pts':>5}")
    print(f"  {'-'*94}")

    solved = 0
    for r in results:
        if r['succ'] > 0.5:
            status = "✅"
            solved += 1
        elif r['best'] >= 0.8:
            status = "🟡"
        else:
            status = "❌"
        print(f"  {status} P{r['pid']:<5} {r['name']:<42} {r['len']:>4} "
              f"{r['r_struct']:>6.3f} {r['best']:>6.3f} {r['succ']:>5.0%} "
              f"{r['r_gc']:>6.3f} {r['r_mfe']:>6.3f} {r['n_points']:>5}")

    print(f"  {'-'*94}")
    print(f"  Solved (>50% success): {solved}/{len(results)}")
    return solved


# Analyze available runs
print("RNA Inverse Folding — Training Results Analysis")

for suffix in ['1', '2', '3']:
    log_base = './tensorboard_logs/'
    dirs = glob.glob(os.path.join(log_base, f'ppo_puzzle*_seed42_{suffix}'))
    if dirs:
        label = {
            '1': 'Run 1 (50K fixed steps)',
            '2': 'Run 2 (50K fixed, one-hot fix)',
            '3': 'Run 3 (adaptive timesteps)',
        }.get(suffix, f'Run {suffix}')
        results = analyze_run(suffix)
        print_table(label, results)

print(f"\n{'=' * 100}")
print("  Config: α=0.5, β=0.2, γ=0.1, δ=0.2 | Algo: PPO | Seed: 42")
print(f"{'=' * 100}")
