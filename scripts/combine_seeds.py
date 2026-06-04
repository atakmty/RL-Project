"""
combine_seeds.py -- merge per-seed evaluation CSVs into a best-of-seeds result.

For distributed multi-seed runs: each collaborator trains and evaluates ONE
seed on their own machine and shares the small CSV (models are gitignored and
large, so we share CSVs, not models). This script takes several such CSVs and,
for each (puzzle, algo), keeps the best row -- preferring a fully-solved
sequence, then higher closeness. The SAME rule is applied to PPO and DQN, so
the PPO-vs-DQN comparison stays fair (best-of-seeds restart budget, identical
for both algorithms).

Works on either evaluate_deterministic.py output or repair_sequences.py output
(both expose puzzle_id, algo, fully_solved, closeness columns). All other
columns are passed through from the winning row.

Usage:
    python scripts/combine_seeds.py --out combined.csv \
        results_seed42.csv results_seed43.csv results_seed44.csv
"""

import argparse
import csv


def _load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _score(row):
    # best = fully_solved first (1 > 0), then higher closeness
    solved = int(float(row.get("fully_solved", 0) or 0))
    close = float(row.get("closeness", 0) or 0)
    return (solved, close)


def main():
    ap = argparse.ArgumentParser(
        description="Merge per-seed evaluation CSVs into a best-of-seeds result.")
    ap.add_argument("--out", default="combined.csv")
    ap.add_argument("csvs", nargs="+", help="per-seed CSV files to merge")
    args = ap.parse_args()

    files = [_load(p) for p in args.csvs]

    # Union of all column names (preserve first-seen order) so CSVs with
    # slightly different schemas (e.g. eval vs repaired) still merge cleanly.
    fieldnames = []
    seen = set()
    for rows in files:
        if rows:
            for k in rows[0].keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    best = {}            # (puzzle_id, algo) -> winning row
    for rows in files:
        for row in rows:
            key = (row["puzzle_id"], row["algo"])
            if key not in best or _score(row) > _score(best[key]):
                best[key] = row

    if not best:
        print("  No rows found in the given CSVs.")
        return

    out_rows = sorted(best.values(), key=lambda r: (r["algo"], int(r["puzzle_id"])))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print("=" * 60)
    print(f"  Combined {len(args.csvs)} CSV(s) -> {args.out}")
    for algo in ("ppo", "dqn"):
        arows = [r for r in out_rows if r["algo"] == algo]
        if not arows:
            continue
        solved = sum(1 for r in arows if int(float(r.get("fully_solved", 0) or 0)))
        print(f"  {algo.upper()}: fully solved {solved}/{len(arows)}")
    total = sum(1 for r in out_rows if int(float(r.get("fully_solved", 0) or 0)))
    print(f"  TOTAL best-of-seeds: {total}/{len(out_rows)} fully solved")
    # show which seed won the solved puzzles, if a 'seed' column is present
    if "seed" in (fieldnames or []):
        solved_rows = [r for r in out_rows if int(float(r.get("fully_solved", 0) or 0))]
        if solved_rows:
            print("  Solved (puzzle/algo/seed):")
            for r in solved_rows:
                print(f"    P{r['puzzle_id']:<3} {r['algo'].upper():<3} seed={r.get('seed','?')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
