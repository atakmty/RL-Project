#!/usr/bin/env python3
"""
repair_sequences.py -- Post-hoc deterministic repair of canonical RNA sequences.

Takes the canonical sequences produced by trained RL agents (from the
deterministic evaluation CSV) and applies structure-preserving edits to:
  1. Break homopolymer runs > 4  (paired: G<->C or A<->U swap; unpaired: substitute)
  2. Adjust GC content into [0.40, 0.60]  (unpaired positions first, then paired)
  3. Verify fold is preserved after each change via ViennaRNA

Only sequences with R_struct == 1.0 (perfect structural match) are repaired.
This is the "Option C" from the HANDOFF: no retraining, fastest path to the
first fully-solved puzzles.

Usage (WSL, conda env rlrna):
    python scripts/repair_sequences.py
    python scripts/repair_sequences.py --input evaluation_results_g0.4.csv \\
                                       --output evaluation_results_repaired.csv
"""

import os
import sys
import csv
import re
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import RNA

    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False
    print(
        "WARNING: ViennaRNA not available. Fold verification will be SKIPPED."
    )
    print(
        "         For accurate results, run in WSL with: conda activate rlrna"
    )

from eterna100 import get_train_structures

# ---------------------------------------------------------------------------
# Constants  (must match evaluate_deterministic.py and environment.py)
# ---------------------------------------------------------------------------
HOMO_K = 4  # max allowed homopolymer run
GC_LOW, GC_HIGH = 0.40, 0.60
TAU_MFE = 0.3  # kcal/mol/nt

VALID_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("G", "C"),
    ("C", "G"),
    ("G", "U"),
    ("U", "G"),
}

# For a nucleotide at position i, what partner nucleotides at position j
# produce a valid (i,j) base pair?
PARTNER_OPTIONS = {
    "A": ["U"],  # A-U
    "U": ["A", "G"],  # U-A, U-G (wobble)
    "G": ["C", "U"],  # G-C, G-U (wobble)
    "C": ["G"],  # C-G
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_pair_table(structure):
    """Return dict {i: j, j: i} for every base pair in dot-bracket notation."""
    stack = []
    pairs = {}
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            j = stack.pop()
            pairs[i] = j
            pairs[j] = i
    return pairs


def longest_run(seq):
    """Length of the longest homopolymer run in seq."""
    if not seq:
        return 0
    return max(
        (m.end() - m.start() for m in re.finditer(r"(.)\1*", seq)), default=0
    )


def find_long_runs(seq, k=HOMO_K):
    """Return list of (start, end, nucleotide, length) for runs > k."""
    runs = []
    for m in re.finditer(r"(.)\1*", seq):
        rlen = m.end() - m.start()
        if rlen > k:
            runs.append((m.start(), m.end(), m.group()[0], rlen))
    return runs


def gc_fraction(seq):
    """GC content as a fraction in [0, 1]."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in "GC") / len(seq)


def local_run_at(seq_list, pos):
    """Length of the run of identical nucleotides containing position pos."""
    n = len(seq_list)
    nuc = seq_list[pos]
    left = pos
    while left > 0 and seq_list[left - 1] == nuc:
        left -= 1
    right = pos
    while right < n - 1 and seq_list[right + 1] == nuc:
        right += 1
    return right - left + 1


def fold_and_check(seq, target_structure):
    """Fold seq with ViennaRNA.  Returns (r_struct, mfe_val).
    If ViennaRNA is unavailable returns (1.0, -0.5*n) as optimistic mock."""
    n = len(seq)
    if not HAS_VIENNARNA:
        return 1.0, -0.5 * n
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    hamming = sum(1 for a, b in zip(struct, target_structure) if a != b)
    r_struct = 1.0 - hamming / n
    return r_struct, mfe


def compute_objectives(seq, target_structure):
    """Score all four objectives + closeness for a sequence.
    Matches evaluate_deterministic.py / environment.py exactly."""
    n = len(seq)
    r_struct, mfe_val = fold_and_check(seq, target_structure)
    hamming = round((1.0 - r_struct) * n)

    gc = gc_fraction(seq)
    gc_dev = max(0.0, abs(gc - 0.5) - 0.1)
    r_gc = 1.0 - 2.0 * (gc_dev / 0.4)

    max_run = longest_run(seq)

    # p_homo (quadratic over margin-3, matching environment.py)
    p_homo = 0.0
    for m in re.finditer(r"(.)\1+", seq):
        rlen = m.end() - m.start()
        excess = max(0, rlen - 3)
        p_homo += excess * excess
    p_homo /= n

    mfe_per_nt = abs(mfe_val) / n

    struct_ok = hamming == 0
    gc_ok = GC_LOW <= gc <= GC_HIGH
    homo_ok = max_run <= HOMO_K
    mfe_ok = mfe_per_nt >= TAU_MFE
    fully_solved = struct_ok and gc_ok and homo_ok and mfe_ok

    s_struct = r_struct
    s_gc = max(0.0, min(1.0, r_gc))
    s_homo = max(0.0, 1.0 - p_homo)
    s_mfe = min(1.0, mfe_per_nt / TAU_MFE) if TAU_MFE > 0 else 1.0
    closeness = (s_struct + s_gc + s_homo + s_mfe) / 4.0

    return {
        "sequence": seq,
        "n": n,
        "r_struct": r_struct,
        "hamming": hamming,
        "gc_frac": gc,
        "r_gc": r_gc,
        "max_run": max_run,
        "p_homo": p_homo,
        "mfe": mfe_val,
        "mfe_per_nt": mfe_per_nt,
        "struct_ok": struct_ok,
        "gc_ok": gc_ok,
        "homo_ok": homo_ok,
        "mfe_ok": mfe_ok,
        "fully_solved": fully_solved,
        "closeness": closeness,
    }


# ---------------------------------------------------------------------------
# Phase 1: Homopolymer repair
# ---------------------------------------------------------------------------
def repair_homopolymer(seq_list, pair_table):
    """Break all homopolymer runs > HOMO_K in-place.

    Strategy: for each oversized run, substitute every (HOMO_K+1)-th
    nucleotide.  For paired positions the complementary partner is swapped
    to maintain a valid base pair.  The replacement nucleotide is chosen to
    steer GC content towards 0.50.

    Returns True if all runs were successfully broken to <= HOMO_K.
    """
    MAX_ITER = 50  # safety cap

    for _it in range(MAX_ITER):
        seq_str = "".join(seq_list)
        long_runs = find_long_runs(seq_str)
        if not long_runs:
            return True

        # Attack the longest run first
        long_runs.sort(key=lambda r: -r[3])
        start, end, _nuc, _rlen = long_runs[0]

        # Position to change: keep the first HOMO_K nucleotides, break after
        pos = start + HOMO_K
        if pos >= end:
            return False  # shouldn't happen

        current_gc = gc_fraction(seq_str)
        applied = False

        # --- Try all candidates for this position --------------------------
        candidates = _candidates_for_position(seq_list, pos, pair_table, current_gc)

        if candidates:
            best = candidates[0]  # sorted: lowest overall max_run, then best GC
            seq_list[:] = best["seq"]
            applied = True

        # --- Fallback: try other positions in the run ----------------------
        if not applied:
            for alt_pos in range(pos + 1, end):
                cands = _candidates_for_position(
                    seq_list, alt_pos, pair_table, current_gc
                )
                if cands:
                    seq_list[:] = cands[0]["seq"]
                    applied = True
                    break

        if not applied:
            return False  # exhausted all options for this run

    return longest_run("".join(seq_list)) <= HOMO_K


def _candidates_for_position(seq_list, pos, pair_table, current_gc):
    """Generate valid replacement candidates for a single position.

    Each candidate is a dict with keys 'seq' (modified list), 'max_run',
    'gc_dist'.  Returned sorted by (max_run ascending, gc_dist ascending).
    """
    old_nuc = seq_list[pos]
    candidates = []
    current_max = longest_run("".join(seq_list))

    if pos in pair_table:
        partner = pair_table[pos]
        for new_nuc in "ACGU":
            if new_nuc == old_nuc:
                continue
            for pnuc in PARTNER_OPTIONS.get(new_nuc, []):
                test = seq_list[:]
                test[pos] = new_nuc
                test[partner] = pnuc

                # Reject if new long runs appear at pos or partner
                if local_run_at(test, pos) > HOMO_K:
                    continue
                if local_run_at(test, partner) > HOMO_K:
                    continue

                test_str = "".join(test)
                new_max = longest_run(test_str)
                if new_max >= current_max:
                    # Must strictly improve or at least not worsen
                    if new_max > current_max:
                        continue
                new_gc = gc_fraction(test_str)
                gc_dist = abs(new_gc - 0.5)
                candidates.append(
                    {"seq": test, "max_run": new_max, "gc_dist": gc_dist}
                )
    else:
        # Unpaired: free to choose any nucleotide
        for new_nuc in "ACGU":
            if new_nuc == old_nuc:
                continue
            test = seq_list[:]
            test[pos] = new_nuc

            if local_run_at(test, pos) > HOMO_K:
                continue

            test_str = "".join(test)
            new_max = longest_run(test_str)
            if new_max > current_max:
                continue
            new_gc = gc_fraction(test_str)
            gc_dist = abs(new_gc - 0.5)
            candidates.append(
                {"seq": test, "max_run": new_max, "gc_dist": gc_dist}
            )

    candidates.sort(key=lambda c: (c["max_run"], c["gc_dist"]))
    return candidates


# ---------------------------------------------------------------------------
# Phase 2: GC content repair
# ---------------------------------------------------------------------------
def repair_gc(seq_list, pair_table, target_structure):
    """Push GC fraction into [GC_LOW, GC_HIGH].

    First pass:  change unpaired positions (safest).
    Second pass: swap paired G-C <-> A-U if unpaired alone is not enough.
    After each change, verify no new homopolymer runs > HOMO_K and
    (if ViennaRNA is available) the fold is preserved.

    Returns True if GC is now within the acceptable band.
    """
    n = len(seq_list)

    # ---- Pass 1: unpaired positions --------------------------------------
    _gc_adjust_unpaired(seq_list, pair_table, target_structure, n)

    # ---- Pass 2: paired positions (only if still out of range) ------------
    gc = gc_fraction("".join(seq_list))
    if not (GC_LOW <= gc <= GC_HIGH):
        _gc_adjust_paired(seq_list, pair_table, target_structure, n)

    gc = gc_fraction("".join(seq_list))
    return GC_LOW <= gc <= GC_HIGH


def _gc_adjust_unpaired(seq_list, pair_table, target_structure, n):
    """Change unpaired positions to move GC towards [0.40, 0.60]."""
    gc = gc_fraction("".join(seq_list))
    unpaired = [i for i in range(n) if i not in pair_table]

    if gc > GC_HIGH:
        # Reduce GC: change unpaired G/C -> A/U
        gc_positions = [i for i in unpaired if seq_list[i] in "GC"]
        for pos in gc_positions:
            if gc_fraction("".join(seq_list)) <= GC_HIGH:
                break
            _try_substitute(seq_list, pos, ["A", "U"], target_structure)

    elif gc < GC_LOW:
        # Increase GC: change unpaired A/U -> G/C
        au_positions = [i for i in unpaired if seq_list[i] in "AU"]
        for pos in au_positions:
            if gc_fraction("".join(seq_list)) >= GC_LOW:
                break
            _try_substitute(seq_list, pos, ["G", "C"], target_structure)


def _gc_adjust_paired(seq_list, pair_table, target_structure, n):
    """Swap paired G-C <-> A-U to adjust GC content.

    Each swap changes 2 GC nucleotides at once (both partners).
    Verified with ViennaRNA fold check after each swap.
    """
    gc = gc_fraction("".join(seq_list))

    # Collect paired position indices (only one side of each pair, i < j)
    paired_indices = sorted(
        {min(i, pair_table[i]) for i in pair_table}
    )

    if gc > GC_HIGH:
        # Change G-C pairs to A-U pairs
        gc_pairs = [
            i
            for i in paired_indices
            if (seq_list[i] in "GC" and seq_list[pair_table[i]] in "GC")
        ]
        for i in gc_pairs:
            if gc_fraction("".join(seq_list)) <= GC_HIGH:
                break
            j = pair_table[i]
            _try_pair_swap(seq_list, i, j, "AU", target_structure)

    elif gc < GC_LOW:
        # Change A-U pairs to G-C pairs
        au_pairs = [
            i
            for i in paired_indices
            if (seq_list[i] in "AU" and seq_list[pair_table[i]] in "AU")
        ]
        for i in au_pairs:
            if gc_fraction("".join(seq_list)) >= GC_LOW:
                break
            j = pair_table[i]
            _try_pair_swap(seq_list, i, j, "GC", target_structure)


def _try_substitute(seq_list, pos, preferred_nucs, target_structure):
    """Try substituting seq_list[pos] with a preferred nucleotide.
    Accepts the first option that doesn't create a long run and preserves fold."""
    old = seq_list[pos]
    for new_nuc in preferred_nucs:
        if new_nuc == old:
            continue
        seq_list[pos] = new_nuc
        if local_run_at(seq_list, pos) > HOMO_K:
            seq_list[pos] = old
            continue
        # Fold check
        if HAS_VIENNARNA:
            r_s, _ = fold_and_check("".join(seq_list), target_structure)
            if r_s < 1.0:
                seq_list[pos] = old
                continue
        return True  # accepted
    return False


def _try_pair_swap(seq_list, i, j, target_type, target_structure):
    """Swap a base pair at positions (i,j) to the target_type ('GC' or 'AU').

    Tries both orientations (e.g. G-C and C-G) and picks the one that
    preserves fold and doesn't create homopolymer runs.
    """
    old_i, old_j = seq_list[i], seq_list[j]

    if target_type == "GC":
        options = [("G", "C"), ("C", "G")]
    else:  # AU
        options = [("A", "U"), ("U", "A")]

    for ni, nj in options:
        if (ni, nj) not in VALID_PAIRS:
            continue
        seq_list[i] = ni
        seq_list[j] = nj

        if local_run_at(seq_list, i) > HOMO_K or local_run_at(seq_list, j) > HOMO_K:
            seq_list[i], seq_list[j] = old_i, old_j
            continue

        if HAS_VIENNARNA:
            r_s, _ = fold_and_check("".join(seq_list), target_structure)
            if r_s < 1.0:
                seq_list[i], seq_list[j] = old_i, old_j
                continue

        return True  # accepted

    seq_list[i], seq_list[j] = old_i, old_j
    return False


# ---------------------------------------------------------------------------
# Main repair pipeline
# ---------------------------------------------------------------------------
def repair_sequence(seq, target_structure):
    """Full repair pipeline for one sequence.

    Returns a result dict with before/after objectives and metadata.
    """
    pair_table = build_pair_table(target_structure)
    seq_list = list(seq)

    # Phase 1: break homopolymer runs
    homo_ok = repair_homopolymer(seq_list, pair_table)

    # Phase 2: adjust GC content
    gc_ok = repair_gc(seq_list, pair_table, target_structure)

    repaired = "".join(seq_list)
    changes = sum(1 for a, b in zip(seq, repaired) if a != b)

    # Final scoring
    obj = compute_objectives(repaired, target_structure)
    obj["changes"] = changes
    obj["original"] = seq
    obj["homo_fixed"] = homo_ok
    obj["gc_fixed"] = gc_ok

    return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_eval_csv(path):
    """Read the evaluation CSV into a list of dicts."""
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def flag(b):
    return "Y" if b else "-"


def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc repair of canonical RNA sequences"
    )
    ap.add_argument(
        "--input",
        default="./evaluation_results_g0.4.csv",
        help="Evaluation CSV from evaluate_deterministic.py",
    )
    ap.add_argument(
        "--output",
        default="./evaluation_results_repaired.csv",
        help="Output CSV with repaired sequences",
    )
    ap.add_argument("--tau-mfe", type=float, default=0.3)
    args = ap.parse_args()

    global TAU_MFE
    TAU_MFE = args.tau_mfe

    # Target structures
    targets = get_train_structures()
    struct_by_pid = {pid: struct for pid, _, struct in targets}
    name_by_pid = {pid: name for pid, name, _ in targets}

    # Read evaluation results
    rows = read_eval_csv(args.input)

    print("=" * 100)
    print("  RNA Inverse Folding -- POST-HOC SEQUENCE REPAIR (Option C)")
    print(f"  Input : {args.input}")
    print(
        f"  Criteria: R_struct==1.0 | GC in [{GC_LOW},{GC_HIGH}] "
        f"| max_run <= {HOMO_K} | |MFE|/n >= {TAU_MFE}"
    )
    if not HAS_VIENNARNA:
        print(
            "  *** ViennaRNA NOT available -- fold verification SKIPPED ***"
        )
    print("=" * 100)

    results = []
    csv_rows = []

    for algo in ("ppo", "dqn"):
        algo_rows = [r for r in rows if r["algo"] == algo]

        print(f"\n  {algo.upper()} -- Repair Results")
        print(
            f"  {'Puzzle':<7}{'Len':>4} | "
            f"{'--- BEFORE ---':^25} | "
            f"{'--- AFTER ---':^25} | "
            f"{'Chg':>3} | {'SOLVED':>6}"
        )
        print(
            f"  {'':7}{'':>4} | "
            f"{'Rs':>5} {'GC':>5} {'Run':>4} {'MFE':>6} | "
            f"{'Rs':>5} {'GC':>5} {'Run':>4} {'MFE':>6} | "
            f"{'':>3} | {'':>6}"
        )
        print("  " + "-" * 90)

        for row in algo_rows:
            pid = int(row["puzzle_id"])
            seq = row["sequence"].strip()
            r_struct_orig = float(row["r_struct"])
            n = int(row["length"])

            if pid not in struct_by_pid:
                continue

            target = struct_by_pid[pid]

            # Original metrics (from CSV)
            gc_orig = float(row["gc_fraction"])
            run_orig = int(row["max_run"])
            mfe_orig = float(row["mfe_per_nt"])

            if r_struct_orig < 1.0:
                # Skip -- structure is not perfect, repair is too risky
                print(
                    f"  P{pid:<6}{n:>4} | "
                    f"{r_struct_orig:>5.3f} {gc_orig:>5.2f} {run_orig:>4} {mfe_orig:>6.3f} | "
                    f"{'--- skipped (Rs<1) ---':>25} |     |"
                )
                results.append(
                    {
                        "pid": pid,
                        "algo": algo,
                        "repaired": False,
                        "fully_solved": False,
                    }
                )
                csv_rows.append(
                    {
                        "puzzle_id": pid,
                        "name": name_by_pid[pid],
                        "algo": algo,
                        "length": n,
                        "repaired": 0,
                        "r_struct": f"{r_struct_orig:.4f}",
                        "gc_fraction": f"{gc_orig:.4f}",
                        "max_run": run_orig,
                        "mfe_per_nt": f"{mfe_orig:.4f}",
                        "mfe": row["mfe"],
                        "fully_solved": 0,
                        "closeness": row["closeness"],
                        "changes": 0,
                        "original_sequence": seq,
                        "repaired_sequence": seq,
                    }
                )
                continue

            # ---- REPAIR ----
            result = repair_sequence(seq, target)

            status = "YES *" if result["fully_solved"] else "no"

            # Diff display
            diff_chars = []
            for a, b in zip(seq, result["sequence"]):
                diff_chars.append("^" if a != b else " ")
            diff_str = "".join(diff_chars)

            print(
                f"  P{pid:<6}{n:>4} | "
                f"{r_struct_orig:>5.3f} {gc_orig:>5.2f} {run_orig:>4} {mfe_orig:>6.3f} | "
                f"{result['r_struct']:>5.3f} {result['gc_frac']:>5.2f} "
                f"{result['max_run']:>4} {result['mfe_per_nt']:>6.3f} | "
                f"{result['changes']:>3} | {status:>6}"
            )
            if result["changes"] > 0:
                print(f"          Original : {seq}")
                print(f"          Repaired : {result['sequence']}")
                print(f"          Changes  : {diff_str}")

            results.append(
                {
                    "pid": pid,
                    "algo": algo,
                    "repaired": True,
                    "fully_solved": result["fully_solved"],
                }
            )
            csv_rows.append(
                {
                    "puzzle_id": pid,
                    "name": name_by_pid[pid],
                    "algo": algo,
                    "length": n,
                    "repaired": 1,
                    "r_struct": f"{result['r_struct']:.4f}",
                    "gc_fraction": f"{result['gc_frac']:.4f}",
                    "max_run": result["max_run"],
                    "mfe_per_nt": f"{result['mfe_per_nt']:.4f}",
                    "mfe": f"{result['mfe']:.2f}",
                    "fully_solved": int(result["fully_solved"]),
                    "closeness": f"{result['closeness']:.4f}",
                    "changes": result["changes"],
                    "original_sequence": seq,
                    "repaired_sequence": result["sequence"],
                }
            )

        print("  " + "-" * 90)
        algo_results = [r for r in results if r["algo"] == algo]
        solved = sum(1 for r in algo_results if r["fully_solved"])
        total = len(algo_results)
        repaired_ct = sum(1 for r in algo_results if r.get("repaired"))
        print(
            f"  Fully solved: {solved}/{total} | "
            f"Attempted repairs: {repaired_ct}/{total}"
        )

    # ---- Write CSV --------------------------------------------------------
    if args.output:
        fields = [
            "puzzle_id", "name", "algo", "length", "repaired",
            "r_struct", "gc_fraction", "max_run", "mfe", "mfe_per_nt",
            "fully_solved", "closeness", "changes",
            "original_sequence", "repaired_sequence",
        ]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"\n  Results written to: {args.output}")

    # ---- Final summary ----------------------------------------------------
    total_solved = sum(1 for r in results if r["fully_solved"])
    total_all = len(results)
    print(f"\n  === TOTAL FULLY SOLVED: {total_solved}/{total_all} ===")

    if total_solved > 0:
        print("\n  Fully-solved sequences:")
        for r, cr in zip(results, csv_rows):
            if r["fully_solved"]:
                print(
                    f"    [{cr['algo'].upper()}] P{cr['puzzle_id']} "
                    f"({cr['name']}): {cr['repaired_sequence']}"
                )
    else:
        print("    (no fully-solved puzzles yet)")

    print("=" * 100)


if __name__ == "__main__":
    main()
