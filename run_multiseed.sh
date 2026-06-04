#!/bin/bash
# ----------------------------------------------------------------------
# run_multiseed.sh -- multi-seed training, comparison-preserving.
#
# Trains ALL 15 puzzles for BOTH algorithms (PPO and DQN) across the given
# seeds. The SAME seed set is applied to both algorithms, so the PPO-vs-DQN
# comparison stays fair -- multi-seed only adds restart robustness (the
# standard "solved within a restart budget" criterion for inverse folding),
# it does not advantage either algorithm.
#
# Usage:
#   bash run_multiseed.sh                 # seeds 43 44, homo_step_scale 0 (struct-focus)
#   bash run_multiseed.sh "43 44" 0       # explicit
#   bash run_multiseed.sh "42 43 44" 0.15 # all seeds, dense penalty on
#
# Then evaluate best-of-seeds (must match --homo-step-scale to the training):
#   python scripts/evaluate_deterministic.py --seeds 42,43,44 --homo-step-scale 0 \
#       --csv evaluation_results_multiseed.csv
# ...and (optionally) repair the structurally-correct sequences:
#   python scripts/repair_sequences.py --input evaluation_results_multiseed.csv \
#       --output evaluation_results_multiseed_repaired.csv
# ----------------------------------------------------------------------
set -e

SEEDS="${1:-43 44}"
HOMO_SCALE="${2:-0}"

echo "================================================================"
echo "  Multi-seed training"
echo "  Seeds            : $SEEDS"
echo "  homo_step_scale  : $HOMO_SCALE  (0 = structure-focused; repair fixes homopolymers)"
echo "  Algorithms       : ppo dqn   (same seeds -> fair comparison)"
echo "================================================================"

for seed in $SEEDS; do
  for algo in ppo dqn; do
    echo ""
    echo "================ $algo  seed=$seed  h=$HOMO_SCALE ================"
    python train_multi_target.py --algo "$algo" --seed "$seed" \
        --weight-config 0 --homo-step-scale "$HOMO_SCALE"
  done
done

echo ""
echo "All multi-seed training done. Evaluate with:"
echo "  python scripts/evaluate_deterministic.py --seeds $(echo $SEEDS | tr ' ' ',') --homo-step-scale $HOMO_SCALE --csv evaluation_results_multiseed.csv"
