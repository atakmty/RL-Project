#!/bin/bash
# RNA Inverse Folding — Full Grid Search (3 weight configs × DQN)
# Usage: bash run_grid_search_dqn.sh

echo "================================================================="
echo "  RNA Inverse Folding - DQN Grid Search"
echo "================================================================="

echo -e "\n>>> Step 1/3: Config 0 (Balanced: 0.5, 0.2, 0.1, 0.2)"
python train_multi_target.py --algo dqn --seed 42 --weight-config 0

echo -e "\n>>> Step 2/3: Config 1 (Structure-focused: 0.6, 0.15, 0.1, 0.15)"
python train_multi_target.py --algo dqn --seed 42 --weight-config 1

echo -e "\n>>> Step 3/3: Config 2 (Biophysics-focused: 0.4, 0.2, 0.15, 0.25)"
python train_multi_target.py --algo dqn --seed 42 --weight-config 2

echo -e "\n================================================================="
echo "  ALL TRAINING COMPLETED SUCCESSFULLY!"
echo "================================================================="
