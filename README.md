# RNA Inverse Folding — Multi-Objective Deep Reinforcement Learning

> **Work in Progress** — This project is actively being developed. Results and code are updated regularly.

Solving the RNA inverse folding problem with PPO and DQN, optimizing for structural accuracy, GC-content, thermodynamic stability, and homopolymer avoidance simultaneously.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![ViennaRNA](https://img.shields.io/badge/Folding-ViennaRNA-orange.svg)](https://www.tbi.univie.ac.at/RNA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The RNA inverse folding problem asks: *given a target secondary structure, find a nucleotide sequence that folds into it*. This is an NP-hard combinatorial optimization problem with a search space of **4ⁿ** candidates.

Existing DRL frameworks like [LEARNA](https://github.com/automl/learna) optimize for structural match alone. Our pipeline extends this with **four simultaneous objectives**:

| Objective | Description | Weight |
|-----------|-------------|--------|
| **R_struct** | Structural accuracy (normalized Hamming distance) | α |
| **R_GC** | GC-content fitness — full reward in [40%, 60%] band | β |
| **P_homo** | Homopolymer penalty — penalizes runs > 4 identical bases | γ |
| **R_MFE** | Thermodynamic stability — normalized \|MFE\| per nucleotide | δ |

**Compound reward:** `R = α·R_struct + β·R_GC − γ·P_homo + δ·R_MFE`

### Preliminary Findings

- 100% success rate on short structures (P1, P8)
- 48% success rate on P10 Frog Foot (len=45, 3 hairpins) — up from initial 0%
- PPO shows better performance than DQN on longer structures (n > 30)
- Partner-Aware observation space improved DQN R_struct from 0.27 → 0.46 on P54 (92 nt)

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Eterna100   │────▶│  LearnaEnv       │────▶│  PPO / DQN    │
│  Benchmark   │     │  (Gymnasium)     │     │  (SB3)        │
│  (20 targets)│     │                  │     │               │
└──────────────┘     │  • One-hot obs   │     │  • [128,128]  │
                     │  • Partner-aware │     │  • ent=0.02   │
                     │  • Reward shaping│     │  • GAE / Replay│
                     └────────┬─────────┘     └───────┬───────┘
                              │                       │
                     ┌────────▼─────────┐     ┌───────▼───────┐
                     │  ViennaRNA       │     │  TensorBoard  │
                     │  MFE Folding     │     │  Logging      │
                     └──────────────────┘     └───────────────┘
```

### Three-Phase Adaptive Weight Scheduling

| Phase | Steps | Strategy |
|-------|-------|----------|
| **A** (0–30%) | Structure-dominant | α=1.0, β=0.7·β*, γ=γ*, δ=0 |
| **B** (30–70%) | Linear ramp | Smooth transition to target weights |
| **C** (70–100%) | Joint optimization | All weights at target values |

> **Critical insight:** Phase A must include non-zero β and γ from step 0 to prevent GC-content collapse.

## Project Structure

```
RL-Project/
├── environment.py              # Gymnasium RL environment (Partner-Aware obs, 4-objective reward)
├── eterna100.py                # Eterna100-V2 dataset (15 train + 5 test targets)
├── train_multi_target.py       # Main training pipeline (curriculum learning, weight scheduling)
├── run_grid_search_ppo.sh      # Grid search runner — 3 weight configs × PPO
├── run_grid_search_dqn.sh      # Grid search runner — 3 weight configs × DQN
├── scripts/
│   ├── analyze_ppo_vs_dqn.py   # PPO vs DQN comparison from TensorBoard logs
│   ├── analyze_dqn.py          # DQN-specific result analysis
│   ├── evaluate_deterministic.py  # Deterministic evaluation (ε=0, no exploration noise)
│   └── preflight_check.py      # Environment verification (ViennaRNA, GPU, etc.)
├── models/                     # Trained model checkpoints (gitignored)
├── tensorboard_logs/           # Training logs (gitignored)
├── environment.yml             # Conda environment definition
├── requirements.txt            # pip dependencies
├── LICENSE
└── README.md
```

### What each file does

| File | Purpose |
|------|---------|
| `environment.py` | Defines the `LearnaEnv` Gymnasium environment. The agent places one nucleotide (A/C/G/U) per step. At the final step, ViennaRNA folds the sequence and returns a 4-objective reward. Intermediate steps use Ng et al. (1999) potential-based reward shaping. |
| `eterna100.py` | Contains the 20 selected Eterna100-V2 target structures (15 train + 5 test). Each target is a dot-bracket string like `((((((......))))))`. |
| `train_multi_target.py` | The main training script. Trains a **separate PPO or DQN model for each puzzle** sequentially. Includes the 3-phase adaptive weight scheduler and adaptive episode scaling for longer sequences. |
| `run_grid_search_ppo.sh` | Runs `train_multi_target.py` 3 times with PPO for each weight configuration (Balanced, Structure-heavy, Thermodynamic-focused). |
| `run_grid_search_dqn.sh` | Same as above but with DQN. |
| `scripts/evaluate_deterministic.py` | Loads saved models and evaluates them with `deterministic=True` (no exploration noise), revealing the true learned policy performance. |
| `scripts/analyze_ppo_vs_dqn.py` | Reads TensorBoard logs and prints a side-by-side PPO vs DQN comparison table. |

## Quick Start

### Prerequisites

- Python 3.10+
- Conda (for ViennaRNA installation)

### Installation

> **Platform note — ViennaRNA is Linux/macOS only.** The bioconda channel does not publish `viennarna` for `win-64`, so `conda env create` will fail on native Windows with `PackagesNotFoundError: viennarna`. **Windows users must use WSL2** (instructions below). macOS and Linux users can skip straight to the standard install.

#### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/atakmty/RL-Project.git
cd RL-Project

# 2. Create environment with ALL dependencies (single command)
conda env create -f environment.yml
conda activate rlrna
```

> `environment.yml` installs Python 3.10, ViennaRNA, GSL, and all pip packages automatically. No separate `pip install` needed.

#### Windows (via WSL2)

**1. Install WSL2 + Ubuntu.** In an **administrator PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted. Ubuntu will launch and ask you to create a Linux username and password.

**2. Open Ubuntu** (Start menu → "Ubuntu", or run `wsl` in any terminal) and install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# accept defaults; when it asks about running conda init, say yes
```

Close and reopen the Ubuntu terminal — your prompt should now start with `(base)`.

**3. Clone and create the environment** (same commands as Linux):

```bash
git clone https://github.com/atakmty/RL-Project.git
cd RL-Project
conda env create -f environment.yml
conda activate rlrna
```

> **Tip:** `conda env create` is a **one-time** setup. After it succeeds, every new Ubuntu session only needs `cd ~/RL-Project && conda activate rlrna`.

### Training

```bash
# Train PPO on all 15 targets (balanced config)
python train_multi_target.py --algo ppo --seed 42 --weight-config 0

# Train DQN on all 15 targets
python train_multi_target.py --algo dqn --seed 42 --weight-config 0

# Run full grid search (3 weight configs)
bash run_grid_search_ppo.sh   # PPO × 3 configs
bash run_grid_search_dqn.sh   # DQN × 3 configs
```

> Each run trains **15 separate models** (one per puzzle). Models are saved under `models/` and logs under `tensorboard_logs/`.

### Weight Configurations (Grid Search)

| Config | α | β | γ | δ | Strategy |
|--------|---|---|---|---|----------|
| 0 | 0.5 | 0.2 | 0.1 | 0.2 | Balanced |
| 1 | 0.6 | 0.15 | 0.1 | 0.15 | Structure-heavy |
| 2 | 0.4 | 0.2 | 0.15 | 0.25 | Thermodynamic-focused |

### Evaluation

```bash
# Deterministic evaluation (epsilon=0, no exploration noise)
python scripts/evaluate_deterministic.py

# Compare PPO vs DQN from TensorBoard logs
python scripts/analyze_ppo_vs_dqn.py

# Analyze DQN results
python scripts/analyze_dqn.py
```

### TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs/
```


## Technical Details

### Search Space

The search space is **4ⁿ** — for a target of length *n*, there are 4ⁿ possible nucleotide sequences (A, C, G, U at each position). For example, the longest puzzle P54 (n=92) has a search space of 4⁹² ≈ 2.4 × 10⁵⁵ candidates, making brute-force search infeasible.

### Observation Space (7n + 10 dimensions)

The observation vector has two parts: a **base encoding** of 7n + 1 dimensions and a **Partner-Aware extension** of 9 dimensions.

**Base encoding (7n + 1):**

| Component | Dims | Description |
|-----------|------|-------------|
| Sequence one-hot | 4n | A/C/G/U at each placed position |
| Target one-hot | 3n | ./(/) at each position |
| Progress | 1 | current_step / n |

**Partner-Aware extension (+9 → total +10 with progress):**

RNA structures contain base pairs — positions marked `(` are paired with positions marked `)`. The agent needs to know about its partner when placing a nucleotide at a paired position. Without this context, the agent has no way to choose a complementary base (e.g., G for a partner C).

| Component | Dims | Description |
|-----------|------|-------------|
| Local target char | 3 | One-hot of target structure at current step: `.` / `(` / `)` |
| is_paired | 1 | 1.0 if current position has a base-pair partner, 0.0 otherwise |
| partner_placed | 1 | 1.0 if the partner's nucleotide has already been placed |
| partner_nucleotide | 4 | One-hot of partner's nucleotide (A/C/G/U), zeros if not yet placed |

> These 9 extra dimensions give the agent explicit structural context. For example, when placing position 15 (a `)`) whose partner position 3 (a `(`) already has `C`, the agent sees `partner_nucleotide = [0,1,0,0]` and can learn to place `G` for a valid C-G base pair.

### Action Space

**Discrete(4)** — at each step, the agent selects one nucleotide. The action is encoded as one-hot:

| Action | Nucleotide | One-hot |
|--------|------------|----------|
| 0 | A (Adenine) | `[1, 0, 0, 0]` |
| 1 | C (Cytosine) | `[0, 1, 0, 0]` |
| 2 | G (Guanine) | `[0, 0, 1, 0]` |
| 3 | U (Uracil) | `[0, 0, 0, 1]` |

### Reward Shaping (Ng et al., 1999)

Potential-based shaping provides dense intermediate rewards without altering the optimal policy:

```
Φ(s) = correct_pairs / checked_pairs
F(s, a, s') = 0.1 × (0.99 × Φ(s') − Φ(s))
```

### Algorithm Comparison

| Feature | PPO | DQN |
|---------|-----|-----|
| Policy type | On-policy | Off-policy |
| Credit assignment | Multi-step (GAE) | 1-step TD bootstrap |
| Long-horizon (n>30) | Strong | Weak |
| Exploration | Entropy bonus | ε-greedy (1.0 → 0.08) |

**Preliminary finding:** PPO with GAE appears superior to DQN for sequential combinatorial problems with sparse terminal rewards and n > 30. Further experiments are in progress.

## Current Best Results (Eterna100-V2 Subset)

> These are preliminary results from ongoing experiments. Final results will be updated.

| Puzzle | Length | Type | PPO Best R_struct | Current Status |
|--------|--------|------|-------------------|----------------|
| P1 Simple Hairpin | 18 | Basic | 1.000 | Solved |
| P8 G-C Placement | 12 | Basic | 1.000 | Solved |
| P10 Frog Foot | 45 | Multi-stem | 0.860 | Improving |
| P13 Square | 67 | Nested | 0.760 | In progress |
| P54 7-Multiloop | 92 | Complex | 0.420 | Needs more training |

## Roadmap

- [ ] Complete grid search across all 3 weight configurations
- [ ] Run DQN experiments with optimized hyperparameters for all puzzles
- [ ] Deterministic evaluation of all trained models
- [ ] Test set evaluation (5 held-out targets)
- [ ] Final comparative analysis (PPO vs DQN)

## References

- Runge, F., Stoll, D., Falkner, S., & Hutter, F. (2019). *Learning to Design RNA*. ICLR.
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations*. ICML.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Lorenz, R., et al. (2011). *ViennaRNA Package 2.0*. Algorithms for Molecular Biology.

## Authors

- **Utku Bora Döke** — Department of Health Informatics
- **Ata Kamutay** — Department of Health Informatics

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
