# 🧬 RNA Inverse Folding — Multi-Objective Deep Reinforcement Learning

> Solving the RNA inverse folding problem with PPO and DQN, optimizing for structural accuracy, GC-content, thermodynamic stability, and homopolymer avoidance simultaneously.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![ViennaRNA](https://img.shields.io/badge/Folding-ViennaRNA-orange.svg)](https://www.tbi.univie.ac.at/RNA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

The RNA inverse folding problem asks: *given a target secondary structure, find a nucleotide sequence that folds into it*. This is an NP-hard combinatorial optimization problem with a search space of 4ⁿ candidates.

Existing DRL frameworks like [LEARNA](https://github.com/automl/learna) optimize for structural match alone. Our pipeline extends this with **four simultaneous objectives**:

| Objective | Description | Weight |
|-----------|-------------|--------|
| **R_struct** | Structural accuracy (normalized Hamming distance) | α |
| **R_GC** | GC-content fitness — full reward in [40%, 60%] band | β |
| **P_homo** | Homopolymer penalty — penalizes runs > 4 identical bases | γ |
| **R_MFE** | Thermodynamic stability — normalized \|MFE\| per nucleotide | δ |

**Compound reward:** `R = α·R_struct + β·R_GC − γ·P_homo + δ·R_MFE`

### Key Results

- ✅ **100% success rate** on short structures (P1, P8)
- ✅ **48% success rate** on P10 Frog Foot (len=45, 3 hairpins) — up from 0%
- 📊 PPO consistently outperforms DQN on structures with n > 30
- 🔬 Partner-Aware observation space boosted DQN R_struct from 0.27 → 0.46 on P54 (92 nt)

## 🏗️ Architecture

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

> **Critical insight:** Phase A must include non-zero β and γ from step 0 to prevent GC-content collapse (see Section 5.5 of our report).

## 📁 Project Structure

```
RL-Project/
├── environment.py           # Gymnasium environment (Partner-Aware obs)
├── eterna100.py             # Eterna100-V2 dataset (15 train + 5 test)
├── train_multi_target.py    # Full training pipeline with curriculum
├── baseline.py              # PPO baseline (structure-only Phase A)
├── baseline_dqn.py          # DQN baseline (structure-only Phase A)
├── generate_final_report.py # PDF report generator (ReportLab)
├── run_grid_search.sh       # Grid search runner (3 weight configs)
├── scripts/
│   ├── analyze_ppo_vs_dqn.py       # PPO vs DQN comparison from TB logs
│   ├── analyze_dqn.py              # DQN-specific result analysis
│   ├── analyze_results.py          # General result analyzer
│   ├── evaluate_deterministic.py   # Deterministic evaluation (ε=0)
│   ├── colab_ppo_train.py          # Self-contained Colab training script
│   ├── check_logs.py               # TensorBoard log inspector
│   ├── preflight_check.py          # Environment verification
│   ├── quick_test.py               # Quick sanity check
│   └── generate_report_pdf.py      # Alternative report generator
├── docs/
│   ├── Ata_Kamutay_Utku_Bora_Döke_RL_Proposal.pdf  # Project proposal
│   └── Utku Bora Döke_Ata_Kamutay_RLmakale (1).pdf # Research paper
├── codes/                   # Legacy development scripts
├── models/                  # Trained model checkpoints (gitignored)
├── tensorboard_logs/        # Training logs (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (for ViennaRNA installation)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/RL-Project.git
cd RL-Project

# 2. Create conda environment
conda create -n rlrna python=3.10 -y
conda activate rlrna

# 3. Install ViennaRNA (MUST use conda, not pip)
conda install -c bioconda -c conda-forge viennarna -y
conda install -c conda-forge gsl -y

# 4. Install Python dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train PPO on all 15 targets (balanced config)
python train_multi_target.py --algo ppo --seed 42 --weight-config 0

# Train DQN on all 15 targets
python train_multi_target.py --algo dqn --seed 42 --weight-config 0

# Run full grid search (3 weight configs × PPO)
bash run_grid_search.sh

# Run baseline (single structure, structure-only reward)
python baseline.py
```

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

### Google Colab

For Colab training, use the self-contained script:

```python
# In a Colab cell:
!pip install stable-baselines3 gymnasium tensorboard viennarna
!python colab_ppo_train.py --algo ppo --seed 42 --weight-config 0
```

## 🔬 Technical Details

### Observation Space (7n + 10 dimensions)

| Component | Dims | Description |
|-----------|------|-------------|
| Sequence one-hot | 4n | A/C/G/U at each position |
| Target one-hot | 3n | ./(/\) at each position |
| Progress | 1 | current_step / n |
| Local target char | 3 | One-hot of target at current step |
| is_paired | 1 | Whether current position has a base-pair partner |
| partner_placed | 1 | Whether partner is already placed |
| partner_nucleotide | 4 | One-hot of partner's nucleotide |

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
| Long-horizon (n>30) | ✅ Strong | ⚠️ Weak |
| Exploration | Entropy bonus | ε-greedy (1.0 → 0.08) |

**Finding:** PPO with GAE is empirically superior for sequential combinatorial problems with sparse terminal rewards and n > 30.

## 📊 Benchmark Results (Eterna100-V2 Subset)

| Puzzle | Length | Type | PPO R_struct | Status |
|--------|--------|------|-------------|--------|
| P1 Simple Hairpin | 18 | Basic | 1.000 | ✅ Solved |
| P8 G-C Placement | 12 | Basic | 1.000 | ✅ Solved |
| P10 Frog Foot | 45 | Multi-stem | 0.860 | 🟡 Close |
| P13 Square | 67 | Nested | 0.760 | 🟡 Partial |
| P54 7-Multiloop | 92 | Complex | 0.420 | ❌ Insufficient |

## 📚 References

- Runge, F., Stoll, D., Falkner, S., & Hutter, F. (2019). *Learning to Design RNA*. ICLR.
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations*. ICML.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Lorenz, R., et al. (2011). *ViennaRNA Package 2.0*. Algorithms for Molecular Biology.

## 👥 Authors

- **Utku Bora Döke** — Department of Health Informatics
- **Ata Kamutay** — Department of Health Informatics

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
