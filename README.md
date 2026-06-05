# RNA Inverse Folding — Multi-Objective Deep Reinforcement Learning

> **Work in Progress** — actively developed; results and code are updated regularly.

Solving the RNA inverse folding problem with PPO and DQN, optimizing **structural accuracy,
GC-content, thermodynamic stability, and homopolymer avoidance simultaneously** — and satisfying
all four with the RL policy itself, **with no post-hoc repair**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![ViennaRNA](https://img.shields.io/badge/Folding-ViennaRNA-orange.svg)](https://www.tbi.univie.ac.at/RNA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The RNA inverse folding problem asks: *given a target secondary structure, find a nucleotide
sequence that folds into it*. This is an NP-hard combinatorial optimization problem with a search
space of **4ⁿ** candidates.

Existing DRL frameworks like [LEARNA](https://github.com/automl/learna) optimize for structural
match alone. We extend this to **four simultaneous objectives**:

| Objective | Description | Weight |
|-----------|-------------|--------|
| **R_struct** | Structural accuracy (normalized Hamming distance) | α |
| **R_GC** | GC-content fitness — full reward in the [40%, 60%] band | β |
| **P_homo** | Homopolymer penalty — penalizes runs > 4 identical bases | γ |
| **R_MFE** | Thermodynamic stability — normalized \|MFE\| per nucleotide | δ |

**Compound reward:** `R = α·R_struct + β·R_GC − γ·P_homo + δ·R_MFE + B(φ)`

A purely additive reward lets the policy *trade* objectives (e.g. buying structure with
out-of-band GC). We therefore **shape the reward so the simultaneously-satisfying sequence is its
optimum** (see [Reward Design](#reward-design)): the MFE term saturates at a stability threshold,
the GC-band penalty is steepened, and a **multiplicative joint-satisfaction bonus `B`** is near
zero unless all four objectives are met at once.

## Key Results (seed 44, deterministic evaluation, no post-hoc repair)

| Metric | PPO | DQN |
|--------|-----|-----|
| **Fully solved** (all 4 objectives) | **4/15** | 0/15 |
| Mean R_struct | **0.80** | 0.54 |
| Mean R_struct, long targets (n ≥ 40) | **0.71** | 0.46 |
| GC content within [0.40, 0.60] | **15/15** | 6/15 |
| Mean closeness | **0.90** | 0.60 |

- **PPO** solves **P1, P8, P26, P30** with all four objectives satisfied by the policy itself,
  and keeps GC in-band on **all 15** targets (the GC-drift of structure-only training is gone).
- **DQN** reaches no full solution; its structure collapses once the biophysical objectives fully
  activate (its best is P30 at R_struct = 0.935, one structural step short).
- The PPO advantage is largest on long targets (n ≥ 40), consistent with DQN's O(n)
  credit-assignment bottleneck.
- **Partner-aware** observations raise DQN's structural reward by up to **70%** on long targets
  (0.27 → 0.46 on the 92-nt P54; 0.55 → 0.82 on the 40-nt P65).

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Eterna100   │────▶│  LearnaEnv       │────▶│  PPO / DQN    │
│  Benchmark   │     │  (Gymnasium)     │     │  (SB3)        │
│ (15+5 targets)│    │                  │     │               │
└──────────────┘     │  • One-hot obs   │     │  • [128,128]  │
                     │  • Partner-aware │     │  • ent=0.05   │
                     │  • Reward shaping│     │  • GAE / Replay│
                     └────────┬─────────┘     └───────┬───────┘
                              │                       │
                     ┌────────▼─────────┐     ┌───────▼───────┐
                     │  ViennaRNA       │     │  TensorBoard  │
                     │  MFE Folding     │     │  Logging      │
                     └──────────────────┘     └───────────────┘
```

### Reward Design

The compound reward is built so that the policy itself satisfies all four objectives:

- **R_struct** = 1 − (Hamming distance / n).
- **R_GC** — a *signed, steepened* band reward: 1.0 inside [0.40, 0.60] and decreasing to a floor
  of −1 just outside it, so structure cannot be cheaply "bought" with out-of-band GC.
- **P_homo** — a terminal homopolymer penalty, **plus a dense per-step penalty** (`--homo-step-scale`,
  default 0.05) that discourages extending a run during generation.
- **R_MFE** — **saturated** at a stability threshold: `R_MFE = min(1, (|MFE|/n) / τ)`, τ = 0.30, so
  the agent is not rewarded for over-stabilizing (which would drive GC out of band).
- **B(φ)** — joint-satisfaction bonus `B = λ·(s_struct·s_GC·s_homo·s_MFE) + ρ·𝟙[all four gates pass]`
  (λ = ρ = 0.5). The product is near zero unless every objective is met, so partial solutions earn
  almost nothing.

Intermediate steps use **potential-based reward shaping** (Ng et al., 1999) — see
[Reward Shaping](#reward-shaping-ng-et-al-1999).

### Three-Phase Adaptive Weight Curriculum

| Phase | Steps | Strategy |
|-------|-------|----------|
| **A** (0–15%) | Structure-dominant | α=1.0, β=β*, γ=γ*, δ=0 |
| **B** (15–70%) | Linear ramp | Smooth transition to target weights |
| **C** (70–100%) | Joint optimization | All weights held at target values |

> **Critical insight:** Phase A must keep β and γ active from step 0 (combined with PPO's entropy
> bonus, `ent_coef = 0.05`) to prevent an irreversible all-GC collapse.

## Project Structure

```
RL-Project/
├── environment.py              # Gymnasium env (Partner-Aware obs, 4-objective reward + joint bonus)
├── eterna100.py                # Eterna100 dataset (15 train + 5 test targets)
├── train_multi_target.py       # Main training pipeline (curriculum, weight scheduling, CLI flags)
├── run_multiseed.sh            # Train all puzzles for both algorithms at a given seed/scale
├── scripts/
│   ├── evaluate_deterministic.py  # Four-objective deterministic evaluation (ε=0)
│   ├── analyze_ppo_vs_dqn.py   # PPO vs DQN comparison from TensorBoard logs
│   └── analyze_dqn.py          # DQN-specific result analysis
├── models/                     # Trained model checkpoints (gitignored)
├── tensorboard_logs/           # Training logs (gitignored)
├── environment.yml             # Conda environment definition
├── requirements.txt            # pip dependencies
├── LICENSE
└── README.md
```

> `run_grid_search_{ppo,dqn}.sh` remains in the repo but is **not part of the current pipeline**: we
> report a single (balanced) weight configuration, and full solutions are produced by the policy
> itself — there is **no post-hoc repair**.

### What each file does

| File | Purpose |
|------|---------|
| `environment.py` | The `LearnaEnv` Gymnasium environment. The agent places one nucleotide (A/C/G/U) per step; at the final step ViennaRNA folds the sequence and returns the 4-objective reward plus the joint-satisfaction bonus. Intermediate steps use potential-based shaping. |
| `eterna100.py` | The 20 selected Eterna100 target structures (15 train + 5 held-out test). Each target is a dot-bracket string like `((((((......))))))`. |
| `train_multi_target.py` | Main training script. Trains a **separate PPO or DQN specialist per puzzle**, with the 3-phase curriculum and adaptive episode scaling for longer sequences. |
| `scripts/evaluate_deterministic.py` | Loads saved models and evaluates them deterministically (PPO `deterministic=True`, DQN ε=0) against the four-objective criteria. Best-of-seeds via `--seeds` (a puzzle counts as solved if any seed solves it; the same seed set is applied to both algorithms). |

## Quick Start

### Prerequisites

- Python 3.10+
- Conda (for ViennaRNA installation)

### Installation

> **Platform note — ViennaRNA is Linux/macOS only.** The bioconda channel does not publish
> `viennarna` for `win-64`, so `conda env create` fails on native Windows with
> `PackagesNotFoundError: viennarna`. **Windows users must use WSL2** (instructions below).
> macOS and Linux users can use the standard install.

#### Linux / macOS

```bash
git clone https://github.com/atakmty/RL-Project.git
cd RL-Project
conda env create -f environment.yml
conda activate rlrna
```

> `environment.yml` installs Python 3.10, ViennaRNA, GSL, and all pip packages. No separate
> `pip install` needed.

#### Windows (via WSL2)

**1. Install WSL2 + Ubuntu.** In an **administrator PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Reboot when prompted; Ubuntu will ask you to create a Linux username and password.

**2. Open Ubuntu** and install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   # accept defaults; say yes to conda init
```

Close and reopen the Ubuntu terminal — your prompt should start with `(base)`.

**3. Clone and create the environment** (same as Linux):

```bash
git clone https://github.com/atakmty/RL-Project.git
cd RL-Project
conda env create -f environment.yml
conda activate rlrna
```

> **Tip:** `conda env create` is a **one-time** setup. Afterwards each session only needs
> `cd ~/RL-Project && conda activate rlrna`.

### Training

The defaults reproduce the reported run (seed 44, `--homo-step-scale 0.05`, balanced config):

```bash
# Train all 15 targets — one specialist per puzzle
python train_multi_target.py --algo ppo
python train_multi_target.py --algo dqn
```

> Each run trains **15 separate models** (one per puzzle). Models are saved under `models/`, logs
> under `tensorboard_logs/`. Only `--algo` differs between the two runs; everything else defaults to
> the reported configuration.

**Multi-seed (no repair).** The per-target specialist setup admits a restart budget. Train extra
seeds and take the best per puzzle (the same seed set is applied to both algorithms, keeping the
comparison fair):

```bash
python train_multi_target.py --algo ppo --seed 43
python train_multi_target.py --algo dqn --seed 43
python scripts/evaluate_deterministic.py --seeds 43,44 --csv combined.csv
```

### Key CLI flags (`train_multi_target.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--algo` | `ppo` | `ppo` or `dqn` |
| `--seed` | `44` | random seed (the reported run) |
| `--homo-step-scale` | `0.05` | dense per-step homopolymer penalty strength |
| `--weight-config` | `0` | weight config index (see below) |
| `--mfe-tau` | `0.3` | MFE/nt stability threshold (the MFE reward saturates here) |
| `--gc-falloff` | `0.2` | smaller = steeper out-of-band GC penalty |
| `--solve-bonus` / `--solve-jackpot` | `0.5` / `0.5` | joint-satisfaction bonus scale / completion reward |
| `--alpha-floor` | `0.0` | soft floor on α in Phase B/C (0 = off; ~0.7 protects DQN's structure) |
| `--puzzles` | all | comma-separated subset, e.g. `"1,8,26,30"` |
| `--test` | off | train the 5 held-out test puzzles instead of the 15 training puzzles |

### Weight Configurations

We report the **balanced** configuration. (Structure-heavy / thermodynamic-focused settings
produced near-identical trajectories in preliminary runs; a differentiated grid search is future
work.)

| Config | α | β | γ | δ | Strategy |
|--------|---|---|---|---|----------|
| **0** | 0.5 | 0.2 | 0.2 | 0.2 | **Balanced (reported)** |
| 1 | 0.6 | 0.15 | 0.1 | 0.15 | Structure-heavy |
| 2 | 0.4 | 0.2 | 0.15 | 0.25 | Thermodynamic-focused |

### Evaluation

The evaluator defaults match training (seed 44, scale 0.05):

```bash
# Four-objective deterministic evaluation — no post-hoc repair
python scripts/evaluate_deterministic.py --csv results_seed44_v2.csv

# Compare PPO vs DQN from TensorBoard logs
python scripts/analyze_ppo_vs_dqn.py
```

A target is **structurally solved** when R_struct = 1, and **fully solved** when it additionally
satisfies GC ∈ [0.40, 0.60], longest run ≤ 4, and |MFE|/n ≥ 0.30 — all produced by the policy
itself.

### TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs/
```

## Technical Details

### Search Space

For a target of length *n* there are **4ⁿ** possible sequences. The longest puzzle, P54 (n=92),
has 4⁹² ≈ 2.4 × 10⁵⁵ candidates, making brute force infeasible.

### Observation Space (7n + 10 dimensions)

A **base encoding** of 7n + 1 dimensions plus a **Partner-Aware extension** of 9 dimensions.

**Base encoding (7n + 1):**

| Component | Dims | Description |
|-----------|------|-------------|
| Sequence one-hot | 4n | A/C/G/U at each placed position |
| Target one-hot | 3n | ./(/) at each position |
| Progress | 1 | current_step / n |

**Partner-Aware extension (+9):**

| Component | Dims | Description |
|-----------|------|-------------|
| Local target char | 3 | One-hot of target structure at current step: `.` / `(` / `)` |
| is_paired | 1 | 1.0 if the current position has a base-pair partner |
| partner_placed | 1 | 1.0 if the partner's nucleotide has already been placed |
| partner_nucleotide | 4 | One-hot of the partner's nucleotide (A/C/G/U), zeros if not yet placed |

> These dimensions expose complementarity at closing-bracket positions, so the agent can place a
> complementary base (e.g. `G` for a partner `C`) without recovering partner identity from the full
> sequence vector.

### Action Space

**Discrete(4)** — one nucleotide per step: `0=A`, `1=C`, `2=G`, `3=U`.

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
| Exploration | Entropy bonus (ent_coef = 0.05) | ε-greedy (1.0 → 0.08) |
| Network | [128, 128] | [256, 256] |

### Fully-Solved Canonical Sequences (PPO, seed 44)

| Puzzle | Len | GC | Max run | \|MFE\|/n | Sequence |
|--------|-----|----|---------|-----------|----------|
| P1  Simple Hairpin        | 18 | 0.56 | 2 | 0.46 | `GGCGCCAAUUAAGGUGCU` |
| P8  G-C Placement         | 12 | 0.58 | 2 | 0.38 | `GGCUUAAGGCCA` |
| P26 Stickshift            | 26 | 0.54 | 3 | 0.37 | `AAGGGCCGCGAUUUACGACGGCUUAA` |
| P30 Corner Bulge Training | 31 | 0.52 | 4 | 0.45 | `UUUGAGAGCCCCAAAGGGGCAAGAUCUCGAA` |

## Authors

- **Utku Bora Döke** — Department of Health Informatics
- **Ata Kamutay** — Department of Health Informatics

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
