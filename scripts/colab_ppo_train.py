"""
colab_ppo_train.py — Single-file PPO training script for Google Colab Pro.
Contains Eterna100 dataset, Partner-Aware Environment, and Training Pipeline.

Instructions for Colab:
1. Open a new notebook and select a high-RAM CPU or GPU runtime.
2. Upload this file to the Colab environment.
3. Install dependencies in a cell:
   !pip install stable-baselines3 gymnasium tensorboard viennarna
4. Run the training in the next cell:
   !python colab_ppo_train.py --algo ppo --seed 42 --weight-config 0
"""

import os
import re
import time
import sys
import argparse
import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False
    print("WARNING: ViennaRNA (RNA module) not found. Install via: pip install viennarna")
    print("         The environment will run in MOCK mode (random folding).")

# ======================================================================
# 1. Eterna100 Data
# ======================================================================
ETERNA100_SELECTED = [
    (1,  "Simple Hairpin", "((((((......))))))"),
    (8,  "G-C Placement", "((((...))))."),
    (10, "Frog Foot", "..........((((....))))((((....))))((((...))))"),
    (13, "square", "((((((((((((((((((((((((((...))))))....)))))))....))))))....)))))))"),
    (15, "Small and Easy 6", "(((((.....))..((.........)))))" ),
    (23, "Shortie 4", "((....)).((....))" ),
    (25, "The Ministry", "(((.....(((.....(((.....(((.....(((........))).))).))).))).)))"),
    (26, "stickshift", "..((((((((.....)).)))))).." ),
    (30, "Corner bulge training", ".(((((((((((...)))))....))))))." ),
    (40, "Tripod5", "..((((((((.....))))((((.....)))))))).." ),
    (41, "Shortie 6", "((....)).((....)).((....)).((....))" ),
    (45, "[CloudBeta] 5 Adjacent Stack Multi-Branch Loop", "(((((((((....))))(((((....)))))(((((....)))))((((....)))))))))." ),
    (47, "Misfolded Aptamer", "((((......(((((...))).((....)).........)).....))))" ),
    (54, "7 multiloop", "(((((((((....)))))(((((....)))))(((((....)))))(((((....)))))(((((....)))))(((((....)))))))))" ),
    (65, "Branching Loop", ".(((((........)((((....))))..))))......." ),
    (3,  "Prion Pseudoknot", "((((((.((((....))))))).))).........." ),
    (11, "InfoRNA test 16", "((((((.((((((((....))))).)).).))))))" ),
    (20, "InfoRNA bulge test 9", "(((((((.(.(.(.(((((((....)))))))))))))))))" ),
    (33, "Worm 1", ".......(.(.(.(.(.((.((.(....).)).)).).).).).)" ),
    (59, "hard Y", ".....((((((.((((((((....))))))).)))((((((((((....))))))))))))))....." ),
]

TRAIN_TARGETS = ETERNA100_SELECTED[:15]

def get_train_structures():
    return [(pid, name, struct) for pid, name, struct in TRAIN_TARGETS]

# ======================================================================
# 2. Environment (Partner-Aware)
# ======================================================================
def mock_fold(sequence: str):
    n = len(sequence)
    return '.' * n, -0.5 * n

class LearnaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target_structure: str, alpha: float=0.5, beta: float=0.2, gamma: float=0.1, delta: float=0.2, discount: float=0.99):
        super().__init__()
        self.target_structure = target_structure
        self.n = len(target_structure)
        self.action_space = spaces.Discrete(4)
        
        # 7n + 10 dims for partner awareness
        obs_size = 4 * self.n + 3 * self.n + 1 + 9
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.nucleotides = ["A", "C", "G", "U"]
        self._nuc_to_idx = {c: i for i, c in enumerate(self.nucleotides)}
        self._struct_to_idx = {".": 0, "(": 1, ")": 2}

        self._target_onehot = np.zeros(3 * self.n, dtype=np.float32)
        for i, ch in enumerate(self.target_structure):
            idx = self._struct_to_idx.get(ch, 0)
            self._target_onehot[3 * i + idx] = 1.0

        self._pair_table = self._build_pair_table(target_structure)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.discount = discount
        self.shaping_scale = 0.1
        self.current_seq = ""
        self.current_step = 0
        self.last_potential = 0.0

    @staticmethod
    def _build_pair_table(structure: str):
        stack, pairs = [], {}
        for i, ch in enumerate(structure):
            if ch == "(": stack.append(i)
            elif ch == ")":
                if stack:
                    j = stack.pop()
                    pairs[i], pairs[j] = j, i
        return pairs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_seq = ""
        self.current_step = 0
        self.last_potential = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        seq_onehot = np.zeros(4 * self.n, dtype=np.float32)
        for i, ch in enumerate(self.current_seq):
            seq_onehot[4 * i + self._nuc_to_idx[ch]] = 1.0

        progress = np.array([self.current_step / self.n], dtype=np.float32)
        
        local_context = np.zeros(9, dtype=np.float32)
        if self.current_step < self.n:
            ch = self.target_structure[self.current_step]
            local_context[self._struct_to_idx.get(ch, 0)] = 1.0
            
            if self.current_step in self._pair_table:
                local_context[3] = 1.0
                partner = self._pair_table[self.current_step]
                if partner < self.current_step:
                    local_context[4] = 1.0
                    nuc_idx = self._nuc_to_idx.get(self.current_seq[partner], 0)
                    local_context[5 + nuc_idx] = 1.0

        return np.concatenate([seq_onehot, self._target_onehot, progress, local_context])

    def _potential_function(self) -> float:
        if not self.current_seq or not self._pair_table: return 0.0
        valid_pairs = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
        placed_len = len(self.current_seq)
        correct, total_checked = 0, 0
        for i in range(placed_len):
            if i in self._pair_table:
                partner = self._pair_table[i]
                if partner < placed_len:
                    total_checked += 1
                    if (self.current_seq[i], self.current_seq[partner]) in valid_pairs:
                        correct += 1
        return correct / total_checked if total_checked > 0 else 0.0

    def step(self, action):
        self.current_seq += self.nucleotides[action]
        self.current_step += 1
        terminated = self.current_step >= self.n
        
        if not terminated:
            new_potential = self._potential_function()
            reward = self.shaping_scale * (self.discount * new_potential - self.last_potential)
            self.last_potential = new_potential
            return self._get_obs(), reward, terminated, False, {}
        else:
            if HAS_VIENNARNA:
                fc = RNA.fold_compound(self.current_seq)
                mfe_struct, mfe_val = fc.mfe()
            else:
                mfe_struct, mfe_val = mock_fold(self.current_seq)

            hamming = sum(1 for a, b in zip(mfe_struct, self.target_structure) if a != b)
            r_struct = max(0.0, 1.0 - (hamming / self.n))
            
            gc_ratio = (self.current_seq.count("G") + self.current_seq.count("C")) / self.n
            r_gc = max(0.0, 1.0 - (max(0.0, abs(gc_ratio - 0.5) - 0.1) / 0.4))
            
            p_homo = sum(max(0, m.end() - m.start() - 4) for m in re.finditer(r"(.)\1+", self.current_seq)) / self.n
            r_mfe = abs(mfe_val) / self.n

            terminal_reward = self.alpha * r_struct + self.beta * r_gc - self.gamma * p_homo + self.delta * r_mfe
            reward = terminal_reward + self.shaping_scale * (0.0 - self.last_potential)

            info = {
                "r_struct": r_struct, "r_gc": r_gc, "p_homo": p_homo, "r_mfe": r_mfe,
                "mfe_val": mfe_val, "gc_ratio": gc_ratio, "sequence": self.current_seq,
                "is_success": float(hamming == 0),
            }
            return self._get_obs(), reward, True, False, info

# ======================================================================
# 3. Training Pipeline
# ======================================================================
class AdaptiveWeightScheduler:
    def __init__(self, total_timesteps: int, target_alpha=0.5, target_beta=0.2, target_gamma=0.1, target_delta=0.2):
        self.total = total_timesteps
        self.ta, self.tb, self.tg, self.td = target_alpha, target_beta, target_gamma, target_delta
        self.phase_a_end = int(0.30 * total_timesteps)
        self.phase_b_end = int(0.70 * total_timesteps)

    def get_weights(self, current_step: int):
        if current_step <= self.phase_a_end: return 1.0, 0.0, 0.0, 0.0
        elif current_step <= self.phase_b_end:
            p = (current_step - self.phase_a_end) / (self.phase_b_end - self.phase_a_end)
            return 1.0 - p*(1.0 - self.ta), p*self.tb, p*self.tg, p*self.td
        else: return self.ta, self.tb, self.tg, self.td

    def get_phase_name(self, current_step: int) -> str:
        if current_step <= self.phase_a_end: return "A (struct)"
        elif current_step <= self.phase_b_end: return "B (ramp)"
        else: return "C (joint)"

class TrainingCallback(BaseCallback):
    def __init__(self, scheduler: AdaptiveWeightScheduler, target_name: str, total_timesteps: int, print_interval: int = 2000, verbose=0):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.target_name = target_name
        self.total_timesteps = total_timesteps
        self.print_interval = print_interval
        self._episode_count = 0
        self._start_time = None
        self._last_print_step = 0
        self._recent_r_struct = deque(maxlen=50)
        self._recent_success = deque(maxlen=50)

    def _on_training_start(self): self._start_time = time.time()

    def _on_step(self) -> bool:
        alpha, beta, gamma, delta = self.scheduler.get_weights(self.num_timesteps)
        env = self.training_env.envs[0]
        env.alpha, env.beta, env.gamma, env.delta = alpha, beta, gamma, delta

        for info in self.locals.get("infos", []):
            if "r_struct" in info:
                self._episode_count += 1
                self._recent_r_struct.append(info["r_struct"])
                self._recent_success.append(info["is_success"])
                self.logger.record("episode/r_struct", info["r_struct"])
                self.logger.record("episode/is_success", info["is_success"])

        if self.num_timesteps - self._last_print_step >= self.print_interval:
            pct = 100.0 * self.num_timesteps / max(1, self.total_timesteps)
            phase = self.scheduler.get_phase_name(self.num_timesteps)
            avg_struct = np.mean(self._recent_r_struct) if self._recent_r_struct else 0.0
            succ_rate = np.mean(self._recent_success) if self._recent_success else 0.0
            print(f"    [{pct:5.1f}%] Phase {phase} | R_struct={avg_struct:.3f} | Succ={succ_rate:.1%}")
            self._last_print_step = self.num_timesteps
        return True

def train_single_target(algo_name, target_id, target_name, structure, seed, total_timesteps, weight_config, log_dir):
    env = LearnaEnv(structure, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    scheduler = AdaptiveWeightScheduler(total_timesteps, *weight_config)
    run_name = f"{algo_name}_puzzle{target_id}_seed{seed}_colab"

    if algo_name == "ppo":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, seed=seed, n_steps=128, batch_size=64, n_epochs=10, learning_rate=3e-4)
    else:
        model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_dir, seed=seed, learning_rate=5e-4, buffer_size=50_000, learning_starts=1000, batch_size=128, exploration_fraction=0.6, policy_kwargs=dict(net_arch=[256, 256]))

    callback = TrainingCallback(scheduler, target_name, total_timesteps, print_interval=max(2000, total_timesteps // 10))
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=run_name)
    
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", run_name)
    model.save(save_path)
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episodes", type=int, default=0)
    parser.add_argument("--weight-config", type=int, default=0)
    args = parser.parse_args()

    WEIGHT_CONFIGS = [(0.5, 0.2, 0.1, 0.2), (0.6, 0.15, 0.1, 0.15), (0.4, 0.2, 0.15, 0.25)]
    wc = WEIGHT_CONFIGS[args.weight_config]
    train_targets = get_train_structures()

    min_ep = 5000 if args.algo == "dqn" else 3000
    if args.min_episodes > 0: min_ep = args.min_episodes

    print(f"=== Starting Colab Training ({args.algo.upper()}) ===")
    print(f"Weight Config: {wc} | Min Episodes: {min_ep}")

    for idx, (pid, name, struct) in enumerate(train_targets, 1):
        ts = len(struct) * min_ep
        print(f"\n[{idx}/15] Puzzle {pid}: {name} (len={len(struct)}, steps={ts})")
        train_single_target(args.algo, pid, name, struct, args.seed, ts, wc, "./tensorboard_logs/")

    print("\n=== All Training Complete! ===")

if __name__ == "__main__":
    main()
