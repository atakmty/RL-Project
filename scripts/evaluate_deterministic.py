"""
evaluate_deterministic.py — Evaluate trained models with exploration disabled (deterministic=True)
This script measures the TRUE success rate of DQN and PPO by disabling epsilon-greedy 
and stochastic sampling, revealing what the models actually learned.
"""

import os
import glob
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import DQN, PPO
from environment import LearnaEnv
from eterna100 import ETERNA100_SELECTED

def evaluate_model(model_path, structure, is_dqn, episodes=20):
    env = LearnaEnv(structure, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    
    # Load model
    try:
        if is_dqn:
            model = DQN.load(model_path, env=env, custom_objects={'exploration_rate': 0.0})
        else:
            model = PPO.load(model_path, env=env)
    except Exception as e:
        # Ignore old models with different observation spaces
        return 0.0, 0.0
        
    successes = 0
    r_structs = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # deterministic=True turns OFF epsilon-greedy for DQN and stochastic sampling for PPO
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        successes += info['is_success']
        r_structs.append(info['r_struct'])
        
    return (successes / episodes) * 100.0, np.mean(r_structs)

def main():
    print("=" * 90)
    print("  RNA Inverse Folding — DETERMINISTIC EVALUATION (Exploration OFF)")
    print("=" * 90)
    
    # Find all trained models
    model_dir = "./models"
    
    print(f"  {'Puzzle':<8} {'Len':<5} | {'DQN R_str':>9} {'DQN Succ':>9} | {'PPO R_str':>9} {'PPO Succ':>9}")
    print("-" * 85)
    
    dqn_solved = 0
    ppo_solved = 0
    
    for pid, name, structure in ETERNA100_SELECTED:
        # Find latest DQN model (v2)
        dqn_paths = sorted([p for p in glob.glob(os.path.join(model_dir, f"dqn_puzzle{pid}_*")) if not p.endswith('.zip')])
        # Find latest PPO model (Run 3)
        ppo_paths = sorted([p for p in glob.glob(os.path.join(model_dir, f"ppo_puzzle{pid}_*")) if not p.endswith('.zip')])
        
        dqn_succ, dqn_rs = 0.0, 0.0
        ppo_succ, ppo_rs = 0.0, 0.0
        
        if dqn_paths:
            dqn_succ, dqn_rs = evaluate_model(dqn_paths[-1], structure, is_dqn=True)
            if dqn_succ >= 50.0: dqn_solved += 1
            
        if ppo_paths:
            ppo_succ, ppo_rs = evaluate_model(ppo_paths[-1], structure, is_dqn=False)
            if ppo_succ >= 50.0: ppo_solved += 1
            
        dqn_mark = "✅" if dqn_succ >= 50.0 else "❌"
        ppo_mark = "✅" if ppo_succ >= 50.0 else "❌"
        
        print(f"  P{pid:<7} {len(structure):<5} | {dqn_rs:>9.3f} {dqn_succ:>8.1f}% {dqn_mark} | {ppo_rs:>9.3f} {ppo_succ:>8.1f}% {ppo_mark}")

    print("-" * 85)
    print(f"  Total Solved (>50% success):  DQN: {dqn_solved}/15  |  PPO: {ppo_solved}/15")
    print("=" * 90)

if __name__ == "__main__":
    main()
