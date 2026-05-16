"""
quick_test.py — Quick sanity check: run 10 random + 10 trained episodes.
Verifies the environment works correctly after one-hot fix.

Usage: python quick_test.py
"""
from environment import LearnaEnv
import numpy as np

target = "((((....))))"
env = LearnaEnv(target, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)

print(f"Target: {target} (len={len(target)})")
print(f"Obs space shape: {env.observation_space.shape}")
print(f"Expected: ({7 * len(target) + 1},) = ({7*12+1},)")
print()

# Run 10 random episodes
r_structs = []
for ep in range(10):
    obs, info = env.reset()
    total_r = 0
    for _ in range(len(target)):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_r += reward
    r_structs.append(info["r_struct"])

print(f"Random baseline (10 episodes):")
print(f"  R_struct values: {[round(x,3) for x in r_structs]}")
print(f"  Mean R_struct:   {np.mean(r_structs):.4f}")
print(f"  Max  R_struct:   {np.max(r_structs):.4f}")
print()

# Check observation encoding
obs, _ = env.reset()
print(f"Obs after reset (first 20 values): {obs[:20]}")  # Should be all zeros in seq part
action = 0  # Place 'A' at position 0
obs, r, _, _, _ = env.step(action)
print(f"Obs after placing A (first 8 values): {obs[:8]}")  # [1,0,0,0, 0,0,0,0] expected
action = 1  # Place 'C' at position 1
obs, r, _, _, _ = env.step(action)
print(f"Obs after placing C (next 8 values): {obs[:8]}")   # [1,0,0,0, 0,1,0,0] expected
print()
print("If you see [1,0,0,0] for A and [0,1,0,0] for C, one-hot encoding works correctly!")
