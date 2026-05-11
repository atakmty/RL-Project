"""
preflight_check.py — Pre-flight check before running experiments.
Verifies all imports, ViennaRNA, environment, dataset, and training pipeline.

Usage: python preflight_check.py
"""
import sys
import time

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

def check(name, func):
    try:
        result = func()
        print(f"  {PASS} {name}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        return False

print("=" * 70)
print("  Pre-flight Check — RNA Inverse Folding Pipeline")
print("=" * 70)
print()

all_ok = True

# 1. Python version
print("[1/7] Python")
print(f"  {PASS} Version: {sys.version.split()[0]}")
print(f"  {PASS} Path: {sys.executable}")
print()

# 2. Core dependencies
print("[2/7] Core Dependencies")
all_ok &= check("numpy", lambda: __import__("numpy").__version__)
all_ok &= check("gymnasium", lambda: __import__("gymnasium").__version__)
all_ok &= check("stable_baselines3", lambda: __import__("stable_baselines3").__version__)
all_ok &= check("torch", lambda: f"{__import__('torch').__version__} (CUDA: {__import__('torch').cuda.is_available()})")
all_ok &= check("tensorboard", lambda: __import__("tensorboard").__version__)
print()

# 3. ViennaRNA
print("[3/7] ViennaRNA")
all_ok &= check("RNA module", lambda: f"version {__import__('RNA').__version__}")
# Quick fold test
def test_fold():
    import RNA
    seq = "GCGCAAAAGCGC"
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    return f"fold('{seq}') = '{struct}' (MFE={mfe:.2f})"
all_ok &= check("RNA.fold_compound", test_fold)
print()

# 4. Environment
print("[4/7] Environment (LearnaEnv)")
def test_env():
    from environment import LearnaEnv
    env = LearnaEnv("((((....))))")
    obs, _ = env.reset()
    assert obs.shape == (85,), f"Expected (85,), got {obs.shape}"
    # Run one full episode
    for _ in range(12):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert "r_struct" in info, "Missing r_struct in terminal info"
    assert "is_success" in info, "Missing is_success in terminal info"
    return f"obs={obs.shape}, r_struct={info['r_struct']:.3f}, seq={info['sequence']}"
all_ok &= check("LearnaEnv episode", test_env)

# Test observation encoding
def test_obs_encoding():
    from environment import LearnaEnv
    env = LearnaEnv("((..))")
    obs, _ = env.reset()
    assert all(obs[:24] == 0), "Empty positions should be all zeros"
    obs, _, _, _, _ = env.step(0)  # Place A
    assert obs[0] == 1.0 and obs[1] == 0.0, "A should be [1,0,0,0]"
    obs, _, _, _, _ = env.step(2)  # Place G
    assert obs[4] == 0.0 and obs[6] == 1.0, "G should be [0,0,1,0]"
    return "one-hot encoding verified"
all_ok &= check("Observation encoding", test_obs_encoding)
print()

# 5. Dataset
print("[5/7] Eterna100 Dataset")
def test_dataset():
    from eterna100 import get_train_structures, get_test_structures
    train = get_train_structures()
    test = get_test_structures()
    assert len(train) == 15, f"Expected 15 train targets, got {len(train)}"
    assert len(test) == 5, f"Expected 5 test targets, got {len(test)}"
    lengths = [len(s) for _, _, s in train]
    return f"{len(train)} train (len {min(lengths)}-{max(lengths)}), {len(test)} test"
all_ok &= check("Train/Test split", test_dataset)
print()

# 6. Weight Scheduler
print("[6/7] Adaptive Weight Scheduler")
def test_scheduler():
    sys.path.insert(0, ".")
    from train_multi_target import AdaptiveWeightScheduler
    sched = AdaptiveWeightScheduler(100_000, 0.5, 0.2, 0.1, 0.2)
    # Phase A
    a, b, g, d = sched.get_weights(0)
    assert (a, b, g, d) == (1.0, 0.0, 0.0, 0.0), f"Phase A wrong: {a},{b},{g},{d}"
    # Phase B midpoint
    a, b, g, d = sched.get_weights(50_000)
    assert 0.5 < a < 1.0, f"Phase B alpha wrong: {a}"
    assert b > 0, "Phase B beta should be > 0"
    # Phase C
    a, b, g, d = sched.get_weights(100_000)
    assert (a, b, g, d) == (0.5, 0.2, 0.1, 0.2), f"Phase C wrong: {a},{b},{g},{d}"
    return f"A→(1,0,0,0)  B→({a:.1f},ramp)  C→(0.5,0.2,0.1,0.2)"
all_ok &= check("Phase transitions", test_scheduler)
print()

# 7. Quick training smoke test (very short)
print("[7/7] Training Smoke Test (500 steps)")
def test_training():
    from environment import LearnaEnv
    from stable_baselines3 import PPO
    env = LearnaEnv("((..))", alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=32, batch_size=32, seed=42)
    t0 = time.time()
    model.learn(total_timesteps=500)
    elapsed = time.time() - t0
    return f"PPO 500 steps in {elapsed:.1f}s ({500/elapsed:.0f} fps)"
all_ok &= check("PPO smoke test", test_training)
print()

# Summary
print("=" * 70)
if all_ok:
    print(f"  {PASS} ALL CHECKS PASSED — Ready to run experiments!")
    print()
    print("  Run training with:")
    print("    python train_multi_target.py --algo ppo --seed 42 --weight-config 0")
    print()
    print("  Full 18-experiment grid:")
    print("    for algo in ppo dqn; do")
    print("      for config in 0 1 2; do")
    print("        for seed in 42 123 456; do")
    print('          python train_multi_target.py --algo $algo --seed $seed --weight-config $config')
    print("        done")
    print("      done")
    print("    done")
else:
    print(f"  {FAIL} SOME CHECKS FAILED — Fix the errors above before training.")
print("=" * 70)
