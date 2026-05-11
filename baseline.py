"""
baseline.py — Phase A baseline training (structure-only reward).

Trains a PPO agent on a single short Eterna100 target using only R_struct.
Logs per-objective metrics to TensorBoard for monitoring.

Usage (inside WSL with rlrna conda env):
    python baseline.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import LearnaEnv


class PerObjectiveLogger(BaseCallback):
    """
    Custom SB3 callback that logs the 4 reward components separately
    to TensorBoard whenever an episode ends (terminal info is available).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        # SB3 wraps single envs in DummyVecEnv; infos is a list of dicts.
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 auto-resets vec envs and moves terminal info into "terminal_info"
            # (or keeps it at top level if the env just terminated on this step).
            terminal = info.get("terminal_info", info)
            if "r_struct" in terminal:
                self._episode_count += 1
                self.logger.record("episode/r_struct", terminal["r_struct"])
                self.logger.record("episode/r_gc", terminal["r_gc"])
                self.logger.record("episode/p_homo", terminal["p_homo"])
                self.logger.record("episode/r_mfe", terminal["r_mfe"])
                self.logger.record("episode/gc_ratio", terminal.get("gc_ratio", 0.0))
                self.logger.record("episode/is_success", terminal["is_success"])
                self.logger.record("episode/count", self._episode_count)
        return True


def main():
    # ----- Target Structure -----
    # A simple hairpin loop from Eterna100 for quick baseline validation.
    target_structure = "((((....))))"  # 12 nt hairpin

    # ----- Phase A: Structure-only reward -----
    env = LearnaEnv(
        target_structure,
        alpha=1.0,  # full structural weight
        beta=0.0,  # no GC penalty
        gamma=0.0,  # no homopolymer penalty
        delta=0.0,  # no MFE reward
    )

    print(f"Target structure: {target_structure}  (length={len(target_structure)})")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print()

    # ----- Sanity check: manual episode -----
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    total_reward = 0.0
    for i in range(len(target_structure)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(f"Random episode reward: {total_reward:.4f}")
    if "r_struct" in info:
        print(
            f"  R_struct={info['r_struct']:.4f}  GC={info.get('gc_ratio', 0):.2f}  "
            f"MFE={info.get('mfe_val', 0):.2f}  seq={info['sequence']}"
        )
    print()

    # ----- Train PPO -----
    print("Starting PPO training (Phase A — structure only)...")
    model = PPO(
        "MlpPolicy",  # flat Box obs → MlpPolicy (not MultiInputPolicy)
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        seed=42,
        n_steps=128,  # collect 128 steps per rollout (short episodes)
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,  # RL discount factor
    )

    callback = PerObjectiveLogger()
    model.learn(total_timesteps=50_000, callback=callback)

    model.save("ppo_baseline_phaseA")
    print("Training complete. Model saved to ppo_baseline_phaseA.zip")
    print("Run 'tensorboard --logdir ./tensorboard_logs/' to view training curves.")


if __name__ == "__main__":
    main()
