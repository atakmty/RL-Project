"""
baseline_dqn.py — Phase A baseline training with DQN (structure-only reward).

DQN uses experience replay to propagate sparse terminal rewards.
The environment already provides Ng et al. (1999) potential-based
intermediate shaping, which gives DQN denser signals without altering
the optimal policy.

Usage (inside WSL with rlrna conda env):
    python baseline_dqn.py
"""

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment import LearnaEnv


class PerObjectiveLogger(BaseCallback):
    """Log per-objective metrics to TensorBoard at episode boundaries."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
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
    target_structure = "((((....))))"  # 12 nt hairpin

    # ----- Phase A: Structure-only reward -----
    env = LearnaEnv(
        target_structure,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
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
        print(f"  R_struct={info['r_struct']:.4f}  GC={info.get('gc_ratio', 0):.2f}  "
              f"MFE={info.get('mfe_val', 0):.2f}  seq={info['sequence']}")
    print()

    # ----- Train DQN -----
    print("Starting DQN training (Phase A — structure only)...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        seed=42,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=500,       # collect 500 transitions before first update
        batch_size=64,
        tau=0.005,                 # soft update coefficient for target network
        gamma=0.99,                # RL discount factor
        train_freq=4,              # update every 4 steps
        target_update_interval=250,
        exploration_fraction=0.3,  # explore for first 30% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    callback = PerObjectiveLogger()
    model.learn(total_timesteps=50_000, callback=callback)

    model.save("dqn_baseline_phaseA")
    print("Training complete. Model saved to dqn_baseline_phaseA.zip")
    print("Run 'tensorboard --logdir ./tensorboard_logs/' to view DQN vs PPO curves.")


if __name__ == "__main__":
    main()
