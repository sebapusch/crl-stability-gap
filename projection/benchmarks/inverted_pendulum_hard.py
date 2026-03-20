import numpy as np
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv

class InvertedPendulumHard(InvertedPendulumEnv):
    def __init__(self, angle: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.angle = angle

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        terminated = bool(
            not np.isfinite(observation).all() or (np.abs(observation[1]) > self.angle)
        )

        reward = int(not terminated)

        info = {"reward_survive": reward}

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
