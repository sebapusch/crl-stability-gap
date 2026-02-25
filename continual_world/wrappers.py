from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import WrapperObsType
from numpy.typing import NDArray
from stable_baselines3.common.type_aliases import GymEnv


class SuccessToIsSuccess(gym.Wrapper):
    def __init__(self, env: GymEnv):
        super().__init__(env)
        self.success = False

    def step(self, action: NDArray) -> tuple[NDArray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.success = self.success or info.get("success", False)

        info['is_success'] = self.success

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.success = False

        return self.env.reset(seed=seed, options=options)