from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType
from metaworld import SawyerXYZEnv


class SetArmPositionToObjectiveWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.unwrapped: SawyerXYZEnv

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, state = self.env.reset(seed=seed, options=options)
        self.unwrapped.hand_init_pos = self.unwrapped.obj_init_pos

        return obs, state