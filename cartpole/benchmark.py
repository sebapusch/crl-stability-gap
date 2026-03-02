from __future__ import annotations

from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from metaworld.wrappers import OneHotWrapper

from stable_baselines3.common.type_aliases import GymEnv


class ObsSpaceInf(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def observation(self, obs):
        return obs


class ObsLinearTransform(gym.ObservationWrapper):
    def __init__(self, env, projection: np.ndarray, bias: np.ndarray | None = None):
        super().__init__(env)

        assert projection.shape == (4, 4)

        self.projection = np.asarray(projection, dtype=np.float32)
        self.bias = np.zeros(4, dtype=np.float32) \
            if bias is None \
            else np.asarray(bias, dtype=np.float32)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return self.projection @ obs + self.bias


def _random_orthogonal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(4, 4))
    q, _ = np.linalg.qr(m)          # Q is orthonormal
    if np.linalg.det(q) < 0:        # make it a proper rotation (det = +1)
        q[:, 0] *= -1
    return q.astype(np.float32)


class ContinualCartPole(Enum):
    V1 = 1
    V2 = 2
    V3 = 3

    def make(self, render_mode: str | None = None) -> GymEnv:
        match self:
            case ContinualCartPole.V1:
                env = CartPoleEnv(render_mode=render_mode)
                env = ObsSpaceInf(env)
                return env
            case ContinualCartPole.V2:
                env = CartPoleEnv(render_mode=render_mode)
                env = ObsSpaceInf(env)
                env = ObsLinearTransform(env, projection=_random_orthogonal(91))
                return env
            case ContinualCartPole.V3:
                env = CartPoleEnv(render_mode=render_mode)
                env = ObsSpaceInf(env)
                env = ObsLinearTransform(env, projection=_random_orthogonal(92))
                return env
        assert False


def make_env(
        env: ContinualCartPole,
        seed: int = 42,
        task_ix: int = 0,
        render_mode: str | None = None,
) -> GymEnv:
    env = env.make(render_mode)
    env = TimeLimit(env, max_episode_steps=500)
    env.reset(seed=seed + task_ix)
    env.action_space.seed(seed + task_ix)

    return env


def make_benchmark(
        benchmark: list[ContinualCartPole] | None = None,
        seed: int = 42,
        render_mode: str | None = None,
        encode_task: bool = True,
) -> list[GymEnv]:
    benchmark = benchmark or [ContinualCartPole.V1, ContinualCartPole.V2, ContinualCartPole.V3]
    envs = []
    for ix, variant in enumerate(benchmark):
        env = make_env(variant, seed, ix, render_mode)
        if encode_task:
            env = OneHotWrapper(env, ix, len(benchmark))
        envs.append(env)

    return envs
