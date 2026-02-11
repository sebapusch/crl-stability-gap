import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env


class SuccessToIsSuccess(gym.Wrapper):
    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, rew, done, info = out
            if "is_success" not in info and "success" in info:
                info["is_success"] = float(info["success"])
            return obs, rew, done, info
        else:
            obs, rew, terminated, truncated, info = out
            if "is_success" not in info and "success" in info:
                info["is_success"] = float(info["success"])
            return obs, rew, terminated, truncated, info


def make_benchmark(seed: int, benchmark: list[str]) -> (list[Env], list[Env]):
    envs_train = []
    envs_test = []

    for env_name in benchmark:
        env_train = gym.make('Meta-World/MT1', env_name=env_name, seed=seed)
        env_test  = gym.make('Meta-World/MT1', env_name=env_name, seed=seed + 1)

        env_train = SuccessToIsSuccess(env_train)
        env_train = Monitor(env_train)
        env_train.reset()

        env_test = SuccessToIsSuccess(env_test)
        env_test = Monitor(env_test)
        env_test.reset()

        envs_train.append(env_train)
        envs_test.append(env_test)

    return envs_train, envs_test

