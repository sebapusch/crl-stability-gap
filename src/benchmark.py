import gymnasium as gym
from gymnasium.vector import VectorEnv
import metaworld
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


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


def _make_mt1(env_name: str, seed: int) -> VectorEnv:
    env = gym.make('Meta-World/MT1', env_name=env_name, seed=seed)
    env = SuccessToIsSuccess(env)
    env = Monitor(env)
    env = DummyVecEnv(env_fns=[lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )
    env.reset()

    return env


def make_benchmark(seed: int, benchmark: list[str]) -> (list[VectorEnv], list[VectorEnv]):
    envs_train = [_make_mt1(env, seed)     for env in benchmark]
    envs_test  = [_make_mt1(env, seed + 1) for env in benchmark]

    return envs_train, envs_test

