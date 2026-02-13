import gymnasium as gym
from gymnasium.vector import VectorEnv
import metaworld
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np

from wrappers import SuccessCounter, OneHotAdder, RandomizationWrapper

def _make_mt1(env_name: str, seed: int) -> gym.Env:
    saved_random_state = np.random.get_state()
    np.random.seed(1)
    mt1 = metaworld.MT1(env_name)
    np.random.set_state(saved_random_state)
    
    if env_name not in mt1.train_classes:
         raise ValueError(f"Environment {env_name} not found in MT1 train classes")
    
    env_cls = mt1.train_classes[env_name]
    env = env_cls()

    env = RandomizationWrapper(env, mt1.train_tasks, "random_init_all")
    env = OneHotAdder(env, one_hot_idx=0, one_hot_len=1)
    env = TimeLimit(env, max_episode_steps=200)
    env = SuccessCounter(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env

def _make_vec_env(env_name: str, seed: int) -> VectorEnv:
    def make_env():
        return _make_mt1(env_name, seed)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    return env

def make_benchmark(seed: int, benchmark: list[str]) -> tuple[list[VectorEnv], list[VectorEnv]]:
    envs_train = [_make_vec_env(env, seed)      for env in benchmark]
    envs_test  = [_make_vec_env(env, seed + 1)  for env in benchmark]

    return envs_train, envs_test


