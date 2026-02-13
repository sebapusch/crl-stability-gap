import numpy as np
import metaworld
from gymnasium.wrappers import TimeLimit
from metaworld.wrappers import RandomTaskSelectWrapper, AutoTerminateOnSuccessWrapper, OneHotWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.type_aliases import GymEnv

def make_mt1(env_name: str,
             seed: int,
             render_mode: str | None = None) -> GymEnv:
    saved_random_state = np.random.get_state()
    np.random.seed(1)
    mt1 = metaworld.MT1(env_name)
    np.random.set_state(saved_random_state)
    
    if env_name not in mt1.train_classes:
         raise ValueError(f"Environment {env_name} not found in MT1 train classes")
    
    env_cls = mt1.train_classes[env_name]
    env = env_cls(render_mode=render_mode)

    tasks = mt1.train_tasks

    env = RandomTaskSelectWrapper(env, tasks)
    env = AutoTerminateOnSuccessWrapper(env)
    env = TimeLimit(env, max_episode_steps=300)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env

def _make_vec_env(env_name: str, seed: int) -> GymEnv:
    def make_env():
        return make_mt1(env_name, seed)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    return env

def make_benchmark(seed: int, benchmark: list[str]) -> tuple[list[GymEnv], list[GymEnv]]:
    envs_train = [_make_vec_env(env, seed)      for env in benchmark]
    envs_test  = [_make_vec_env(env, seed + 1)  for env in benchmark]

    return envs_train, envs_test


