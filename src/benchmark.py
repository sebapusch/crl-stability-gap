import numpy as np
import metaworld
from gymnasium.wrappers import TimeLimit
from metaworld.wrappers import RandomTaskSelectWrapper, OneHotWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.type_aliases import GymEnv

from wrappers import SuccessToIsSuccess


def make_mt1(
    env_name: str,
    seed: int,
    render_mode: str | None = None,
    max_episode_step: int = 500,
    task_ix: int = 0,
    num_tasks: int = 1,
) -> GymEnv:
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
    env = TimeLimit(env, max_episode_steps=max_episode_step)
    env = SuccessToIsSuccess(env)
    env = OneHotWrapper(env, task_ix, num_tasks)

    env.env.observation_space = env.observation_space

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def _make_vec_env(env_name: str, seed: int, task_ix: int, num_tasks: int) -> GymEnv:
    def make_env():
        return make_mt1(env_name, seed, task_ix=task_ix, num_tasks=num_tasks)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    
    return env

def make_benchmark(seed: int, benchmark: list[str]) -> tuple[list[GymEnv], list[GymEnv]]:
    envs_train, envs_test = [], []
    for i, env in enumerate(benchmark):
        envs_train.append(_make_vec_env(env, seed, i, len(benchmark)))
        envs_test.append(_make_vec_env(env, seed + 1, i, len(benchmark)))

    return envs_train, envs_test


