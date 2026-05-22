from __future__ import annotations

from copy import deepcopy
from typing import Literal

import gymnasium as gym
import metaworld
import torch
import wandb
from gymnasium.wrappers import TimeLimit
from metaworld import RandomTaskSelectWrapper
from wandb.integration.sb3 import WandbCallback

from args import get_args
from continual_world.callbacks import EnvEvalCallback
from continual_world.wrappers import SuccessToIsSuccess
from projection.common import make_logger
from single_task.wrappers import SetArmPositionToObjectiveWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.callbacks import CallbackList

Render = Literal['rgb_array', 'human'] | None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Environment ─────────────────────────────────────────────────────


def make_env(
    env_name: str,
    seed: int,
    max_episode_steps: int,
    render_mode: Render,
    start_arm_at_obj: bool,
) -> tuple[gym.Env, gym.Env]:

    mt1 = metaworld.MT1(env_name, seed=seed)
    env_train = mt1.train_classes[env_name](render_mode=render_mode, camera_id=1)
    env_test  = mt1.test_classes[env_name](render_mode=render_mode, camera_id=1)

    env_train = RandomTaskSelectWrapper(env_train, mt1.train_tasks)
    env_test = RandomTaskSelectWrapper(env_test, mt1.train_tasks)

    env_train = TimeLimit(env_train, max_episode_steps=max_episode_steps)
    env_test = TimeLimit(env_test, max_episode_steps=max_episode_steps)

    env_train = SuccessToIsSuccess(env_train)
    env_test = SuccessToIsSuccess(env_test)

    if start_arm_at_obj:
        env_train = SetArmPositionToObjectiveWrapper(env_train)
        env_test = SetArmPositionToObjectiveWrapper(env_test)

    env_train.reset(seed=seed)
    env_test.reset(seed=seed + 1)

    env_train.action_space.seed(seed)
    env_test.action_space.seed(seed + 1)

    env_train.observation_space.seed(seed)
    env_test.observation_space.seed(seed + 1)

    return env_train, env_test


def build_sac(
    env: gym.Env,
    *,
    lr: float,
    buffer_size: int,
    batch_size: int,
    gamma: float,
    learning_starts: int,
    net_arch: list[int],
    tau: float = 0.005,
    seed: int,
) -> SAC:
    return SAC(
        device=DEVICE,
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=learning_starts,
        seed=seed,
        tau=tau,
        train_freq=(1, 'episode'),
        gradient_steps=-1,
        policy_kwargs={
            'net_arch': net_arch,
            'layer_norm': True,
        },
    )


# ── Training ────────────────────────────────────────────────────────


def train_phase_1(
    model: SAC,
    env_eval: gym.Env,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    gradient_save_freq: int,
) -> None:
    """Phase 1: train with arm starting at object position."""
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([
            WandbCallback(gradient_save_freq=gradient_save_freq, verbose=2),
            EnvEvalCallback('1', env_eval, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes),
        ]),
    )


def train_phase_2(
    model: SAC,
    env_train_2: gym.Env,
    env_eval_1: gym.Env,
    env_eval_2: gym.Env,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    gradient_save_freq: int,
    learning_starts: int,
) -> None:
    """Phase 2: train normally, evaluate on both phases."""
    model.learning_starts = model.num_timesteps + learning_starts
    model.set_env(env_train_2)
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=CallbackList([
            WandbCallback(gradient_save_freq=gradient_save_freq, verbose=2),
            EnvEvalCallback('1', env_eval_1, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes),
            EnvEvalCallback('2', env_eval_2, eval_freq=[(500_000, 10_000), (550_000, 1000), (0, 10_000)], n_eval_episodes=n_eval_episodes),
        ]),
    )


def swap_to_multi_buffer(model: SAC, buffer_size: int) -> None:
    """Store phase-1 buffer and replace with a MultiReplayBuffer for phase 2."""
    buffer_phase_2 = MultiReplayBuffer(
        2, 1, buffer_size,
        model.replay_buffer.observation_space,
        model.replay_buffer.action_space,
    )
    buffer_phase_2.buffers[0] = deepcopy(model.replay_buffer)
    buffer_phase_2.increase_index()
    model.replay_buffer = buffer_phase_2


# ── Main ────────────────────────────────────────────────────────────


def main(
    name_prefix: str,
    project: str,
    seed: int,
    env_name: str,
    max_episode_steps: int,
    total_timesteps: int,
    lr: float,
    buffer_size: int,
    batch_size: int,
    gamma: float,
    learning_starts: int,
    net_arch: list[int],
    eval_freq: int,
    n_eval_episodes: int,
    gradient_save_freq: int,
) -> None:
    # phase 1: train model only with starting state = object position
    #          - on completion: store phase 1 buffer
    # phase 2: reset model buffer and train normally
    #          - sample uniformly from phase 1 and phase 2 buffer
    #          - evaluate on starting state phase 1


    run = wandb.init(name=name_prefix, project=project, monitor_gym=True)

    print(f"TRAINING ON DEVICE: {DEVICE}")

    env_train_1, env_test_1 = make_env(env_name, seed, max_episode_steps, 'rgb_array', True)
    env_train_2, env_test_2 = make_env(env_name, seed + 100, max_episode_steps, 'rgb_array', False)

    model = build_sac(
        env_train_1,
        lr=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=learning_starts,
        net_arch=net_arch,
        seed=seed,
    )

    model.set_logger(make_logger(project, name_prefix))

    print('phase 1...')
    train_phase_1(
        model, env_test_1,
        total_timesteps=total_timesteps // 2,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        gradient_save_freq=gradient_save_freq,
    )
    print('phase 1 completed')

    swap_to_multi_buffer(model, buffer_size)

    print('phase 2...')
    train_phase_2(
        model,
        env_train_2, env_test_1, env_test_2,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        gradient_save_freq=gradient_save_freq,
        learning_starts=learning_starts,
    )
    print('phase 2 completed')

    env_test_1.close()
    env_test_2.close()
    env_train_1.close()
    env_train_2.close()

    run.finish()


if __name__ == "__main__":
    args = vars(get_args())
    main(**args)