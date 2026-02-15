import sys
from typing import Any

import torch.cuda
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from gymnasium import Env

from benchmark import make_benchmark
from integration import WandbWriter
from args import get_args
from callbacks import make_callbacks


def make_logger() -> Logger:
    return Logger(
        folder='../.logs',
        output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()],
    )

def make_model(
        benchmark: list[GymEnv],
        device: str,
        lr: float = 1e-3,
        batch_size: int = 128,
        learning_starts: int = 10_000,
        gamme: int = 0.99,
        train_freq: int | tuple[int, str] = (1, 'episode'), # finish episode
        gradient_steps: int = -1,                           # then do 500 gradient steps
        tau: float = 0.005,
        net_arch: list[int] | None = None,
        layer_norm: bool = False,
        multi_head_output: bool = True,
) -> SAC:
    assert len(benchmark) > 0, 'Invalid benchmark'

    policy_kwargs: dict[str, Any] = {
        'layer_norm': layer_norm,
    }
    if net_arch:
        policy_kwargs['net_arch'] = net_arch

    if multi_head_output and len(benchmark) > 1:
        policy_kwargs['n_heads'] = len(benchmark)

    sac = SAC(
        policy='MlpPolicy',
        policy_kwargs=policy_kwargs,
        env=benchmark[0],
        device=device,
        verbose=1,
        learning_rate=lr,
        batch_size=batch_size,
        learning_starts=learning_starts,
        gamma=gamme,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tau=tau,
        ent_coef='auto',
    )

    sac.set_logger(make_logger())

    return sac


def main(
        benchmark: list[str],
        seed: int,
        total_timesteps: int,
        video_freq: int,
        eval_freq: int,
        n_eval_episodes: int,
        lr: float,
        layer_norm: bool,
        multi_head_output: bool,
) -> None:
    # run = wandb.init(
    #     project='test-crl',
    #     monitor_gym=True,
    # )

    envs_train, envs_test = make_benchmark(seed, benchmark=benchmark)
    model = make_model(
        envs_train,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        net_arch=[256, 256, 256, 256],
        layer_norm=layer_norm,
        lr=lr,
        multi_head_output=multi_head_output,
    )

    callbacks = make_callbacks(
        benchmark=benchmark,
        envs_test=envs_test,
        video_freq=video_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )

    for i, env in enumerate(envs_train):
        model.set_env(env)
        model.replay_buffer.reset()
        model.reset_optim()
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=callbacks(i),
        )

        env.close()

    for env in envs_test:
        env.close()

    run.finish()


if __name__ == '__main__':
    main(**vars(get_args()))

