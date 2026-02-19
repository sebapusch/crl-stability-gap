import os
import sys
from typing import Any

import torch.cuda
import wandb

from stable_baselines3 import SAC
from stable_baselines3.sac.ewc import SAC_EWC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.logger import Logger, HumanOutputFormat, CSVOutputFormat

from benchmark import make_benchmark
from integration import WandbWriter
from args import get_args
from callbacks import make_callbacks


def make_logger(run_name: str) -> Logger:
    csv_path = os.path.abspath(os.path.join(
        __file__, '..', '..', 'output', f'{run_name}.csv')
    )

    return Logger(
        folder='../.logs',
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbWriter(),
            CSVOutputFormat(csv_path)
        ],
    )


def make_model(
        benchmark: list[GymEnv],
        run_name: str,
        device: str,
        method: str = 'fine-tune',
        ewc_lambda: float = 10_000.0,
        lr: float = 1e-3,
        seed: int = 42,
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

    default_kwargs = dict(
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
        seed=seed,
    )

    match method:
        case 'fine-tune':
            model = SAC(**default_kwargs)
        case 'ewc':
            model = SAC_EWC(
                lambda_=ewc_lambda,
                **default_kwargs,
            )
        case _:
            raise ValueError(f'invalid method {method}')

    model.set_logger(make_logger(run_name))

    return model


def main(
        benchmark: list[str],
        method: str,
        seed: int,
        total_timesteps: int,
        learning_starts: int,
        video_freq: int,
        eval_freq: int,
        n_eval_episodes: int,
        lr: float,
        layer_norm: bool,
        multi_head_output: bool,
        wandb_project: str,
        wandb_name: str | None,
        ewc_lambda: float,
) -> None:
    run = wandb.init(
        name=wandb_name,
        project=wandb_project,
        monitor_gym=True,
    )

    envs_train, envs_test = make_benchmark(seed, benchmark=benchmark)
    model = make_model(
        envs_train,
        method=method,
        learning_starts=learning_starts,
        run_name=run.name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        net_arch=[256, 256, 256, 256],
        layer_norm=layer_norm,
        lr=lr,
        multi_head_output=multi_head_output,
        seed=seed,
        ewc_lambda=ewc_lambda,
    )

    callbacks = make_callbacks(
        benchmark=benchmark,
        envs_test=envs_test,
        video_freq=video_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_all=True,
    )

    for task_ix, env in enumerate(envs_train):
        model.set_env(env)
        model.on_task_change(task_ix)

        model.replay_buffer.reset()
        model.reset_optim()
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=callbacks(task_ix),
        )

        env.close()

    for env in envs_test:
        env.close()

    run.finish()


if __name__ == '__main__':
    main(**vars(get_args()))

