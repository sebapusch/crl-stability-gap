import sys

import torch.cuda
import wandb
import math
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from gymnasium import Env
from stable_baselines3.common.type_aliases import GymEnv

from benchmark import make_benchmark
from callbacks import EnvEvalCallback, RegisterVideoCallback
from integration import WandbWriter

BENCHMARK = ['reach-v3', 'push-v3', 'pick-place-v3']
CONFIG = {
    'policy':          'MlpPolicy',
    'architecture':    [256, 256, 256],
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'total_timesteps': 1_000_000,
    'seed':            42,
    'eval_freq':       20_000,
    'video_freq':      20_000,
    'lr':              1e-3,
    'batch_size':      128,
    'learning_starts': 10_000,
    'tau':             0.005,
    'gamma':           0.99,
    'train_freq':      1,
    'gradient_steps':  1,
    'target_output_std': 0.089,
}

def make_logger() -> Logger:
    return Logger(
        folder='../.logs',
        output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()],
    )

def make_model(env: Env) -> SAC:
    # Calculate target entropy based on target_output_std
    # From continual_world/continualworld/sac/sac.py
    target_output_std = CONFIG['target_output_std']
    target_1d_entropy = np.log(target_output_std * math.sqrt(2 * math.pi * math.e))
    target_entropy = float(np.prod(env.action_space.shape) * target_1d_entropy)

    sac = SAC(
        CONFIG['policy'],
        env=env,
        device=CONFIG['device'],
        verbose=1,
        learning_rate=CONFIG['lr'],
        batch_size=CONFIG['batch_size'],
        learning_starts=CONFIG['learning_starts'],
        ent_coef='auto',
        target_entropy=target_entropy,
    #     policy_kwargs={
    #         'net_arch': CONFIG['architecture']
    #     },
    )
    sac.set_logger(make_logger())

    return sac

def make_callbacks(env_ix: int, envs_test: list[GymEnv]) -> list[BaseCallback]:
    callbacks: list[BaseCallback] = [
        WandbCallback(
            gradient_save_freq=1000,
            verbose=2,
        )
    ]

    if CONFIG['video_freq'] > 0:
        callbacks.append(
            RegisterVideoCallback(
                CONFIG['video_freq'],
                BENCHMARK[0],
            ),
        )

    for i in range(env_ix + 1):
        callbacks.append(
            EnvEvalCallback(
                eval_id=BENCHMARK[i],
                eval_env=envs_test[i],
                eval_freq=CONFIG['eval_freq'],
            )
        )

    return callbacks

def main() -> None:
    config = CONFIG

    run = wandb.init(
        project='test-crl',
        config=config,
        monitor_gym=True,
    )

    envs_train, envs_test = make_benchmark(42, BENCHMARK)
    model = make_model(envs_train[0])

    for i, env in enumerate(envs_train):
        model.set_env(env)
        model.replay_buffer.reset()
        model.learn(
            total_timesteps=CONFIG['total_timesteps'],
            reset_num_timesteps=False,
            callback=CallbackList(make_callbacks(i, envs_test)),
        )
        env.close()

    for env in envs_test:
        env.close()

    run.finish()


if __name__ == '__main__':
    main()

