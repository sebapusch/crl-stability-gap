import sys

import torch.cuda
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from gymnasium import Env

from benchmark import make_benchmark
from common import WandbWriter, EnvEvalCallback

import math
import numpy as np
from policy import ContinualWorldMlpPolicy

BENCHMARK = ['hammer-v3','push-back-v3', 'stick-pull-v3']
CONFIG = {
    'policy':          ContinualWorldMlpPolicy,
    'architecture':    [256, 256, 256, 256],
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'total_timesteps': 1_000_000,
    'seed':            42,
    'eval_freq':       20_000,
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
        policy_kwargs={
            'net_arch': CONFIG['architecture']
        },
    )
    sac.set_logger(make_logger())

    return sac

def main() -> None:
    config = CONFIG

    run = wandb.init(
        project='test-crl',
        config=config,
        monitor_gym=True,
    )

    envs_train, envs_test = make_benchmark(42, BENCHMARK)
    model = make_model(envs_train[0])
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f'models/{run.id}',
        verbose=2,
    )

    for i, env in enumerate(envs_train):
        model.set_env(env)
        eval_callbacks = [
            EnvEvalCallback(
                eval_env_id=BENCHMARK[j],
                eval_env=envs_test[j],
                eval_freq=CONFIG['eval_freq']
            ) for j in range(i + 1)
        ]
        model.replay_buffer.reset()
        model.learn(
            total_timesteps=CONFIG['total_timesteps'],
            reset_num_timesteps=False,
            callback=CallbackList(eval_callbacks + [wandb_callback]),
        )
        env.close()

    for env in envs_test:
        env.close()

    run.finish()


if __name__ == '__main__':
    main()

