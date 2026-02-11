import sys
from typing import Any

import metaworld
import numpy as np
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from wandb.integration.sb3 import WandbCallback

# hammer, push-back, stick-pull

class WandbWriter(KVWriter):
    def write(self, key_values: dict[str, Any], key_excluded: dict[str, Any], step: int = 0) -> None:
        log_dict: dict[str, Any] = {}
        for k, v in key_values.items():
            if v is None:
                continue
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            # W&B only logs simple scalars nicely as time series; keep others out
            if isinstance(v, (int, float)):
                log_dict[k] = v

        if log_dict:
            wandb.log(log_dict, step=step)

def make_env(task_name: str, seed: int, train: bool = True):
    # ML1 = single task set, with train/test variants available
    ml1 = metaworld.ML1(task_name, seed=seed)

    # Create the *train* environment for that task
    if train:
        env_cls = ml1.train_classes[task_name]
        task = ml1.train_tasks[0]
    else:
        env_cls = ml1.test_classes[task_name]
        task = ml1.test_tasks[0]

    env = env_cls(render_mode='rgb_array')

    env.set_task(task)

    env = Monitor(env)

    return env

def make_logger() -> Logger:
    return Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()],
    )

def main():
    config = {
        'policy': 'MlpPolicy',
        'device': 'cpu',
        'total_timesteps': 10_000,
        'seed': 42,
    }

    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    env_train = make_env('push-back-v3', config['seed'])
    env_test  = make_env('push-back-v3', config['seed'], False)

    eval_callback  = EvalCallback(
        env_test,
        eval_freq=1000,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f'models/{run.id}',
        verbose=2,
    )

    sac = SAC(
        config['policy'],
        env=env_train,
        device=config['device'],
        verbose=1,
        tensorboard_log=f'runs/{run.id}',
    )

    sac.set_logger(make_logger())

    sac.learn(
        total_timesteps=config['total_timesteps'],
        callback=CallbackList([eval_callback, wandb_callback])
    )

    env_train.close()
    env_test.close()
    run.finish()


if __name__ == "__main__":
    main()
