from typing import Any

import numpy as np
import wandb
import metaworld
from stable_baselines3.common.callbacks import (EventCallback,
                                                BaseCallback,
                                                sync_envs_normalization,
                                                evaluate_policy)
from stable_baselines3.common.type_aliases import GymEnv

from benchmark import make_mt1


class EnvEvalCallback(EventCallback):
    def __init__(
            self,
            eval_id: str,
            eval_env: GymEnv,
            callback_on_new_best: BaseCallback | None = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 1,
    ):
        super().__init__()

        self.eval_id = eval_id
        self.eval_env = eval_env
        self.callback_on_new_best = callback_on_new_best
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []
        self.best_mean_reward = 0.0
        self.verbose = verbose

    def _log_success_callback(self, locals_: dict[str, Any], _: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_['info']

        if locals_['done']:
            maybe_is_success = info.get('success')
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq < 1 or self.n_calls % self.eval_freq != 0:
            return continue_training

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # reset buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=False,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=False,
            callback=self._log_success_callback,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = float(mean_reward)
        success_rate = np.mean(self._is_success_buffer) if len(self._is_success_buffer) > 0 else 0

        self.logger.record(f"eval/{self.eval_id}/mean_reward", float(mean_reward))
        self.logger.record(f"eval/{self.eval_id}/mean_ep_length", mean_ep_length)
        self.logger.record(f"eval/{self.eval_id}/success_rate", success_rate)
        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record(f"time/{self.eval_id}/total_timesteps", self.num_timesteps, exclude="tensorboard")

        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            if self.verbose >= 1:
                print("New best mean reward!")
            # if self.best_model_save_path is not None:
            #    self.model.save(os.path.join(self.best_model_save_path, "best_model"))

            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        return continue_training


class RegisterVideoCallback(EventCallback):
    def __init__(self, frequency: int, env_name: str, seed: int = 1):
        self.frequency = frequency
        self.env = make_mt1(env_name, seed, 'rgb_array')
        self.task = metaworld.MT1(env_name).train_tasks[0]
        self.seed = seed

        super().__init__()

    def _on_step(self) -> bool:
        if self.n_calls < 1 or self.n_calls % self.frequency != 0:
            return True

        frames = []

        obs, _ = self.env.reset(seed=self.seed)
        self.env.unwrapped.set_task(self.task)

        for _ in range(500):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            frame = self.env.render()

            frame = np.moveaxis(frame, -1, 0).astype(np.uint8)
            frames.append(frame)

            if terminated or truncated:
                break
        frames_array = np.array(frames)

        wandb.log({'video': wandb.Video(frames_array, fps=30, format='mp4')})

        return True