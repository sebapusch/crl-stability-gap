from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm, SelfOffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, TrainFreq, GymEnv
from stable_baselines3.continual import ContinualLearning


class OffPolicyJointIncremental(ContinualLearning):
    replay_buffer: MultiReplayBuffer

    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, balanced_sampling: bool = False) -> None:
        self.replay_buffer = MultiReplayBuffer(
            n_envs=n_tasks,
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            balanced_sampling=balanced_sampling,
        )
        self.envs: list[GymEnv] = []


    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            num_timesteps = self.num_timesteps
            for i, env in enumerate(self.envs):
                rollout = self.collect_rollouts(
                    env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer.buffers[i],
                    log_interval=log_interval,
                )
                if not rollout.continue_training:
                    break

                self.num_timesteps = num_timesteps
            self.num_timesteps += self.train_freq.frequency

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps * len(self.envs))

        callback.on_training_end()


        return self

    def on_task_change(
            self,
            task_ix: int,
            env: GymEnv,
            logger: Logger,
    ) -> None:
        self.set_env(env)
        self.envs.append(self.env)
        self.set_logger(logger)

        if task_ix == 0:
            return

        # Reset optimizer momentum / adaptive state
        self.reset_optimizer()

        # Reset exploration counter (target-net update counter)
        self._n_calls = 0

        self.replay_buffer.reset()
        self.replay_buffer.active_index = task_ix


