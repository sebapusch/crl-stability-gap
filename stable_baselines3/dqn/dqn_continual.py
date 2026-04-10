from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.dqn import DQN


class DQN_Continual(DQN):
    """DQN with experience replay across tasks.

    Uses a MultiReplayBuffer that partitions experience by task.
    On task change, resets optimizer state and advances the active
    buffer partition.
    """

    def __init__(
            self,
            *,
            n_tasks: int,
            balanced_sampling: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # Replace the default replay buffer with a multi-task one
        self.replay_buffer = MultiReplayBuffer(
            n_tasks,
            self.buffer_size,
            self.observation_space,
            self.action_space,
            balanced_sampling=balanced_sampling,
        )

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self.set_env(env)
        self.set_logger(logger)

        if task_ix == 0:
            return
            
        # Reset optimizer momentum / adaptive state
        self.policy.optimizer.state.clear()

        # Reset exploration counter (target-net update counter)
        self._n_calls = 0

        # Advance to next partition (keeps all past experience)
        self.replay_buffer.increase_index()
