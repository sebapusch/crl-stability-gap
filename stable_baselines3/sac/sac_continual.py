import torch as th

from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.sac.sac import SAC


class SAC_Continual(SAC):
    """SAC with experience replay across tasks.

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
        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()

        # Reset entropy coefficient to its initial value
        if self.ent_coef_optimizer is not None:
            self.log_ent_coef.data.fill_(0.0)
            self.ent_coef_optimizer.state.clear()

        # Advance to next partition (keeps all past experience)
        self.replay_buffer.increase_index()
