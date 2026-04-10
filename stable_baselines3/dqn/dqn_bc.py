import torch as th

from stable_baselines3.common.buffers import ExpertBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.dqn_fine_tune import DQN_FineTune


class DQN_BC(DQN_FineTune):
    """DQN with behavior cloning auxiliary loss.

    Extends DQN_FineTune: on task change the expert buffer is populated
    with Q-value targets from the target network BEFORE the replay
    buffer is cleared and optimizers are reset.
    """

    def __init__(
            self,
            expert_buffer_size: int,
            n_tasks: int,
            expert_buffer_batch_size: int,
            lambda_: float,
            **kwargs,
    ):
        super().__init__(**kwargs)

        expert_output_size = kwargs["env"].action_space.n
        self.expert_buffer = ExpertBuffer(
            buffer_size=expert_buffer_size,
            n_tasks=n_tasks,
            observation_space=kwargs["env"].observation_space,
            output_size=expert_output_size,
        )
        self.lambda_ = lambda_
        self.expert_buffer_batch_size = expert_buffer_batch_size
        self._task_ix: int = 0

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self._task_ix = task_ix

        if task_ix > 0:
            # Populate expert buffer BEFORE clearing the replay buffer
            self.expert_buffer.populate(
                self.q_net_target,
                self.replay_buffer,
            )

        # Delegate optimizer reset + epsilon reset + buffer clear to FineTune
        super().on_task_change(task_ix, env, logger)

    def get_auxiliary_loss(self) -> th.Tensor:
        if self._task_ix == 0:
            return th.zeros([])

        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)
        curr_q_values = self.q_net(expert_samples.observations)

        return self.lambda_ * th.mean(
            (curr_q_values - expert_samples.outputs) ** 2
        )
