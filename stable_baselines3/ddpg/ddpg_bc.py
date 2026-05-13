from stable_baselines3.common.buffers import ExpertBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.ddpg.ddpg_fine_tune import DDPG_FineTune
import torch as th


class DDPG_BC(DDPG_FineTune):
    def __init__(
        self,
        expert_buffer_size: int,
        n_tasks: int,
        expert_buffer_batch_size: int,
        lambda_: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        expert_output_size = kwargs["env"].action_space.shape[0]
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
            assert self.replay_buffer is not None

            self.expert_buffer.populate(
                self.actor,
                self.replay_buffer,
            )

        super().on_task_change(task_ix, env, logger)

    def get_actor_auxiliary_loss(self) -> th.Tensor:
        if self._task_ix == 0:
            return th.zeros([])

        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)
        out = self.actor(expert_samples.observations)

        return self.lambda_ * th.mean((out - expert_samples.outputs) ** 2)