from collections.abc import Callable

import torch

from stable_baselines3.common.buffers import ExpertBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.sac.sac_fine_tune import SAC_FineTune


def _gaussian_kl(mu_q: torch.Tensor, log_std_q: torch.Tensor, mu_p: torch.Tensor, log_std_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * log_std_q)
    var_p = torch.exp(2 * log_std_p)

    return (
            log_std_p - log_std_q
            + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p)
            - 0.5
    ).mean()

def _l2(mu_q: torch.Tensor, _: torch.Tensor, mu_p: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
    return (
        (mu_q - mu_p) ** 2
    ).mean()



class SAC_BC(SAC_FineTune):
    """SAC with behavior cloning auxiliary loss.

    Extends SAC_FineTune: on task change the expert buffer is populated
    with distributional targets from the current policy BEFORE the
    replay buffer is cleared and optimizers are reset.
    """

    def __init__(
            self,
            expert_buffer_size: int,
            n_tasks: int,
            expert_buffer_batch_size: int,
            lambda_: float,
            loss_fn: str = 'kl',
            **kwargs,
    ):
        super().__init__(**kwargs)

        expert_output_size = 2 * kwargs["env"].action_space.shape[0]
        self.expert_buffer = ExpertBuffer(
            buffer_size=expert_buffer_size,
            n_tasks=n_tasks,
            observation_space=kwargs["env"].observation_space,
            output_size=expert_output_size,
        )
        self.lambda_ = lambda_
        self.expert_buffer_batch_size = expert_buffer_batch_size
        self.loss_fn = _l2 if loss_fn == 'l2' else _gaussian_kl
        self._task_ix: int = 0

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self._task_ix = task_ix

        if task_ix > 0:
            # Populate expert buffer BEFORE clearing the replay buffer
            def get_dist(obs: torch.Tensor) -> torch.Tensor:
                mu, log_std, _ = self.actor.get_action_dist_params(obs)
                return torch.concat([mu, log_std], 1)

            self.expert_buffer.populate(
                get_dist,
                self.replay_buffer,
            )

        # Delegate optimizer reset + buffer clear to FineTune
        super().on_task_change(task_ix, env, logger)

    def get_actor_auxiliary_loss(self) -> torch.Tensor:
        if self._task_ix == 0:
            return torch.zeros([])

        action_dim = self.action_space.shape[0]
        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)

        mu, log_std, _ = self.actor.get_action_dist_params(expert_samples.observations)

        expert_mu = expert_samples.outputs[:, :action_dim]
        expert_log_std = expert_samples.outputs[:, action_dim:]

        return self.lambda_ * self.loss_fn(mu, log_std, expert_mu, expert_log_std)