from collections.abc import Callable

import torch

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ExpertBuffer


def gaussian_kl(mu_q: torch.Tensor, log_std_q: torch.Tensor, mu_p: torch.Tensor, log_std_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * log_std_q)
    var_p = torch.exp(2 * log_std_p)

    return (
            log_std_p - log_std_q
            + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p)
            - 0.5
    ).mean()

def l2(mu_q: torch.Tensor, _: torch.Tensor, mu_p: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
    return (
        (mu_q - mu_p) ** 2
    ).mean()



class SAC_BC(SAC):
    def __init__(
            self,
            expert_buffer: ExpertBuffer,
            expert_buffer_batch_size: int,
            lambda_: float,
            loss_fn: Callable = gaussian_kl,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.expert_buffer = expert_buffer
        self.lambda_ = lambda_
        self.expert_buffer_batch_size = expert_buffer_batch_size
        self.loss_fn = loss_fn

    def on_task_change(self, task_ix: int) -> None:
        def get_dist(obs: torch.Tensor) -> torch.Tensor:
            mu, log_std, _ = self.actor.get_action_dist_params(obs)
            return torch.concat([mu, log_std], 1)

        self.expert_buffer.populate(
            get_dist,
            self.replay_buffer,
        )

    def get_actor_auxiliary_loss(self, task_ix: int) -> torch.Tensor:
        if task_ix == 0:
            return torch.zeros([])

        action_dim = self.action_space.shape[0]
        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)

        mu, log_std, _ = self.actor.get_action_dist_params(expert_samples.observations)

        expert_mu = expert_samples.outputs[:, :action_dim]
        expert_log_std = expert_samples.outputs[:, action_dim:]

        return self.lambda_ * self.loss_fn(mu, log_std, expert_mu, expert_log_std)