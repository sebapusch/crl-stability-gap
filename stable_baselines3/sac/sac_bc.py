import dataclasses

import torch
import torch.nn.functional as F
from gymnasium import Env

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ExpertBuffer


def _gaussian_kl(mu_q: torch.Tensor, log_std_q: torch.Tensor, mu_p: torch.Tensor, log_std_p: torch.Tensor):
    var_q = torch.exp(2 * log_std_q)
    var_p = torch.exp(2 * log_std_p)

    kl = (
            log_std_p - log_std_q
            + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p)
            - 0.5
    )

    return kl.sum(-1)


class SAC_BC(SAC):
    def __init__(
            self,
            expert_buffer: ExpertBuffer,
            expert_buffer_batch_size: int,
            lambda_: float,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.expert_buffer = expert_buffer
        self.lambda_ = lambda_
        self.expert_buffer_batch_size = expert_buffer_batch_size

    def on_task_change(self, task_ix: int) -> None:
        def get_dist(obs: torch.Tensor) -> torch.Tensor:
            mu, log_std, _ = self.actor.get_action_dist_params(obs)
            return torch.concat([mu, log_std], 1)

        self.expert_buffer.populate(
            get_dist,
            self.replay_buffer,
        )


    def get_auxiliary_loss(self, task_ix: int) -> torch.Tensor:
        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)

        mu, log_std, _ = self.actor.get_action_dist_params(expert_samples.observations)

        expert_mu = torch.from_numpy(expert_samples.outputs[:,:,0])
        expert_log_std = torch.from_numpy(expert_samples.outputs[:,0,:])

        kl = _gaussian_kl(expert_mu, expert_log_std, mu, log_std)

        return self.lambda_ * kl