from collections.abc import Callable

import torch

from stable_baselines3.common.buffers import ExpertBuffer
from stable_baselines3.sacd.sacd import SACD


def discrete_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)

    return kl.mean()


class SAC_BC(SACD):
    def __init__(
            self,
            expert_buffer: ExpertBuffer,
            expert_buffer_batch_size: int,
            lambda_: float,
            loss_fn: Callable = discrete_kl,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.expert_buffer = expert_buffer
        self.lambda_ = lambda_
        self.expert_buffer_batch_size = expert_buffer_batch_size
        self.loss_fn = loss_fn

    def get_actor_auxiliary_loss(self, task_ix: int) -> torch.Tensor:
        if task_ix == 0:
            return torch.zeros([])

        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)

        out_p, _ = self.actor.get_action_dist_params(expert_samples.observations)
        out_q = expert_samples.outputs

        return self.lambda_ * self.loss_fn(out_q, out_p)