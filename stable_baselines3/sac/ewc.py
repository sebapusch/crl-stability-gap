from typing import Iterator

import torch
from torch.func import functional_call, vmap, jacrev

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class SAC_EWC(SAC):
    def __init__(self, lambda_: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lambda_ = lambda_
        self.old_params = self.frozen_params
        self.reg_weights = [
            torch.zeros_like(p, requires_grad=False) for p in self.old_params
        ]

    def get_auxiliary_loss(self, task_ix: int) -> torch.Tensor:
        return self._regression_loss(task_ix)


    def on_task_change(self, task_ix: int) -> None:
        if task_ix > 0:
            for old_weight, new_weight in zip(self.old_params, self.shared_params):
                old_weight.data.copy_(new_weight)
            self._update_reg_weights()

    def _regression_loss(self, task_ix: int) -> torch.Tensor:
        if task_ix < 1:
            return torch.zeros([])

        return self.lambda_ * self._regularize()

    def _regularize(self) -> torch.Tensor:
        reg_loss = torch.zeros([])
        for new_param, old_param, weight in zip(
            self.shared_params, self.old_params, self.reg_weights,
        ):
            diff = (new_param - old_param) ** 2
            w_diff = weight * diff
            reg_loss += torch.sum(w_diff)
        return reg_loss

    @property
    def shared_params(self) -> list[torch.Tensor]:
        params = [
            p for p in self.actor.latent_pi.parameters() if p.requires_grad
        ]
        for q_net in self.critic.q_networks_core:
            params.extend(
                p for p in q_net.parameters() if p.requires_grad
            )
        return params

    @property
    def frozen_params(self) -> list[torch.Tensor]:
        frozen_params = [
            p.clone().detach() for p in self.actor.latent_pi.parameters() if p.requires_grad
        ]
        for q_net in self.critic.q_networks_core:
            frozen_params.extend(
                p.clone().detach() for p in q_net.parameters() if p.requires_grad
            )

        return frozen_params

    def _merge_reg_weights(self, new_weights: list[torch.Tensor]):
        merged_weights = [
            old + new for old, new in zip(self.reg_weights, new_weights)
        ]

        for old_weight, new_weight in zip(self.reg_weights, merged_weights):
            old_weight.data.copy_(new_weight)

    def _update_reg_weights(self, n_batches: int = 10, batch_size: int = 256) -> None:
        all_weights = []
        for batch_ix in range(n_batches):
            batch = self.replay_buffer.sample(batch_size)
            all_weights.append(self._get_importance_weights(batch))

        mean_weights = []
        for weights in zip(*all_weights):
            stacked = torch.stack(weights)
            mean_weights.append(torch.mean(stacked, dim=0))

        self._merge_reg_weights(mean_weights)

    def _get_importance_weights(
            self, samples: ReplayBufferSamples,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert len(self.critic.q_networks_core) == 2

        obs = samples.observations
        actions = samples.actions

        with torch.no_grad():
            actor_features = self.actor.latent_pi(obs)
            log_std = self.actor.log_std(actor_features)
            std_standard = torch.exp(log_std)

        # isolate params
        actor_params = dict(self.actor.latent_pi.named_parameters())
        critic1_params = dict(self.critic.q_networks_core[0].named_parameters())
        critic2_params = dict(self.critic.q_networks_core[1].named_parameters())

        # setup functional wrappers for vectorized jacobians
        def actor_mu_std_fn(
                params: dict[str, torch.Tensor], single_obs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            obs_b = single_obs.unsqueeze(0)
            features = functional_call(self.actor.latent_pi, params, (obs_b,))
            mu_b, log_std_b = self.actor.mu(features), self.actor.log_std(features)

            return mu_b.squeeze(0), torch.exp(log_std_b.squeeze(0))

        def critic1_fn(
                params: dict[str, torch.Tensor],
                single_obs: torch.Tensor,
                single_action: torch.Tensor,
        ) -> torch.Tensor:
            obs_b, act_b = single_obs.unsqueeze(0), single_action.unsqueeze(0)
            features = functional_call(self.critic.q_networks_core[0], params, (obs_b, act_b))
            q = self.critic.q_networks_head[0](features)
            return q.squeeze(0)

        def critic2_fn(
                params: dict[str, torch.Tensor],
                single_obs: torch.Tensor,
                single_action: torch.Tensor,
        ) -> torch.Tensor:
            obs_b, act_b = single_obs.unsqueeze(0), single_action.unsqueeze(0)
            features = functional_call(self.critic.q_networks_core[1], params, (obs_b, act_b))
            q = self.critic.q_networks_head[1](features)
            return q.squeeze(0)

        # Compute Per-Sample Jacobians using vmap + jacrev
        batched_actor_jac_fn = vmap(jacrev(actor_mu_std_fn, argnums=0), in_dims=(None, 0))
        actor_mu_gs, actor_std_gs = batched_actor_jac_fn(actor_params, obs)

        batched_q1_jac_fn = vmap(jacrev(critic1_fn, argnums=0), in_dims=(None, 0, 0))
        q1_gs = batched_q1_jac_fn(critic1_params, obs, actions)

        batched_q2_jac_fn = vmap(jacrev(critic2_fn, argnums=0), in_dims=(None, 0, 0))
        q2_gs = batched_q2_jac_fn(critic2_params, obs, actions)

        def compute_fisher_diagonal_trace(jac_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            trace = 0.0
            for param_name, jac in jac_dict.items():
                j_flat = jac.view(jac.shape[0], -1)
                trace += j_flat.pow(2).sum(dim=1)
            return trace

        return (
            compute_fisher_diagonal_trace(actor_mu_gs),
            compute_fisher_diagonal_trace(actor_std_gs),
            compute_fisher_diagonal_trace(q1_gs),
            compute_fisher_diagonal_trace(q2_gs),
            std_standard,
        )




