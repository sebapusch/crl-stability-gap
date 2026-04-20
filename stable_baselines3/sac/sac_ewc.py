import gc

import torch
from torch.func import functional_call, vmap, grad

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import ReplayBufferSamples, GymEnv
from stable_baselines3.sac.sac_fine_tune import SAC_FineTune


class SAC_EWC(SAC_FineTune):
    def __init__(self, lambda_: float, regularize_critic: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lambda_ = lambda_
        self.old_params = self.frozen_params
        self.regularize_critic = regularize_critic
        self.reg_weights = [
            torch.zeros_like(p, requires_grad=False) for p in self.old_params
        ]
        self._task_ix: int = 0

    def get_actor_auxiliary_loss(self) -> torch.Tensor:
        return self._regression_loss()

    def get_critic_auxiliary_loss(self) -> torch.Tensor:
        return self._regression_loss()

    def on_task_change(
            self,
            task_ix: int,
            env: GymEnv,
            logger: Logger,
    ) -> None:
        self._task_ix = task_ix

        if task_ix == 0:
            return

        for old_weight, new_weight in zip(self.old_params, self.shared_params):
            old_weight.data.copy_(new_weight)
        self._update_reg_weights()

        super().on_task_change(task_ix, env, logger)

    def _regression_loss(self) -> torch.Tensor:
        if self._task_ix < 1:
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
        running_mean_weights = [torch.zeros_like(p, device='cpu') for p in self.old_params]     #@todo properly handle device

        for batch_ix in range(n_batches):
            batch = self.replay_buffer.sample(batch_size)

            # 1. Get the flat importance weights
            actor_mu_f, actor_std_f, q1_f, q2_f = self._get_importance_weights(batch)

            # 2. Combine EWC actor components
            actor_f = (actor_mu_f + actor_std_f).detach()
            q1_f = q1_f.detach()
            q2_f = q2_f.detach()

            # Combine all flat vectors in the exact order of self.shared_params
            flat_batch_weights = torch.cat([actor_f, q1_f, q2_f])

            # 3. Unflatten and accumulate as a running average
            idx = 0
            for i, p in enumerate(self.old_params):
                numel = p.numel()
                # Extract the chunk, reshape it to the parameter shape, and add
                param_weight = flat_batch_weights[idx: idx + numel].view_as(p)
                running_mean_weights[i] += (param_weight / n_batches)
                idx += numel

            # 4. Aggressively free memory before the next batch
            del batch, actor_mu_f, actor_std_f, q1_f, q2_f, actor_f, flat_batch_weights
            gc.collect()

        # Apply to your existing merge function
        self._merge_reg_weights(running_mean_weights)

    def _get_importance_weights(
            self, samples: ReplayBufferSamples,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        from stable_baselines3.common.policies import choose_head

        assert len(self.critic.q_networks_core) == 2

        obs = samples.observations
        actions = samples.actions

        n_heads = getattr(self.actor, "n_heads", 1)

        # 1. Forward pass for the required 'std' output and to determine action dimensions
        with torch.no_grad():
            actor_features = self.actor.latent_pi(obs)
            log_std = self.actor.log_std(actor_features)
            log_std_head = choose_head(log_std, obs, n_heads)
            std_standard = torch.exp(log_std_head)

        action_dim = std_standard.shape[-1]

        # 2. Isolate params for functional calls
        actor_params = dict(self.actor.latent_pi.named_parameters())
        critic1_params = dict(self.critic.q_networks_core[0].named_parameters())
        critic2_params = dict(self.critic.q_networks_core[1].named_parameters())

        # -------------------------------------------------------------------------
        # 3. Setup Scalar-Output Functional Wrappers
        # By outputting a scalar (e.g., a specific action index), grad() natively
        # returns vectors of shape [P] instead of massive Jacobian matrices.
        # -------------------------------------------------------------------------
        def actor_mu_k_fn(params: dict[str, torch.Tensor], single_obs: torch.Tensor, k: int) -> torch.Tensor:
            obs_b = single_obs.unsqueeze(0)
            features = functional_call(self.actor.latent_pi, params, (obs_b,))
            mu_b = self.actor.mu(features)
            mu_head = choose_head(mu_b, obs_b, n_heads)
            return mu_head[0, k]

        def actor_std_k_fn(params: dict[str, torch.Tensor], single_obs: torch.Tensor, k: int) -> torch.Tensor:
            obs_b = single_obs.unsqueeze(0)
            features = functional_call(self.actor.latent_pi, params, (obs_b,))
            log_std_b = self.actor.log_std(features)
            log_std_head = choose_head(log_std_b, obs_b, n_heads)
            return torch.exp(log_std_head)[0, k]

        def critic1_fn(
                params: dict[str, torch.Tensor], single_obs: torch.Tensor, single_act: torch.Tensor
        ) -> torch.Tensor:
            obs_b, act_b = single_obs.unsqueeze(0), single_act.unsqueeze(0)
            # Prevent graph accumulation in the feature extractor during mapping
            with torch.no_grad():
                features = self.critic.extract_features(obs_b, self.critic.features_extractor)
                q_in = torch.cat([features, act_b], dim=1)

            features_core = functional_call(self.critic.q_networks_core[0], params, (q_in,))
            q = self.critic.q_networks_head[0](features_core)
            q_head = choose_head(q, obs_b, n_heads)
            return q_head[0, 0]

        def critic2_fn(
                params: dict[str, torch.Tensor], single_obs: torch.Tensor, single_act: torch.Tensor
        ) -> torch.Tensor:
            obs_b, act_b = single_obs.unsqueeze(0), single_act.unsqueeze(0)
            with torch.no_grad():
                features = self.critic.extract_features(obs_b, self.critic.features_extractor)
                q_in = torch.cat([features, act_b], dim=1)

            features_core = functional_call(self.critic.q_networks_core[1], params, (q_in,))
            q = self.critic.q_networks_head[1](features_core)
            q_head = choose_head(q, obs_b, n_heads)
            return q_head[0, 0]

        # -------------------------------------------------------------------------
        # 4. Vectorized Gradient Execution (Empirical & Analytic Fisher)
        # -------------------------------------------------------------------------
        # Critics: Just square the gradients and mean across the batch dimension.
        batched_q1_grad = vmap(grad(critic1_fn, argnums=0), in_dims=(None, 0, 0))
        q1_grads = batched_q1_grad(critic1_params, obs, actions)
        q1_fisher = {name: g.pow(2).mean(dim=0) for name, g in q1_grads.items()}

        batched_q2_grad = vmap(grad(critic2_fn, argnums=0), in_dims=(None, 0, 0))
        q2_grads = batched_q2_grad(critic2_params, obs, actions)
        q2_fisher = {name: g.pow(2).mean(dim=0) for name, g in q2_grads.items()}

        # Actor: Accumulate across the action dimension sequentially.
        actor_mu_fisher = {name: torch.zeros_like(p) for name, p in actor_params.items()}
        actor_std_fisher = {name: torch.zeros_like(p) for name, p in actor_params.items()}

        for k in range(action_dim):
            # We explicitly bind `k` in the closure arguments to avoid Python late-binding loop bugs
            def curr_mu_fn(p, o, k_idx=k):
                return actor_mu_k_fn(p, o, k_idx)

            def curr_std_fn(p, o, k_idx=k):
                return actor_std_k_fn(p, o, k_idx)

            batched_mu_grad = vmap(grad(curr_mu_fn, argnums=0), in_dims=(None, 0))
            mu_grads_k = batched_mu_grad(actor_params, obs)

            batched_std_grad = vmap(grad(curr_std_fn, argnums=0), in_dims=(None, 0))
            std_grads_k = batched_std_grad(actor_params, obs)

            sigma_k = std_standard[:, k]
            mu_weight = 1.0 / (sigma_k.pow(2) + 1e-8)
            std_weight = 2.0 / (sigma_k.pow(2) + 1e-8)

            # Accumulate squared gradients
            for name in actor_params:
                reshape_dims = [-1] + [1] * (mu_grads_k[name].dim() - 1)
                mw = mu_weight.view(*reshape_dims)
                sw = std_weight.view(*reshape_dims)
                
                actor_mu_fisher[name] += (mw * mu_grads_k[name].pow(2)).mean(dim=0)
                actor_std_fisher[name] += (sw * std_grads_k[name].pow(2)).mean(dim=0)

        # -------------------------------------------------------------------------
        # 5. Type-Hint Enforcement & Return
        # -------------------------------------------------------------------------
        # flatten and concatenate the dictionaries into 1D vectors per network.

        def flatten_fisher(fisher_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([f.flatten() for f in fisher_dict.values()])

        critic_coef = 1.0 if self.regularize_critic else 0.0

        return (
            flatten_fisher(actor_mu_fisher),
            flatten_fisher(actor_std_fisher),
            critic_coef * flatten_fisher(q1_fisher),
            critic_coef * flatten_fisher(q2_fisher),
        )



