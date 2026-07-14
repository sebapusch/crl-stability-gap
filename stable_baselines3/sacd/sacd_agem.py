import copy
from collections.abc import Iterable

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sacd.sacd_fine_tune import SACD_FineTune


class SACD_AGEM(SACD_FineTune):
    def __init__(
        self,
        buffer_size: int,
        n_tasks: int,
        env: GymEnv,
        balanced_sampling: bool = False,
        **kwargs,
    ):
        super().__init__(buffer_size=buffer_size, env=env, **kwargs)
        self.gradient_memory_buffer = MultiReplayBuffer(
            n_envs=n_tasks,
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            vec_n_envs=self.n_envs,
            device=self.device,
            balanced_sampling=balanced_sampling,
        )
        self.task_ix: int = 0

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        if task_ix > 0:
            assert self.replay_buffer is not None
            self.gradient_memory_buffer.buffers[self.task_ix] = copy.deepcopy(self.replay_buffer)
            self.gradient_memory_buffer.active_index = task_ix - 1

        self.task_ix = task_ix
        super().on_task_change(task_ix, env, logger)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.task_ix == 0:
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        actor_projection_rates: list[float] = []
        critic_projection_rates: list[float] = []

        n_active_tasks = self.task_ix + 1
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())

        for gradient_step in range(gradient_steps):
            current_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            memory_data = self.gradient_memory_buffer.sample(
                batch_size,
                idx=list(range(self.task_ix)),
                env=self._vec_normalize_env,
            )

            critic_loss = self._critic_loss(current_data)
            critic_losses.append(critic_loss.item())
            new_critic_grads = self._gradients_for_loss(
                critic_loss,
                critic_params,
                self.critic.optimizer,
            )

            memory_critic_loss = self._critic_loss(memory_data)
            old_critic_grads = self._gradients_for_loss(
                memory_critic_loss,
                critic_params,
                self.critic.optimizer,
            )

            critic_projection_rates.append(
                self._project_gradients(
                    n_active_tasks,
                    old_critic_grads,
                    new_critic_grads,
                    critic_params,
                )
            )
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)
            self.critic.optimizer.step()

            actor_loss, new_actor_grads, log_action_prob = self._actor_loss_and_gradients(
                current_data,
                actor_params,
            )
            actor_losses.append(actor_loss.item())

            _, old_actor_grads, _ = self._actor_loss_and_gradients(
                memory_data,
                actor_params,
            )

            actor_projection_rates.append(
                self._project_gradients(
                    n_active_tasks,
                    old_actor_grads,
                    new_actor_grads,
                    actor_params,
                )
            )
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
            self.actor.optimizer.step()

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_action_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                self.take_optimisation_step(self.ent_coef_optimizer, None, ent_coef_loss, None)
                self.ent_coef_tensor = th.exp(self.log_ent_coef.detach())

            ent_coefs.append(self.ent_coef_tensor.item())

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/num_projected_critic", np.mean(critic_projection_rates))
        self.logger.record("train/num_projected_actor", np.mean(actor_projection_rates))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _critic_loss(self, replay_data: ReplayBufferSamples) -> th.Tensor:
        discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

        with th.no_grad():
            action_prob, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            next_q_values = th.stack(self.critic_target(replay_data.next_observations), dim=2)
            next_q_values, _ = th.min(next_q_values, dim=2)
            next_q_values = (action_prob * (next_q_values - self.ent_coef_tensor * next_log_prob)).sum(
                dim=1
            ).unsqueeze(-1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

        current_q_values = self.critic(replay_data.observations)
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q.gather(1, replay_data.actions), target_q_values)
            for current_q in current_q_values
        )
        critic_loss += self.get_critic_auxiliary_loss()
        assert isinstance(critic_loss, th.Tensor)
        return critic_loss

    def _actor_loss(self, replay_data: ReplayBufferSamples) -> tuple[th.Tensor, th.Tensor]:
        action_prob, log_action_prob = self.actor.action_log_prob(replay_data.observations)
        q_values_pi = th.stack(self.critic(replay_data.observations), dim=2)
        min_qf_pi, _ = th.min(q_values_pi, dim=2)
        inside_term = self.ent_coef_tensor * log_action_prob - min_qf_pi
        actor_loss = (action_prob * inside_term).sum(dim=1).mean()
        actor_loss += self.get_actor_auxiliary_loss()
        assert isinstance(actor_loss, th.Tensor)
        return actor_loss, log_action_prob

    def _gradients_for_loss(
        self,
        loss: th.Tensor,
        params: Iterable[Parameter],
        optimizer: th.optim.Optimizer,
    ) -> list[th.Tensor]:
        param_list = list(params)
        optimizer.zero_grad()
        loss.backward()
        return [
            param.grad.detach().clone() if param.grad is not None else th.zeros_like(param)
            for param in param_list
        ]

    def _actor_loss_and_gradients(
        self,
        replay_data: ReplayBufferSamples,
        params: Iterable[Parameter],
    ) -> tuple[th.Tensor, list[th.Tensor], th.Tensor]:
        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad_(False)
        try:
            actor_loss, log_action_prob = self._actor_loss(replay_data)
            gradients = self._gradients_for_loss(actor_loss, params, self.actor.optimizer)
            return actor_loss, gradients, log_action_prob
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)

    def _project_gradients(
        self,
        n_active_tasks: int,
        old_grads: list[th.Tensor],
        new_grads: list[th.Tensor],
        params: Iterable[Parameter],
    ) -> float:
        total = 0
        projected = 0

        for old_grad, new_grad, param in zip(old_grads, new_grads, params):
            joint_grad = (1 / n_active_tasks) * new_grad + (1 - 1 / n_active_tasks) * old_grad
            dot = th.sum(joint_grad * old_grad)
            old_norm_sq = th.sum(old_grad * old_grad)

            if dot.item() < 0 and old_norm_sq.item() > 0:
                joint_grad = joint_grad - dot / old_norm_sq * old_grad
                projected += 1

            param.grad = joint_grad.detach().clone()
            total += 1

        return projected / total if total > 0 else 0.0
