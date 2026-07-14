from collections.abc import Iterable

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.dqn.dqn_joint_icremental import DQN_JointIncremental


class DQN_JointIncremental_AGEM(DQN_JointIncremental):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.task_ix == 0:
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        n_active_tasks = self.task_ix + 1
        q_net_params = list(self.policy.q_net.parameters())

        losses: list[float] = []
        memory_losses: list[float] = []
        projection_rates: list[float] = []

        for _ in range(gradient_steps):
            current_loss = self._sample_loss(self.task_ix, batch_size)
            losses.append(current_loss.item())
            new_grads = self._gradients_for_loss(
                current_loss,
                q_net_params,
                self.policy.optimizer,
            )

            memory_loss = self._sample_loss(list(range(self.task_ix)), batch_size)
            memory_losses.append(memory_loss.item())
            old_grads = self._gradients_for_loss(
                memory_loss,
                q_net_params,
                self.policy.optimizer,
            )

            projection_rates.append(
                self._project_gradients(
                    n_active_tasks,
                    old_grads,
                    new_grads,
                    q_net_params,
                )
            )

            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/memory_loss", np.mean(memory_losses))
        self.logger.record("train/num_projected", np.mean(projection_rates))

    def _sample_loss(self, idx: int | list[int], batch_size: int) -> th.Tensor:
        replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
            batch_size,
            idx=idx,
            env=self._vec_normalize_env,
        )
        return self._td_loss(replay_data)

    def _td_loss(self, replay_data: ReplayBufferSamples) -> th.Tensor:
        discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

        with th.no_grad():
            next_q_values = self.q_net_target(replay_data.next_observations)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

        current_q_values = self.q_net(replay_data.observations)
        current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

        return F.smooth_l1_loss(current_q_values, target_q_values)

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
