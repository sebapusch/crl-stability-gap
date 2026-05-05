import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.dqn.dqn_bc import DQN_BC


class DQN_AEGEM(DQN_BC):
    task_ix: int

    def target(self, batch_size: int) -> th.Tensor:
        assert self.replay_buffer is not None

        replay_data = self.replay_buffer.sample(batch_size)
        discounts = (
            replay_data.discounts if replay_data.discounts is not None else self.gamma
        )

        with th.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = self.q_net_target(replay_data.next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * discounts * next_q_values
            )

        current_q_values = self.q_net(replay_data.observations)

        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = th.gather(
            current_q_values, dim=1, index=replay_data.actions.long()
        )

        # Compute Huber loss (less sensitive to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        return loss

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.task_ix == 0:
            return super().train(gradient_steps, batch_size)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        n_active_tasks = self.task_ix + 1

        losses = []

        for _ in range(gradient_steps):
            grad_new: list[th.Tensor] = []
            grad_old: list[th.Tensor] = []

            loss = self.target(batch_size)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()

            for param in self.policy.q_net.parameters():
                assert param.grad is not None
                grad_new.append(param.grad.clone())

            loss = self.get_auxiliary_loss()
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()

            for param in self.policy.q_net.parameters():
                assert param.grad is not None
                grad_old.append(param.grad.clone())

            for g_new, g_old, param in zip(
                grad_new, grad_old, self.policy.q_net.parameters()
            ):
                grad_joint = (1 / n_active_tasks) * g_new + (
                    1 - 1 / n_active_tasks
                ) * g_old

                cos_sim = (grad_joint * g_old).sum()

                if cos_sim >= 0:
                    param.grad = grad_joint
                else:
                    param.grad = grad_joint - cos_sim / (g_old * g_old).sum() * g_old

            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
