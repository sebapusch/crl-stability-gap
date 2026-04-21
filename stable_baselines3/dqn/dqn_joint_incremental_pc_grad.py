import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.pc_grad import pc_grad_losses

from stable_baselines3.dqn.dqn_joint_icremental import DQN_JointIncremental


class DQN_JointIncremental_PCGrad(DQN_JointIncremental):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        n_active_tasks = self.task_ix + 1

        losses = []

        for _ in range(gradient_steps):
            task_losses = []

            for task_ix in range(self.task_ix + 1):
                task_batch_size = batch_size // n_active_tasks
                if task_ix == self.task_ix:
                    task_batch_size += batch_size % n_active_tasks

                replay_data = self.replay_buffer.buffers[task_ix].sample(task_batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                with th.no_grad():
                    # Compute the next Q-values using the target network
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    # 1-step TD target
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                # Get current Q-values estimates
                current_q_values = self.q_net(replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                task_losses.append(loss)

            pc_grad_losses(
                task_losses,
                self.policy.optimizer,
                self.policy.parameters(),
            )

            losses.append(np.mean([l.item() for l in task_losses]))

            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def on_task_change(
            self,
            task_ix: int,
            env: GymEnv,
            logger: Logger,
    ) -> None:
        super().on_task_change(task_ix, env, logger)


        