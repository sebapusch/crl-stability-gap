from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.continual_learning import ContinualLearning
from stable_baselines3.ddpg.ddpg import DDPG


class DDPG_FineTune(DDPG, ContinualLearning):  # pyright: ignore[reportIncompatibleMethodOverride]
    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self.set_env(env)
        self.set_logger(logger)

        if self.replay_buffer:
            self.replay_buffer.reset()

        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()
