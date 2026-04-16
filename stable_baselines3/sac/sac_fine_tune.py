import torch as th

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual import ContinualLearning
from stable_baselines3.sac.sac import SAC


class SAC_FineTune(SAC, ContinualLearning):
    """SAC with weight transfer between tasks.

    On task change, resets optimizer state and entropy coefficient
    to match the behavior of constructing a fresh model, while
    retaining the learned network weights.
    """

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self.set_env(env)
        self.set_logger(logger)
        
        if task_ix == 0:
            return
            
        # Reset optimizer momentum / adaptive state
        self.actor.optimizer.state.clear()
        self.critic.optimizer.state.clear()

        # Reset entropy coefficient to its initial value
        if self.ent_coef_optimizer is not None:
            self.log_ent_coef.data.fill_(0.0)
            self.ent_coef_optimizer.state.clear()

        # Clear replay buffer (each task starts collecting fresh)
        self.replay_buffer.reset()
