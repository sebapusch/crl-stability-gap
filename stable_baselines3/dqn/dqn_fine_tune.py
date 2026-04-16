from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual import ContinualLearning
from stable_baselines3.dqn.dqn import DQN


class DQN_FineTune(DQN, ContinualLearning):
    """DQN with weight transfer between tasks.

    On task change, resets optimizer state, exploration counters,
    and replay buffer to match the behavior of constructing a fresh
    model while retaining the learned network weights.
    """

    def on_task_change(self, task_ix: int, env: GymEnv, logger: Logger) -> None:
        self.set_env(env)
        self.set_logger(logger)

        if task_ix == 0:
            return
            
        # Reset optimizer momentum / adaptive state
        self.policy.optimizer.state.clear()

        # Reset exploration counter (target-net update counter)
        self._n_calls = 0

        # Clear replay buffer (each task starts collecting fresh)
        self.replay_buffer.reset()
