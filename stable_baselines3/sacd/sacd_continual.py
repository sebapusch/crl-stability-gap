import torch as th

from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental
from stable_baselines3.sacd.sacd import SACD


class SACD_JointIncremental(OffPolicyJointIncremental, SACD):
    """SACD with experience replay across tasks.

    Uses a MultiReplayBuffer that partitions experience by task.
    On task change, resets optimizer state and advances the active
    buffer partition.
    """
    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, **kwargs) -> None:
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env)
        kwargs['env'] = env
        SACD.__init__(self, **kwargs)
