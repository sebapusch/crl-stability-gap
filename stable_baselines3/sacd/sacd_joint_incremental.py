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
        kwargs['env'] = env
        SACD.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env)
