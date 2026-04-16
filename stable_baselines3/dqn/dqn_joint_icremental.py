from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental


class DQN_JointIncremental(OffPolicyJointIncremental, DQN):
    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, **kwargs) -> None:
        kwargs['env'] = env
        DQN.__init__(self, **kwargs)
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env)
