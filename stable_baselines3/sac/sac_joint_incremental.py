from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.continual.off_policy_joint_incremental import OffPolicyJointIncremental


class SAC_JointIncremental(OffPolicyJointIncremental, SAC):
    def __init__(self, buffer_size: int, n_tasks: int, env: GymEnv, **kwargs) -> None:
        OffPolicyJointIncremental.__init__(self, buffer_size, n_tasks, env)
        kwargs['env'] = env
        SAC.__init__(self, **kwargs)