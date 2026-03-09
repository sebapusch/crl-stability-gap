import wandb
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium import Env

from callbacks import make_callbacks
from cartpole.benchmarks.permuted_env_benchmark import PermutedEnvBenchmark
from cartpole.common import make_logger
from stable_baselines3.common.buffers import MultiReplayBuffer, ExpertBuffer
from stable_baselines3.dqn import DQN
from args import get_args

def get_benchmark(
        base_env: str | type[Env],
        benchmark: list[str],
        seed: int,
        encode: bool = True,
) -> PermutedEnvBenchmark:
    benchmark = [int(v.strip('V')) for v in benchmark]

    if isinstance(base_env, str):
        match base_env:
            case 'cartpole':
                base_env_cls = CartPoleEnv
            case 'inverted_pendulum':
                base_env_cls = InvertedPendulumEnv
            case _:
                raise ValueError(f'Unknown env {base_env}')
    else:
        base_env_cls = base_env

    return PermutedEnvBenchmark(
        base_env_cls,               # type: ignore
        benchmark,
        encode,
        seed,
        500
    )


def main(
        benchmark: list[str] | None = None,
        seed: int = 42,
        name_prefix: str = '',
        project: str = '',
        method: str = 'sequential',
        eval_freq: int = 5000,
        video_freq: int = 0,
        n_eval_episodes: int = 15,
        lr: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        target_update: int = 1000,
        learning_starts: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frac: float = 0.1,
        total_timesteps: int = 200_000,
        encode_task: bool = False,
        balanced_sampling: bool = False,
        behavior_cloning_coefficient: float = 100
):
    experience_replay = method == 'continual'

    benchmark = get_benchmark(
        CartPoleEnv,
        benchmark or ['V1', 'V2', 'V3'],
        seed,
        encode_task,
    )

    envs_train, envs_test = benchmark.make()

    if experience_replay:
        buffer = MultiReplayBuffer(
            len(benchmark),
            buffer_size,
            envs_train[0].observation_space,
            envs_train[0].action_space,
            balanced_sampling=balanced_sampling,
        )
    else:
        buffer = None

    expert_buffer = None
    if method == 'behavior_cloning':
        expert_buffer = ExpertBuffer(
            buffer_size=1000,
            n_tasks=len(benchmark),
            observation_space=envs_train[0].observation_space,
            output_size=2,
        )

    q_state, q_target_state = None, None

    for ix, v in enumerate(envs_train):
        version = f'V{benchmark.benchmark[ix]}'

        run = wandb.init(
            name=f'{name_prefix}-{version}',
            project=project,
            tags=[version, str(seed), method],
        )

        model = DQN(
            'MlpPolicy',
            envs_train[ix],
            verbose=1,
            learning_rate=lr,
            learning_starts=learning_starts,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_interval=target_update,
            exploration_initial_eps=epsilon_start,
            exploration_final_eps=epsilon_end,
            exploration_fraction=epsilon_decay_frac,
            policy_kwargs={
                'net_arch': [128, 128],
            },
            seed=seed,
            expert_buffer=expert_buffer,
            expert_buffer_batch_size=128,
            behavior_cloning=method == 'behavior_cloning' and ix > 0,
            behavior_cloning_coefficient=behavior_cloning_coefficient,
        )
        model.set_logger(make_logger(run.name))

        if buffer:
            model.replay_buffer = buffer  # type: ignore

        if q_state is not None:
            model.policy.q_net.load_state_dict(q_state)
            model.policy.q_net_target.load_state_dict(q_target_state)

        callbacks = make_callbacks(
            benchmark=benchmark,
            envs_test=envs_test,
            eval_freq=eval_freq,
            video_freq=video_freq,
            n_eval_episodes=n_eval_episodes,
            eval_all=False,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks(ix),
            reset_num_timesteps=False,
        )

        if method in ['continual', 'fine_tune', 'behavior_cloning']:
            q_state = model.policy.q_net.state_dict()
            q_target_state = model.policy.q_net_target.state_dict()

        if method == 'behavior_cloning' and ix < len(benchmark) - 1:
            expert_buffer.populate(
                network=model.policy.q_net_target,
                buffer=model.replay_buffer,
            )

        if experience_replay and ix < len(benchmark) - 1:
            buffer.increase_index()

        run.finish()


if __name__ == '__main__':
    main(**vars(get_args()))
