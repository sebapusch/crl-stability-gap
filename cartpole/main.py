import wandb

from benchmark import make_benchmark
from callbacks import make_callbacks
from cartpole.benchmark import ContinualCartPole
from cartpole.common import make_logger
from stable_baselines3.common.buffers import MultiReplayBuffer
from stable_baselines3.dqn import DQN
from args import get_args


def main(
        benchmark: list[str] | None = None,
        seed: int = 42,
        name_prefix: str = '',
        project: str = '',
        method: str = 'sequential',
        eval_freq: int = 500,
        video_freq: int = 10_000,
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
):
    experience_replay = method == 'continual'

    benchmark = [ContinualCartPole[version] for version in (benchmark if benchmark else ['V1', 'V2', 'V3'])]

    envs_train = make_benchmark(benchmark, encode_task=encode_task, seed=seed)
    envs_test = make_benchmark(benchmark, encode_task=encode_task, seed=seed + 1)

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

    q_state, q_target_state = None, None

    for ix, v in enumerate(benchmark):
        run = wandb.init(
            name=f'{name_prefix}-{v.name}',
            project=project,
            tags=[v.name, seed, method],
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
            eval_all=True,
            encode_task=encode_task,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks(ix),
            reset_num_timesteps=False,
        )

        if method in ['continual', 'fine_tune']:
            q_state = model.policy.q_net.state_dict()
            q_target_state = model.policy.q_net_target.state_dict()

        if experience_replay:
            buffer.increase_index()

        run.finish()


if __name__ == '__main__':
    main(**vars(get_args()))
