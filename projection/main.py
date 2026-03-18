from collections.abc import Callable

import torch
import wandb
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium import Env

from args import get_args, parse_eval_freq
from callbacks import make_callbacks
from projection.benchmarks.projected_env_benchmark import ProjectedEnvBenchmark
from projection.common import make_logger
from stable_baselines3.common.buffers import MultiReplayBuffer, ExpertBuffer
from stable_baselines3.dqn import DQN
from stable_baselines3.sac import SAC
from stable_baselines3.sac.sac_bc import SAC_BC, gaussian_kl, l2

# ── Environment registry ────────────────────────────────────────────
ENV_REGISTRY: dict[str, tuple[type[Env], int]] = {
    'cartpole':           (CartPoleEnv,         500),
    'inverted_pendulum':  (InvertedPendulumEnv, 1000),
}

TRANSFER_METHODS = {'fine_tune', 'continual', 'behavior_cloning'}


def get_benchmark(
        env: str,
        benchmark: list[str],
        seed: int,
        encode: bool = True,
) -> ProjectedEnvBenchmark:
    env_cls, time_limit = ENV_REGISTRY[env]
    versions = [int(v.strip('V')) for v in benchmark]

    return ProjectedEnvBenchmark(
        env_cls,                     # type: ignore
        versions,
        encode,
        seed,
        time_limit,
    )


# ── Algorithm construction ──────────────────────────────────────────
def _build_dqn(
        train_env: Env,
        *,
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        target_update: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_frac: float,
        seed: int,
        expert_buffer: ExpertBuffer | None,
        behavior_cloning: bool,
        behavior_cloning_coefficient: float,
) -> DQN:
    return DQN(
        'MlpPolicy',
        train_env,
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
        policy_kwargs={'net_arch': [128, 128]},
        seed=seed,
        expert_buffer=expert_buffer,
        expert_buffer_batch_size=128,
        behavior_cloning=behavior_cloning,
        behavior_cloning_coefficient=behavior_cloning_coefficient,
    )


def _build_sac(
        train_env: Env,
        *,
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        seed: int,
        method: str,
        num_tasks: int,
        behavior_cloning_coefficient: float,
        expert_buffer: ExpertBuffer | None,
        bc_loss_fn: str,
) -> SAC:
    common_kwargs = dict(
        policy='MlpPolicy',
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs={'net_arch': [256, 256]},
        seed=seed,
    )

    if method == 'behavior_cloning':
        assert expert_buffer is not None

        return SAC_BC(
            expert_buffer=expert_buffer,
            expert_buffer_batch_size=128,
            lambda_=behavior_cloning_coefficient,
            bc_loss_fn=gaussian_kl if bc_loss_fn == 'kl' else l2,
            **common_kwargs,
        )

    return SAC(**common_kwargs)


# ── Weight transfer helpers ─────────────────────────────────────────
def _save_weights(model: DQN | SAC) -> dict:
    """Snapshot the network weights needed to warm-start the next task."""
    if isinstance(model, DQN):
        return {
            'q_net': model.policy.q_net.state_dict(),
            'q_net_target': model.policy.q_net_target.state_dict(),
        }
    return {
        'actor': model.actor.state_dict(),
        'critic': model.critic.state_dict(),
        'critic_target': model.critic_target.state_dict(),
    }


def _load_weights(model: DQN | SAC, state: dict) -> None:
    """Restore previously saved weights into *model*."""
    if isinstance(model, DQN):
        model.policy.q_net.load_state_dict(state['q_net'])
        model.policy.q_net_target.load_state_dict(state['q_net_target'])
    else:
        model.actor.load_state_dict(state['actor'])
        model.critic.load_state_dict(state['critic'])
        model.critic_target.load_state_dict(state['critic_target'])


def _get_expert_targets(model: SAC) -> Callable[[torch.Tensor], torch.Tensor]:
    def get_targets(obs: torch.Tensor) -> torch.Tensor:
        mu, log_std, _ = model.actor.get_action_dist_params(obs)
        return torch.concat([mu, log_std], 1)
    return get_targets


# ── Main training loop ──────────────────────────────────────────────
def main(
        benchmark: list[str] | None = None,
        env: str = 'cartpole',
        seed: int = 42,
        name_prefix: str = '',
        project: str = '',
        method: str = 'sequential',
        eval_freq: int | list[tuple[int, int]] = 500,
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
        behavior_cloning_coefficient: float = 100,
        expert_buffer_size: int = 1000,
        eval_all: bool = True,
        bc_loss_fn: str = 'kl'
):
    use_dqn = env == 'cartpole'
    experience_replay = method == 'continual'

    bench = get_benchmark(env, benchmark or ['V1', 'V2', 'V3'], seed, encode_task)
    envs_train, envs_test = bench.make()

    # ── Shared replay buffer (continual learning only) ──────────────
    buffer = None
    if experience_replay:
        buffer = MultiReplayBuffer(
            len(bench),
            buffer_size,
            envs_train[0].observation_space,
            envs_train[0].action_space,
            balanced_sampling=balanced_sampling,
        )

    # ── DQN-specific expert buffer (behavior cloning only) ──────────
    expert_buffer = None
    if method == 'behavior_cloning':
        if use_dqn:
            expert_output_size = envs_train[0].action_space.n
        else:
            expert_output_size = 2 * envs_train[0].action_space.shape[0]
        expert_buffer = ExpertBuffer(
            buffer_size=expert_buffer_size,
            n_tasks=len(bench),
            observation_space=envs_train[0].observation_space,
            output_size=expert_output_size,
        )

    saved_weights: dict | None = None

    for ix, train_env in enumerate(envs_train):
        version = f'V{bench.benchmark[ix]}'

        run = wandb.init(
            name=f'{name_prefix}-{version}',
            project=project,
            tags=[version, str(seed), method],
        )

        # ── Build algorithm ─────────────────────────────────────────
        if use_dqn:
            model = _build_dqn(
                train_env,
                lr=lr,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                target_update=target_update,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_frac=epsilon_decay_frac,
                seed=seed,
                expert_buffer=expert_buffer,
                behavior_cloning=method == 'behavior_cloning' and ix > 0,
                behavior_cloning_coefficient=behavior_cloning_coefficient,
            )
        else:
            model = _build_sac(
                train_env,
                lr=lr,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                seed=seed,
                method=method,
                num_tasks=len(bench),
                behavior_cloning_coefficient=behavior_cloning_coefficient,
                expert_buffer=expert_buffer,
                bc_loss_fn=bc_loss_fn,
            )

        model.set_logger(make_logger(run.name))

        if buffer:
            model.replay_buffer = buffer  # type: ignore

        if saved_weights is not None:
            _load_weights(model, saved_weights)

        # ── Train ───────────────────────────────────────────────────
        callbacks = make_callbacks(
            benchmark=bench,
            envs_test=envs_test,
            eval_freq=eval_freq,
            video_freq=video_freq,
            n_eval_episodes=n_eval_episodes,
            eval_all=eval_all,
        )

        learn_kwargs = {
            'total_timesteps': total_timesteps,
            'callback': callbacks(ix),
            'reset_num_timesteps': False,
        }

        if not use_dqn:
            learn_kwargs['task_ix'] = ix

        model.learn(**learn_kwargs)

        # ── Post-training bookkeeping ───────────────────────────────
        if method in TRANSFER_METHODS:
            saved_weights = _save_weights(model)

        is_last_task = ix >= len(bench) - 1

        if method == 'behavior_cloning' and not is_last_task:
            network = model.q_net_target if use_dqn else _get_expert_targets(model)
            expert_buffer.populate(
                network=network,
                buffer=model.replay_buffer,
            )

        # if not use_dqn and isinstance(model, SAC_BC) and not is_last_task:
        #     model.on_task_change(ix)

        if experience_replay and not is_last_task:
            buffer.increase_index()

        run.finish()


if __name__ == '__main__':
    args = vars(get_args())
    args['eval_freq'] = parse_eval_freq(args['eval_freq'], args['total_timesteps'])
    main(**args)
