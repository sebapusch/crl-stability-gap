import wandb
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium import Env
from torch.optim import Adam, SGD, RMSprop, AdamW, Optimizer

from args import get_args, parse_eval_freq
from callbacks import make_callbacks
from projection.benchmarks.inverted_pendulum_hard import InvertedPendulumHard
from projection.benchmarks.projected_env_benchmark import ProjectedEnvBenchmark
from projection.common import make_logger
from stable_baselines3.continual import ContinualLearning
from stable_baselines3.dqn.dqn_bc import DQN_BC
from stable_baselines3.dqn.dqn_fine_tune import DQN_FineTune
from stable_baselines3.dqn.dqn_joint_icremental import DQN_JointIncremental
from stable_baselines3.sac.sac_bc import SAC_BC
from stable_baselines3.sac.sac_fine_tune import SAC_FineTune
from stable_baselines3.sac.sac_joint_incremental import SAC_JointIncremental
from stable_baselines3.sacd.sacd_bc import SACD_BC
from stable_baselines3.sacd.sacd_joint_incremental import SACD_JointIncremental
from stable_baselines3.sacd.sacd_fine_tune import SACD_FineTune

# ── Environment registry ────────────────────────────────────────────
ENV_REGISTRY: dict[str, tuple[type[Env], int]] = {
    'cartpole':                 (CartPoleEnv,         500),
    'inverted_pendulum':        (InvertedPendulumEnv, 1000),
    'inverted_pendulum_hard':   (InvertedPendulumHard, 1000),
}

# ── Optimizer registry ───────────────────────────────────────────────
OPTIMIZERS: dict[str, tuple[type[Optimizer], dict]] = {
    'adam':             (Adam, {}),
    'sgd':              (SGD, {}),
    'sgd_momentum':     (SGD, {'momentum': 0.9}),
    'rmsprop':          (RMSprop, {}),
    'adamw':            (AdamW, {})
}

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
        method: str,
        behavior_cloning_coefficient: float,
        expert_buffer_size: int,
        expert_buffer_batch_size: int,
        tau: float,
        network_size: int,
        n_tasks: int,
        balanced_sampling: bool,
) -> ContinualLearning:
    common_kwargs = dict(
        policy='MlpPolicy',
        env=train_env,
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
        policy_kwargs={'net_arch': [network_size, network_size]},
        seed=seed,
        tau=tau,
    )

    match method:
        case 'behavior_cloning':
            return DQN_BC(
                expert_buffer_size=expert_buffer_size,
                n_tasks=n_tasks,
                expert_buffer_batch_size=expert_buffer_batch_size,
                lambda_=behavior_cloning_coefficient,
                **common_kwargs,
            )
        case 'fine_tune':
            return DQN_FineTune(**common_kwargs)
        case 'joint_incremental':
            return DQN_JointIncremental(
                n_tasks=n_tasks,
                **common_kwargs
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


def _build_sacd(
        train_env: Env,
        *,
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        seed: int,
        method: str,
        behavior_cloning_coefficient: float,
        expert_buffer_size: int,
        expert_buffer_batch_size: int,
        ent_coef: float | None,
        network_size: int,
        n_tasks: int,
        balanced_sampling: bool,
) -> ContinualLearning:
    common_kwargs = dict(
        policy='MlpPolicy',
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs={'net_arch': [network_size, network_size]},
        seed=seed,
        ent_coef='auto' if ent_coef is None else ent_coef,
    )

    match method:
        case 'behavior_cloning':
            return SACD_BC(
                expert_buffer_size=expert_buffer_size,
                n_tasks=n_tasks,
                expert_buffer_batch_size=expert_buffer_batch_size,
                lambda_=behavior_cloning_coefficient,
                **common_kwargs,
            )
        case 'fine_tune':
            return SACD_FineTune(**common_kwargs)
        case 'joint_incremental':
            return SACD_JointIncremental(
                n_tasks=n_tasks,
                **common_kwargs,
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


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
        behavior_cloning_coefficient: float,
        expert_buffer_size: int,
        expert_buffer_batch_size: int,
        bc_loss_fn: str,
        ent_coef: float | None,
        network_size: int,
        n_tasks: int,
        balanced_sampling: bool,
) -> ContinualLearning:
    common_kwargs = dict(
        policy='MlpPolicy',
        env=train_env,
        verbose=1,
        learning_rate=lr,
        learning_starts=learning_starts,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs={'net_arch': [network_size, network_size]},
        seed=seed,
        ent_coef='auto' if ent_coef is None else ent_coef,
    )

    match method:
        case 'behavior_cloning':
            return SAC_BC(
                expert_buffer_size=expert_buffer_size,
                n_tasks=n_tasks,
                expert_buffer_batch_size=expert_buffer_batch_size,
                lambda_=behavior_cloning_coefficient,
                loss_fn=bc_loss_fn,
                **common_kwargs,
            )
        case 'fine_tune':
            return SAC_FineTune(**common_kwargs)
        case 'joint_incremental':
            return SAC_JointIncremental(
                n_tasks=n_tasks,
                **common_kwargs,
            )
        case _:
            raise ValueError(f'Unknown method "{method}"')


# ── Main training loop ──────────────────────────────────────────────
def main(
        benchmark: list[str] | None = None,
        env: str = 'cartpole',
        seed: int = 42,
        name_prefix: str = '',
        project: str = '',
        method: str = 'fine_tune',
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
        expert_buffer_batch_size: int = 128,
        expert_buffer_size: int = 1000,
        eval_all: bool = True,
        bc_loss_fn: str = 'kl',
        algorithm: str = 'dqn',
        ent_coef: float | None = None,
        dqn_tau: float = 1.0,
        network_size: int | None = None,
        ewc_lambda: float = 1.0,
        optimizer: str = 'adam',
):
    bench = get_benchmark(env, benchmark or ['V1', 'V2', 'V3'], seed, encode_task)
    envs_train, envs_test = bench.make()

    # ── Common builder kwargs ────────────────────────────────────────
    common_build_kwargs = dict(
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        seed=seed,
        method=method,
        behavior_cloning_coefficient=behavior_cloning_coefficient,
        expert_buffer_size=expert_buffer_size,
        expert_buffer_batch_size=expert_buffer_batch_size,
        network_size=network_size,
        n_tasks=len(bench),
        balanced_sampling=balanced_sampling,
        policy_kwargs=dict(
            optimizer=OPTIMIZERS[optimizer][0],
            optimizer_kwargs=OPTIMIZERS[optimizer][1],
        )
    )

    dqn_build_kwargs = dict(
        **common_build_kwargs,
        target_update=target_update,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frac=epsilon_decay_frac,
        tau=dqn_tau,
    )

    sac_build_kwargs = dict(
        **common_build_kwargs,
        bc_loss_fn=bc_loss_fn,
        ent_coef=ent_coef,
        ewc_lambda=ewc_lambda,
    )

    sacd_build_kwargs = dict(
        **common_build_kwargs,
        ent_coef=ent_coef,
    )

    # ── Build model ─────────────────────────────────────
    train_env_init = envs_train[0]
    match algorithm:
        case 'dqn':
            config = dqn_build_kwargs
            model = _build_dqn(train_env_init, **dqn_build_kwargs)
        case 'sac':
            config = sac_build_kwargs
            model = _build_sac(train_env_init, **sac_build_kwargs)
        case 'sacd':
            config = sacd_build_kwargs
            model = _build_sacd(train_env_init, **sacd_build_kwargs)
        case _:
            raise ValueError(f'Unknown algorithm "{algorithm}"')

    for ix, train_env in enumerate(envs_train):
        version = f'V{bench.benchmark[ix]}'

        run = wandb.init(
            name=f'{name_prefix}-{version}',
            project=project,
            config=config,
            tags=[version, str(seed), method],
        )

        model.on_task_change(ix, train_env, make_logger(project, run.name))

        # ── Train ───────────────────────────────────────────────────
        callbacks = make_callbacks(
            benchmark=bench,
            envs_test=envs_test,
            eval_freq=eval_freq,
            video_freq=video_freq,
            n_eval_episodes=n_eval_episodes,
            eval_all=eval_all,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks(ix),
            reset_num_timesteps=True,
        )

        run.finish()


if __name__ == '__main__':
    args = vars(get_args())
    args['eval_freq'] = parse_eval_freq(args['eval_freq'], args['total_timesteps'])
    main(**args)
