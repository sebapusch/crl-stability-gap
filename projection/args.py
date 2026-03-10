import argparse
from argparse import Namespace


METHODS = ['sequential', 'fine_tune', 'continual', 'behavior_cloning']
BENCHMARK = ['V1', 'V2', 'V3']
ENVS = ['cartpole', 'inverted_pendulum']

def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--benchmark',
        default=BENCHMARK,
        choices=BENCHMARK,
        nargs='*',
    )
    parser.add_argument(
        '--env',
        choices=ENVS,
        default=ENVS[0],
        type=str,
        help='Environment to use: cartpole (DQN) or inverted_pendulum (SAC)',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    parser.add_argument(
        '--behavior_cloning_coefficient',
        default=100,
        type=float,
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
    )
    parser.add_argument(
        '--name_prefix',
        default='',
        type=str,
    )
    parser.add_argument(
        '--project',
        default='cartpole',
        type=str,
    )
    parser.add_argument(
        '--method',
        choices=METHODS,
        default=METHODS[0],
        type=str,
    )

    parser.add_argument(
        '--encode_task',
        action='store_true',
        help='Append a one-hot encoding of the current task index to the observation',
    )
    parser.add_argument(
        '--balanced_sampling',
        action='store_true',
        help='Whether to maintain original batch size per task on when mode is "continual"',
    )
    parser.add_argument(
        '--eval_all',
        action='store_false',
        default=True,
        help='Whether to only evaluate all environments in the benchmark on every evaluation step',
    )

    # ── Training hyperparameters ────────────────────────────────────
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--video_freq', default=0, type=int)
    parser.add_argument('--n_eval_episodes', default=15, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--buffer_size', default=50_000, type=int)
    parser.add_argument('--target_update', default=1000, type=int)
    parser.add_argument('--learning_starts', default=1000, type=int)
    parser.add_argument('--total_timesteps', default=200_000, type=int)

    # ── DQN-specific (epsilon-greedy) ───────────────────────────────
    parser.add_argument('--epsilon_start', default=1.0, type=float)
    parser.add_argument('--epsilon_end', default=0.05, type=float)
    parser.add_argument('--epsilon_decay_frac', default=0.1, type=float)

    return parser.parse_args()