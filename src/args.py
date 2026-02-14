import argparse
from argparse import Namespace

BENCHMARK = ['reach-v3', 'push-v3', 'pick-place-v3']


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--benchmark',
        default=BENCHMARK,
        choices=BENCHMARK,
        nargs='*',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    parser.add_argument(
        '--total_timesteps',
        default=1_000_000,
        type=int,
    )
    parser.add_argument(
        '--video_freq',
        default=20_000,
        type=int,
    )
    parser.add_argument(
        '--eval_freq',
        default=20_000,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
    )

    return parser.parse_args()