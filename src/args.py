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
    parser.add_argument(
        '--n_eval_episodes',
        default=15,
        type=int,
    )
    parser.add_argument(                # In the continual_world paper the architecture is:
        '--layer_norm',    # Input -> FC(256) -> LayerNorm -> FC(256) -> FC(256) -> FC(256)
        default=True,                   # This argument controls whether to add layer norm
        type=bool,                      # to our network.
    )

    return parser.parse_args()