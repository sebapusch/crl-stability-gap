import argparse
from argparse import Namespace

BENCHMARK = ['reach-v3', 'hammer-v3', 'peg-unplug-side-v3']
METHODS   = ['fine-tune', 'ewc']


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
        '--learning_starts',
        default=10_000,
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
        '--n_eval_episodes',  # how many episodes to run for the evaluation rounds
        default=15,
        type=int,
    )
    parser.add_argument(                    # In the continual_world paper the architecture is:
        '--layer_norm',        # Input -> FC(256) -> LayerNorm -> FC(256) -> FC(256) -> FC(256)
        default=True,                       # This argument controls whether to add layer norm
        type=bool,                          # to our network.
    )
    parser.add_argument(                    # if True: one output head per benchmark env
        '--multi_head_output',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--wandb_project',
        default='crl',
        type=str,
    )
    parser.add_argument(
        '--wandb_name',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--method',
        default='fine-tune',
        choices=METHODS,
        type=str,
    )
    parser.add_argument(
        '--ewc_lambda',
        default=10_000,
        type=float,
    )

    return parser.parse_args()