import argparse
from argparse import Namespace


METHODS = ['sequential', 'fine_tune', 'continual']


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        default=42,
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

    return parser.parse_args()