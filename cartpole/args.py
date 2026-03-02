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
        '--method',
        allowed=METHODS,
        default=METHODS[0],
        type=str,
    )


    return parser.parse_args()