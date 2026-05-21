import argparse
from argparse import Namespace


ENV_NAMES = [
    "pick-place-v3",
    "reach-v3",
    "push-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "window-open-v3",
    "window-close-v3",
]


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # ── Experiment identification ────────────────────────────────────
    parser.add_argument("--name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--seed", type=int)

    # ── Environment ─────────────────────────────────────────────────
    parser.add_argument(
        "--env_name",
        default="pick-place-v3",
        type=str,
        choices=ENV_NAMES,
    )
    parser.add_argument("--max_episode_steps", default=1000, type=int)

    # ── Training hyperparameters ────────────────────────────────────
    parser.add_argument("--total_timesteps", default=1_000_000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--buffer_size", default=1_000_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--learning_starts", default=1000, type=int)

    # ── Network architecture ────────────────────────────────────────
    parser.add_argument(
        "--net_arch",
        default=[256, 256, 256, 256],
        type=int,
        nargs="+",
        help="Hidden layer sizes for the MLP policy",
    )

    # ── Evaluation ──────────────────────────────────────────────────
    parser.add_argument("--eval_freq", default=10_000, type=int)
    parser.add_argument("--n_eval_episodes", default=15, type=int)
    parser.add_argument("--gradient_save_freq", default=1000, type=int)

    return parser.parse_args()
