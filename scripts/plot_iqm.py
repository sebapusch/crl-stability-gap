#!/usr/bin/env python3
"""
Generate IQM plots with 95% CI for each test environment.

Produces 3 plots (one per test env V1, V2, V3). Each plot has lines
(one per method) showing the IQM of the mean reward with a shaded
95% confidence interval across 5 seeds.

Usage:
    python plot_iqm.py --methods continual sequential fine_tune
    python plot_iqm.py --methods continual_encode --prefix encode
    python plot_iqm.py  # defaults to continual, sequential, fine_tune
"""

import argparse
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "output" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "plots"
CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"

DEFAULT_METHODS = ["continual", "sequential", "fine_tune"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_ENVS = ["V1", "V2", "V3"]
TEST_ENVS = ["V1", "V2", "V3"]
TIMESTEPS_PER_ENV = 40_000

# Known labels; unknown methods get auto-generated labels
METHOD_LABELS = {
    "continual": "Continual",
    "sequential": "Sequential",
    "fine_tune": "Fine-tune",
    "continual_encode": "Continual (encode)",
}

# A palette that cycles for any number of methods
COLOR_PALETTE = [
    "#2196F3",  # blue
    "#FF9800",  # orange
    "#4CAF50",  # green
    "#E91E63",  # pink
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
]

N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95


def get_label(method: str) -> str:
    """Return a display label for a method, auto-generating if unknown."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return method.replace("_", " ").title()


def get_color(method: str, index: int) -> str:
    """Return a color for a method, cycling through the palette."""
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def interquartile_mean(values: np.ndarray) -> float:
    """Compute the interquartile mean (IQM) of a 1-D array."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    if q3_idx <= q1_idx:
        return np.mean(sorted_vals)
    return np.mean(sorted_vals[q1_idx:q3_idx])


def interquartile_mean_batch(values: np.ndarray) -> np.ndarray:
    """Compute IQM along axis=1 of a 2-D array (vectorised)."""
    sorted_vals = np.sort(values, axis=1)
    n = sorted_vals.shape[1]
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    if q3_idx <= q1_idx:
        return np.mean(sorted_vals, axis=1)
    return np.mean(sorted_vals[:, q1_idx:q3_idx], axis=1)


def bootstrap_iqm(seed_values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP):
    """
    Compute IQM and 95% CI via bootstrap over seeds.

    Parameters
    ----------
    seed_values : array of shape (n_seeds,) or (n_timesteps, n_seeds)
    n_bootstrap : number of bootstrap resamples

    Returns
    -------
    iqm : float or array
    ci_low : float or array
    ci_high : float or array
    """
    if seed_values.ndim == 1:
        # --- single-timestep path (original behaviour) ---
        n_seeds = len(seed_values)
        if n_seeds == 0:
            return np.nan, np.nan, np.nan

        rng = np.random.default_rng(42)
        # Vectorised: draw all bootstrap indices at once
        indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
        boot_samples = seed_values[indices]          # (n_bootstrap, n_seeds)
        boot_iqms = interquartile_mean_batch(boot_samples)

        alpha = (1 - CONFIDENCE) / 2
        ci_low = np.percentile(boot_iqms, 100 * alpha)
        ci_high = np.percentile(boot_iqms, 100 * (1 - alpha))
        iqm = interquartile_mean(seed_values)
        return iqm, ci_low, ci_high

    # --- batched path: seed_values is (n_timesteps, n_seeds) ---
    n_ts, n_seeds = seed_values.shape
    rng = np.random.default_rng(42)
    # Same bootstrap indices for every timestep (matches original per-row RNG
    # reset to seed 42).
    indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    # boot_samples: (n_ts, n_bootstrap, n_seeds)
    boot_samples = seed_values[:, indices]  # advanced indexing broadcasts

    # Compute IQM for each (ts, bootstrap) pair
    sorted_bs = np.sort(boot_samples, axis=2)
    q1_idx = int(np.floor(n_seeds * 0.25))
    q3_idx = int(np.ceil(n_seeds * 0.75))
    if q3_idx <= q1_idx:
        boot_iqms = np.mean(sorted_bs, axis=2)      # (n_ts, n_bootstrap)
    else:
        boot_iqms = np.mean(sorted_bs[:, :, q1_idx:q3_idx], axis=2)

    alpha = (1 - CONFIDENCE) / 2
    ci_low = np.percentile(boot_iqms, 100 * alpha, axis=1)
    ci_high = np.percentile(boot_iqms, 100 * (1 - alpha), axis=1)

    # Point IQM per timestep
    sorted_sv = np.sort(seed_values, axis=1)
    if q3_idx <= q1_idx:
        iqm = np.mean(sorted_sv, axis=1)
    else:
        iqm = np.mean(sorted_sv[:, q1_idx:q3_idx], axis=1)

    return iqm, ci_low, ci_high


def load_eval_data(method: str, seed: int, test_env: str) -> pd.DataFrame:
    """
    Load and concatenate eval data for a given method, seed, and test env
    across all 3 training environments.

    Returns a DataFrame with columns ['timestep', 'reward'].
    """
    all_timesteps = []

    reward_col = f"eval/{test_env}/mean_reward"
    timestep_col = f"time/{test_env}/total_timesteps"

    for env_idx, train_env in enumerate(TRAIN_ENVS):
        filename = f"{method}-{train_env}.csv".replace('<s>', str(seed))
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.")
            continue

        if filepath.stat().st_size == 0:
            print(f"Warning: {filepath} is empty, skipping.")
            continue

        df = pd.read_csv(filepath)

        if reward_col not in df.columns or timestep_col not in df.columns:
            continue

        # Extract only rows with eval data for this test env
        mask = df[reward_col].notna() & df[timestep_col].notna()
        subset = df.loc[mask, [timestep_col, reward_col]].copy()
        subset.columns = ["timestep", "reward"]

        # Offset timesteps by training env index
        subset["timestep"] = subset["timestep"] + env_idx * TIMESTEPS_PER_ENV

        all_timesteps.append(subset)

    if not all_timesteps:
        return pd.DataFrame(columns=["timestep", "reward"])

    result = pd.concat(all_timesteps, ignore_index=True)
    result = result.sort_values("timestep").reset_index(drop=True)
    return result


def compute_iqm_curve(method: str, test_env: str):
    """
    Compute IQM + 95% CI curve for a method on a test environment.

    Returns
    -------
    timesteps : array
    iqm_values : array
    ci_low : array
    ci_high : array
    """
    # Load all seeds
    seed_frames = []
    for seed in SEEDS:
        df = load_eval_data(method, seed, test_env)
        if not df.empty:
            df = df.copy()
            df["seed"] = seed
            seed_frames.append(df)

    if not seed_frames:
        return np.array([]), np.array([]), np.array([]), np.array([])

    combined = pd.concat(seed_frames, ignore_index=True)

    # Pivot to (timestep × seed) matrix; NaN where a seed has no data
    pivot = combined.pivot_table(
        index="timestep", columns="seed", values="reward", aggfunc="first"
    )
    # Keep only timesteps with >= 2 seeds present
    seed_counts = pivot.notna().sum(axis=1)
    pivot = pivot.loc[seed_counts >= 2]

    if pivot.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    valid_ts = pivot.index.values

    # Group rows by their set of available seeds so we can batch-bootstrap
    # rows that share the same seed availability pattern together.
    present_mask = pivot.notna().values  # (n_ts, n_seed_cols)
    # Encode each row's pattern as a hashable tuple
    patterns = [tuple(row) for row in present_mask]
    unique_patterns = list(set(patterns))

    iqm_values = np.empty(len(valid_ts))
    ci_lows = np.empty(len(valid_ts))
    ci_highs = np.empty(len(valid_ts))

    for pat in unique_patterns:
        row_indices = np.array([i for i, p in enumerate(patterns) if p == pat])
        col_mask = np.array(pat)
        # Extract the dense (n_rows, n_present_seeds) sub-matrix
        seed_matrix = pivot.values[np.ix_(row_indices, col_mask)]
        iqm, cl, ch = bootstrap_iqm(seed_matrix)
        iqm_values[row_indices] = iqm
        ci_lows[row_indices] = cl
        ci_highs[row_indices] = ch

    return valid_ts, iqm_values, ci_lows, ci_highs


def make_cache_key(methods: list[str], prefix: str) -> str:
    """Generate a short hash from the sorted methods + prefix combination."""
    canonical = "|".join(sorted(methods)) + "||" + prefix
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def cache_path_for(cache_key: str, method: str, test_env: str) -> Path:
    """Return the CSV cache path for a specific method/test_env under a cache key."""
    return CACHE_DIR / cache_key / f"{method}_{test_env}.csv"


def save_to_cache(cache_key: str, method: str, test_env: str,
                  ts: np.ndarray, iqm: np.ndarray,
                  ci_lo: np.ndarray, ci_hi: np.ndarray):
    """Save computed IQM curve to a CSV cache file."""
    path = cache_path_for(cache_key, method, test_env)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "timestep": ts, "iqm": iqm, "ci_low": ci_lo, "ci_high": ci_hi,
    })
    df.to_csv(path, index=False)


def load_from_cache(cache_key: str, method: str, test_env: str):
    """Load cached IQM curve. Returns (ts, iqm, ci_lo, ci_hi) or None."""
    path = cache_path_for(cache_key, method, test_env)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return (
        df["timestep"].values, df["iqm"].values,
        df["ci_low"].values, df["ci_high"].values,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IQM plots with 95% CI for each test environment."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Method names to plot (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filename prefix for output plots, e.g. 'encode' -> iqm_encode_V1.png",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument(
        "--envs", "--env_order",
        nargs="+",
        default=TEST_ENVS,
        help=f"Environments to plot and their training order (default: {TEST_ENVS})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Seeds to include (default: {SEEDS})",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS_PER_ENV,
        help=f"Timesteps per environment (default: {TIMESTEPS_PER_ENV})",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help=f"Base name of the environment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Subdirectory inside output/plots to save plots (created if not exists)",
    )
    return parser.parse_args()


def main():
    global TRAIN_ENVS, TEST_ENVS, SEEDS, TIMESTEPS_PER_ENV
    args = parse_args()
    methods = args.methods
    prefix = args.prefix
    use_cache = not args.no_cache
    TRAIN_ENVS = args.envs
    TEST_ENVS = args.envs
    SEEDS = args.seeds
    TIMESTEPS_PER_ENV = args.timesteps
    env_name = args.env_name
    output_subdir = args.output_dir

    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR

    plot_output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = make_cache_key(methods, prefix)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")
    
    for test_env in TEST_ENVS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, method in enumerate(methods):
            # Try loading from cache first
            cached = None
            if use_cache:
                cached = load_from_cache(cache_key, method, test_env)

            if cached is not None:
                ts, iqm, ci_lo, ci_hi = cached
                print(f"Loaded cached IQM for {method} on {test_env}")
            else:
                print(f"Computing IQM for {method} on {test_env}...")
                ts, iqm, ci_lo, ci_hi = compute_iqm_curve(method, test_env)
                if len(ts) > 0:
                    save_to_cache(cache_key, method, test_env, ts, iqm, ci_lo, ci_hi)

            if len(ts) == 0:
                print(f"  No data for {method}/{test_env}")
                continue

            label = get_label(prefix)
            color = get_color(method, idx)
            ax.plot(ts, iqm, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        # Add vertical lines at environment boundaries
        for i in range(1, len(TRAIN_ENVS)):
            ax.axvline(
                x=i * TIMESTEPS_PER_ENV,
                color="black",
                linestyle="-",
                alpha=0.6,
                linewidth=0.5,
                zorder=5,
            )

        # Add env labels just above the plot area
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i, env in enumerate(TRAIN_ENVS):
            center = (i + 0.5) * TIMESTEPS_PER_ENV
            ax.text(
                center,
                1.02,
                f"Train {env}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
                transform=trans,
            )

        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("IQM Episodic Return")
        ax.set_title(f"Evaluation on {env_name}-{test_env}", pad=25)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Build output filename
        parts = ["iqm"]
        if prefix:
            parts.append(prefix)
        parts.append(test_env)
        out_path = plot_output_dir / f"{'_'.join(parts)}.png"

        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
