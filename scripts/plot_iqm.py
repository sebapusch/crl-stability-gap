#!/usr/bin/env python3
"""
Generate IQM plots with 95% CI for each test environment.

Produces 3 plots (one per test env V1, V2, V3). Each plot has 3 lines
(one per method: continual, sequential, fine_tune) showing the IQM of
the mean reward with a shaded 95% confidence interval across 5 seeds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "output" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "plots"

METHODS = ["continual"]
SEEDS = [1, 2, 3, 4, 5]
TRAIN_ENVS = ["V1", "V2", "V3"]
TEST_ENVS = ["V1", "V2", "V3"]
TIMESTEPS_PER_ENV = 200_000

METHOD_LABELS = {
    "continual": "Continual",
    "sequential": "Sequential",
    "fine_tune": "Fine-tune",
}

N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95


def interquartile_mean(values: np.ndarray) -> float:
    """Compute the interquartile mean (IQM) of an array."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    if q3_idx <= q1_idx:
        return np.mean(sorted_vals)
    return np.mean(sorted_vals[q1_idx:q3_idx])


def bootstrap_iqm(seed_values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP):
    """
    Compute IQM and 95% CI via bootstrap over seeds.

    Parameters
    ----------
    seed_values : array of shape (n_seeds,)
    n_bootstrap : number of bootstrap resamples

    Returns
    -------
    iqm : float
    ci_low : float
    ci_high : float
    """
    n_seeds = len(seed_values)
    if n_seeds == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(42)
    boot_iqms = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(seed_values, size=n_seeds, replace=True)
        boot_iqms[b] = interquartile_mean(sample)

    alpha = (1 - CONFIDENCE) / 2
    ci_low = np.percentile(boot_iqms, 100 * alpha)
    ci_high = np.percentile(boot_iqms, 100 * (1 - alpha))
    iqm = interquartile_mean(seed_values)
    return iqm, ci_low, ci_high


def load_eval_data(method: str, seed: int, test_env: str) -> pd.DataFrame:
    """
    Load and concatenate eval data for a given method, seed, and test env
    across all 3 training environments.

    Returns a DataFrame with columns ['timestep', 'reward'].
    """
    all_timesteps = []
    all_rewards = []

    reward_col = f"eval/{test_env}/mean_reward"
    timestep_col = f"time/{test_env}/total_timesteps"

    for env_idx, train_env in enumerate(TRAIN_ENVS):
        filename = f"{method}-s{seed}-{train_env}.csv"
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.")
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
    seed_data = {}
    for seed in SEEDS:
        df = load_eval_data(method, seed, test_env)
        if not df.empty:
            seed_data[seed] = df

    if not seed_data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Get union of all timesteps across seeds
    all_ts = set()
    for df in seed_data.values():
        all_ts.update(df["timestep"].values)
    all_ts = np.sort(list(all_ts))

    # For each timestep, collect rewards from each seed (use nearest available)
    iqm_values = []
    ci_lows = []
    ci_highs = []
    valid_ts = []

    for ts in tqdm.tqdm(all_ts):
        rewards = []
        for seed, df in seed_data.items():
            # Find the exact timestep match
            match = df.loc[df["timestep"] == ts, "reward"]
            if len(match) > 0:
                rewards.append(match.values[0])

        if len(rewards) >= 2:  # Need at least 2 seeds for meaningful IQM
            rewards = np.array(rewards)
            iqm, cl, ch = bootstrap_iqm(rewards)
            iqm_values.append(iqm)
            ci_lows.append(cl)
            ci_highs.append(ch)
            valid_ts.append(ts)

    return np.array(valid_ts), np.array(iqm_values), np.array(ci_lows), np.array(ci_highs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "continual": "#2196F3",
        "sequential": "#FF9800",
        "fine_tune": "#4CAF50",
    }

    for test_env in TEST_ENVS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for method in METHODS:
            print(f"Computing IQM for {method} on {test_env}...")
            ts, iqm, ci_lo, ci_hi = compute_iqm_curve(method, test_env)

            if len(ts) == 0:
                print(f"  No data for {method}/{test_env}")
                continue

            label = METHOD_LABELS[method]
            color = colors[method]
            ax.plot(ts, iqm, label=label, color=color, linewidth=1.5)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        # Add vertical lines at environment boundaries
        for i in range(1, len(TRAIN_ENVS)):
            ax.axvline(
                x=i * TIMESTEPS_PER_ENV,
                color="gray",
                linestyle="--",
                alpha=0.5,
                linewidth=0.8,
            )

        # Add env labels at top
        for i, env in enumerate(TRAIN_ENVS):
            center = (i + 0.5) * TIMESTEPS_PER_ENV
            ax.text(
                center,
                ax.get_ylim()[1],
                f"Train {env}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
            )

        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("IQM Episodic Return")
        ax.set_title(f"Evaluation on {test_env}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"iqm_{test_env}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
