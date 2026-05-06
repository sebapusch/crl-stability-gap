#!/usr/bin/env python3
"""
Compute aggregated Final Average Performance P(T) for RL ablation experiments.

Follows the Continual World benchmark metric:
  1. For each task i in seed s, smooth the final score p_si(T) by averaging the
     last X evaluation points.
  2. Per-seed average: P_s(T) = (1/N) * sum_i p_si(T)
  3. Aggregate across seeds: mean of P_s(T), with 90% bootstrap CI.

Usage:
  python compute_final_performance.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml
  python compute_final_performance.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml --smooth 10
  python compute_final_performance.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml --output_dir output/tables
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


DATA_DIR = Path(__file__).resolve().parent.parent / "output" / "output"
DEFAULT_SMOOTH = 5
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.90


# ---------------------------------------------------------------------------
# Value formatting — mirrors dispatch_yaml.py format_value()
# ---------------------------------------------------------------------------

def format_value(val) -> str:
    """Format a hyperparameter value the same way dispatch_yaml.py does for filenames."""
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        else:
            return str(val).rstrip("0").rstrip(".").replace(".", "")
    if isinstance(val, str) and "." in val:
        try:
            f_val = float(val)
            if f_val.is_integer():
                return str(int(f_val))
            else:
                return str(f_val).rstrip("0").rstrip(".").replace(".", "")
        except ValueError:
            pass
    return str(val)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_config(config_path: str) -> dict:
    """Load a YAML experiment config and return a structured dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ablations = cfg.get("ablations", {})
    benchmark = cfg.get("benchmark", [])
    name_prefix = cfg.get("name_prefix", "experiment")

    # Preserve the original YAML key order (including seed) for filename reconstruction
    all_ablation_keys = list(ablations.keys())

    # Separate seeds from other ablation axes
    seeds = ablations.get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    # Remaining ablation axes (hyperparameters to sweep)
    hp_keys = [k for k in all_ablation_keys if k != "seed"]
    hp_values = [ablations[k] if isinstance(ablations[k], list) else [ablations[k]] for k in hp_keys]

    return {
        "name_prefix": name_prefix,
        "benchmark": benchmark,
        "seeds": seeds,
        "hp_keys": hp_keys,
        "hp_values": hp_values,
        "n_eval_episodes": cfg.get("n_eval_episodes", 15),
        "all_ablation_keys": all_ablation_keys,
    }


def enumerate_combinations(cfg: dict):
    """
    Yield dicts of {hp_name: value, ...} for every hyperparameter combination
    (Cartesian product of the non-seed ablation axes).
    """
    if not cfg["hp_keys"]:
        yield {}
        return
    for combo in itertools.product(*cfg["hp_values"]):
        yield dict(zip(cfg["hp_keys"], combo))


# ---------------------------------------------------------------------------
# Filename construction
# ---------------------------------------------------------------------------

def build_suffix(hp_combo: dict, seed: int, all_ablation_keys: list[str]) -> str:
    """
    Build the ablation suffix string in the original YAML key order.

    E.g. for keys [behavior_cloning_coefficient, seed, lr] with values
    {bc: 0.1, seed: 0, lr: 0.0001} -> "b_01-s_0-l_00001"
    """
    parts = []
    for k in all_ablation_keys:
        if k == "seed":
            v = seed
        else:
            v = hp_combo[k]
        parts.append(f"{k[0]}_{format_value(v)}")
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_final_reward(filepath: Path, eval_env: str, n_smooth: int) -> float | None:
    """
    Load a training-env CSV and extract the smoothed final evaluation score
    for a specific eval environment.

    Returns the mean of the last `n_smooth` non-NaN evaluation points for
    `eval/{eval_env}/mean_reward`, or None if insufficient data.
    """
    if not filepath.exists():
        return None
    if filepath.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    reward_col = f"eval/{eval_env}/mean_reward"
    if reward_col not in df.columns:
        return None

    rewards = df[reward_col].dropna().values
    if len(rewards) == 0:
        return None

    n = min(n_smooth, len(rewards))
    return float(np.mean(rewards[-n:]))


def compute_per_env_final_score(
    name_prefix: str,
    hp_combo: dict,
    seed: int,
    benchmark: list[str],
    eval_env: str,
    n_smooth: int,
    data_dir: Path,
    all_ablation_keys: list[str],
) -> float | None:
    """
    Compute the smoothed final score p_si(T) for a single (seed, eval_env) pair.

    Since training happens sequentially across training environments, the final
    score is taken from the CSV of the *last* training environment in the
    benchmark sequence.
    """
    last_train_env = benchmark[-1]
    suffix = build_suffix(hp_combo, seed, all_ablation_keys)
    filename = f"{name_prefix}-{suffix}-{last_train_env}.csv"

    filepath = data_dir / name_prefix / filename
    return load_final_reward(filepath, eval_env, n_smooth)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(values: np.ndarray, confidence: float = CONFIDENCE,
                      n_bootstrap: int = N_BOOTSTRAP):
    """
    Compute the mean and a non-parametric bootstrap confidence interval.

    Returns (mean, ci_low, ci_high).
    """
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan

    observed_mean = np.mean(values)

    rng = np.random.default_rng(42)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(values[boot_indices], axis=1)

    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))

    return observed_mean, ci_low, ci_high


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all(cfg: dict, n_smooth: int, data_dir: Path) -> pd.DataFrame:
    """
    Compute Final Average Performance for every hyperparameter combination.

    Returns a DataFrame with columns:
      - One column per hyperparameter
      - One column per benchmark env (per-env final score)
      - P(T)_mean, P(T)_ci_low, P(T)_ci_high
      - n_seeds_found
    """
    benchmark = cfg["benchmark"]
    seeds = cfg["seeds"]
    name_prefix = cfg["name_prefix"]
    all_ablation_keys = cfg["all_ablation_keys"]

    rows = []

    for hp_combo in enumerate_combinations(cfg):
        # Collect per-seed average performance P_s(T)
        seed_avg_perfs = []
        # Also collect per-env scores across seeds for individual env reporting
        per_env_seed_scores = {env: [] for env in benchmark}

        for seed in seeds:
            env_scores = []
            for eval_env in benchmark:
                score = compute_per_env_final_score(
                    name_prefix, hp_combo, seed, benchmark, eval_env,
                    n_smooth, data_dir, all_ablation_keys,
                )
                if score is not None:
                    env_scores.append(score)
                    per_env_seed_scores[eval_env].append(score)

            if len(env_scores) == len(benchmark):
                # P_s(T) = mean across all tasks
                P_s = np.mean(env_scores)
                seed_avg_perfs.append(P_s)

        seed_avg_perfs = np.array(seed_avg_perfs)

        # Compute aggregated stats
        if len(seed_avg_perfs) > 0:
            mean_val, ci_low, ci_high = bootstrap_mean_ci(seed_avg_perfs)
        else:
            mean_val, ci_low, ci_high = np.nan, np.nan, np.nan

        row = dict(hp_combo)
        row["n_seeds_found"] = len(seed_avg_perfs)

        # Per-environment final scores (mean ± CI across seeds)
        for env in benchmark:
            env_arr = np.array(per_env_seed_scores[env])
            if len(env_arr) > 0:
                e_mean, e_ci_lo, e_ci_hi = bootstrap_mean_ci(env_arr)
                row[f"{env}_mean"] = e_mean
                row[f"{env}_ci_low"] = e_ci_lo
                row[f"{env}_ci_high"] = e_ci_hi
            else:
                row[f"{env}_mean"] = np.nan
                row[f"{env}_ci_low"] = np.nan
                row[f"{env}_ci_high"] = np.nan

        row["P(T)_mean"] = mean_val
        row["P(T)_ci_low"] = ci_low
        row["P(T)_ci_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_ci(mean: float, ci_low: float, ci_high: float, precision: int = 2) -> str:
    """Format as 'mean [ci_low, ci_high]'."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} [{ci_low:.{precision}f}, {ci_high:.{precision}f}]"


def results_to_markdown(df: pd.DataFrame, cfg: dict) -> str:
    """Convert the results DataFrame into a Markdown table."""
    benchmark = cfg["benchmark"]
    hp_keys = cfg["hp_keys"]

    lines = []
    lines.append(f"# Final Average Performance P(T) — {cfg['name_prefix']}")
    lines.append("")
    lines.append(f"Smoothing window: last evaluation points | "
                 f"Seeds: {len(cfg['seeds'])} | "
                 f"Confidence: {CONFIDENCE*100:.0f}%")
    lines.append("")

    # Build header
    header_parts = [k for k in hp_keys]
    for env in benchmark:
        header_parts.append(f"{env}")
    header_parts.append("P(T)")
    header_parts.append("Seeds")

    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

    # Sort by P(T) descending
    df_sorted = df.sort_values("P(T)_mean", ascending=False)

    for _, row in df_sorted.iterrows():
        parts = []
        for k in hp_keys:
            parts.append(str(row[k]))
        for env in benchmark:
            parts.append(format_ci(row[f"{env}_mean"], row[f"{env}_ci_low"], row[f"{env}_ci_high"]))
        parts.append(format_ci(row["P(T)_mean"], row["P(T)_ci_low"], row["P(T)_ci_high"]))
        parts.append(str(int(row["n_seeds_found"])))
        lines.append("| " + " | ".join(parts) + " |")

    return "\n".join(lines)


def results_to_csv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Build a clean CSV-friendly DataFrame with formatted CI strings and
    individual columns.
    """
    benchmark = cfg["benchmark"]
    hp_keys = cfg["hp_keys"]

    out_rows = []
    df_sorted = df.sort_values("P(T)_mean", ascending=False)

    for _, row in df_sorted.iterrows():
        out = {}
        for k in hp_keys:
            out[k] = row[k]
        for env in benchmark:
            out[f"{env}_mean"] = row[f"{env}_mean"]
            out[f"{env}_ci_low"] = row[f"{env}_ci_low"]
            out[f"{env}_ci_high"] = row[f"{env}_ci_high"]
            out[f"{env}_formatted"] = format_ci(
                row[f"{env}_mean"], row[f"{env}_ci_low"], row[f"{env}_ci_high"]
            )
        out["P(T)_mean"] = row["P(T)_mean"]
        out["P(T)_ci_low"] = row["P(T)_ci_low"]
        out["P(T)_ci_high"] = row["P(T)_ci_high"]
        out["P(T)_formatted"] = format_ci(
            row["P(T)_mean"], row["P(T)_ci_low"], row["P(T)_ci_high"]
        )
        out["n_seeds"] = int(row["n_seeds_found"])
        out_rows.append(out)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Final Average Performance P(T) for RL ablation experiments.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--smooth", type=int, default=DEFAULT_SMOOTH,
        help=f"Number of last evaluation points to average for smoothing (default: {DEFAULT_SMOOTH}).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Override the data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save output files. If not set, prints to stdout only.",
    )
    parser.add_argument(
        "--confidence", type=float, default=CONFIDENCE,
        help=f"Confidence level for bootstrap CI (default: {CONFIDENCE}).",
    )
    return parser.parse_args()


def main():
    global CONFIDENCE

    args = parse_args()
    CONFIDENCE = args.confidence

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    cfg = parse_config(args.config)

    print(f"Config: {args.config}")
    print(f"  Name prefix:  {cfg['name_prefix']}")
    print(f"  Benchmark:    {cfg['benchmark']}")
    print(f"  Seeds:        {cfg['seeds']}")
    print(f"  HP axes:      {cfg['hp_keys']}")
    n_combos = 1
    for v in cfg["hp_values"]:
        n_combos *= len(v)
    print(f"  Combinations: {n_combos}")
    print(f"  Smoothing:    last {args.smooth} eval points")
    print(f"  Confidence:   {CONFIDENCE*100:.0f}%")
    print(f"  Data dir:     {data_dir}")
    print()

    df = compute_all(cfg, args.smooth, data_dir)

    # Print Markdown table
    md = results_to_markdown(df, cfg)
    print(md)
    print()

    # Save outputs if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"{cfg['name_prefix']}_final_performance.md"
        md_path.write_text(md)
        print(f"Saved Markdown: {md_path}")

        csv_df = results_to_csv(df, cfg)
        csv_path = out_dir / f"{cfg['name_prefix']}_final_performance.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV:      {csv_path}")


if __name__ == "__main__":
    main()
