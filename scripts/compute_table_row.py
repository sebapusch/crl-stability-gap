#!/usr/bin/env python3
r"""
Compute combined LaTeX table rows with per-environment final performance P(Vi),
aggregate P(T), and min-ACC for each hyperparameter combination.

Combines the logic of compute_final_performance.py and compute_min_acc.py,
outputting one LaTeX row per combination in the format:

  & \makecell{<P(V1)> \footnotesize (<ci_low>,<ci_high>)}
  & \makecell{<P(V2)> \footnotesize (<ci_low>,<ci_high>)}
  & \makecell{<P(V3)> \footnotesize (<ci_low>,<ci_high>)}
  & \makecell{<P(T)> \footnotesize (<ci_low>,<ci_high>)}
  & \makecell{<min-ACC> \footnotesize (<ci_low>,<ci_high>)} \\

Usage:
  python compute_table_row.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml
  python compute_table_row.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml --smooth 10
  python compute_table_row.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml --iqm
  python compute_table_row.py --config dispatch/experiments/cartpole/dqn_cp_bc.yaml --output_dir output/tables
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
CONFIDENCE = 0.95


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

    all_ablation_keys = list(ablations.keys())

    seeds = ablations.get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    hp_keys = [k for k in all_ablation_keys if k != "seed"]
    hp_values = [ablations[k] if isinstance(ablations[k], list) else [ablations[k]] for k in hp_keys]

    return {
        "name_prefix": name_prefix,
        "benchmark": benchmark,
        "seeds": seeds,
        "hp_keys": hp_keys,
        "hp_values": hp_values,
        "all_ablation_keys": all_ablation_keys,
        "env": cfg.get("env"),
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
    parts = []
    for k in all_ablation_keys:
        v = seed if k == "seed" else hp_combo[k]
        parts.append(f"{k[0]}_{format_value(v)}")
    return "-".join(parts)


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
# Environment normalization
# ---------------------------------------------------------------------------

def get_env_max_return(env_name: str) -> float:
    """Return the theoretical maximum return for the environment."""
    if "inverted_pendulum" in env_name:
        return 1000.0
    elif "cartpole" in env_name:
        return 500.0
    return 1.0


# ---------------------------------------------------------------------------
# Final Performance: data loading & computation
# ---------------------------------------------------------------------------

def compute_final_performance(
        cfg: dict, n_smooth: int, data_dir: Path, env_name: str,
        use_iqm: bool = True,
) -> pd.DataFrame:
    """
    Compute Final Average Performance for every hyperparameter combination.

    Aggregates across seeds at each timestep (using IQM or Mean), then takes
    the smoothed final score from the aggregated curve.  Bootstrap CIs are
    generated by resampling seeds before aggregation.

    Returns a DataFrame with per-env scores and P(T).
    """
    benchmark = cfg["benchmark"]
    seeds = cfg["seeds"]
    name_prefix = cfg["name_prefix"]
    all_ablation_keys = cfg["all_ablation_keys"]
    last_train_env = benchmark[-1]
    env_max = get_env_max_return(env_name)

    rows = []

    for hp_combo in enumerate_combinations(cfg):
        # Load full eval curves for each seed from the last training env CSV
        valid_seeds = []
        seed_curves = {env: [] for env in benchmark}

        for seed in seeds:
            suffix = build_suffix(hp_combo, seed, all_ablation_keys)
            filename = f"{name_prefix}-{suffix}-{last_train_env}.csv"
            filepath = data_dir / name_prefix / filename

            if not filepath.exists() or filepath.stat().st_size == 0:
                continue

            try:
                df = pd.read_csv(filepath)
            except Exception:
                continue

            seed_valid = True
            temp_curves = {}
            for eval_env in benchmark:
                reward_col = f"eval/{eval_env}/mean_reward"
                if reward_col not in df.columns:
                    seed_valid = False
                    break
                vals = df[reward_col].dropna().values
                if len(vals) == 0:
                    seed_valid = False
                    break
                # Normalize to 0-100
                temp_curves[eval_env] = (vals / env_max) * 100.0

            if seed_valid:
                valid_seeds.append(seed)
                for env in benchmark:
                    seed_curves[env].append(temp_curves[env])

        if len(valid_seeds) == 0:
            row = dict(hp_combo)
            row["n_seeds_perf"] = 0
            for env in benchmark:
                row[f"{env}_mean"] = np.nan
                row[f"{env}_ci_low"] = np.nan
                row[f"{env}_ci_high"] = np.nan
            row["P(T)_mean"] = np.nan
            row["P(T)_ci_low"] = np.nan
            row["P(T)_ci_high"] = np.nan
            rows.append(row)
            continue

        # Stack curves, truncating to the shortest seed
        stacked = {}
        for env in benchmark:
            min_len = min(len(c) for c in seed_curves[env])
            stacked[env] = np.array([c[:min_len] for c in seed_curves[env]])

        n_valid = len(valid_seeds)
        lowercut = int(0.25 * n_valid)
        uppercut = n_valid - lowercut
        do_iqm = use_iqm and lowercut > 0 and uppercut > lowercut

        def calc_perf_vectorized(seed_indices_2d):
            """Compute per-env final scores for batches of seed resamplings."""
            N = seed_indices_2d.shape[0]
            env_scores = np.zeros((len(benchmark), N))

            for idx, env in enumerate(benchmark):
                curves = stacked[env][seed_indices_2d]  # (N, n_valid, T)

                if do_iqm:
                    curves = np.partition(curves, uppercut - 1, axis=1)
                    curves[:, :uppercut, :] = np.partition(
                        curves[:, :uppercut, :], lowercut, axis=1
                    )
                    agg_curve = np.mean(curves[:, lowercut:uppercut, :], axis=1)
                else:
                    agg_curve = np.mean(curves, axis=1)  # (N, T)

                # Final score: average of last n_smooth points
                n_pts = min(n_smooth, agg_curve.shape[1])
                env_scores[idx, :] = np.mean(agg_curve[:, -n_pts:], axis=1)

            return env_scores  # (len(benchmark), N)

        # Observed statistic
        obs_env_scores = calc_perf_vectorized(np.arange(n_valid)[None, :])  # (n_envs, 1)

        # Bootstrap
        rng = np.random.default_rng(42)
        boot_indices = rng.integers(0, n_valid, size=(N_BOOTSTRAP, n_valid))
        boot_env_scores = calc_perf_vectorized(boot_indices)  # (n_envs, N_BOOTSTRAP)

        # P(T) = mean across envs
        obs_pt = float(np.mean(obs_env_scores[:, 0]))
        boot_pt = np.mean(boot_env_scores, axis=0)  # (N_BOOTSTRAP,)

        alpha = (1 - CONFIDENCE) / 2

        row = dict(hp_combo)
        row["n_seeds_perf"] = n_valid

        for idx, env in enumerate(benchmark):
            row[f"{env}_mean"] = float(obs_env_scores[idx, 0])
            row[f"{env}_ci_low"] = float(np.percentile(boot_env_scores[idx], 100 * alpha))
            row[f"{env}_ci_high"] = float(np.percentile(boot_env_scores[idx], 100 * (1 - alpha)))

        row["P(T)_mean"] = obs_pt
        row["P(T)_ci_low"] = float(np.percentile(boot_pt, 100 * alpha))
        row["P(T)_ci_high"] = float(np.percentile(boot_pt, 100 * (1 - alpha)))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# min-ACC computation
# ---------------------------------------------------------------------------

def compute_min_acc(cfg: dict, k_idx: int, data_dir: Path, env_name: str, use_iqm: bool = False) -> pd.DataFrame:
    """
    Compute min-ACC for each hyperparameter combination.
    """
    benchmark = cfg["benchmark"]
    seeds = cfg["seeds"]
    name_prefix = cfg["name_prefix"]
    all_ablation_keys = cfg["all_ablation_keys"]

    rows = []

    for hp_combo in enumerate_combinations(cfg):
        valid_seeds = []
        seed_data_collection = {benchmark[i]: [] for i in range(k_idx)}
        phase_boundaries = {benchmark[i]: None for i in range(k_idx)}

        for seed in seeds:
            seed_valid = True
            temp_seed_data = {}
            temp_boundaries = {}

            suffix = build_suffix(hp_combo, seed, all_ablation_keys)

            for i in range(k_idx):
                eval_env = benchmark[i]
                curve_parts = []
                lengths = []

                for j in range(i, k_idx + 1):
                    train_env = benchmark[j]
                    filename = f"{name_prefix}-{suffix}-{train_env}.csv"
                    filepath = data_dir / name_prefix / filename

                    if not filepath.exists() or filepath.stat().st_size == 0:
                        seed_valid = False
                        break

                    try:
                        df = pd.read_csv(filepath)
                        reward_col = f"eval/{eval_env}/mean_reward"
                        if reward_col not in df.columns:
                            seed_valid = False
                            break

                        vals = df[reward_col].dropna().values
                        if len(vals) == 0:
                            seed_valid = False
                            break

                        curve_parts.append(vals)
                        lengths.append(len(vals))
                    except Exception:
                        seed_valid = False
                        break

                if not seed_valid:
                    break

                temp_seed_data[eval_env] = np.concatenate(curve_parts)
                temp_boundaries[eval_env] = lengths[0]

            if seed_valid:
                valid_seeds.append(seed)
                for i in range(k_idx):
                    eval_env = benchmark[i]
                    seed_data_collection[eval_env].append(temp_seed_data[eval_env])
                    if phase_boundaries[eval_env] is None:
                        phase_boundaries[eval_env] = temp_boundaries[eval_env]

        if len(valid_seeds) == 0:
            row = dict(hp_combo)
            row["n_seeds_minacc"] = 0
            row["min-ACC_stat"] = np.nan
            row["min-ACC_ci_low"] = np.nan
            row["min-ACC_ci_high"] = np.nan
            rows.append(row)
            continue

        # Ensure uniform lengths
        stacked_curves = {}
        for i in range(k_idx):
            eval_env = benchmark[i]
            min_len = min(len(c) for c in seed_data_collection[eval_env])
            stacked_curves[eval_env] = np.array([c[:min_len] for c in seed_data_collection[eval_env]])
            phase_boundaries[eval_env] = min(phase_boundaries[eval_env], min_len)

        n_valid = len(valid_seeds)
        lowercut = int(0.25 * n_valid)
        uppercut = n_valid - lowercut
        do_iqm_trim = use_iqm and lowercut > 0 and uppercut > lowercut

        def calc_metric_vectorized(seed_indices_2d):
            N = seed_indices_2d.shape[0]
            task_accs = np.zeros((k_idx, N))

            for i in range(k_idx):
                eval_env = benchmark[i]
                curves = stacked_curves[eval_env][seed_indices_2d]

                if do_iqm_trim:
                    curves = np.partition(curves, uppercut - 1, axis=1)
                    curves[:, :uppercut, :] = np.partition(
                        curves[:, :uppercut, :], lowercut, axis=1
                    )
                    agg_curve = np.mean(curves[:, lowercut:uppercut, :], axis=1)
                else:
                    agg_curve = np.mean(curves, axis=1)

                max_score = np.max(agg_curve, axis=1)
                start_idx = phase_boundaries[eval_env]
                min_score = np.min(agg_curve[:, start_idx:], axis=1)

                # Normalize relative to the peak performance (0-100 scale)
                max_score_safe = np.where(max_score > 0, max_score, 1.0)
                norm = (min_score / max_score_safe) * 100.0
                norm[max_score <= 0] = 0.0
                task_accs[i, :] = norm

            return np.mean(task_accs, axis=0)

        observed_stat = calc_metric_vectorized(np.arange(n_valid)[None, :])[0]

        rng = np.random.default_rng(42)
        boot_indices = rng.integers(0, n_valid, size=(N_BOOTSTRAP, n_valid))
        boot_stats = calc_metric_vectorized(boot_indices)

        alpha = (1 - CONFIDENCE) / 2
        ci_low = np.percentile(boot_stats, 100 * alpha)
        ci_high = np.percentile(boot_stats, 100 * (1 - alpha))

        row = dict(hp_combo)
        row["n_seeds_minacc"] = n_valid
        row["min-ACC_stat"] = observed_stat
        row["min-ACC_ci_low"] = ci_low
        row["min-ACC_ci_high"] = ci_high
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX output formatting
# ---------------------------------------------------------------------------

def format_makecell(mean: float, ci_low: float, ci_high: float, precision: int = 2) -> str:
    """Format as \\makecell{<mean> \\footnotesize (<ci_low>,<ci_high>)}."""
    if np.isnan(mean):
        return r"\makecell{N/A}"
    return (
        rf"\makecell{{{mean:.{precision}f} "
        rf"\footnotesize ({ci_low:.{precision}f},{ci_high:.{precision}f})}}"
    )


def build_latex_rows(perf_df: pd.DataFrame, minacc_df: pd.DataFrame, cfg: dict, precision: int = 1) -> str:
    """
    Merge final-performance and min-ACC results and produce one LaTeX row
    per hyperparameter combination.
    """
    benchmark = cfg["benchmark"]
    hp_keys = cfg["hp_keys"]

    # Merge on hp keys (or on index if no hp keys)
    if hp_keys:
        merged = pd.merge(perf_df, minacc_df, on=hp_keys, how="outer")
    else:
        merged = pd.concat([perf_df, minacc_df], axis=1)

    lines = []
    for _, row in merged.iterrows():
        # Configuration label for the first column
        if hp_keys:
            hp_label = ", ".join(f"{k}={row[k]}" for k in hp_keys)
        else:
            hp_label = cfg["name_prefix"]

        parts = []
        # Per-env final performance P(Vi)
        for env in benchmark:
            parts.append(format_makecell(
                row.get(f"{env}_mean", np.nan),
                row.get(f"{env}_ci_low", np.nan),
                row.get(f"{env}_ci_high", np.nan),
                precision,
            ))

        # P(T)
        parts.append(format_makecell(
            row.get("P(T)_mean", np.nan),
            row.get("P(T)_ci_low", np.nan),
            row.get("P(T)_ci_high", np.nan),
            precision,
        ))

        # min-ACC
        parts.append(format_makecell(
            row.get("min-ACC_stat", np.nan),
            row.get("min-ACC_ci_low", np.nan),
            row.get("min-ACC_ci_high", np.nan),
            precision,
        ))

        # Build the line: config & cell1 & cell2 ... \\
        cells = " & ".join(f" {p}" for p in parts)
        line = f"{hp_label} & {cells} \\\\"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute combined LaTeX table rows with P(Vi), P(T), and min-ACC.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML experiment config file.",
    )
    parser.add_argument(
        "--smooth", type=int, default=DEFAULT_SMOOTH,
        help=f"Number of last evaluation points to average for final performance smoothing (default: {DEFAULT_SMOOTH}).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Override the data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save output file. If not set, prints to stdout only.",
    )
    parser.add_argument(
        "--confidence", type=float, default=CONFIDENCE,
        help=f"Confidence level for bootstrap CI (default: {CONFIDENCE}).",
    )
    parser.add_argument(
        "--mean", action="store_true",
        help="Use Mean (instead of IQM) across seeds for timestep aggregation (applies to both P(T) and min-ACC).",
    )
    parser.add_argument(
        "--precision", type=int, default=1,
        help="Number of decimal places for formatted values (default: 1).",
    )
    return parser.parse_args()


def main():
    global CONFIDENCE

    args = parse_args()
    CONFIDENCE = args.confidence
    use_iqm = not args.mean

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    cfg = parse_config(args.config)

    env_name = cfg["env"]
    benchmark = cfg["benchmark"]

    k_target = 3
    k_idx = k_target - 1

    print(f"Config: {args.config}", file=sys.stderr)
    print(f"  Name prefix:  {cfg['name_prefix']}", file=sys.stderr)
    print(f"  Benchmark:    {benchmark}", file=sys.stderr)
    print(f"  Seeds:        {cfg['seeds']}", file=sys.stderr)
    print(f"  Smoothing:    last {args.smooth} eval points", file=sys.stderr)
    print(f"  Confidence:   {CONFIDENCE * 100:.0f}%", file=sys.stderr)
    print(f"  Aggregation:  {'IQM' if use_iqm else 'Mean'} by timestep", file=sys.stderr)
    print(file=sys.stderr)

    if len(benchmark) <= k_idx:
        print(f"Error: Benchmark needs at least {k_target} tasks for min-ACC at k={k_target}.",
              file=sys.stderr)
        sys.exit(1)

    # Compute both metrics
    print("Computing final performance...", file=sys.stderr)
    perf_df = compute_final_performance(cfg, args.smooth, data_dir, env_name, use_iqm=use_iqm)

    print("Computing min-ACC...", file=sys.stderr)
    minacc_df = compute_min_acc(cfg, k_idx, data_dir, env_name, use_iqm=use_iqm)

    # Build and print LaTeX rows
    latex = build_latex_rows(perf_df, minacc_df, cfg, precision=args.precision)
    print(latex)

    # Save if requested
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tex_path = out_dir / f"{cfg['name_prefix']}_table_rows.tex"
        tex_path.write_text(latex)
        print(f"\nSaved LaTeX: {tex_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
