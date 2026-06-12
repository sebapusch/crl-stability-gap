#!/usr/bin/env python3
"""
Generate IQM plots with 95% CI for each test environment.

Supports two modes:

1. CLI mode (original): produces one plot per test env.
   python plot_iqm.py --methods continual sequential fine_tune

2. YAML config mode: produces a grid of subplots from a config file.
   python plot_iqm.py --config path/to/config.yaml
"""

import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from common import (
    compute_iqm_curve,
    smooth_peak_aware as _smooth,
)

try:
    import yaml
except ImportError:
    yaml = None

DATA_DIR = Path(__file__).resolve().parent.parent / "output" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "plots"
CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "cache"

DEFAULT_METHODS = ["continual", "sequential", "fine_tune"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_ENVS = ["V1", "V2", "V3"]
TEST_ENVS = ["V1", "V2", "V3"]
TIMESTEPS_PER_ENV = 40_000

# Known labels
METHOD_LABELS = {
    "continual": "Continual",
    "sequential": "Sequential",
    "fine_tune": "Fine-tune",
    "continual_encode": "Continual (encode)",
}

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


def get_label(method: str) -> str:
    """Return a display label for a method, auto-generating if unknown."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return method.replace("_", " ").title()


def get_color(method: str, index: int) -> str:
    """Return a color for a method, cycling through the palette."""
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def make_cache_key(methods: list[str], prefix: str) -> str:
    """Generate a short hash from the sorted methods + prefix combination."""
    canonical = "|".join(sorted(methods)) + "||" + prefix
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def cache_path_for(cache_key: str, method: str, test_env: str) -> Path:
    """Return the CSV cache path for a specific method/test_env under a cache key."""
    safe_method = method.replace("/", "__")
    return CACHE_DIR / cache_key / f"{safe_method}_{test_env}.csv"


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


def load_config(path: str) -> dict:
    """Load and validate a YAML config file for grid plotting."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config mode. Install with: pip install pyyaml"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "plots" not in cfg or not isinstance(cfg["plots"], list):
        raise ValueError("YAML config must contain a 'plots' list.")

    for i, plot in enumerate(cfg["plots"]):
        if "lines" not in plot or not isinstance(plot["lines"], list):
            raise ValueError(f"Plot entry {i} must contain a 'lines' list.")
        if "test_env" not in plot:
            raise ValueError(f"Plot entry {i} must specify 'test_env'.")
        for j, line in enumerate(plot["lines"]):
            if "method" not in line:
                raise ValueError(f"Line {j} in plot {i} must specify 'method'.")
    return cfg


def _nice_floor(value: float) -> float:
    """Round *value* down to the nearest 'nice' number in {1, 2, 5} * 10^n."""
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    for nice in (5, 2, 1):
        if mantissa >= nice:
            return nice * base
    return base


def _decorate_ax(ax, train_envs, timesteps_per_env, title=None, test_env=None, zoomed=False):
    """Add environment boundary lines, labels, and grid to an axis."""
    fs = 2.3 if zoomed else 1.84
    x_lo, x_hi = ax.get_xlim()

    test_env_idx = None
    if test_env and test_env in train_envs:
        test_env_idx = train_envs.index(test_env)

    # Boundary lines
    for i in range(1, len(train_envs)):
        boundary = i * timesteps_per_env
        if x_lo < boundary < x_hi:
            ax.axvline(
                x=boundary,
                color="black",
                linestyle="-",
                alpha=1,
                linewidth=0.5,
                zorder=5,
            )

    # Task labels
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, env in enumerate(train_envs):
        env_start = i * timesteps_per_env
        env_end = (i + 1) * timesteps_per_env
        if env_end <= x_lo or env_start >= x_hi:
            continue
        if test_env_idx is not None and test_env_idx > i:
            continue
        center = (i + 0.5) * timesteps_per_env
        ax.text(
            center,
            1.02,
            f"Task {i + 1}",
            ha="center",
            va="bottom",
            fontsize=9 * fs,
            color="gray",
            transform=trans,
        )

    if zoomed:
        ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend:
            legend.remove()
        ax.tick_params(axis='both', labelsize=10 * fs)
        y_lo, y_hi = ax.get_ylim()
        y_max = _nice_floor(y_hi)
        yticks = [0, y_max / 2, y_max]
        ax.set_yticks(yticks)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
    else:
        ax.set_xlabel("Total Timesteps", fontsize=10 * fs)
        ax.set_ylabel("IQM Episodic Return", fontsize=10 * fs)
        if title:
            ax.set_title(title, pad=25, fontsize=12 * fs)
        ax.legend(loc="lower right", fontsize=10 * fs)
        ax.tick_params(axis='both', labelsize=10 * fs)
        y_lo, y_hi = ax.get_ylim()
        y_max = _nice_floor(y_hi)
        yticks = [0, y_max / 2, y_max]
        ax.set_yticks(yticks)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)


def plot_grid(config: dict, use_cache: bool):
    """Main driver for YAML-config grid plotting."""
    defaults = config.get("defaults", {})
    plots = config["plots"]

    seeds = defaults.get("seeds", SEEDS)
    timesteps = defaults.get("timesteps", TIMESTEPS_PER_ENV)
    envs = defaults.get("envs", TRAIN_ENVS)
    env_name = defaults.get("env_name", "")
    output_subdir = defaults.get("output_dir", "")
    output_file = defaults.get("output_file", "iqm_grid")
    dpi = defaults.get("dpi", 300)

    n_plots = len(plots)
    grid = defaults.get("grid", None)
    if grid:
        nrows, ncols = grid
    elif n_plots == 1:
        nrows, ncols = 1, 1
    else:
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)

    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    all_methods = []
    for p in plots:
        for line in p["lines"]:
            all_methods.append(line["method"])
    cache_key = make_cache_key(all_methods, output_file)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    any_zoomed = any(
        p.get("t_start", defaults.get("t_start")) is not None
        or p.get("t_end", defaults.get("t_end")) is not None
        for p in plots
    )

    if any_zoomed:
        fig_w = 7 * ncols
        fig_h = 5 * nrows
    else:
        fig_w = 10 * ncols
        fig_h = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for plot_idx, plot_cfg in enumerate(plots):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row][col]

        test_env = plot_cfg["test_env"]
        title = plot_cfg.get("title", None)
        if title is None and env_name:
            title = f"Evaluation on {env_name}-{test_env}"

        for line_idx, line_cfg in enumerate(plot_cfg["lines"]):
            method = line_cfg["method"]
            label = line_cfg.get("label", get_label(method))
            color = line_cfg.get("color", get_color(method, line_idx))

            cached = None
            if use_cache:
                cached = load_from_cache(cache_key, method, test_env)

            if cached is not None:
                ts, iqm, ci_lo, ci_hi = cached
                print(f"Loaded cached IQM for {label} on {test_env}")
            else:
                print(f"Computing IQM for {label} on {test_env}...")
                ts, iqm, ci_lo, ci_hi = compute_iqm_curve(
                    method, test_env,
                    seeds=seeds,
                    train_envs=envs,
                    timesteps_per_env=timesteps,
                    data_dir=DATA_DIR,
                )
                if len(ts) > 0:
                    save_to_cache(cache_key, method, test_env, ts, iqm, ci_lo, ci_hi)

            if len(ts) == 0:
                print(f"  No data for {method}/{test_env}")
                continue

            smooth = plot_cfg.get("smooth", defaults.get("smooth"))
            iqm = _smooth(iqm, smooth)
            ci_lo = _smooth(ci_lo, smooth)
            ci_hi = _smooth(ci_hi, smooth)

            ax.plot(ts, iqm, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        t_start = plot_cfg.get("t_start", defaults.get("t_start"))
        t_end = plot_cfg.get("t_end", defaults.get("t_end"))
        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        _decorate_ax(ax, envs, timesteps, title=title, test_env=test_env, zoomed=zoomed)

    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    plt.tight_layout()
    out_path = plot_output_dir / f"{output_file}.svg"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IQM plots with 95% CI for each test environment.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file for grid plotting.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help=f"Method names to plot (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Filename prefix for output plots, e.g. 'encode' -> iqm_encode_V1.png",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument(
        "--envs", "--env_order", nargs="+", default=TEST_ENVS,
        help=f"Environments to plot and their training order (default: {TEST_ENVS})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds to include (default: {SEEDS})",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TIMESTEPS_PER_ENV,
        help=f"Timesteps per environment (default: {TIMESTEPS_PER_ENV})",
    )
    parser.add_argument(
        "--env_name", type=str, help="Base name of the environment",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Subdirectory inside output/plots to save plots",
    )
    parser.add_argument(
        "--t_start", type=int, default=None,
        help="Start of global timestep range to zoom into.",
    )
    parser.add_argument(
        "--t_end", type=int, default=None,
        help="End of global timestep range to zoom into.",
    )
    parser.add_argument(
        "--smooth", type=int, default=None,
        help="Window size for peak-aware moving-average smoothing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cache = not args.no_cache

    if args.config:
        cfg = load_config(args.config)
        defaults = cfg.setdefault("defaults", {})
        if args.env_name and "env_name" not in defaults:
            defaults["env_name"] = args.env_name
        if args.output_dir and "output_dir" not in defaults:
            defaults["output_dir"] = args.output_dir
        if args.smooth is not None and "smooth" not in defaults:
            defaults["smooth"] = args.smooth
        plot_grid(cfg, use_cache)
        print("Done!")
        return

    methods = args.methods
    prefix = args.prefix
    train_envs = args.envs
    test_envs = args.envs
    seeds = args.seeds
    timesteps = args.timesteps
    env_name = args.env_name
    output_subdir = args.output_dir
    t_start = args.t_start
    t_end = args.t_end
    smooth = args.smooth

    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = make_cache_key(methods, prefix)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    for test_env in test_envs:
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, method in enumerate(methods):
            cached = None
            if use_cache:
                cached = load_from_cache(cache_key, method, test_env)

            if cached is not None:
                ts, iqm, ci_lo, ci_hi = cached
                print(f"Loaded cached IQM for {method} on {test_env}")
            else:
                print(f"Computing IQM for {method} on {test_env}...")
                ts, iqm, ci_lo, ci_hi = compute_iqm_curve(
                    method, test_env,
                    seeds=seeds,
                    train_envs=train_envs,
                    timesteps_per_env=timesteps,
                    data_dir=DATA_DIR,
                )
                if len(ts) > 0:
                    save_to_cache(cache_key, method, test_env, ts, iqm, ci_lo, ci_hi)

            if len(ts) == 0:
                print(f"  No data for {method}/{test_env}")
                continue

            label = get_label(prefix)
            color = get_color(method, idx)
            iqm = _smooth(iqm, smooth)
            ci_lo = _smooth(ci_lo, smooth)
            ci_hi = _smooth(ci_hi, smooth)
            ax.plot(ts, iqm, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        _decorate_ax(
            ax, train_envs, timesteps,
            title=f"Evaluation on {env_name}-{test_env}",
            test_env=test_env,
            zoomed=zoomed,
        )

        plt.tight_layout()
        parts = ["iqm"]
        if prefix:
            parts.append(prefix)
        parts.append(test_env)
        out_path = plot_output_dir / f"{'_'.join(parts)}.png"

        fig.savefig(out_path, dpi=1000)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
