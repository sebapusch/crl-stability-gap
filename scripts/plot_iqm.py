#!/usr/bin/env python3
"""
Generate IQM plots with 95% CI for each test environment.

Supports two modes:

1. CLI mode (original): produces one plot per test env.
   python plot_iqm.py --methods continual sequential fine_tune
   python plot_iqm.py --methods continual_encode --prefix encode

2. YAML config mode: produces a grid of subplots from a config file.
   python plot_iqm.py --config path/to/config.yaml
"""

import argparse
import hashlib
import math
from scipy.ndimage import uniform_filter1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # Deferred error if --config is actually used


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


def load_eval_data(
    method: str,
    seed: int,
    test_env: str,
    *,
    train_envs: list[str] | None = None,
    timesteps_per_env: int | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate eval data for a given method, seed, and test env
    across all training environments.

    Returns a DataFrame with columns ['timestep', 'reward'].
    """
    _train_envs = train_envs if train_envs is not None else TRAIN_ENVS
    _timesteps = timesteps_per_env if timesteps_per_env is not None else TIMESTEPS_PER_ENV

    all_timesteps = []

    reward_col = f"eval/{test_env}/mean_reward"
    timestep_col = f"time/{test_env}/total_timesteps"

    for env_idx, train_env in enumerate(_train_envs):
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
        subset["timestep"] = subset["timestep"] + env_idx * _timesteps

        all_timesteps.append(subset)

    if not all_timesteps:
        return pd.DataFrame(columns=["timestep", "reward"])

    result = pd.concat(all_timesteps, ignore_index=True)
    result = result.sort_values("timestep").reset_index(drop=True)
    return result


def compute_iqm_curve(
    method: str,
    test_env: str,
    *,
    seeds: list[int] | None = None,
    train_envs: list[str] | None = None,
    timesteps_per_env: int | None = None,
):
    """
    Compute IQM + 95% CI curve for a method on a test environment.

    Returns
    -------
    timesteps : array
    iqm_values : array
    ci_low : array
    ci_high : array
    """
    _seeds = seeds if seeds is not None else SEEDS

    # Load all seeds
    seed_frames = []
    for seed in _seeds:
        df = load_eval_data(
            method, seed, test_env,
            train_envs=train_envs,
            timesteps_per_env=timesteps_per_env,
        )
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
    # Sanitise method for filesystem (replace / with _)
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


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """
    Load and validate a YAML config file for grid plotting.

    Expected structure:
        defaults:          # optional
            env_name: str
            seeds: list[int]
            timesteps: int
            envs: list[str]
            output_dir: str
            output_file: str
            grid: [rows, cols]   # optional explicit grid size
            dpi: int             # optional, default 300
            t_start: int         # optional, zoom start (global timestep)
            t_end: int           # optional, zoom end (global timestep)
        plots:
            - title: str
              test_env: str
              t_start: int       # optional per-plot override
              t_end: int         # optional per-plot override
              lines:
                - method: str
                  label: str       # optional
                  color: str       # optional
    """
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
                raise ValueError(
                    f"Line {j} in plot {i} must specify 'method'."
                )

    return cfg
def _nice_floor(value: float) -> float:
    """Round *value* down to the nearest 'nice' number in {1, 2, 5} × 10^n."""
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base  # will be in [1, 10)
    # Pick the largest nice mantissa that is <= mantissa
    for nice in (5, 2, 1):
        if mantissa >= nice:
            return nice * base
    return base


def _decorate_ax(ax, train_envs, timesteps_per_env, title=None, test_env=None, zoomed=False):
    """Add environment boundary lines, labels, and grid to an axis."""
    # Font scale factor: 2.3× when zoomed
    fs = 2.3 if zoomed else 1.0

    # Determine visible data range
    x_lo, x_hi = ax.get_xlim()

    # Index of the test env in the training order (for filtering labels)
    test_env_idx = None
    if test_env and test_env in train_envs:
        test_env_idx = train_envs.index(test_env)

    # Vertical lines at environment boundaries (only if within data range)
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

    # Env labels just above the plot area
    # Skip labels for training phases before the eval env (test_env_idx > i)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, env in enumerate(train_envs):
        env_start = i * timesteps_per_env
        env_end = (i + 1) * timesteps_per_env
        # Skip if this env region doesn't overlap with the data range
        if env_end <= x_lo or env_start >= x_hi:
            continue
        # Skip if eval env comes after this training env
        if test_env_idx is not None and test_env_idx > i:
            continue
        center = (i + 0.5) * timesteps_per_env
        ax.text(
            center,
            1.02,
            f"Train {env}",
            ha="center",
            va="bottom",
            fontsize=9 * fs,
            color="gray",
            transform=trans,
        )

    if zoomed:
        # No legend, title, or axis labels in zoomed mode
        ax.get_legend_handles_labels()  # ensure legend data exists but don't show
        legend = ax.get_legend()
        if legend:
            legend.remove()
        ax.tick_params(axis='both', labelsize=10 * fs)

        # Round y_max down to a nice number ({1, 2, 5} × 10^n)
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
        ax.grid(alpha=0.3)


def _smooth(arr: np.ndarray, window: int | None) -> np.ndarray:
    """Apply a uniform moving-average filter if *window* is set."""
    if window is None or window <= 1:
        return arr
    return uniform_filter1d(arr.astype(float), size=window)


def plot_grid(config: dict, use_cache: bool):
    """
    Main driver for YAML-config grid plotting.

    Creates a grid of subplots as specified in the config and saves a single
    combined figure.
    """
    defaults = config.get("defaults", {})
    plots = config["plots"]

    # Resolve defaults
    seeds = defaults.get("seeds", SEEDS)
    timesteps = defaults.get("timesteps", TIMESTEPS_PER_ENV)
    envs = defaults.get("envs", TRAIN_ENVS)
    env_name = defaults.get("env_name", "")
    output_subdir = defaults.get("output_dir", "")
    output_file = defaults.get("output_file", "iqm_grid")
    dpi = defaults.get("dpi", 300)

    # Determine grid layout
    n_plots = len(plots)
    grid = defaults.get("grid", None)
    if grid:
        nrows, ncols = grid
    elif n_plots == 1:
        nrows, ncols = 1, 1
    else:
        # Auto-arrange: prefer wider layouts (more columns than rows)
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)

    # Output directory
    if output_subdir:
        plot_output_dir = OUTPUT_DIR / output_subdir
    else:
        plot_output_dir = OUTPUT_DIR
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all method strings for cache key
    all_methods = []
    for p in plots:
        for line in p["lines"]:
            all_methods.append(line["method"])
    cache_key = make_cache_key(all_methods, output_file)
    if use_cache:
        print(f"Cache key: {cache_key}  (use --no-cache to force recompute)")

    # Create figure
    fig_w = 7 * ncols
    fig_h = 5 * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for plot_idx, plot_cfg in enumerate(plots):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row][col]

        test_env = plot_cfg["test_env"]

        # Build title
        title = plot_cfg.get("title", None)
        if title is None and env_name:
            title = f"Evaluation on {env_name}-{test_env}"

        for line_idx, line_cfg in enumerate(plot_cfg["lines"]):
            method = line_cfg["method"]
            label = line_cfg.get("label", get_label(method))
            color = line_cfg.get("color", get_color(method, line_idx))

            # Try cache
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
                )
                if len(ts) > 0:
                    save_to_cache(
                        cache_key, method, test_env,
                        ts, iqm, ci_lo, ci_hi,
                    )

            if len(ts) == 0:
                print(f"  No data for {method}/{test_env}")
                continue

            # Per-plot smooth overrides defaults
            smooth = plot_cfg.get("smooth", defaults.get("smooth"))
            iqm = _smooth(iqm, smooth)
            ci_lo = _smooth(ci_lo, smooth)
            ci_hi = _smooth(ci_hi, smooth)

            ax.plot(ts, iqm, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        # Apply zoom (per-plot overrides defaults)
        t_start = plot_cfg.get("t_start", defaults.get("t_start"))
        t_end = plot_cfg.get("t_end", defaults.get("t_end"))
        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        _decorate_ax(ax, envs, timesteps, title=title, test_env=test_env, zoomed=zoomed)

    # Hide unused subplot cells
    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    plt.tight_layout()

    out_path = plot_output_dir / f"{output_file}.svg"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IQM plots with 95% CI for each test environment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file for grid plotting. "
             "When provided, most other CLI args are ignored in favour of the YAML.",
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
    parser.add_argument(
        "--t_start",
        type=int,
        default=None,
        help="Start of global timestep range to zoom into (inclusive).",
    )
    parser.add_argument(
        "--t_end",
        type=int,
        default=None,
        help="End of global timestep range to zoom into (inclusive).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=None,
        help="Window size for uniform moving-average smoothing (default: none).",
    )
    return parser.parse_args()


def main():
    global TRAIN_ENVS, TEST_ENVS, SEEDS, TIMESTEPS_PER_ENV
    args = parse_args()
    use_cache = not args.no_cache

    # ---- YAML config mode ----
    if args.config:
        cfg = load_config(args.config)
        # CLI overrides for defaults
        defaults = cfg.setdefault("defaults", {})
        if args.no_cache:
            pass  # use_cache already set
        # Allow CLI to override specific defaults if not set in YAML
        if args.env_name and "env_name" not in defaults:
            defaults["env_name"] = args.env_name
        if args.output_dir and "output_dir" not in defaults:
            defaults["output_dir"] = args.output_dir
        if args.smooth is not None and "smooth" not in defaults:
            defaults["smooth"] = args.smooth

        plot_grid(cfg, use_cache)
        print("Done!")
        return

    # ---- Original CLI mode ----
    methods = args.methods
    prefix = args.prefix
    TRAIN_ENVS = args.envs
    TEST_ENVS = args.envs
    SEEDS = args.seeds
    TIMESTEPS_PER_ENV = args.timesteps
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
            iqm = _smooth(iqm, smooth)
            ci_lo = _smooth(ci_lo, smooth)
            ci_hi = _smooth(ci_hi, smooth)
            ax.plot(ts, iqm, label=label, color=color, linewidth=0.7)
            ax.fill_between(ts, ci_lo, ci_hi, alpha=0.2, color=color)

        # Apply zoom if specified
        zoomed = t_start is not None or t_end is not None
        if zoomed:
            ax.set_xlim(left=t_start, right=t_end)

        _decorate_ax(
            ax, TRAIN_ENVS, TIMESTEPS_PER_ENV,
            title=f"Evaluation on {env_name}-{test_env}",
            test_env=test_env,
            zoomed=zoomed,
        )

        plt.tight_layout()

        # Build output filename
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
