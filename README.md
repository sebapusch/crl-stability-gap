# CRL Stability Gap

This repository contains continual reinforcement-learning experiments for
studying stability gaps under projected observation spaces. The main experiment
path is the `projection` package, backed by a local fork of
`stable_baselines3` that adds continual-learning variants of DQN, SAC, SACD,
and DDPG.

## Projection Experiments

The projection experiments train one policy over a sequence of task versions.
Each version uses the same base environment dynamics, but changes the
observation space by applying a deterministic orthogonal projection. Later
versions can also add a bias term. This gives a controlled continual-learning
benchmark where the task identity changes through the observation
representation.

Supported projection environments are defined in `projection/main.py`:

- `cartpole`: Gymnasium `CartPoleEnv`, normally used with DQN or SACD.
- `inverted_pendulum`: Gymnasium MuJoCo `InvertedPendulumEnv`, normally used
  with SAC or DDPG.
- `inverted_pendulum_hard`: a stricter termination variant implemented in
  `projection/benchmarks/inverted_pendulum_hard.py`.

Task versions are passed as `V1`, `V2`, `V3`, etc. `V1` is the unprojected
environment. Versions greater than `V1` receive a seeded random orthogonal
observation transform; versions greater than `V5` also receive a bias term.

## Project Structure

```text
projection/
  main.py                         Experiment entry point. Builds benchmarks,
                                  selects the algorithm/method, starts W&B
                                  runs, and trains in continual or multitask
                                  mode.
  args.py                         CLI definitions and eval-frequency schedule
                                  parsing.
  callbacks.py                    W&B, evaluation, video, and DQN Q-value
                                  tracking callbacks.
  common.py                       Logger and model-weight output helpers.
  benchmarks/
    projected_env_benchmark.py    Creates train/test environments for each
                                  projected task version.
    wrappers.py                   Observation wrappers for orthogonal
                                  transforms, infinite Box bounds, and
                                  optional one-hot task encoding.
    inverted_pendulum_hard.py     Harder InvertedPendulum variant with a
                                  smaller allowed pole-angle threshold.

stable_baselines3/
  continual/
    continual_learning.py         Protocol shared by continual-learning
                                  implementations.
    off_policy_joint_incremental.py
                                  Shared joint-incremental off-policy replay
                                  and round-robin rollout logic.
    pc_grad.py                    PCGrad helper for projecting conflicting
                                  task gradients.
  common/buffers.py               Includes the added `MultiReplayBuffer` and
                                  `ExpertBuffer` used by continual methods.
  dqn/                            DQN baseline plus continual DQN variants.
  sac/                            SAC baseline plus continual SAC variants.
  sacd/                           Discrete-action SAC plus continual SACD
                                  variants.
  ddpg/                           DDPG baseline plus continual DDPG variants.

dispatch/
  dispatch_projection.sh          SLURM wrapper for `projection/main.py`.
  dispatch_yaml.py                Expands YAML experiment grids into `sbatch`
                                  commands.
  experiments*/                   Projection experiment configurations.

scripts/                          Plotting, metric computation, and analysis
                                  helpers for generated outputs.
output/                           Generated logs, models, plots, cache files,
                                  and metrics.
```

## Continual-Learning Methods

All projection methods implement the `ContinualLearning` protocol, which
provides `on_task_change(task_ix, env, logger)` and `learn(...)`. The projection
runner calls `on_task_change` before each task so the method can switch
environments, reset optimizer or replay state, and preserve any method-specific
memory.

Available CLI methods are:

- `fine_tune`: transfers network weights to the next task, resets optimizer
  state and replay buffer.
- `joint_incremental`: keeps one replay buffer per task and trains over all
  active task buffers.
- `behavior_cloning`: stores previous-policy outputs in an expert buffer and
  adds a cloning loss while learning the next task.
- `joint_incremental_pc_grad`: DQN joint-incremental training with PCGrad.
- `joint_incremental_a_gem`: DQN joint-incremental training with A-GEM-style
  gradient constraints. SAC also accepts this name as an alias for its A-GEM
  implementation.
- `a_gem`: A-GEM-style gradient constraints for DDPG, SAC, and SACD.
- `a_egem`: behavior-cloning extension using expert-gradient constraints for
  DQN, DDPG, SAC, and SACD.

Algorithm support in `projection/main.py` is:

| Algorithm | Implementations |
| --- | --- |
| `dqn` | `fine_tune`, `joint_incremental`, `behavior_cloning`, `joint_incremental_pc_grad`, `joint_incremental_a_gem`, `a_egem` |
| `sacd` | `fine_tune`, `joint_incremental`, `behavior_cloning`, `a_gem`, `a_egem` |
| `sac` | `fine_tune`, `joint_incremental`, `behavior_cloning`, `a_gem`, `joint_incremental_a_gem`, `a_egem` |
| `ddpg` | `joint_incremental`, `behavior_cloning`, `a_gem`, `a_egem` |

The concrete implementations live in:

- `stable_baselines3/dqn/dqn_fine_tune.py`
- `stable_baselines3/dqn/dqn_joint_icremental.py`
- `stable_baselines3/dqn/dqn_bc.py`
- `stable_baselines3/dqn/dqn_joint_incremental_pc_grad.py`
- `stable_baselines3/dqn/dqn_joint_incremental_a_gem.py`
- `stable_baselines3/dqn/dqn_a_egem.py`
- `stable_baselines3/sac/sac_fine_tune.py`
- `stable_baselines3/sac/sac_joint_incremental.py`
- `stable_baselines3/sac/sac_bc.py`
- `stable_baselines3/sac/sac_agem.py`
- `stable_baselines3/sac/sac_aegem.py`
- `stable_baselines3/sac/sac_ewc.py`
- `stable_baselines3/sacd/sacd_fine_tune.py`
- `stable_baselines3/sacd/sacd_joint_incremental.py`
- `stable_baselines3/sacd/sacd_bc.py`
- `stable_baselines3/sacd/sacd_agem.py`
- `stable_baselines3/sacd/sacd_aegem.py`
- `stable_baselines3/ddpg/ddpg_fine_tune.py`
- `stable_baselines3/ddpg/ddpg_joint_incremental.py`
- `stable_baselines3/ddpg/ddpg_bc.py`
- `stable_baselines3/ddpg/ddpg_agem.py`
- `stable_baselines3/ddpg/ddpg_aegem.py`

`stable_baselines3/sac/sac_ewc.py` also contains a SAC EWC implementation, but
it is not currently exposed by the `projection/main.py` method selector.

## Running Projection Experiments

Install dependencies with the project environment manager, then run from the
repository root:

```bash
uv sync
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python projection/main.py \
  --env cartpole \
  --algorithm dqn \
  --method joint_incremental \
  --benchmark V1 V2 V3 \
  --project cartpole \
  --name_prefix dqn_cp_ji
```

For continuous control:

```bash
python projection/main.py \
  --env inverted_pendulum \
  --algorithm sac \
  --method behavior_cloning \
  --benchmark V1 V2 V3 \
  --project inverted_pendulum \
  --name_prefix sac_ip_bc
```

Useful options:

- `--mode continual`: train tasks sequentially, starting a separate W&B run for
  each task version.
- `--mode multitask`: register all task environments, then train jointly.
- `--encode_task`: append a one-hot task id to observations.
- `--balanced_sampling`: keep the requested batch size per active task for
  joint-incremental replay.
- Evaluation runs on all benchmark versions by default. Passing `--eval_all`
  currently disables that behavior because the CLI flag uses `store_false`.
- `--eval_freq`: accepts either a single frequency or a schedule such as
  `--eval_freq 10000 500 50000 2500 10000`.
- `--store_weights`: save trained model checkpoints under `output/models/`.
- `--exploration_strategy boltzmann`: use Boltzmann exploration for DQN.

## Dispatching YAML Experiments

Projection experiment grids are stored under `dispatch/experiments/` and
`dispatch/experiments_2-8-9/`. Preview the generated SLURM commands with:

```bash
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/dqn_cp_ji.yaml --dry
```

Submit them by omitting `--dry`:

```bash
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/dqn_cp_ji.yaml
```

The SLURM wrapper `dispatch/dispatch_projection.sh` activates the environment,
sets `MUJOCO_GL=egl`, adds the repository to `PYTHONPATH`, and calls
`projection/main.py`.

## Outputs

Runs log to W&B and CSV files through `projection/common.py`. Generated files
are written under `output/`, including:

- `output/<project>/*.csv`: scalar logs from experiment runs.
- `output/models/<project>/*.zip`: optional saved SB3 checkpoints.
- `output/plots/`, `output/tables/`, `output/metrics/`: analysis artifacts.
- `output/cache/`: cached intermediate analysis data.
