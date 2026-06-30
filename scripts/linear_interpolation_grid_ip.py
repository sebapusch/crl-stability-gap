import argparse
import zipfile
from functools import reduce
from os import path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

type MLP = list[tuple[jax.Array, jax.Array]]

N_STEPS = 200
N_ENV_STEPS = 1000
N_EVAL = 15
MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "__", "models"))
if not path.exists(MODEL_PATH):
    MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "models"))


# Physics constants matching MuJoCo InvertedPendulum-v5
M = 10.47197551
m = 5.01859164
x0 = 0.0005
z0 = 0.3
I = 0.18874977
g = 9.81
J = I + m * (x0**2 + z0**2)


def generate_combinations() -> jax.Array:
    vals = jnp.linspace(-0.5, 1.5, N_STEPS)
    X, Y = jnp.meshgrid(vals, vals)

    grid_matrix = jnp.stack([X, Y], axis=-1)
    grid_matrix = grid_matrix.reshape((-1, 2))

    return grid_matrix


def forward(policy: MLP, x: jax.Array) -> jax.Array:
    Wo, bo = policy[-1]
    latent = reduce(lambda xo, l: jax.nn.relu(l[0] @ xo + l[1]), policy[:-1], x)
    mean = Wo @ latent + bo
    # SAC/DDPG squashes actions to [-1, 1] then scales to [-3, 3] for InvertedPendulum
    return 3.0 * jnp.tanh(mean)


def policy_fn(policy: MLP, batch_obs: jax.Array) -> jax.Array:
    vmap_forward = jax.vmap(forward, in_axes=(None, 0))
    actions = vmap_forward(policy, batch_obs)
    return actions


def dynamics(state: jax.Array, action: float) -> tuple[jax.Array, jax.Array]:
    x, theta, x_dot, theta_dot = state
    F = 100.0 * action

    # Mass matrix and right hand side
    m11 = M + m
    m12 = m * (z0 * jnp.cos(theta) - x0 * jnp.sin(theta))
    m22 = J

    a_prime = -m * (z0 * jnp.sin(theta) + x0 * jnp.cos(theta))
    b1 = F - x_dot - a_prime * (theta_dot**2)
    b2 = -theta_dot - g * a_prime

    # Solve system: Mass * [x_ddot, theta_ddot]^T = [b1, b2]^T
    det = m11 * m22 - m12**2
    x_ddot = (m22 * b1 - m12 * b2) / det
    theta_ddot = (-m12 * b1 + m11 * b2) / det

    return x_ddot, theta_ddot


def rk4_step(state: jax.Array, action: float, dt: float = 0.02) -> jax.Array:
    def f(s):
        x_val, theta_val, x_dot_val, theta_dot_val = s
        x_ddot, theta_ddot = dynamics(s, action)
        return jnp.stack([x_dot_val, theta_dot_val, x_ddot, theta_ddot])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_2steps(state: jax.Array, action: jax.Array) -> jax.Array:
    action_val = action[0] if action.ndim > 0 else action
    state1 = rk4_step(state, action_val, 0.02)
    state2 = rk4_step(state1, action_val, 0.02)
    return state2


def evaluate(
    policy: MLP,
    onehot: jax.Array,
    proj_mat: jax.Array,
    proj_bias: jax.Array,
    angle_limit: float,
    key: jax.Array,
) -> jax.Array:
    # Reset model uniformly matching Gymnasium
    vmap_reset = jax.vmap(lambda k: jax.random.uniform(k, minval=-0.01, maxval=0.01, shape=(4,)))

    batch_onehot = jnp.tile(onehot, (N_EVAL, 1))

    def process_obs(obs_batch: jax.Array) -> jax.Array:
        proj_obs = obs_batch @ proj_mat.T + proj_bias
        return jnp.concatenate([proj_obs, batch_onehot], axis=-1)

    rng, reset_rng = jax.random.split(key)
    reset_keys = jax.random.split(reset_rng, N_EVAL)

    init_state = vmap_reset(reset_keys)
    init_obs_processed = process_obs(init_state)

    def scan_step(carry, _):
        current_state, current_obs_processed, already_done = carry

        actions = policy_fn(policy, current_obs_processed)

        vmap_step_env = jax.vmap(rk4_step_2steps, in_axes=(0, 0))
        next_state = vmap_step_env(current_state, actions)

        # Check termination condition
        angles = next_state[:, 1]
        is_finite = jnp.isfinite(next_state).all(axis=-1)
        dones = jnp.logical_or(~is_finite, jnp.abs(angles) > angle_limit)

        next_obs_processed = process_obs(next_state)

        rewards = 1.0 - already_done
        masked_rewards = jnp.where(already_done, 0.0, rewards)

        next_already_done = jnp.logical_or(already_done, dones)

        next_carry = (next_state, next_obs_processed, next_already_done)

        return next_carry, masked_rewards

    initial_already_done = jnp.zeros(N_EVAL, dtype=bool)
    initial_carry = (init_state, init_obs_processed, initial_already_done)

    final_carry, reward_history = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(N_ENV_STEPS)
    )

    return reward_history


def combine(a: MLP, b: MLP, c: MLP, d: MLP, alpha: jax.Array, beta: jax.Array) -> MLP:
    comb: MLP = []
    for al, bl, cl, dl in zip(a, b, c, d):
        comb.append((
            (beta * ((1 - alpha) * al[0] + alpha * bl[0]) + (1 - beta) * ((1 - alpha) * cl[0] + alpha * dl[0])),
            (beta * ((1 - alpha) * al[1] + alpha * bl[1]) + (1 - beta) * ((1 - alpha) * cl[1] + alpha * dl[1])),
        ))

    return comb


def evaluate_combination(
    policies: tuple[MLP, MLP, MLP, MLP],
    vals: jax.Array,
    onehot: jax.Array,
    proj_mat: jax.Array,
    proj_bias: jax.Array,
    angle_limit: float,
) -> jax.Array:
    comb = combine(*policies, alpha=vals[0], beta=vals[1])

    key = jax.random.key(0)

    rh = evaluate(
        comb,
        onehot,
        proj_mat=proj_mat,
        proj_bias=proj_bias,
        angle_limit=angle_limit,
        key=key
    )

    return rh.sum(axis=0)


MAT_V2 = jnp.array([
    [ 0.14022304,  0.4014328 , -0.9011977  ,-0.08385649],
    [ 0.6846759 , -0.34279321, -0.10520617 , 0.63454187],
    [-0.70954596, -0.14529827, -0.2354287  , 0.64807891],
    [ 0.09000526,  0.83679922,  0.34834996 , 0.41269653],
])


def get_projection(version_str: str, num_tasks: int, benchmark: list[str]) -> tuple[jax.Array, jax.Array, jax.Array]:
    version = int(version_str.strip("V"))
    task_idx = benchmark.index(version_str)
    
    # One-hot representation of this task
    onehot = np.zeros(num_tasks, dtype=np.float32)
    onehot[task_idx] = 1.0

    if version == 1:
        return jnp.eye(4, dtype=jnp.float32), jnp.zeros(4, dtype=jnp.float32), jnp.array(onehot)
    elif version == 2:
        return MAT_V2, jnp.zeros(4, dtype=jnp.float32), jnp.array(onehot)

    seed = range(90, 200)[version - 1]
    has_bias = version > 5

    rng = np.random.default_rng(seed)
    m = rng.normal(size=(4, 4))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    q = q.astype(np.float32)

    if has_bias:
        b = rng.random(size=4).astype(np.float32)
    else:
        b = np.zeros(4, dtype=np.float32)

    return jnp.array(q), jnp.array(b), jnp.array(onehot)



def load_policies(model_path: str, benchmark: list[str]) -> tuple[MLP, MLP, MLP, MLP]:
    models = {}

    for v in benchmark:
        with zipfile.ZipFile(f"{model_path}-{v}.zip") as archive:
            with archive.open("policy.pth", mode="r") as param_file:
                th_object = torch.load(param_file, weights_only=True)

                # Continuous policies (SAC/DDPG) use latent_pi.0, latent_pi.2, and mu layers
                models[v] = [
                    (jnp.array(th_object["actor.latent_pi.0.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.0.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.latent_pi.2.weight"].cpu().numpy()), jnp.array(th_object["actor.latent_pi.2.bias"].cpu().numpy())),
                    (jnp.array(th_object["actor.mu.weight"].cpu().numpy()), jnp.array(th_object["actor.mu.bias"].cpu().numpy())),
                ]

    t1, t2, t3 = benchmark[0], benchmark[1], benchmark[2]
    d = [(lc[0] + lb[0] - la[0], lc[1] + lb[1] - la[1]) for la, lb, lc in zip(models[t1], models[t2], models[t3])]

    return models[t1], models[t2], models[t3], d


def main(
    seeds: list[int],
    model_path: str,
    output_dir: str,
    benchmark: list[str] = ["V2", "V8", "V9"],
    eval_task: str | None = None,
    hard: bool = False,
    chunk_size: int = 2000,
) -> None:
    if eval_task is None:
        eval_task = benchmark[0]

    angle_limit = 0.1 if hard else 0.2
    print(f"Evaluating task {eval_task} with angle limit {angle_limit} radians")

    proj_mat, proj_bias, onehot = get_projection(eval_task, len(benchmark), benchmark)

    for s in tqdm.tqdm(seeds):
        try:
            policies = load_policies(
                path.join(MODEL_PATH, model_path.replace('<s>', str(s))),
                benchmark
            )
        except (FileNotFoundError, zipfile.BadZipFile, KeyError) as e:
            print(f"Warning: Skipping seed {s} due to missing or corrupt model file: {e}")
            continue

        combinations = generate_combinations()

        
        # Partially apply env details to evaluate_combination
        eval_fn = lambda pols, comb: evaluate_combination(
            pols, comb, onehot, proj_mat, proj_bias, angle_limit
        )
        
        eval_vmap = jax.vmap(eval_fn, in_axes=(None, 0))

        results = []
        for i in range(0, len(combinations), chunk_size):
            chunk = combinations[i : i + chunk_size]
            res_chunk = eval_vmap(policies, chunk)
            res_chunk = res_chunk.mean(axis=-1)
            results.append(res_chunk)
            
        res = jnp.concatenate(results, axis=0)

        data = jnp.column_stack((combinations, res))

        np.savetxt(path.join(output_dir, f"data_{s}.csv"), np.asarray(data), delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--benchmark', nargs='+', type=str, default=["V2", "V8", "V9"])
    parser.add_argument('--eval_task', type=str, default=None)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=2000)

    main(**parser.parse_args().__dict__)

