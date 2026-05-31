import zipfile
from functools import reduce
from os import path

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import torch

type MLP = list[tuple[jax.Array, jax.Array]]


N_STEPS = 1000
N_ENV_STEPS = 500
N_EVAL = 15
GAMMA = 0.99
BENCHMARK = ["V1", "V2", "V3"]
MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "models"))

def generate_combinations() -> jax.Array:
    vals = jnp.linspace(-0.5, 1, N_STEPS)
    X, Y = jnp.meshgrid(vals, vals)

    grid_matrix = jnp.stack([X, Y], axis=-1)
    grid_matrix = grid_matrix.reshape((-1, 2))

    return grid_matrix


def forward(policy: MLP, x: jax.Array) -> jax.Array:
    Wo, bo = policy[-1]

    return Wo @ reduce(lambda xo, l: jax.nn.relu(l[0] @ xo + l[1]), policy[:-1], x) + bo


def policy_fn(policy: MLP, batch_obs: jax.Array) -> jax.Array:
    vmap_forward = jax.vmap(forward, in_axes=(None, 0))
    q_values = vmap_forward(policy, batch_obs)

    return jnp.argmax(q_values, axis=-1)


def evaluate(policy: MLP, onehot: jax.Array, proj_mat: jax.Array, key: jax.Array) -> jax.Array:
    env, env_params = gymnax.make('CartPole-v1')
    env_params = env_params.replace(max_steps_in_episode=N_ENV_STEPS)

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    batch_onehot = jnp.tile(onehot, (N_EVAL, 1))

    def process_obs(obs_batch: jax.Array) -> jax.Array:
        proj_obs = obs_batch @ proj_mat

        return jnp.concatenate([proj_obs, batch_onehot], axis=-1)

    rng, reset_rng = jax.random.split(key)
    reset_keys = jax.random.split(reset_rng, N_EVAL)

    init_obs, init_state = vmap_reset(reset_keys, env_params)
    init_obs = process_obs(init_obs)

    def scan_step(carry, _):
        current_obs, current_state, current_rng, already_done = carry

        current_rng, step_rng = jax.random.split(current_rng)
        step_keys = jax.random.split(step_rng, N_EVAL)

        actions = policy_fn(policy, current_obs)

        next_obs, next_state, rewards, dones, info = vmap_step(
            step_keys, current_state, actions, env_params
        )

        next_obs = process_obs(next_obs)

        masked_rewards = jnp.where(already_done, 0.0, rewards)

        next_already_done = jnp.logical_or(already_done, dones)

        next_carry = (next_obs, next_state, current_rng, next_already_done)

        return next_carry, masked_rewards

    initial_already_done = jnp.zeros(N_EVAL, dtype=bool)
    initial_carry = (init_obs, init_state, rng, initial_already_done)

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


def evaluate_combination(policies: tuple[MLP, MLP, MLP, MLP], vals: jax.Array) -> jax.Array:
    comb = combine(*policies, alpha=vals[0], beta=vals[1])

    key = jax.random.key(0)

    rh = evaluate(
        comb,
        jnp.array([1, 0, 0], dtype=jnp.float32),
        proj_mat=jnp.identity(4),
        key=key
    )

    return rh.sum(axis=0)


def load_policies(model_path: str) -> tuple[MLP, MLP, MLP, MLP]:
    models = {}

    for v in BENCHMARK:

        with zipfile.ZipFile(f"{model_path}-{v}.zip") as archive:
            with archive.open("policy.pth", mode="r") as param_file:
                th_object = torch.load(param_file, weights_only=True)

                models[v] = [
                    (jnp.array(th_object["q_net.q_net.0.weight"].cpu().numpy()), jnp.array(th_object["q_net.q_net.0.bias"].cpu().numpy())),
                    (jnp.array(th_object["q_net.q_net.2.weight"].cpu().numpy()), jnp.array(th_object["q_net.q_net.2.bias"].cpu().numpy())),
                    (jnp.array(th_object["q_net.q_net.4.weight"].cpu().numpy()), jnp.array(th_object["q_net.q_net.4.bias"].cpu().numpy())),
                ]

    d = [(lc[0] + lb[0] - la[0], lc[1] + lb[1] - la[1]) for la, lb, lc in zip(models['V1'], models['V2'], models['V3'])]

    return models['V1'], models['V2'], models['V3'], d


def main() -> None:
    for s in range(10):
        policies = load_policies(
            path.join(MODEL_PATH, f"dqn_linear_interpolation/dqn_linear_interpolation-s_{s}")
        )

        combinations = generate_combinations()
        eval_vmap = jax.vmap(evaluate_combination, in_axes=(None, 0))

        res = eval_vmap(policies, combinations)
        res = res.mean(axis=-1)

        data = jnp.column_stack((combinations, res))

        np.savetxt(f"data_{s}.csv", np.asarray(data), delimiter=",")


if __name__ == "__main__":
    main()

