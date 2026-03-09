import numpy as np
from gymnasium import Env

from cartpole.benchmarks.wrappers import ObsSpaceInf, ObsLinearTransform, OneHotWrapper


PERMUTATION_SEEDS = range(90, 200)


def _random_orthogonal(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(size, size))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


class PermutedEnvBenchmark:
    def __init__(
            self,
            env_class: type[Env],
            benchmark: list[int],
            encode_task: bool,
            seed: int = 42,
    ) -> None:
        assert len(benchmark) > 0
        assert len(benchmark) == len(set(benchmark))
        assert min(benchmark) > 0
        assert max(benchmark) < len(PERMUTATION_SEEDS)

        self.env_class = env_class
        self.benchmark = benchmark
        self.encode_task = encode_task
        self.seed = seed

    def make_single(self, version: int, test: bool = False, **env_kwargs) -> Env:
        if version not in self.benchmark:
            raise ValueError(f'Invalid version \'{version}\'')

        env = self.env_class(**env_kwargs)      # type: ignore
        env = ObsSpaceInf(env)

        if version > 1:
            env = ObsLinearTransform(
                env,
                _random_orthogonal(PERMUTATION_SEEDS[version - 1], env.observation_space.shape[0])
            )

        if self.encode_task:
            env = OneHotWrapper(env, self.benchmark.index(version), len(self))

        seed = self.seed + version + (1 if test else 0)

        env.reset(seed=seed)
        env.action_space.seed(seed)

        return env

    def make_train(self, **env_kwargs) -> list[Env]:
        return [
            self.make_single(ix, False, **env_kwargs)
            for ix in self.benchmark
        ]

    def make_test(self, **env_kwargs) -> list[Env]:
        return [
            self.make_single(ix, True, **env_kwargs)
            for ix in self.benchmark
        ]

    def make(self, **env_kwargs) -> tuple[list[Env], list[Env]]:
        return (
            self.make_train(**env_kwargs),
            self.make_test(**env_kwargs),
        )

    def __len__(self) -> int:
        return len(self.benchmark)

