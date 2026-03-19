import numpy as np
from gymnasium import Env
from gymnasium.wrappers import TimeLimit

from projection.benchmarks.wrappers import ObsSpaceInf, ObsLinearTransform, OneHotWrapper


PERMUTATION_SEEDS = range(90, 200)


def _random_orthogonal(seed: int, size: int, bias: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(size, size))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    q =  q.astype(np.float32)
    b = None if not bias else rng.random(size=size)

    return q, b


class ProjectedEnvBenchmark:
    def __init__(
            self,
            env_class: type[Env],
            benchmark: list[int],
            encode_task: bool,
            seed: int = 42,
            time_limit: int | None = None
    ) -> None:
        assert len(benchmark) > 0
        assert len(benchmark) == len(set(benchmark))
        assert min(benchmark) > 0
        assert max(benchmark) < len(PERMUTATION_SEEDS)

        self.env_class = env_class
        self.benchmark = benchmark
        self.encode_task = encode_task
        self.seed = seed
        self.time_limit = time_limit

    def make_single(self, version: int, test: bool = False, **env_kwargs) -> Env:
        if version not in self.benchmark:
            raise ValueError(f'Invalid version \'{version}\'')

        env = self.env_class(**env_kwargs)      # type: ignore
        env = ObsSpaceInf(env)

        if version > 1:
            q, b = _random_orthogonal(
                PERMUTATION_SEEDS[version - 1],
                env.observation_space.shape[0],
                version > 5
            )
            env = ObsLinearTransform(env, q, b)

        if self.encode_task:
            env = OneHotWrapper(env, self.benchmark.index(version), len(self))

        seed = self.seed + version + (1 if test else 0)

        if self.time_limit is not None:
            env = TimeLimit(env, self.time_limit)

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

