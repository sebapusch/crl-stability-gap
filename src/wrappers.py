
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.spaces import Box


class SuccessCounter(gym.Wrapper):
    """Helper class to keep count of successes in MetaWorld environments."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['is_success'] = info.get("success", False)
        if info['is_success']:
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def pop_successes(self) -> List[bool]:
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.current_success = False
        return self.env.reset(**kwargs)


class OneHotAdder(gym.Wrapper):
    """Appends one-hot encoding to the observation. Can be used e.g. to encode the task."""

    def __init__(
        self, env: gym.Env, one_hot_idx: int, one_hot_len: int, orig_one_hot_dim: int = 0
    ) -> None:
        super().__init__(env)
        assert 0 <= one_hot_idx < one_hot_len
        self.to_append = np.zeros(one_hot_len)
        self.to_append[one_hot_idx] = 1.0

        orig_obs_low = self.env.observation_space.low
        orig_obs_high = self.env.observation_space.high
        if orig_one_hot_dim > 0:
            orig_obs_low = orig_obs_low[:-orig_one_hot_dim]
            orig_obs_high = orig_obs_high[:-orig_one_hot_dim]
        
        # Ensure we are working with flat arrays for concatenation
        self.observation_space = Box(
            np.concatenate([orig_obs_low, np.zeros(one_hot_len)]),
            np.concatenate([orig_obs_high, np.ones(one_hot_len)]),
            dtype=np.float64 # Explicitly set dtype to avoid mismatch if needed
        )
        self.orig_one_hot_dim = orig_one_hot_dim

    def _append_one_hot(self, obs: np.ndarray) -> np.ndarray:
        if self.orig_one_hot_dim > 0:
            obs = obs[: -self.orig_one_hot_dim]
        return np.concatenate([obs, self.to_append])

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._append_one_hot(obs), reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._append_one_hot(obs), info


class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind
        
        # Metaworld environments usually have set_task
        if hasattr(env, 'set_task') and subtasks:
             env.set_task(subtasks[0])
             
        if kind == "random_init_all":
            # Some metaworld envs imply this by default or have specific flags
            # Checking implementation, _freeze_rand_vec is a property in some MW versions
            if hasattr(env, '_freeze_rand_vec'):
                env._freeze_rand_vec = False

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            # This logic depends on env internal properties being accessible
            if hasattr(env, '_random_reset_space'):
                diff = env._random_reset_space.high - env._random_reset_space.low
                self.reset_space_low = env._random_reset_space.low + 0.45 * diff
                self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        if self.kind == "random_init_fixed20":
            if hasattr(self.env, 'set_task'):
                self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            if hasattr(self.env, '_last_rand_vec'):
                rand_vec = np.random.uniform(
                    self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
                )
                self.env._last_rand_vec = rand_vec

        return self.env.reset(**kwargs)
