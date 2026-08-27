"""Provide from gym functionality."""

import functools

import elements
import embodied
import gym
import numpy as np


class FromGym(embodied.Env):
    """Represent from gym."""

    def __init__(self, env, obs_key="image", act_key="action", **kwargs):
        """Initialize the from gym.

        Args:
            env: Environment value.
            obs_key: Observation key value.
            act_key: Act key value.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        if isinstance(env, str):
            self._env = gym.make(env, **kwargs)
        else:
            assert not kwargs, kwargs
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

    @property
    def env(self):
        """Handle environment.

        Returns:
            Result of the operation.
        """
        return self._env

    @property
    def info(self):
        """Handle info.

        Returns:
            Result of the operation.
        """
        return self._info

    @functools.cached_property
    def obs_space(self):
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        """Handle act space.

        Returns:
            Result of the operation.
        """
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = elements.Space(bool)
        return spaces

    def step(self, action):
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        if action["reset"] or self._done:
            self._done = False
            obs = self._env.reset()
            if isinstance(obs, tuple):  # Gym >= 0.26 returns (obs, info).
                obs, self._info = obs
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            # 'reset' belongs to the embodied action space, not to the wrapped env.
            action = self._unflatten({k: v for k, v in action.items() if k != "reset"})
        else:
            action = action[self._act_key]
        result = self._env.step(action)
        if len(result) == 5:  # Gym >= 0.26 returns (obs, rew, term, trunc, info).
            obs, reward, terminated, truncated, self._info = result
            self._done = terminated or truncated
            is_terminal = bool(self._info.get("is_terminal", terminated))
        else:
            obs, reward, self._done, self._info = result
            is_terminal = bool(self._info.get("is_terminal", self._done))
        return self._obs(obs, reward, is_last=bool(self._done), is_terminal=is_terminal)

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return obs

    def render(self):
        """Render state.

        Returns:
            Result of the operation.
        """
        image = self._env.render("rgb_array")
        assert image is not None, "Expected image not to be None."
        return image

    def close(self):
        """Close state."""
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)
