"""Provide crafter functionality."""

from __future__ import annotations
from typing import Any


import json

import crafter
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):
    """Represent crafter."""

    def __init__(
        self,
        task: Any,
        size: tuple[Any, ...] = (64, 64),
        logs: bool = False,
        logdir: Any | None = None,
        seed: Any | None = None,
    ) -> None:
        """Initialize the crafter.

        Args:
            task: Task value.
            size: Requested number of elements.
            logs: Logs value.
            logdir: Logging directory value.
            seed: Random seed.
        """
        assert task in (
            "reward",
            "noreward",
        ), 'Expected task to be present in ("reward", "noreward").'
        self._env = crafter.Env(size=size, reward=(task == "reward"), seed=seed)
        self._logs = logs
        self._logdir = logdir and elements.Path(logdir)
        self._logdir and self._logdir.mkdir()
        self._episode = 0
        self._length = None
        self._reward = None
        self._achievements = crafter.constants.achievements.copy()
        self._done = True

    @property
    def obs_space(self) -> Any:
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        spaces = {
            "image": elements.Space(np.uint8, self._env.observation_space.shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/reward": elements.Space(np.float32),
        }
        if self._logs:
            spaces.update(
                {
                    f"log/achievement_{k}": elements.Space(np.int32)
                    for k in self._achievements
                }
            )
        return spaces

    @property
    def act_space(self) -> dict[Any, Any]:
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return {
            "action": elements.Space(np.int32, (), 0, self._env.action_space.n),
            "reset": elements.Space(bool),
        }

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        if action["reset"] or self._done:
            self._episode += 1
            self._length = 0
            self._reward = 0
            self._done = False
            image = self._env.reset()
            return self._obs(image, 0.0, {}, is_first=True)
        image, reward, self._done, info = self._env.step(action["action"])
        self._reward += reward
        self._length += 1
        if self._done and self._logdir:
            self._write_stats(self._length, self._reward, info)
        return self._obs(
            image, reward, info, is_last=self._done, is_terminal=info["discount"] == 0
        )

    def _obs(
        self,
        image: Any,
        reward: Any,
        info: Any,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> Any:
        obs = dict(
            image=image,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            **{"log/reward": np.float32(info["reward"] if info else 0.0)},
        )
        if self._logs:
            log_achievements = {
                f"log/achievement_{k}": info["achievements"][k] if info else 0
                for k in self._achievements
            }
            obs.update({k: np.int32(v) for k, v in log_achievements.items()})
        return obs

    def _write_stats(self, length: Any, reward: Any, info: Any) -> None:
        stats = {
            "episode": self._episode,
            "length": length,
            "reward": round(reward, 1),
            **{f"achievement_{k}": v for k, v in info["achievements"].items()},
        }
        filename = self._logdir / "stats.jsonl"
        lines = filename.read() if filename.exists() else ""
        lines += json.dumps(stats) + "\n"
        filename.write(lines, mode="w")
        print(f"Wrote stats: {filename}")
