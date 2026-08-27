"""Provide utils functionality."""

from __future__ import annotations
from typing import Any


import time

import elements
import zerofun
import numpy as np


class TestAgent:
    """Represent test agent."""

    def __init__(self, obs_space: Any, act_space: Any, addr: Any | None = None) -> None:
        """Initialize the test agent.

        Args:
            obs_space: Observation space value.
            act_space: Act space value.
            addr: Address value.
        """
        self.obs_space = obs_space
        self.act_space = act_space
        if addr:
            self.client = zerofun.Client(addr, connect=True)
            self.should_stats = elements.when.Clock(1)
        else:
            self.client = None
        self._stats = {
            "env_steps": 0,
            "replay_steps": 0,
            "reports": 0,
            "saves": 0,
            "loads": 0,
            "created": time.time(),
        }

    def _watcher(self) -> None:
        while True:
            if self.queue.empty():
                self.queue.put(self.stats())
            else:
                time.sleep(0.01)

    def stats(self) -> Any:
        """Handle statistics.

        Returns:
            Result of the operation.
        """
        stats = self._stats.copy()
        stats["lifetime"] = time.time() - stats.pop("created")
        return stats

    def init_policy(self, batch_size: Any) -> tuple[Any, ...]:
        """Handle init policy.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return (np.zeros(batch_size),)

    def init_train(self, batch_size: Any) -> tuple[Any, ...]:
        """Handle init train.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return (np.zeros(batch_size),)

    def init_report(self, batch_size: Any) -> tuple[Any, ...]:
        """Handle init report.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return ()

    def policy(self, carry: Any, obs: Any, mode: str = "train") -> tuple[Any, ...]:
        """Handle policy.

        Args:
            carry: Carry value.
            obs: Observation value.
            mode: Mode value.

        Returns:
            Result of the operation.
        """
        assert set(obs.keys()) == set(
            self.obs_space.keys()
        ), "Expected keys in obs keys() to equal set(self.obs_space.keys())."
        B = len(obs["is_first"])
        self._stats["env_steps"] += B
        (carry,) = carry
        carry = np.asarray(carry)

        assert carry.shape == (B,), "Expected carry shape to equal (B,)."
        assert not any(
            k.startswith("log/") for k in obs.keys()
        ), "Expected no elements to violate the invariant."

        target = (carry + 1) * (1 - obs["is_first"])
        assert (obs["count"] == target).all(), "Expected obs count to equal target."
        carry = target

        if self.client and self.should_stats():
            self.client.report(self.stats())

        act = {
            k: np.stack([v.sample() for _ in range(B)])
            for k, v in self.act_space.items()
            if k != "reset"
        }
        return (carry,), act, {}

    def train(self, carry: Any, data: Any) -> tuple[Any, ...]:
        """Train state.

        Args:
            carry: Carry value.
            data: Data to process.

        Returns:
            Result of the operation.
        """
        expected = sorted(set(self.obs_space | self.act_space) | {"stepid"})
        assert sorted(data.keys()) == expected, (sorted(data.keys()), expected)
        B, T = data["count"].shape
        (carry,) = carry
        assert carry.shape == (B,), "Expected carry shape to equal (B,)."
        assert not any(
            k.startswith("log/") for k in data.keys()
        ), "Expected no elements to violate the invariant."
        self._stats["replay_steps"] += B * T
        for t in range(T):
            current = data["count"][:, t]
            reset = data["is_first"][:, t]
            target = (1 - reset) * (carry + 1) + reset * current
            assert (current == target).all(), "Expected current to equal target."
            carry = current

        outs = {}
        metrics = {}
        return (carry,), outs, metrics

    def report(self, carry: Any, data: Any) -> tuple[Any, ...]:
        """Handle report.

        Args:
            carry: Carry value.
            data: Data to process.

        Returns:
            Result of the operation.
        """
        self._stats["reports"] += 1
        return carry, {
            "scalar": np.float32(0),
            "vector": np.zeros(10),
            "image1": np.zeros((64, 64, 1)),
            "image3": np.zeros((64, 64, 3)),
            "video": np.zeros((10, 64, 64, 3)),
        }

    def dataset(self, generator: Any) -> Any:
        """Handle dataset.

        Args:
            generator: Generator value.

        Returns:
            Result of the operation.
        """
        return generator()

    def save(self) -> Any:
        """Save state.

        Returns:
            Result of the operation.
        """
        self._stats["saves"] += 1
        return self._stats

    def load(self, data: Any) -> None:
        """Load state.

        Args:
            data: Data to process.
        """
        self._stats = data
        self._stats["loads"] += 1
