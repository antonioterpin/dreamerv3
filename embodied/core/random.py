"""Provide random functionality."""

import numpy as np


class RandomAgent:
    """Represent random agent."""

    def __init__(self, obs_space, act_space):
        """Initialize the random agent.

        Args:
            obs_space: Observation space value.
            act_space: Act space value.
        """
        self.obs_space = obs_space
        self.act_space = act_space

    def init_policy(self, batch_size):
        """Handle init policy.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return ()

    def init_train(self, batch_size):
        """Handle init train.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return ()

    def init_report(self, batch_size):
        """Handle init report.

        Args:
            batch_size: Batch size value.

        Returns:
            Result of the operation.
        """
        return ()

    def policy(self, carry, obs, mode="train"):
        """Handle policy.

        Args:
            carry: Carry value.
            obs: Observation value.
            mode: Mode value.

        Returns:
            Result of the operation.
        """
        batch_size = len(obs["is_first"])
        act = {
            k: np.stack([v.sample() for _ in range(batch_size)])
            for k, v in self.act_space.items()
            if k != "reset"
        }
        return carry, act, {}

    def train(self, carry, data):
        """Train state.

        Args:
            carry: Carry value.
            data: Data to process.

        Returns:
            Result of the operation.
        """
        return carry, {}, {}

    def report(self, carry, data):
        """Handle report.

        Args:
            carry: Carry value.
            data: Data to process.

        Returns:
            Result of the operation.
        """
        return carry, {}

    def stream(self, st):
        """Handle stream.

        Args:
            st: St value.

        Returns:
            Result of the operation.
        """
        return st

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return None

    def load(self, data=None):
        """Load state.

        Args:
            data: Data to process.
        """
        pass
