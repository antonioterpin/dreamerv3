"""Provide base functionality."""


class Agent:
    """Represent agent."""

    def __init__(self, obs_space, act_space, config):
        """Initialize the agent.

        Args:
            obs_space: Observation space value.
            act_space: Act space value.
            config: Runtime configuration.
        """
        pass

    def init_train(self, batch_size):
        """Handle init train.

        Args:
            batch_size: Batch size value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("init_train(batch_size) -> carry")

    def init_report(self, batch_size):
        """Handle init report.

        Args:
            batch_size: Batch size value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("init_report(batch_size) -> carry")

    def init_policy(self, batch_size):
        """Handle init policy.

        Args:
            batch_size: Batch size value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("init_policy(batch_size) -> carry")

    def train(self, carry, data):
        """Train state.

        Args:
            carry: Carry value.
            data: Data to process.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("train(carry, data) -> carry, out, metrics")

    def report(self, carry, data):
        """Handle report.

        Args:
            carry: Carry value.
            data: Data to process.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("report(carry, data) -> carry, metrics")

    def policy(self, carry, obs, mode):
        """Handle policy.

        Args:
            carry: Carry value.
            obs: Observation value.
            mode: Mode value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("policy(carry, obs, mode) -> carry, act, out")

    def stream(self, st):
        """Handle stream.

        Args:
            st: St value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("stream(st) -> st")

    def save(self):
        """Save state.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("save() -> data")

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("load(data) -> None")


class Env:
    """Represent environment."""

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"obs_space={self.obs_space}, "
            f"act_space={self.act_space})"
        )

    @property
    def obs_space(self):
        # The observation space must contain the keys is_first, is_last, and
        # is_terminal. Commonly, it also contains the keys reward and image. By
        # convention, keys starting with 'log/' are not consumed by the agent.
        """Handle observation space.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("Returns: dict of spaces")

    @property
    def act_space(self):
        # The action space must contain the reset key as well as any actions.
        """Handle act space.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("Returns: dict of spaces")

    def step(self, action):
        """Advance state.

        Args:
            action: Action value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError("Returns: dict")

    def close(self):
        """Close state."""
        pass


class Stream:
    """Represent stream."""

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def save(self):
        """Save state.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def load(self, state):
        """Load state.

        Args:
            state: State value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError
