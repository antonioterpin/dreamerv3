"""Provide test from gym functionality."""

from __future__ import annotations
from typing import Any


import gym
import numpy as np
import pytest

import embodied
from embodied.envs.from_gym import FromGym


class _LegacyEnv(gym.Env):
    """Gym < 0.26 API: reset() -> obs, step() -> (obs, rew, done, info)."""

    observation_space = gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
        -1, 1, (2,), np.float32
    )
    action_space = gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
        -1, 1, (1,), np.float32
    )

    def __init__(self) -> None:
        """Initialize the legacy environment."""
        self.t = 0

    def reset(self) -> Any:
        """Reset state.

        Returns:
            Result of the operation.
        """
        self.t = 0
        return np.zeros((2,), np.float32)

    def step(self, action: Any) -> tuple[Any, ...]:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        self.t += 1
        done = self.t >= 2
        return np.full((2,), self.t, np.float32), 1.0, done, {}


class _ModernEnv(gym.Env):
    """Gym >= 0.26 API: reset() -> (obs, info), step() -> 5-tuple."""

    observation_space = gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
        -1, 1, (2,), np.float32
    )
    action_space = gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
        -1, 1, (1,), np.float32
    )

    def __init__(self, truncate: bool = False) -> None:
        """Initialize the modern environment.

        Args:
            truncate: Truncate value.
        """
        self.t = 0
        self.truncate = truncate

    def reset(
        self, *, seed: Any | None = None, options: Any | None = None
    ) -> tuple[Any, ...]:
        """Reset state.

        Args:
            seed: Random seed.
            options: Options value.

        Returns:
            Result of the operation.
        """
        self.t = 0
        return np.zeros((2,), np.float32), {"source": "reset"}

    def step(self, action: Any) -> tuple[Any, ...]:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        self.t += 1
        last = self.t >= 2
        terminated = last and not self.truncate
        truncated = last and self.truncate
        return np.full((2,), self.t, np.float32), 1.0, terminated, truncated, {}


def _rollout(env: Any) -> Any:
    env = FromGym(env)
    act = {k: v.sample() for k, v in env.act_space.items()}
    obs = [env.step({**act, "reset": True})]
    while not obs[-1]["is_last"]:
        obs.append(env.step({**act, "reset": False}))
    return obs


def test_legacy_api_unchanged() -> None:
    """Verify legacy api unchanged."""
    obs = _rollout(_LegacyEnv())
    assert [o["is_first"] for o in obs] == [
        True,
        False,
        False,
    ], "Expected [o is first for o in obs] to equal [True, False, False]."
    assert [o["is_last"] for o in obs] == [
        False,
        False,
        True,
    ], "Expected [o is last for o in obs] to equal [False, False, True]."
    assert [o["is_terminal"] for o in obs] == [
        False,
        False,
        True,
    ], "Expected [o is terminal for o in obs] to equal [False, False, True]."
    assert obs[0]["image"].tolist() == [
        0,
        0,
    ], "Expected obs[0] image tolist() to equal [0, 0]."
    assert obs[-1]["image"].tolist() == [
        2,
        2,
    ], "Expected obs[-1] image tolist() to equal [2, 2]."


@pytest.mark.parametrize("truncate", [False, True])
def test_modern_api_reset_tuple_and_five_tuple_step(truncate: Any) -> None:
    """Verify modern api reset tuple and five tuple step.

    Args:
        truncate: Truncate value.
    """
    env = _ModernEnv(truncate=truncate)
    wrapped = FromGym(env)
    act = {k: v.sample() for k, v in wrapped.act_space.items()}
    first = wrapped.step({**act, "reset": True})
    assert first["is_first"] and first["image"].tolist() == [
        0,
        0,
    ], 'Expected all parts of the first["is_first"] and first["image"].tolist() == [0, 0] invariant to hold.'
    assert wrapped.info == {
        "source": "reset"
    }, 'Expected wrapped info to equal {"source": "reset"}.'
    obs = [first]
    while not obs[-1]["is_last"]:
        obs.append(wrapped.step({**act, "reset": False}))
    assert len(obs) == 3, "Expected number of obs to equal 3."
    assert obs[-1]["is_last"], "Expected obs[-1] is last to be initialized or truthy."
    # A truncated episode ends without being terminal; a terminated one is.
    assert obs[-1]["is_terminal"] == (
        not truncate
    ), "Expected obs[-1] is terminal to equal not truncate."
    assert obs[-1]["reward"] == np.float32(
        1.0
    ), "Expected obs[-1] reward to equal np.float32(1.0)."
    # The episode end triggers an automatic reset on the next step.
    again = wrapped.step({**act, "reset": False})
    assert again["is_first"] and again["image"].tolist() == [
        0,
        0,
    ], 'Expected all parts of the again["is_first"] and again["image"].tolist() == [0, 0] invariant to hold.'


def test_modern_api_info_overrides_is_terminal() -> None:
    """Verify modern api info overrides is terminal.

    Returns:
        Result of the operation.
    """

    class Env(_ModernEnv):
        """Represent environment."""

        def step(self, action: Any) -> tuple[Any, ...]:
            """Advance state.

            Args:
                action: Action value.

            Returns:
                Result of the operation.
            """
            obs, rew, term, trunc, info = super().step(action)
            return obs, rew, term, trunc, {"is_terminal": False}

    obs = _rollout(Env())
    assert (
        obs[-1]["is_last"] and not obs[-1]["is_terminal"]
    ), 'Expected all parts of the obs[-1]["is_last"] and (not obs[-1]["is_terminal"]) invariant to hold.'


def test_reset_key_is_not_forwarded_to_dict_action_envs() -> None:
    """Verify reset key is not forwarded to dict action envs.

    Returns:
        Result of the operation.
    """

    class DictActionEnv(gym.Env):
        """Represent dict action environment."""

        observation_space = gym.spaces.Dict(  # pyright: ignore[reportAttributeAccessIssue]
            {
                "pos": gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
                    -1, 1, (2,), np.float32
                )
            }
        )
        action_space = gym.spaces.Dict(  # pyright: ignore[reportAttributeAccessIssue]
            {
                "move": gym.spaces.Box(  # pyright: ignore[reportAttributeAccessIssue]
                    -1, 1, (1,), np.float32
                )
            }
        )

        def __init__(self) -> None:
            """Initialize the dict action environment."""
            self.received = []

        def reset(self) -> dict[Any, Any]:
            """Reset state.

            Returns:
                Result of the operation.
            """
            return {"pos": np.zeros((2,), np.float32)}

        def step(self, action: Any) -> tuple[Any, ...]:
            """Advance state.

            Args:
                action: Action value.

            Returns:
                Result of the operation.
            """
            self.received.append(action)
            return {"pos": np.ones((2,), np.float32)}, 0.0, True, {}

    raw = DictActionEnv()
    env = FromGym(raw)
    assert set(env.act_space) == {
        "move",
        "reset",
    }, 'Expected keys in env act space to equal {"move", "reset"}.'
    act = {k: v.sample() for k, v in env.act_space.items()}
    env.step({**act, "reset": True})
    env.step({**act, "reset": False})
    assert raw.received and all(set(a) == {"move"} for a in raw.received), raw.received
