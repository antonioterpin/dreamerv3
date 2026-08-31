"""When the actor installs the parameters staged by the learner."""

from __future__ import annotations
from typing import Any


import threading

import elements
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax import internal
from embodied.jax.agent import Agent as JaxAgent
from embodied.jax.agent import Options

P = jax.sharding.PartitionSpec


def _agent(mode: Any, steps: int = 1) -> Any:
    """Create a JAX agent skeleton with a real policy path on one CPU device.

    Returns:
        Agent configured for policy synchronization tests.
    """
    agent = object.__new__(JaxAgent)
    agent.jaxcfg = Options(policy_sync_mode=mode, policy_sync_steps=steps)
    agent.config = elements.Config(seed=0)
    devices = np.array(jax.devices("cpu")[:1]).reshape((1, 1, 1))
    mesh = jax.sharding.Mesh(devices, ("d", "f", "t"))
    agent.policy_mesh = mesh
    agent.policy_sharded = jax.sharding.NamedSharding(mesh, P(("d", "f")))
    agent.policy_mirrored = jax.sharding.NamedSharding(mesh, P())
    agent.obs_space = {
        "sensor": elements.Space(np.float32, (1,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    agent.act_space = {"action": elements.Space(np.float32, (1,), -1, 1)}
    agent.policy_lock = threading.Lock()
    agent.n_actions = elements.Counter()
    agent.pending_sync = None
    agent.policy_params_sharding = {"w": agent.policy_mirrored}
    agent.policy_params = internal.move(
        {"w": jnp.float32(1.0)}, agent.policy_params_sharding
    )
    agent._split = jax.jit(
        lambda xs: jax.tree.map(lambda x: list(x), xs),
        in_shardings=internal.local_sharding(agent.policy_sharded),
        out_shardings=internal.local_sharding(agent.policy_mirrored),
    )
    agent._stack = jax.jit(
        lambda xs: jax.tree.map(jnp.stack, xs, is_leaf=lambda x: isinstance(x, list)),
        in_shardings=internal.local_sharding(agent.policy_mirrored),
        out_shardings=internal.local_sharding(agent.policy_sharded),
    )

    def _policy(
        params: Any, seed: Any, carry: Any, obs: Any, mode: Any
    ) -> tuple[Any, ...]:
        # The action reveals which parameters produced it.
        return carry, {"action": obs["sensor"] * params["w"]}, {}

    agent._policy = _policy
    return agent


def _obs(last: bool = False, batch: int = 1) -> dict[Any, Any]:
    return {
        "sensor": np.ones((batch, 1), np.float32),
        "reward": np.zeros((batch,), np.float32),
        "is_first": np.zeros((batch,), bool),
        "is_last": np.full((batch,), last),
        "is_terminal": np.full((batch,), last),
    }


def _act(agent: Any, last: bool = False) -> Any:
    carry = {"state": [np.zeros((1,), np.float32)]}
    _, acts, _ = agent.policy(carry, _obs(last))
    return float(acts["action"][0, 0])


def _stage(agent: Any, value: Any) -> None:
    agent._stage_policy_sync({"w": jnp.float32(value)})


def test_options_default_to_upstream_behavior() -> None:
    """Verify options default to upstream behavior."""
    assert (
        Options().policy_sync_mode == "immediate"
    ), 'Expected Options() policy sync mode to equal "immediate".'
    assert (
        Options().policy_sync_steps == 1
    ), "Expected Options() policy sync steps to equal 1."


@pytest.mark.parametrize(
    "mode, steps", [("immediate", 1), ("episode", 1), ("steps", 3)]
)
def test_sync_due_per_mode(mode: Any, steps: Any) -> None:
    """Verify sync due per mode.

    Args:
        mode: Mode value.
        steps: Steps value.
    """
    agent = _agent(mode, steps)
    due = lambda last, counter: agent._policy_sync_due(_obs(last), counter)
    if mode == "immediate":
        assert all(
            due(last, i) for last in (False, True) for i in range(4)
        ), "Expected all elements to satisfy the invariant."
    elif mode == "episode":
        assert not any(
            due(False, i) for i in range(4)
        ), "Expected no elements to violate the invariant."
        assert all(
            due(True, i) for i in range(4)
        ), "Expected all elements to satisfy the invariant."
        assert agent._policy_sync_due(
            _obs(False, batch=3) | {"is_last": np.array([False, True, False])}, 0
        ), "Any environment of the batch ending its episode counts"
    else:
        assert [due(False, i) for i in range(7)] == [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
        ], "Expected [due(False, i) for i in range(7)] to equal [False, False, True, False, False, True, False]."


def test_immediate_mode_installs_on_the_next_call() -> None:
    """Verify immediate mode installs on the next call."""
    agent = _agent("immediate")
    assert _act(agent) == 1.0, "Expected act(agent) to equal 1.0."
    _stage(agent, 2.0)
    assert _act(agent) == 1.0, "the call that installs still uses old params"
    assert agent.pending_sync is None, "Expected agent pending sync to be None."
    assert _act(agent) == 2.0, "Expected act(agent) to equal 2.0."


def test_episode_mode_installs_after_an_episode_end() -> None:
    """Verify episode mode installs after an episode end."""
    agent = _agent("episode")
    _stage(agent, 2.0)
    assert (
        _act(agent) == 1.0 and _act(agent) == 1.0
    ), "Expected all parts of the _act(agent) == 1.0 and _act(agent) == 1.0 invariant to hold."
    assert agent.pending_sync is not None, "mid-episode calls never install"
    assert _act(agent, last=True) == 1.0, "the terminal step uses old params"
    assert agent.pending_sync is None, "Expected agent pending sync to be None."
    assert _act(agent) == 2.0, "the next episode starts with the new params"


def test_steps_mode_installs_every_nth_call() -> None:
    """Verify steps mode installs every nth call."""
    agent = _agent("steps", steps=3)
    _stage(agent, 2.0)
    assert [_act(agent) for _ in range(3)] == [
        1.0,
        1.0,
        1.0,
    ], "Expected [ act(agent) for in range(3)] to equal [1.0, 1.0, 1.0]."
    assert agent.pending_sync is None, "installed after the third call"
    assert _act(agent) == 2.0, "Expected act(agent) to equal 2.0."
    _stage(agent, 3.0)  # Staged between calls 3 and 4; call 5 installs.
    assert _act(agent) == 2.0, "Expected act(agent) to equal 2.0."
    assert agent.pending_sync is not None, "only every third call installs"
    assert _act(agent) == 2.0, "Expected act(agent) to equal 2.0."
    assert agent.pending_sync is None, "Expected agent pending sync to be None."
    assert _act(agent) == 3.0, "Expected act(agent) to equal 3.0."


def test_immediate_mode_keeps_the_first_staged_params() -> None:
    """Verify immediate mode keeps the first staged parameters."""
    agent = _agent("immediate")
    _stage(agent, 2.0)
    first = agent.pending_sync["w"]
    newer = {"w": jnp.float32(3.0)}
    agent._stage_policy_sync(newer)
    assert (
        agent.pending_sync["w"] is first
    ), "Expected agent pending sync w to be first."
    assert newer["w"].is_deleted(), "the dropped params are released"
    assert (
        _act(agent) == 1.0 and _act(agent) == 2.0
    ), "Expected all parts of the _act(agent) == 1.0 and _act(agent) == 2.0 invariant to hold."


def test_scheduled_modes_keep_the_newest_staged_params() -> None:
    """Verify scheduled modes keep the newest staged parameters."""
    agent = _agent("episode")
    _stage(agent, 2.0)
    first = agent.pending_sync["w"]
    _stage(agent, 3.0)
    assert first.is_deleted(), "the replaced staged params are released"
    assert (
        float(agent.pending_sync["w"]) == 3.0
    ), "Expected float(agent pending sync w) to equal 3.0."
    assert _act(agent, last=True) == 1.0, "Expected act(agent, last=True) to equal 1.0."
    assert _act(agent) == 3.0, "the episode starts with the newest params"


def test_precompile_does_not_disturb_scheduled_sync() -> None:
    """Verify precompile does not disturb scheduled sync."""
    agent = _agent("steps", steps=3)
    agent.init_policy = lambda batch: {"state": [np.zeros((1,), np.float32)] * batch}
    agent.precompile_policy(1)
    assert int(agent.n_actions) == 0, "Expected int(agent n actions) to equal 0."
    _stage(agent, 2.0)
    assert [_act(agent) for _ in range(3)] == [
        1.0,
        1.0,
        1.0,
    ], "Expected [ act(agent) for in range(3)] to equal [1.0, 1.0, 1.0]."
    assert _act(agent) == 2.0, "Expected act(agent) to equal 2.0."
