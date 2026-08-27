"""Policy warmup between checkpoint restoration and actor readiness."""

import threading
import types

import elements
import jax
import numpy as np
import pytest

import embodied
from embodied.jax.agent import Agent as JaxAgent


def _client(*args, **kwargs):
    return types.SimpleNamespace(close=lambda: None, stats=lambda: {})


def _server_factory(events):

    def make_server(*args, **kwargs):
        events.append("server_construct")
        return types.SimpleNamespace(
            bind=lambda *a, **k: None,
            start=lambda **k: events.append("server_start"),
            close=lambda: None,
            stats=lambda: {},
        )

    return make_server


def _args(actor_batch):
    return types.SimpleNamespace(
        actor_batch=actor_batch,
        log_every=0,
        actor_threads=1,
        logger_addr="logger",
        replay_addr="replay",
        actor_addr="actor",
    )


def _run_actor_startup(monkeypatch, events, warmup_error=None):

    class Agent:

        def init_policy(self, batch_size):
            events.append("init_policy")
            return {"state": np.zeros((batch_size, 1), np.float32)}

        def precompile_policy(self, batch_size):
            events.append(f"warmup:{batch_size}")
            if warmup_error is not None:
                raise warmup_error
            events.append("warmup_complete")

    waits = [0]

    def wait():
        waits[0] += 1
        events.append("checkpoint_restored" if waits[0] == 1 else "learner_continues")

    monkeypatch.setattr(embodied.run.parallel.portal, "Client", _client)
    monkeypatch.setattr(
        embodied.run.parallel.portal, "BatchServer", _server_factory(events)
    )
    embodied.run.parallel.parallel_actor(
        Agent(),
        types.SimpleNamespace(
            wait=wait, abort=lambda: events.append("barrier_aborted")
        ),
        _args(3),
    )


def test_actor_warms_policy_after_restore_before_serving(monkeypatch):
    events = []
    _run_actor_startup(monkeypatch, events)
    assert events == [
        "init_policy",
        "checkpoint_restored",
        "warmup:3",
        "warmup_complete",
        "learner_continues",
        "server_construct",
        "server_start",
    ], events


def test_failed_warmup_aborts_barrier_and_never_serves(monkeypatch):
    events = []
    with pytest.raises(RuntimeError, match="warmup failed"):
        _run_actor_startup(monkeypatch, events, RuntimeError("warmup failed"))
    assert "server_construct" not in events, events
    assert "barrier_aborted" in events, events


def test_learner_waits_for_warmup(monkeypatch):
    events = []
    warmup_started = threading.Event()
    finish_warmup = threading.Event()
    barrier = threading.Barrier(2)
    errors = []

    class Agent:

        def init_policy(self, batch_size):
            return {"state": np.zeros((batch_size, 1), np.float32)}

        def precompile_policy(self, batch_size):
            events.append("warmup_begin")
            warmup_started.set()
            assert finish_warmup.wait(5.0), "test did not release the warmup"
            events.append("warmup_end")

    monkeypatch.setattr(embodied.run.parallel.portal, "Client", _client)
    monkeypatch.setattr(
        embodied.run.parallel.portal, "BatchServer", _server_factory(events)
    )

    def actor():
        try:
            embodied.run.parallel.parallel_actor(Agent(), barrier, _args(2))
        except BaseException as e:
            errors.append(e)

    def learner():
        events.append("checkpoint_restored")
        barrier.wait()
        barrier.wait()
        events.append("learner_continues")

    threads = [threading.Thread(target=actor), threading.Thread(target=learner)]
    [t.start() for t in threads]
    assert warmup_started.wait(5.0), "actor never entered policy warmup"
    assert "learner_continues" not in events, events
    assert "server_start" not in events, events
    finish_warmup.set()
    [t.join(5.0) for t in threads]
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors
    assert events.index("warmup_end") < events.index("learner_continues")
    assert events.index("warmup_end") < events.index("server_start")


def test_jax_warmup_compiles_real_shape_without_state_leaks():
    agent = object.__new__(JaxAgent)
    agent.obs_space = {
        "sensor": elements.Space(np.float32, (2,)),
        "index": elements.Space(np.int32, (1,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    agent.policy_lock = threading.Lock()
    agent.n_actions = elements.Counter(17)
    agent.pending_sync = None
    traces, observations, modes = [], [], []

    @jax.jit
    def compiled(state, sensor):
        traces.append("trace")
        return state + 1, sensor[:, :1]

    def init_policy(batch_size):
        return {"state": np.zeros((batch_size, 1), np.float32)}

    def policy(carry, obs, mode="train"):
        with agent.n_actions.lock:
            agent.n_actions.value += 1
        observations.append({k: v.copy() for k, v in obs.items()})
        modes.append(mode)
        state, action = compiled(carry["state"], obs["sensor"])
        return {"state": state}, {"action": action}, {}

    agent.init_policy = init_policy
    agent.policy = policy

    agent.precompile_policy(4)
    real_carry = agent.init_policy(4)
    agent.policy(real_carry, agent._zeros(agent.obs_space, (4,)), mode="train")

    dummy = observations[0]
    assert set(dummy) == set(agent.obs_space)
    for key, space in agent.obs_space.items():
        assert dummy[key].shape == (4, *space.shape), key
        assert dummy[key].dtype == space.dtype, key
    assert np.all(dummy["is_first"]) and not np.any(dummy["is_last"])
    assert not np.any(dummy["is_terminal"]) and np.all(dummy["reward"] == 0)
    assert modes == ["train", "train"]
    assert traces == ["trace"], "the real call must reuse the warmup compilation"
    assert int(agent.n_actions) == 18, "warmup must not consume an action index"
    assert np.all(real_carry["state"] == 0), "dummy carry must not leak"
    assert agent.pending_sync is None
