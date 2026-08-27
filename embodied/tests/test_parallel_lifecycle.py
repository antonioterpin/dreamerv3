"""Graceful completion of finite runs in the parallel runner."""

import functools
import multiprocessing
import time
import warnings

import elements
import numpy as np
import portal
import pytest

import embodied


class RunDone(Exception):
    """Raised by a finite environment when asked to reset past its last episode."""


class FiniteEnv:
    """One-step episodes; the reset after the second episode ends the run."""

    def __init__(self, events, failure=None):
        """Initialize the finite environment.

        Args:
            events: Events value.
            failure: Failure value.
        """
        self.events = events
        self.failure = failure
        self.resets = 0
        self.steps = 0
        self.obs_space = {
            "obs": elements.Space(np.float32),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        self.act_space = {
            "action": elements.Space(np.float32, (1,), -1, 1),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.

        Raises:
            self.failure or RunDone: If the operation cannot be completed.
        """
        if action["reset"]:
            self.resets += 1
            self.events.append(f"reset:{self.resets}")
            if self.resets == 2:
                raise (self.failure or RunDone)("run complete")
            return self._obs(first=True)
        self.steps += 1
        self.events.append(f"step:{self.steps}")
        return self._obs(last=True)

    def _obs(self, first=False, last=False):
        return {
            "obs": np.float32(self.steps),
            "reward": np.float32(1),
            "is_first": first,
            "is_last": last,
            "is_terminal": last,
        }

    def close(self):
        """Close state."""
        self.events.append("env_closed")


class Agent:
    """Represent agent."""

    def __init__(self, events, prefetch=False):
        """Initialize the agent.

        Args:
            events: Events value.
            prefetch: Prefetch value.
        """
        self.events = events
        self.prefetch = prefetch

    def init_policy(self, batch):
        """Handle init policy.

        Args:
            batch: Batch of values to process.

        Returns:
            Result of the operation.
        """
        return {"state": np.zeros((batch, 1), np.float32)}

    def policy(self, carry, obs, mode="train"):
        """Handle policy.

        Args:
            carry: Carry value.
            obs: Observation value.
            mode: Mode value.

        Returns:
            Result of the operation.
        """
        self.events.append("policy")
        batch = obs["reward"].shape[0]
        return carry, {"action": np.zeros((batch, 1), np.float32)}, {}

    def init_train(self, batch):
        """Handle init train.

        Args:
            batch: Batch of values to process.

        Returns:
            Result of the operation.
        """
        return None

    def init_report(self, batch):
        """Handle init report.

        Args:
            batch: Batch of values to process.

        Returns:
            Result of the operation.
        """
        return None

    def stream(self, stream):
        # The JAX agent wraps its streams in Prefetch threads; mimic that.
        """Handle stream.

        Args:
            stream: Stream value.

        Returns:
            Result of the operation.
        """
        return embodied.streams.Prefetch(stream) if self.prefetch else stream

    def train(self, carry, batch):
        """Train state.

        Args:
            carry: Carry value.
            batch: Batch of values to process.

        Returns:
            Result of the operation.
        """
        self.events.append("train")
        return carry, {}, {}

    def report(self, carry, batch):
        """Handle report.

        Args:
            carry: Carry value.
            batch: Batch of values to process.

        Returns:
            Result of the operation.
        """
        return carry, {}

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        self.events.append("checkpoint")
        return {"value": np.asarray(1)}

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.

        Returns:
            Result of the operation.
        """
        return None


class Replay:
    """Represent replay."""

    length = 1

    def __init__(self, events):
        """Initialize the replay.

        Args:
            events: Events value.
        """
        self.events = events
        self.items = []

    def add(self, item, envid):
        """Add state.

        Args:
            item: Item value.
            envid: Envid value.
        """
        self.items.append(item)
        self.events.append(f'replay:{bool(item["is_last"])}')

    def update(self, data):
        """Update state.

        Args:
            data: Data to process.

        Returns:
            Result of the operation.
        """
        return None

    def stats(self):
        """Handle statistics.

        Returns:
            Result of the operation.
        """
        return {}

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return {"items": len(self.items)}

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.

        Returns:
            Result of the operation.
        """
        return None


class Logger:
    """Represent logger."""

    def __init__(self, events):
        """Initialize the logger.

        Args:
            events: Events value.
        """
        self.events = events
        self.step = elements.Counter()

    def add(self, metrics, prefix=None):
        """Add state.

        Args:
            metrics: Metrics value.
            prefix: Prefix value.
        """
        self.events.append("logger_submission")

    def write(self):
        """Write state."""
        self.events.append("logger_write")

    def close(self):
        """Close state."""
        self.events.append("logger_closed")


def make_env(events, failure, envid):
    """Create environment.

    Args:
        events: Events value.
        failure: Failure value.
        envid: Envid value.

    Returns:
        Result of the operation.
    """
    return FiniteEnv(events, failure)


def make_stream(replay, source):
    """Create stream.

    Args:
        replay: Replay value.
        source: Source value.
    """
    while True:
        while not replay.items:
            time.sleep(0.001)
        yield {k: np.asarray([[v]]) for k, v in replay.items[-1].items()}


def make_args(tmp_path, **overrides):
    """Create args.

    Args:
        tmp_path: Tmp path value.
        overrides: Overrides value.

    Returns:
        Result of the operation.
    """
    args = dict(
        actor_batch=1,
        envs=1,
        eval_envs=0,
        actor_addr="localhost:{auto}",
        replay_addr="localhost:{auto}",
        logger_addr="localhost:{auto}",
        agent_process=True,
        remote_envs=False,
        remote_replay=False,
        actor_threads=1,
        log_every=0,
        report_every=0,
        save_every=0,
        save_on_episode=False,
        usage={
            "psutil": False,
            "nvsmi": False,
            "gputil": False,
            "malloc": False,
            "gc": False,
        },
        batch_size=1,
        batch_length=1,
        train_ratio=1,
        from_checkpoint="",
        from_checkpoint_regex=".*",
        logdir=str(tmp_path),
        consec_report=1,
        report_batches=1,
        episode_timeout=10,
    )
    args.update(overrides)
    return elements.Config(**args)


def run_combined(tmp_path, events, failure=None, prefetch=False, **overrides):
    """Run combined.

    Args:
        tmp_path: Tmp path value.
        events: Events value.
        failure: Failure value.
        prefetch: Prefetch value.
        overrides: Overrides value.
    """
    args = make_args(tmp_path)
    if overrides:
        args = args.update(overrides)
    portal.reset()
    portal.setup(ipv6=False)
    embodied.run.parallel.combined(
        functools.partial(Agent, events, prefetch),
        functools.partial(Replay, events),
        functools.partial(Replay, events),
        functools.partial(make_env, events, failure),
        functools.partial(make_env, events, failure),
        make_stream,
        functools.partial(Logger, events),
        args,
        run_done_error=RunDone,
    )


def test_finite_run_flushes_and_exits_cleanly(tmp_path):
    """Verify finite run flushes and exits cleanly.

    Args:
        tmp_path: Tmp path value.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        run_combined(tmp_path, events)
        recorded = list(events)

    assert (
        recorded.count("policy") == 2
    ), "The actor must serve exactly the initial and terminal observations"
    assert (
        "reset:2" in recorded
    ), "The environment must attempt the reset that signals run exhaustion"
    assert (
        "reset:3" not in recorded
    ), "Graceful shutdown must not fabricate another episode"
    assert (
        "replay:True" in recorded
    ), "The terminal transition must reach replay before shutdown"
    assert (
        "logger_submission" in recorded
    ), "The terminal transition must reach the logger before shutdown"
    last_checkpoint = len(recorded) - 1 - recorded[::-1].index("checkpoint")
    assert (
        recorded.index("replay:True") < last_checkpoint
    ), "A final agent checkpoint must follow the terminal transition"
    assert last_checkpoint < recorded.index(
        "logger_closed"
    ), "The final checkpoint must be persisted before the logger shuts down"
    assert "env_closed" in recorded, 'Expected "env closed" to be present in recorded.'


def test_unrelated_environment_error_still_crashes_the_worker(tmp_path):
    """Verify unrelated environment error still crashes the worker.

    Args:
        tmp_path: Tmp path value.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        with pytest.raises(RuntimeError, match=r"parallel_env.*crashed"):
            run_combined(tmp_path, events, failure=ValueError)


def test_finite_run_waits_for_every_environment(tmp_path):
    """Verify finite run waits for every environment.

    Args:
        tmp_path: Tmp path value.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        run_combined(tmp_path, events, envs=2)
        recorded = list(events)

    assert (
        recorded.count("reset:2") == 2
    ), "Both environments must reach the reset that reports exhaustion"
    assert (
        recorded.count("env_closed") == 2
    ), "The actor must not close its server before every environment is done"


def test_agent_with_prefetching_streams_exits_cleanly(tmp_path):
    """Verify agent with prefetching streams exits cleanly.

    Args:
        tmp_path: Tmp path value.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        # The agent runs in its own process, so a prefetch thread crashing at
        # shutdown would surface as a crashed parallel_agent worker.
        run_combined(tmp_path, events, prefetch=True)
        recorded = list(events)
    assert "train" in recorded, "The learner must have trained on the stream"
    assert (
        "replay:True" in recorded
    ), 'Expected "replay:True" to be present in recorded.'


def test_crash_with_in_process_agent_raises_instead_of_hanging(tmp_path):
    """Verify crash with in process agent raises instead of hanging.

    Args:
        tmp_path: Tmp path value.
    """
    import threading

    outcome = []

    def run():
        # Portal warns about plain threads; this one only isolates a potential
        # hang from the test process.
        """Run state."""
        warnings.simplefilter("ignore", UserWarning)
        try:
            with multiprocessing.Manager() as manager:
                run_combined(
                    tmp_path, manager.list(), failure=ValueError, agent_process=False
                )
        except BaseException as e:
            outcome.append(e)
        else:
            outcome.append(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), (
        "A worker crash must propagate; joining the un-killable agent thread "
        "would hang here"
    )
    assert isinstance(outcome[0], RuntimeError), outcome
    assert "crashed" in str(
        outcome[0]
    ), 'Expected "crashed" to be present in str(outcome[0]).'


def test_prefetch_treats_source_exhaustion_as_normal_completion():
    """Verify prefetch treats source exhaustion as normal completion."""
    stream = embodied.streams.Prefetch(iter(()))
    with pytest.raises(StopIteration):
        next(iter(stream))
    stream.worker.join()
    assert (
        stream.worker.exitcode == 0
    ), "Source exhaustion must let the prefetch worker exit successfully"


def test_episode_checkpoints_are_opt_in(tmp_path):
    """Verify episode checkpoints are optimizer in.

    Args:
        tmp_path: Tmp path value.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        run_combined(tmp_path / "off", events)
        assert not (
            tmp_path / "off" / "ckpt" / "agent_best"
        ).exists(), "No best checkpoint is written unless save_on_episode is set"
        run_combined(tmp_path / "on", events, save_on_episode=True)
    assert (
        tmp_path / "on" / "ckpt" / "agent"
    ).exists(), 'Expected (tmp path / "on" / "ckpt" / "agent") exists() to be initialized or truthy.'
    assert (
        tmp_path / "on" / "ckpt" / "agent_best"
    ).exists(), "The first completed episode is the best one so far"


def test_actor_requests_latest_every_episode_and_best_on_improvement(monkeypatch):
    """Verify actor requests latest every episode and best on improvement.

    Args:
        monkeypatch: Monkeypatch value.

    Returns:
        Result of the operation.
    """
    import threading
    import types

    save_events = {"latest": threading.Event(), "best": threading.Event()}
    requests = []

    class Agent:
        """Represent agent."""

        def init_policy(self, batch_size):
            """Handle init policy.

            Args:
                batch_size: Batch size value.

            Returns:
                Result of the operation.
            """
            return {"state": np.zeros((batch_size, 1), np.float32)}

        def policy(self, carry, obs, mode="train"):
            """Handle policy.

            Args:
                carry: Carry value.
                obs: Observation value.
                mode: Mode value.

            Returns:
                Result of the operation.
            """
            return carry, {"action": np.zeros((len(obs["reward"]), 1))}, {}

    def batch(reward, first=False, last=False):
        """Handle batch.

        Args:
            reward: Reward value.
            first: First value.
            last: Last value.

        Returns:
            Result of the operation.
        """
        return {
            "envid": np.array([0]),
            "is_eval": np.array([False]),
            "reward": np.array([reward], np.float32),
            "is_first": np.array([first]),
            "is_last": np.array([last]),
            "is_terminal": np.array([last]),
        }

    # Three one-step episodes with mean rewards 1.0, 0.5, and 2.0.
    script = [
        batch(0.0, first=True),
        batch(1.0, last=True),
        batch(0.0, first=True),
        batch(0.5, last=True),
        batch(0.0, first=True),
        batch(2.0, last=True),
    ]

    def make_server(*args, **kwargs):
        """Create server.

        Args:
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result of the operation.
        """
        bound = {}

        def bind(name, workfn, postfn, *args):
            """Handle bind.

            Args:
                name: Name value.
                workfn: Workfn value.
                postfn: Postfn value.
                args: Positional arguments forwarded to the wrapped callable.
            """
            bound["workfn"] = workfn

        def start(**kwargs):
            """Start state.

            Args:
                kwargs: Keyword arguments forwarded to the wrapped callable.
            """
            for obs in script:
                last = bool(obs["is_last"][0])
                bound["workfn"](dict(obs))
                if last:
                    requests.append(
                        (save_events["latest"].is_set(), save_events["best"].is_set())
                    )
                    save_events["latest"].clear()
                    save_events["best"].clear()

        return types.SimpleNamespace(
            bind=bind, start=start, close=lambda: None, stats=lambda: {}
        )

    monkeypatch.setattr(embodied.run.parallel.portal, "BatchServer", make_server)
    monkeypatch.setattr(
        embodied.run.parallel.portal,
        "Client",
        lambda *a, **k: types.SimpleNamespace(close=lambda: None, stats=lambda: {}),
    )
    embodied.run.parallel.parallel_actor(
        Agent(),
        types.SimpleNamespace(wait=lambda: None),
        types.SimpleNamespace(
            actor_batch=1,
            log_every=0,
            actor_threads=1,
            logger_addr="logger",
            replay_addr="replay",
            actor_addr="actor",
        ),
        save_events=save_events,
    )
    assert requests == [
        (True, True),
        (True, False),
        (True, True),
    ], "latest is requested at every episode end, best only on improvement"
