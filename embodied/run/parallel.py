"""Provide parallel functionality."""

from __future__ import annotations
from collections.abc import Iterator
from typing import Any


import collections
import threading
import time
from functools import partial as bind

import cloudpickle
import elements
import embodied
import numpy as np
import portal

prefix = lambda d, p: {f"{p}/{k}": v for k, v in d.items()}


def _run_workers(workers: Any) -> None:
    """Run workers and, once all of them finished, join them.

    Graceful finite-run completion lets workers terminate normally, so the
    parent waits for their process/thread teardown instead of leaving it to
    interpreter shutdown. portal.run returns only when every worker exited
    cleanly; on a crash it kills the others and raises, and nothing is joined
    (a thread blocked in a wait cannot be killed and would hang the join).

    Args:
      workers: Portal processes or threads to run and subsequently join.
    """
    portal.run(workers)
    for worker in workers:
        if worker.started:
            worker.join()


def combined(
    make_agent: Any,
    make_replay_train: Any,
    make_replay_eval: Any,
    make_env_train: Any,
    make_env_eval: Any,
    make_stream: Any,
    make_logger: Any,
    args: Any,
    run_done_error: Any | None = None,
) -> None:
    """Run the complete parallel Dreamer topology.

    Args:
      make_agent: Factory for the Dreamer agent.
      make_replay_train: Factory for training replay storage.
      make_replay_eval: Factory for evaluation replay storage.
      make_env_train: Factory for training environments.
      make_env_eval: Factory for evaluation environments.
      make_stream: Factory for replay sample streams.
      make_logger: Factory for the experiment logger.
      args: Parallel-run configuration.
      run_done_error: Optional typed exception that signals finite environment
        exhaustion; every environment raises it from env.step on its own final
        reset. Without it, the original indefinitely running behavior is
        preserved.
    """
    if args.actor_batch <= 0:
        args = args.update(actor_batch=max(1, args.envs // 2))
    assert args.actor_batch <= args.envs, (args.actor_batch, args.envs)
    for key in ("actor_addr", "replay_addr", "logger_addr"):
        if "{auto}" in args[key]:
            args = args.update({key: args[key].format(auto=portal.free_port())})

    make_agent = cloudpickle.dumps(make_agent)
    make_replay_train = cloudpickle.dumps(make_replay_train)
    make_replay_eval = cloudpickle.dumps(make_replay_eval)
    make_env_train = cloudpickle.dumps(make_env_train)
    make_env_eval = cloudpickle.dumps(make_env_eval)
    make_stream = cloudpickle.dumps(make_stream)
    make_logger = cloudpickle.dumps(make_logger)

    lifecycle = None
    if run_done_error is not None:
        # These process-safe primitives encode the finite-run handoff. Every
        # environment releases 'requested' once on its final reset, the actor
        # confirms final replay/logger RPCs drained after all of them did, and the
        # learner confirms the final checkpoint save returned.
        lifecycle = {
            "requested": portal.context.mp.Semaphore(0),
            "env_count": args.envs + max(0, args.eval_envs),
            "actor_flushed": portal.context.mp.Event(),
            "checkpoint_persisted": portal.context.mp.Event(),
        }

    workers = []
    if args.agent_process:
        workers.append(portal.Process(parallel_agent, make_agent, args, lifecycle))
    else:
        workers.append(portal.Thread(parallel_agent, make_agent, args, lifecycle))
    workers.append(portal.Process(parallel_logger, make_logger, args, lifecycle))

    if not args.remote_envs:
        for i in range(args.envs):
            workers.append(
                portal.Process(
                    parallel_env,
                    make_env_train,
                    i,
                    args,
                    False,
                    lifecycle,
                    run_done_error,
                )
            )
        for i in range(args.envs, args.envs + args.eval_envs):
            workers.append(
                portal.Process(
                    parallel_env,
                    make_env_eval,
                    i,
                    args,
                    True,
                    lifecycle,
                    run_done_error,
                )
            )

    if not args.remote_replay:
        workers.append(
            portal.Process(
                parallel_replay,
                make_replay_train,
                make_replay_eval,
                make_stream,
                args,
                lifecycle,
            )
        )

    _run_workers(workers)


def parallel_agent(make_agent: Any, args: Any, lifecycle: Any | None = None) -> None:
    """Run the agent in parallel with the actor and learner.

    Args:
      make_agent: A callable that returns a new agent instance.
      args: The arguments for the agent and its components.
      lifecycle: Optional process-safe events for finite-run shutdown.
    """
    if isinstance(make_agent, bytes):
        make_agent = cloudpickle.loads(make_agent)
    agent = make_agent()
    barrier = threading.Barrier(2)
    save_events = None
    if args.save_on_episode:
        # The actor observes episode boundaries, the learner owns the checkpoints.
        save_events = {"latest": threading.Event(), "best": threading.Event()}
    workers = []
    workers.append(
        portal.Thread(parallel_actor, agent, barrier, args, lifecycle, save_events)
    )
    workers.append(
        portal.Thread(parallel_learner, agent, barrier, args, lifecycle, save_events)
    )
    _run_workers(workers)


@elements.timer.section("actor")
def parallel_actor(
    agent: Any,
    barrier: Any,
    args: Any,
    lifecycle: Any | None = None,
    save_events: Any | None = None,
) -> None:
    """Run the actor in parallel with the learner and agent.

    Args:
      agent: The agent instance to use for acting.
      barrier: Two-party barrier shared with the learner; crossed twice, after
        checkpoint restoration and after policy warmup.
      args: Actor and parallel-run configuration.
      lifecycle: Optional process-safe events for finite-run shutdown.
      save_events: Optional thread events ('latest', 'best') the actor sets at
        episode boundaries to request checkpoints from the learner.
    """
    islist = lambda x: isinstance(x, list)
    ep_sums = collections.defaultdict(float)
    ep_steps = collections.defaultdict(int)
    best_score = [float("-inf")]
    initial = agent.init_policy(args.actor_batch)
    initial = elements.tree.map(lambda x: x[0], initial, isleaf=islist)
    carries = collections.defaultdict(lambda: initial)
    # First rendezvous: the learner restored the checkpoint, so warmup sees the
    # restored parameters and action counter.
    barrier.wait()
    try:
        precompile = getattr(agent, "precompile_policy", None)
        if precompile is not None:
            precompile(args.actor_batch)
    except Exception:
        barrier.abort()
        raise
    # Second rendezvous: the learner neither trains nor stages a policy sync
    # until warmup completed, and the actor server starts only after it, so an
    # environment's actor.connect() means the policy is immediately executable.
    barrier.wait()
    fps = elements.FPS()

    should_log = embodied.LocalClock(args.log_every)
    backlog = 8 * args.actor_threads
    logger = portal.Client(args.logger_addr, "ActorLogger", maxinflight=backlog)
    replay = portal.Client(args.replay_addr, "ActorReplay", maxinflight=backlog)

    @elements.timer.section("workfn")
    def workfn(obs: Any) -> tuple[Any, ...]:
        """Handle workfn.

        Args:
            obs: Observation value.

        Returns:
            Result of the operation.
        """
        envid = obs.pop("envid")
        assert envid.shape == (
            args.actor_batch,
        ), "Expected envid shape to equal (args.actor_batch,)."
        is_eval = obs.pop("is_eval")
        fps.step(obs["is_first"].size)
        with elements.timer.section("get_states"):
            carry = [carries[a] for a in envid]
            carry = elements.tree.map(lambda *xs: list(xs), *carry)
        logs = {k: v for k, v in obs.items() if k.startswith("log/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
        carry, acts, outs = agent.policy(carry, obs)
        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))
        with elements.timer.section("put_states"):
            for i, a in enumerate(envid):
                carries[a] = elements.tree.map(lambda x: x[i], carry, isleaf=islist)
        trans = {"envid": envid, "is_eval": is_eval, **obs, **acts, **outs, **logs}
        [x.setflags(write=False) for x in trans.values()]
        acts = {**acts, "reset": obs["is_last"].copy()}
        if save_events is not None:
            # Track each environment's mean reward per step and request a 'latest'
            # checkpoint at every episode end, plus a 'best' one when the mean
            # improves on every episode seen so far.
            for i, a in enumerate(envid):
                key = int(a)
                if obs["is_first"][i]:
                    ep_sums[key] = 0.0
                    ep_steps[key] = 0
                ep_sums[key] += float(obs["reward"][i])
                ep_steps[key] += 1
                if obs["is_last"][i]:
                    score = ep_sums[key] / max(1, ep_steps[key])
                    save_events["latest"].set()
                    if score > best_score[0]:
                        best_score[0] = score
                        save_events["best"].set()
        return acts, trans

    @elements.timer.section("donefn")
    def postfn(trans: Any) -> None:
        """Submit a completed transition to replay and logging services."""
        logs = {k: v for k, v in trans.items() if k.startswith("log/")}
        trans = {k: v for k, v in trans.items() if not k.startswith("log/")}
        replay_future = replay.add_batch(trans)
        logger_future = logger.tran({**trans, **logs})
        if lifecycle is not None and trans["is_last"].any():
            # The environment may discover run exhaustion on its next reset. Resolve
            # both RPCs now so actor_flushed cannot overtake the terminal transition.
            replay_future.result()
            logger_future.result()
        if should_log():
            stats = {}
            stats["fps/policy"] = fps.result()
            stats["parallel/ep_states"] = len(carries)
            stats.update(prefix(server.stats(), "server/actor"))
            stats.update(prefix(logger.stats(), "client/actor_logger"))
            stats.update(prefix(replay.stats(), "client/actor_replay"))
            logger.add(stats)

    server = portal.BatchServer(args.actor_addr, name="Actor")
    server.bind("act", workfn, postfn, args.actor_batch, args.actor_threads)
    if lifecycle is None:
        server.start()
        return
    server.start(block=False)
    # Wait for every environment, so none is cut off mid-episode by the close.
    for _ in range(lifecycle["env_count"]):
        lifecycle["requested"].acquire()
    # BatchServer.close() drains accepted work before returning. Only advertise
    # actor completion after its downstream replay and logger clients also close.
    server.close()
    replay.close()
    logger.close()
    lifecycle["actor_flushed"].set()


@elements.timer.section("learner")
def parallel_learner(
    agent: Any,
    barrier: Any,
    args: Any,
    lifecycle: Any | None = None,
    save_events: Any | None = None,
) -> None:
    """Train the agent and acknowledge persistence of the final checkpoint.

    Args:
      agent: Dreamer agent shared with the actor thread.
      barrier: Two-party barrier shared with the actor; crossed twice, after
        checkpoint restoration and after the actor's policy warmup.
      args: Learner and parallel-run configuration.
      lifecycle: Optional process-safe events for finite-run shutdown.
      save_events: Optional thread events set by the actor at episode
        boundaries; 'latest' saves ckpt/agent, 'best' saves ckpt/agent_best.
    """
    agg = elements.Agg()
    usage = elements.Usage(**args.usage)
    should_log = embodied.GlobalClock(args.log_every)
    should_report = embodied.GlobalClock(args.report_every)
    should_save = embodied.GlobalClock(args.save_every)
    fps = elements.FPS()
    batch_steps = args.batch_size * args.batch_length

    cp = elements.Checkpoint(elements.Path(args.logdir) / "ckpt/agent")
    cp.agent = agent
    if save_events is not None:
        best_cp = elements.Checkpoint(elements.Path(args.logdir) / "ckpt/agent_best")
        best_cp.agent = agent
    if args.from_checkpoint:
        elements.checkpoint.load(
            args.from_checkpoint,
            dict(agent=bind(agent.load, regex=args.from_checkpoint_regex)),
        )
    cp.load_or_save()
    logger = portal.Client(args.logger_addr, "LearnerLogger", maxinflight=1)
    updater = portal.Client(args.replay_addr, "LearnerReplayUpdater", maxinflight=8)
    # Release the actor once checkpoint restoration installed the parameters
    # and counters its policy warmup exercises, then hold training until the
    # warmup is complete (see parallel_actor).
    barrier.wait()
    barrier.wait()

    replays = {}
    received = collections.defaultdict(int)

    def parallel_stream(source: Any, prefetch: int = 2) -> Iterator[Any]:
        """Handle parallel stream.

        Args:
            source: Source value.
            prefetch: Prefetch value.

        Raises:
            Disconnected: If replay disconnects before the actor finishes.
        """
        replay = portal.Client(args.replay_addr, f"LearnerReplay{source.title()}")
        replays[source] = replay
        call = getattr(replay, f"sample_batch_{source}")
        futures = collections.deque([call() for _ in range(prefetch)])
        while True:
            try:
                futures.append(call())
                with elements.timer.section(f"stream_{source}_response"):
                    data = futures.popleft().result()
            except portal.Disconnected:
                if lifecycle is not None and lifecycle["actor_flushed"].is_set():
                    # The learner closed this client during finite-run shutdown; end
                    # the stream so its prefetch thread exits instead of crashing.
                    return
                raise
            if lifecycle is not None and data.get("__run_done__", False):
                return
            received[source] += 1
            yield data

    def evaluate(stream: Any) -> Any:
        """Evaluate state.

        Args:
            stream: Stream value.

        Returns:
            Result of the operation.
        """
        carry = agent.init_report(args.batch_size)
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
            batch = next(stream)
            carry, metrics = agent.report(carry, batch)
            agg.add(metrics)
        return agg.result()

    stream_train = iter(
        agent.stream(embodied.streams.Stateless(parallel_stream("train")))
    )
    stream_report = iter(
        agent.stream(embodied.streams.Stateless(parallel_stream("report")))
    )
    stream_eval = (
        iter(agent.stream(embodied.streams.Stateless(parallel_stream("eval"))))
        if args.eval_envs > 0
        else None
    )
    carry = agent.init_train(args.batch_size)

    while True:

        with elements.timer.section("batch_next"):
            try:
                batch = next(stream_train)
            except StopIteration:
                assert lifecycle is not None, "Expected lifecycle not to be None."
                assert lifecycle[
                    "actor_flushed"
                ].is_set(), "Expected lifecycle actor flushed is set() to be initialized or truthy."
                break
        with elements.timer.section("train_step"):
            carry, outs, mets = agent.train(carry, batch)
        if "replay" in outs:
            with elements.timer.section("replay_update"):
                updater.update(outs["replay"])

        time.sleep(0.0001)
        agg.add(mets)
        fps.step(batch_steps)

        if should_report(skip=not received["report"]):
            print("Report started...")
            with elements.timer.section("report"):
                logger.add(prefix(evaluate(stream_report), "report"))
                if stream_eval is not None and received["eval"]:
                    logger.add(prefix(evaluate(stream_eval), "eval"))
            print("Report finished!")

        if should_log():
            with elements.timer.section("metrics"):
                stats = {}
                stats["fps/train"] = fps.result()
                stats["timer/agent"] = elements.timer.stats()["summary"]
                stats.update(prefix(agg.result(), "train"))
                stats.update(prefix(usage.stats(), "usage/agent"))
                stats.update(prefix(logger.stats(), "client/learner_logger"))
                for source, client in replays.items():
                    stats.update(prefix(client.stats(), f"client/replay_{source}"))
            logger.add(stats)

        if save_events is not None:
            # Episode-boundary checkpoints requested by the actor, serviced between
            # train steps.
            if save_events["latest"].is_set():
                save_events["latest"].clear()
                cp.save()
            if save_events["best"].is_set():
                save_events["best"].clear()
                best_cp.save()
        if should_save():
            cp.save()

    if save_events is not None and save_events["best"].is_set():
        # The actor may have reported a best episode right before its final RPC.
        save_events["best"].clear()
        best_cp.save()
    # The run is complete: persist the final agent state before releasing the
    # replay and logger workers, which wait on checkpoint_persisted.
    cp.save()
    updater.close()
    logger.close()
    for replay in replays.values():
        replay.close()
    lifecycle["checkpoint_persisted"].set()


def parallel_replay(
    make_replay_train: Any,
    make_replay_eval: Any,
    make_stream: Any,
    args: Any,
    lifecycle: Any | None = None,
) -> None:
    """Serve replay operations until the learner persists its final checkpoint.

    Args:
      make_replay_train: Factory for training replay storage.
      make_replay_eval: Factory for evaluation replay storage.
      make_stream: Factory for replay sample streams.
      args: Replay and parallel-run configuration.
      lifecycle: Optional process-safe events for finite-run shutdown.
    """
    if isinstance(make_replay_train, bytes):
        make_replay_train = cloudpickle.loads(make_replay_train)
    if isinstance(make_replay_eval, bytes):
        make_replay_eval = cloudpickle.loads(make_replay_eval)
    if isinstance(make_stream, bytes):
        make_stream = cloudpickle.loads(make_stream)

    replay_train = make_replay_train()
    replay_eval = make_replay_eval()

    stream_train = iter(make_stream(replay_train, "train"))
    stream_report = iter(make_stream(replay_train, "report"))
    stream_eval = iter(make_stream(replay_eval, "eval"))

    should_log = embodied.LocalClock(args.log_every)
    logger = portal.Client(args.logger_addr, "ReplayLogger", maxinflight=1)
    usage = elements.Usage(**args.usage.update(nvsmi=False))
    limit_agg = elements.Agg()
    active = elements.Counter()

    limiter = embodied.limiters.SamplesPerInsert(
        args.train_ratio / args.batch_length,
        tolerance=4 * args.batch_size,
        minsize=args.batch_size * replay_train.length,
    )

    def add_batch(data: Any) -> dict[Any, Any]:
        """Add batch.

        Args:
            data: Data to process.

        Returns:
            Result of the operation.
        """
        active.increment()
        for i, envid in enumerate(data.pop("envid")):
            tran = {k: v[i] for k, v in data.items()}
            if tran.pop("is_eval", False):
                replay_eval.add(tran, envid)
                continue
            with elements.timer.section("replay_insert_wait"):
                dur = embodied.limiters.wait(
                    limiter.want_insert, "Replay insert waiting", limiter.__dict__
                )
                limit_agg.add("insert_wait_dur", dur, agg="sum")
                limit_agg.add("insert_wait_count", dur > 0, agg="sum")
                limit_agg.add("insert_wait_frac", dur > 0, agg="avg")
                limiter.insert()
                replay_train.add(tran, envid)
        return {}

    def sample_batch_train() -> Any:
        """Sample batch train.

        Returns:
            Result of the operation.
        """
        active.increment()
        with elements.timer.section("replay_sample_wait"):
            for _ in range(args.batch_size):
                dur = embodied.limiters.wait(
                    lambda: (
                        limiter.want_sample()
                        or (
                            lifecycle is not None
                            and lifecycle["actor_flushed"].is_set()
                        )
                    ),
                    "Replay sample waiting",
                    limiter.__dict__,
                )
                if lifecycle is not None and lifecycle["actor_flushed"].is_set():
                    # Unblock a learner waiting below replay's minimum fill after the
                    # finite actor has submitted its final transition.
                    return {"__run_done__": np.asarray(True)}
                limit_agg.add("sample_wait_dur", dur, agg="sum")
                limit_agg.add("sample_wait_count", dur > 0, agg="sum")
                limit_agg.add("sample_wait_frac", dur > 0, agg="avg")
                limiter.sample()
        return next(stream_train)

    def sample_batch_report() -> Any:
        """Sample batch report.

        Returns:
            Result of the operation.
        """
        active.increment()
        return next(stream_report)

    def sample_batch_eval() -> Any:
        """Sample batch eval.

        Returns:
            Result of the operation.
        """
        active.increment()
        return next(stream_eval)

    should_save = embodied.LocalClock(args.save_every)
    cp = elements.Checkpoint(elements.Path(args.logdir) / "ckpt/replay")
    cp.replay_train = replay_train
    cp.replay_eval = replay_eval
    cp.limiter = limiter
    cp.load_or_save()

    server = portal.Server(args.replay_addr, name="Replay")
    server.bind("add_batch", add_batch, workers=1)
    server.bind("sample_batch_train", sample_batch_train, workers=1)
    server.bind("sample_batch_report", sample_batch_report, workers=1)
    server.bind("sample_batch_eval", sample_batch_eval, workers=1)
    server.bind("update", lambda data: replay_train.update(data), workers=1)
    server.start(block=False)
    while lifecycle is None or not lifecycle["checkpoint_persisted"].is_set():
        if should_save() and active > 0:
            active.reset()
            cp.save()
        if should_log():
            stats = {}
            stats["timer/replay"] = elements.timer.stats()["summary"]
            stats.update(prefix(limit_agg.result(), "limiter"))
            stats.update(prefix(replay_train.stats(), "replay"))
            stats.update(prefix(replay_eval.stats(), "replay_eval"))
            stats.update(prefix(usage.stats(), "usage/replay"))
            stats.update(prefix(logger.stats(), "client/replay_logger"))
            stats.update(prefix(server.stats(), "server/replay"))
            logger.add(stats)
        time.sleep(1)
    # Persist replay only after the learner has acknowledged its agent checkpoint.
    cp.save()
    for replay in (replay_train, replay_eval):
        # Chunk writes are asynchronous; finish them before the process exits.
        workers = getattr(replay, "workers", None)
        if workers is not None:
            workers.shutdown(wait=True)
    server.close()
    logger.close()


@elements.timer.section("logger")
def parallel_logger(make_logger: Any, args: Any, lifecycle: Any | None = None) -> None:
    """Serve logging RPCs and flush output during finite-run shutdown.

    Args:
      make_logger: Factory for the experiment logger.
      args: Logger and parallel-run configuration.
      lifecycle: Optional process-safe events for finite-run shutdown.
    """
    if isinstance(make_logger, bytes):
        make_logger = cloudpickle.loads(make_logger)

    logger = make_logger()
    should_log = embodied.LocalClock(args.log_every)
    usage = elements.Usage(**args.usage.update(nvsmi=False))

    active = elements.Counter()
    should_save = embodied.LocalClock(args.save_every)
    cp = elements.Checkpoint(elements.Path(args.logdir) / "ckpt/logger")
    cp.step = logger.step
    cp.load_or_save()

    parallel = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    updated = collections.defaultdict(lambda: None)
    dones = collections.defaultdict(lambda: True)

    @elements.timer.section("addfn")
    def addfn(metrics: Any) -> None:
        """Handle addfn.

        Args:
            metrics: Metrics value.
        """
        active.increment()
        logger.add(metrics)

    @elements.timer.section("tranfn")
    def tranfn(trans: Any) -> None:
        """Handle tranfn.

        Args:
            trans: Trans value.
        """
        active.increment()
        now = time.time()
        envid = trans.pop("envid")
        logger.step.increment((~trans["is_eval"]).sum())
        parallel.add("ep_starts", trans["is_first"].sum(), agg="sum")
        parallel.add("ep_ends", trans["is_last"].sum(), agg="sum")

        for i, addr in enumerate(envid):
            tran = {k: v[i] for k, v in trans.items()}

            updated[addr] = now  # pyright: ignore[reportArgumentType]
            episode = episodes[addr]
            if tran["is_first"]:
                episode.reset()
                parallel.add("ep_abandoned", int(not dones[addr]), agg="sum")
            dones[addr] = tran["is_last"]

            episode.add("score", tran["reward"], agg="sum")
            episode.add("length", 1, agg="sum")
            episode.add("rewards", tran["reward"], agg="stack")

            first_addr = next(iter(episodes.keys()))
            for key, value in tran.items():
                if value.dtype == np.uint8 and value.ndim == 3:
                    if addr == first_addr:
                        episode.add(f"policy_{key}", value, agg="stack")
                elif key.startswith("log/"):
                    assert value.ndim == 0, (key, value.shape, value.dtype)
                    episode.add(key + "/avg", value, agg="avg")
                    episode.add(key + "/max", value, agg="max")
                    episode.add(key + "/sum", value, agg="sum")
            if tran["is_last"]:
                result = episode.result()
                logger.add(
                    {
                        "score": result.pop("score"),
                        "length": result.pop("length") - 1,
                    },
                    prefix="episode",
                )
                rew = result.pop("rewards")
                if len(rew) > 1:
                    result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
                epstats.add(result)

        for addr, last in list(updated.items()):
            if (
                now - last  # pyright: ignore[reportOperatorIssue]
                >= args.episode_timeout
            ):
                print("Dropping episode statistics due to timeout.")
                del episodes[addr]
                del updated[addr]

    server = portal.Server(args.logger_addr, "Logger")
    server.bind("add", addfn)
    server.bind("tran", tranfn)
    server.start(block=False)
    last_step = int(logger.step)
    while lifecycle is None or not lifecycle["checkpoint_persisted"].is_set():
        time.sleep(1)
        if should_log() and active > 0:
            active.reset()
            with elements.timer.section("metrics"):
                logger.add({"timer/logger": elements.timer.stats()["summary"]})
                logger.add(parallel.result(), prefix="parallel")
                logger.add(epstats.result(), prefix="epstats")
                logger.add(usage.stats(), prefix="usage/logger")
                logger.add(server.stats(), prefix="server/logger")
            if logger.step == last_step:
                continue
            logger.write()
            last_step = int(logger.step)
        if should_save():
            cp.save()
    # Closing the server first prevents new submissions while checkpoint and
    # output state are flushed.
    server.close()
    cp.save()
    logger.close()


@elements.timer.section("env")
def parallel_env(
    make_env: Any,
    envid: Any,
    args: Any,
    is_eval: bool = False,
    lifecycle: Any | None = None,
    run_done_error: Any | None = None,
) -> None:
    """Drive one environment, translating finite exhaustion into normal exit.

    Args:
      make_env: Factory accepting the numeric environment identifier.
      envid: Non-negative environment identifier.
      args: Environment and parallel-run configuration.
      is_eval: Whether transitions belong to evaluation replay.
      lifecycle: Optional process-safe events for finite-run shutdown.
      run_done_error: Optional exception type denoting authoritative finite-run
        exhaustion.

    Raises:
      Exception: Any environment exception other than ``run_done_error``.
    """
    if isinstance(make_env, bytes):
        make_env = cloudpickle.loads(make_env)
    assert envid >= 0, envid
    name = f"Env{envid:05}"
    print = lambda x: elements.print(f"[{name}] {x}", flush=True)

    should_log = embodied.LocalClock(args.log_every)
    fps = elements.FPS()
    if envid == 0:
        logger = portal.Client(args.logger_addr, f"{name}Logger", maxinflight=1)
        usage = elements.Usage(**args.usage.update(nvsmi=False))

    print("Make env")
    env = make_env(envid)
    actor = portal.Client(args.actor_addr, name, autoconn=False)
    actor.connect()

    done = True
    while True:

        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        scope_name = (
            "reset"
            if act["reset"]  # pyright: ignore[reportOptionalSubscript]
            else "step"
        )
        try:
            with elements.timer.section(scope_name):
                obs = env.step(act)
        except Exception as exc:
            if run_done_error is None or not isinstance(exc, run_done_error):
                raise
            assert (
                lifecycle is not None
            ), "Lifecycle must be provided for graceful shutdown."
            # The terminal transition was processed before this attempted reset.
            # Report exhaustion without fabricating a reset observation, then keep
            # the environment alive until the final checkpoint is durable.
            lifecycle["requested"].release()
            lifecycle["checkpoint_persisted"].wait()
            actor.close()
            if envid == 0:
                logger.close()
            env.close()
            return
        obs = {k: np.asarray(v, order="C") for k, v in obs.items()}
        obs["is_eval"] = is_eval  # pyright: ignore[reportArgumentType]
        score += obs["reward"]
        length += 1
        fps.step(1)
        done = obs["is_last"]
        if done and envid == 0:
            print(f"Episode of length {length} with score {score:.2f}")

        try:
            with elements.timer.section("request"):
                future = actor.act({"envid": envid, **obs})
            with elements.timer.section("response"):
                act = future.result()
        except portal.Disconnected:
            print("Env lost connection to agent")
            actor.connect()
            done = True

        if should_log() and envid == 0:
            stats = {}
            stats["fps/env"] = fps.result()
            stats["timer/env"] = elements.timer.stats()["summary"]
            stats.update(prefix(usage.stats(), "usage/env"))
            stats.update(prefix(logger.stats(), "client/env_logger"))
            stats.update(prefix(actor.stats(), "client/env_actor"))
            logger.add(stats)


def parallel_envs(make_env: Any, make_env_eval: Any, args: Any) -> None:
    """Handle parallel envs.

    Args:
        make_env: Make environment value.
        make_env_eval: Make environment eval value.
        args: Positional arguments forwarded to the wrapped callable.
    """
    workers = []
    for i in range(args.envs):
        workers.append(portal.Process(parallel_env, make_env, i, args))
    for i in range(args.envs, args.envs + args.eval_envs):
        workers.append(portal.Process(parallel_env, make_env_eval, i, args, True))
    _run_workers(workers)
