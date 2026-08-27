"""Provide test train functionality."""

from __future__ import annotations
from typing import Any


from functools import partial as bind

import elements
import embodied
import numpy as np

import utils


class TestTrain:
    """Represent test train."""

    def test_run_loop(self, tmpdir: Any) -> None:
        """Verify run loop.

        Args:
            tmpdir: Tmpdir value.
        """
        args = self._make_args(tmpdir)
        agent = self._make_agent()
        embodied.run.train(  # pyright: ignore[reportCallIssue]
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            self._make_logger,
            args,
        )
        stats = agent.stats()
        print("Stats:", stats)
        replay_steps = args.steps * args.train_ratio
        assert (
            stats["lifetime"] >= 1
        ), "Expected stats lifetime to be at least 1."  # Otherwise decrease log and ckpt interval.
        assert np.allclose(
            stats["env_steps"], args.steps, 100, 0.1
        ), "Expected np allclose(stats env steps, args steps, 100, 0 1) to be initialized or truthy."
        assert np.allclose(
            stats["replay_steps"], replay_steps, 100, 0.1
        ), "Expected np allclose(stats replay steps, replay steps, 100, 0 1) to be initialized or truthy."
        assert stats["reports"] >= 1, "Expected stats reports to be at least 1."
        assert stats["saves"] >= 2, "Expected stats saves to be at least 2."
        assert stats["loads"] == 0, "Expected stats loads to equal 0."
        args = args.update(steps=2 * args.steps)
        embodied.run.train(  # pyright: ignore[reportCallIssue]
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            self._make_logger,
            args,
        )
        stats = agent.stats()
        assert stats["loads"] == 1, "Expected stats loads to equal 1."
        assert np.allclose(
            stats["env_steps"], args.steps, 100, 0.1
        ), "Expected np allclose(stats env steps, args steps, 100, 0 1) to be initialized or truthy."

    def _make_agent(self) -> Any:
        env = self._make_env(0)
        agent = utils.TestAgent(env.obs_space, env.act_space)
        env.close()
        return agent

    def _make_env(self, index: Any) -> Any:
        from embodied.envs import dummy

        return dummy.Dummy("disc", size=(64, 64), length=100)

    def _make_replay(self, args: Any) -> Any:
        kwargs = {"length": args.batch_length, "capacity": 1e4}
        return embodied.replay.Replay(**kwargs)

    def _make_logger(self) -> Any:
        return elements.Logger(
            elements.Counter(),
            [
                elements.logger.TerminalOutput(),
            ],
        )

    def _make_args(self, logdir: Any) -> Any:
        return elements.Config(
            steps=1000,
            train_ratio=32.0,
            log_every=0.1,
            report_every=0.2,
            save_every=0.2,
            report_batches=1,
            from_checkpoint="",
            usage=dict(psutil=True),
            debug=False,
            logdir=str(logdir),
            envs=4,
            batch_size=8,
            batch_length=16,
            replay_context=0,
            report_length=8,
        )
