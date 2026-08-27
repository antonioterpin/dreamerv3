"""Provide test driver functionality."""

from __future__ import annotations
from typing import Any


from functools import partial as bind

import embodied
import numpy as np


class TestDriver:
    """Represent test driver."""

    def test_episode_length(self) -> None:
        """Verify episode length."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=1)
        assert len(seq) == 11, "Expected number of seq to equal 11."

    def test_first_step(self) -> None:
        """Verify first step."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=2)
        for index in [0, 11]:
            assert (
                seq[index]["is_first"].item() is True
            ), "Expected seq[index] is first item() to be True."
            assert (
                seq[index]["is_last"].item() is False
            ), "Expected seq[index] is last item() to be False."
        for index in [1, 10, 12]:
            assert (
                seq[index]["is_first"].item() is False
            ), "Expected seq[index] is first item() to be False."

    def test_last_step(self) -> None:
        """Verify last step."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=2)
        for index in [10, 21]:
            assert (
                seq[index]["is_last"].item() is True
            ), "Expected seq[index] is last item() to be True."
            assert (
                seq[index]["is_first"].item() is False
            ), "Expected seq[index] is first item() to be False."
        for index in [0, 1, 9, 11, 20]:
            assert (
                seq[index]["is_last"].item() is False
            ), "Expected seq[index] is last item() to be False."

    def test_env_reset(self) -> None:
        """Verify environment reset."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=5)])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        action = {"act_disc": np.ones(1, int), "act_cont": np.zeros((1, 6), float)}
        policy = lambda carry, obs: (carry, action, {})
        driver(policy, episodes=2)
        assert len(seq) == 12, "Expected number of seq to equal 12."
        seq = {k: np.array([seq[i][k] for i in range(len(seq))]) for k in seq[0]}
        assert (
            seq["is_first"] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        ).all(), "Expected seq is first to equal [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]."
        assert (
            seq["is_last"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        ).all(), "Expected seq is last to equal [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]."
        assert (
            seq["reset"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        ).all(), "Expected seq reset to equal [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]."
        assert (
            seq["act_disc"] == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
        ).all(), "Expected seq act disc to equal [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]."

    def test_agent_inputs(self) -> None:
        """Verify agent inputs.

        Returns:
            Result of the operation.
        """
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        inputs = []
        states = []

        def policy(carry: Any, obs: Any, mode: str = "train") -> tuple[Any, ...]:
            """Handle policy.

            Args:
                carry: Carry value.
                obs: Observation value.
                mode: Mode value.

            Returns:
                Result of the operation.
            """
            inputs.append(obs)
            states.append(carry)
            _, act, _ = agent.policy(carry, obs, mode)
            return "carry", act, {}

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(policy, episodes=2)
        assert len(seq) == 22, "Expected number of seq to equal 22."
        assert states == (
            [()] + ["carry"] * 21
        ), 'Expected states to equal [()] + ["carry"] * 21.'
        for index in [0, 11]:
            assert (
                inputs[index]["is_first"].item() is True
            ), "Expected inputs[index] is first item() to be True."
        for index in [1, 10, 12, 21]:
            assert (
                inputs[index]["is_first"].item() is False
            ), "Expected inputs[index] is first item() to be False."
        for index in [10, 21]:
            assert (
                inputs[index]["is_last"].item() is True
            ), "Expected inputs[index] is last item() to be True."
        for index in [0, 1, 9, 11, 20]:
            assert (
                inputs[index]["is_last"].item() is False
            ), "Expected inputs[index] is last item() to be False."

    def test_unexpected_reset(self) -> None:
        """Verify unexpected reset.

        Returns:
            Result of the operation.
        """

        class UnexpectedReset(embodied.Wrapper):
            """Send is_first without preceeding is_last."""

            def __init__(self, env: Any, when: Any) -> None:
                """Initialize the unexpected reset.

                Args:
                    env: Environment value.
                    when: When value.
                """
                super().__init__(env)
                self._when = when
                self._step = 0

            def step(self, action: Any) -> Any:
                """Advance state.

                Args:
                    action: Action value.

                Returns:
                    Result of the operation.
                """
                if self._step == self._when:
                    action = action.copy()
                    action["reset"] = np.ones_like(action["reset"])
                self._step += 1
                return self.env.step(action)

        env = self._make_env(length=4)
        env = UnexpectedReset(env, when=3)
        agent = self._make_agent()
        driver = embodied.Driver([lambda: env])
        driver.reset(agent.init_policy)
        steps = []
        driver.on_step(lambda tran, _: steps.append(tran))
        driver(agent.policy, episodes=1)
        assert len(steps) == 8, "Expected number of steps to equal 8."
        steps = {k: np.array([x[k] for x in steps]) for k in steps[0]}
        assert (
            steps["reset"] == [0, 0, 0, 0, 0, 0, 0, 1]
        ).all(), "Expected steps reset to equal [0, 0, 0, 0, 0, 0, 0, 1]."
        assert (
            steps["is_first"] == [1, 0, 0, 1, 0, 0, 0, 0]
        ).all(), "Expected steps is first to equal [1, 0, 0, 1, 0, 0, 0, 0]."
        assert (
            steps["is_last"] == [0, 0, 0, 0, 0, 0, 0, 1]
        ).all(), "Expected steps is last to equal [0, 0, 0, 0, 0, 0, 0, 1]."

    def _make_env(self, length: int = 10) -> Any:
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)

    def _make_agent(self) -> Any:
        env = self._make_env()
        agent = embodied.RandomAgent(env.obs_space, env.act_space)
        env.close()
        return agent
