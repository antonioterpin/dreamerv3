"""Provide bsuite functionality."""

from __future__ import annotations
from typing import Any


import time

import embodied
import numpy as np


class BSuite(embodied.Env):
    """Represent bsuite."""

    def __init__(self, task: Any) -> None:
        """Initialize the bsuite.

        Args:
            task: Task value.
        """
        print(
            "Warning: BSuite result logging is stateful and therefore training "
            + "runs cannot be interrupted or restarted."
        )
        np.int = int  # Patch deprecated Numpy alias used inside BSuite.
        from . import from_dm

        if "/" not in task:
            task = f"{task}/0"
        import bsuite

        env = bsuite.from_checkpoint_id(task)
        self.num_episodes = 0
        self.max_episodes = env.bsuite_num_episodes
        self.exit_after = None
        env = from_dm.FromDM(env)
        env = embodied.wrappers.ForceDtypes(env)
        env = embodied.wrappers.FlattenTwoDimObs(env)
        self.env = env

    @property
    def obs_space(self) -> Any:
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        return self.env.obs_space

    @property
    def act_space(self) -> Any:
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return self.env.act_space

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.

        Raises:
            RuntimeError: If the operation cannot be completed.
        """
        obs = self.env.step(action)
        if obs["is_last"]:
            self.num_episodes += 1
        if self.num_episodes >= self.max_episodes:
            # After reaching the target number of episodes, continue running for 10
            # minutes to make sure logs are flushed and then raise an exception to
            # terminate the program.
            if not self.exit_after:
                self.exit_after = time.time() + 600
            if time.time() > self.exit_after:
                if self.xm:
                    wu = self.xm.get_current_work_unit()
                    wu.stop(mark_as_completed=True, message="BSuite run complete")
                else:
                    raise RuntimeError("BSuite run complete")
        return obs
