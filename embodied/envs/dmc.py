"""Provide dmc functionality."""

from __future__ import annotations
from typing import Any


import functools
import os

import elements
import embodied
import numpy as np
from dm_control import manipulation  # pyright: ignore[reportMissingImports]
from dm_control import suite  # pyright: ignore[reportMissingImports]
from dm_control.locomotion.examples import basic_rodent_2020  # pyright: ignore[reportMissingImports]  # fmt: skip

from . import from_dm


class DMC(embodied.Env):
    """Represent dmc."""

    DEFAULT_CAMERAS = dict(
        quadruped=2,
        rodent=4,
    )

    def __init__(
        self,
        env: Any,
        repeat: int = 1,
        size: tuple[Any, ...] = (64, 64),
        proprio: bool = True,
        image: bool = True,
        camera: Any = -1,
    ) -> None:
        """Initialize the dmc.

        Args:
            env: Environment value.
            repeat: Repeat value.
            size: Requested number of elements.
            proprio: Proprio value.
            image: Image value.
            camera: Camera value.
        """
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if isinstance(env, str):
            domain, task = env.split("_", 1)
            if camera == -1:
                camera = self.DEFAULT_CAMERAS.get(domain, 0)
            if domain == "cup":  # Only domain with multiple words.
                domain = "ball_in_cup"
            if domain == "manip":
                env = manipulation.load(task + "_vision")
            elif domain == "rodent":
                # camera 0: topdown map
                # camera 2: shoulder
                # camera 4: topdown tracking
                # camera 5: eyes
                env = getattr(basic_rodent_2020, task)()
            else:
                env = suite.load(domain, task)
        self._dmenv = env
        self._env = from_dm.FromDM(self._dmenv)
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
        self._size = size
        self._proprio = proprio
        self._image = image
        self._camera = camera

    @functools.cached_property
    def obs_space(self) -> Any:
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        basic = ("is_first", "is_last", "is_terminal", "reward")
        spaces = self._env.obs_space.copy()
        if not self._proprio:
            spaces = {k: spaces[k] for k in basic}
        key = "image" if self._image else "log/image"
        spaces[key] = elements.Space(np.uint8, self._size + (3,))
        return spaces

    @functools.cached_property
    def act_space(self) -> Any:
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return self._env.act_space

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        basic = ("is_first", "is_last", "is_terminal", "reward")
        if not self._proprio:
            obs = {k: obs[k] for k in basic}
        key = "image" if self._image else "log/image"
        obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
        for key, space in self.obs_space.items():
            if np.issubdtype(space.dtype, np.floating):
                assert np.isfinite(obs[key]).all(), (key, obs[key])
        return obs
