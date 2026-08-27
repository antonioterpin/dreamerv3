"""Provide procgen functionality."""

import elements
import embodied
import numpy as np
import procgen  # noqa

from PIL import Image


class ProcGen(embodied.Env):
    """Represent proc gen."""

    def __init__(self, task, size=(64, 64), resize="pillow", **kwargs):
        """Initialize the proc gen.

        Args:
            task: Task value.
            size: Requested number of elements.
            resize: Resize value.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        assert resize in ("opencv", "pillow"), resize
        from . import from_gym

        self.size = size
        self.resize = resize
        if self.size == (64, 64):
            self.source = "step"
        else:
            self.source = "info"
        if self.source == "info":
            kwargs["render_mode"] = "rgb_array"
        try:
            self.env = from_gym.FromGym(f"procgen:procgen-{task}-v0", **kwargs)
        except Exception:
            self.env = from_gym.FromGym(f"procgen-{task}-v0", **kwargs)
        if self.source == "info":
            self.inner = self.env
            while not hasattr(self.inner, "get_info"):
                self.inner = self.inner.env

    @property
    def obs_space(self):
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        spaces = self.env.obs_space.copy()
        spaces["image"] = elements.Space(np.uint8, (*self.size, 3))
        return spaces

    @property
    def act_space(self):
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return self.env.act_space

    def step(self, action):
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        obs = self.env.step(action)
        if self.source == "step":
            pass
        elif self.source == "info":
            info = self.inner.get_info()
            assert len(info) == 1
            obs["image"] = self._resize(info[0]["rgb"], self.size, self.resize)
        elif self.source == "render":
            obs["image"] = self._resize(
                self.env.env.render(mode="rgb_array"), self.size, self.resize
            )
        else:
            raise NotImplementedError(self.source)
        return obs

    def _resize(self, image, size, method):
        if method == "opencv":
            import cv2  # pyright: ignore[reportMissingImports]

            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return image
        elif method == "pillow":
            image = Image.fromarray(image)
            image = image.resize((size[1], size[0]), Image.BILINEAR)
            image = np.array(image)
            return image
        else:
            raise NotImplementedError(method)
