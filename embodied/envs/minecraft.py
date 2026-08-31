"""Provide minecraft functionality."""

from __future__ import annotations
from typing import Any


import importlib

import embodied


class Minecraft(embodied.Wrapper):
    """Represent minecraft."""

    def __init__(self, task: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the minecraft.

        Args:
            task: Task value.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        module, cls = {
            "wood": "minecraft_flat:Wood",
            "climb": "minecraft_flat:Climb",
            "diamond": "minecraft_flat:Diamond",
        }[task].split(":")
        module = importlib.import_module(f".{module}", __package__)
        cls = getattr(module, cls)
        env = cls(*args, **kwargs)
        super().__init__(env)
