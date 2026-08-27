"""Provide minecraft functionality."""

import importlib

import embodied


class Minecraft(embodied.Wrapper):
    """Represent minecraft."""

    def __init__(self, task, *args, **kwargs):
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
