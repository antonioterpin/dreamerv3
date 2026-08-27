"""Provide init functionality."""

from __future__ import annotations

__version__ = "2.0.0"

try:
    import colored_traceback

    colored_traceback.add_hook(colors="terminal")
except ImportError:
    pass

from .core import *

from . import envs
from . import jax
from . import run
