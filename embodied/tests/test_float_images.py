"""Provide test float images functionality."""

from __future__ import annotations
from typing import Any


import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3 import rssm


def _encode(space: Any, image: Any) -> Any:
    enc = rssm.Encoder(
        {"image": space}, depth=2, mults=(1, 1), units=4, layers=1, name="enc"
    )

    def fn(obs: Any) -> Any:
        """Handle function.

        Args:
            obs: Observation value.

        Returns:
            Result of the operation.
        """
        reset = jnp.zeros(obs["image"].shape[:2], bool)
        _, _, tokens = enc({}, obs, reset, training=False)
        return tokens

    _, tokens = nj.pure(fn)(  # pyright: ignore[reportAssignmentType]
        {}, {"image": image}, seed=0, create=True
    )
    return np.asarray(tokens, np.float32)


def test_integer_images_other_than_uint8_are_rejected() -> None:
    """Verify integer images other than uint8 are rejected."""
    pixels = np.zeros((1, 2, 16, 16, 3), np.int32)
    with pytest.raises(AssertionError):
        _encode(elements.Space(np.int32, (16, 16, 3), 0, 256), pixels)
