"""Provide test float images functionality."""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3 import rssm


def _encode(space, image):
    enc = rssm.Encoder(
        {"image": space}, depth=2, mults=(1, 1), units=4, layers=1, name="enc"
    )

    def fn(obs):
        """Handle function.

        Args:
            obs: Observation value.

        Returns:
            Result of the operation.
        """
        reset = jnp.zeros(obs["image"].shape[:2], bool)
        _, _, tokens = enc({}, obs, reset, training=False)
        return tokens

    _, tokens = nj.pure(fn)({}, {"image": image}, seed=0, create=True)
    return np.asarray(tokens, np.float32)


def test_float_images_in_unit_range_match_uint8_images():
    """Verify float images in unit range match uint8 images."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (1, 2, 16, 16, 3), np.uint8)
    as_uint8 = _encode(elements.Space(np.uint8, (16, 16, 3)), pixels)
    as_float = _encode(
        elements.Space(np.float32, (16, 16, 3)), pixels.astype(np.float32) / 255
    )
    assert (
        as_uint8.shape == as_float.shape
    ), "Expected as uint8 shape to equal as_float.shape."
    assert np.allclose(
        as_uint8, as_float, atol=1e-2
    ), "A float image in [0, 1] must encode like the same uint8 image"


def test_integer_images_other_than_uint8_are_rejected():
    """Verify integer images other than uint8 are rejected."""
    pixels = np.zeros((1, 2, 16, 16, 3), np.int32)
    with pytest.raises(AssertionError):
        _encode(elements.Space(np.int32, (16, 16, 3), 0, 256), pixels)
