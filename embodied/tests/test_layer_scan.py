"""Provide test layer scan functionality."""

from __future__ import annotations
from typing import Any


import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from embodied.jax import utils

f32 = jnp.float32
i32 = jnp.int32


class Layer(nj.Module):
    """Represent layer."""

    units: int = 8

    def __call__(self, x: Any, c: Any, k: Any) -> Any:
        """Apply the layer.

        Args:
            x: X to process.
            c: C to process.
            k: K to process.

        Returns:
            Result produced by the operation.
        """
        assert x.shape[1:] == (
            self.units,
        ), "Expected x shape[1:] to equal (self.units,)."
        assert c.shape == (7,), "Expected c shape to equal (7,)."
        assert k.shape == (13, 7), "Expected k shape to equal (13, 7)."
        shape = (x.shape[-1], self.units)
        winit = lambda: jax.random.normal(nj.seed(), shape, f32)
        x = x @ self.value("kernel", winit)
        if "outer3" not in nj.context():
            nj.context()["outer3"] = jnp.zeros((), i32)
        nj.context()["outer3"] += 1
        nj.context()["outer1"] += 1
        inner = self.value("inner", jnp.array(0))
        self.write("inner", inner + nj.context()["outer2"])
        return x


class Net(nj.Module):
    """Represent net."""

    layers: int = 4
    units: int = 8

    def __call__(self, x: Any) -> Any:
        """Apply the net.

        Args:
            x: X to process.

        Returns:
            Result produced by the operation.
        """
        if "outer1" not in nj.context():
            nj.context()["outer1"] = jnp.ones((), i32)
        if "outer2" not in nj.context():
            nj.context()["outer2"] = jnp.ones((), i32)
        nj.context()["outer1"] += 1

        module = self.sub("linear", Layer, units=self.units)
        c = jnp.zeros((self.layers, 7))
        k = jnp.zeros((13, 7))
        x = utils.LayerScan(module, self.layers)(x, c, k=k)

        return x

    def loss(self, x: Any) -> Any:
        """Handle loss.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        return self(x).mean()


class TestLayerScan:
    """Represent test layer scan."""

    def test_init(self, L: int = 4, B: int = 2, D: int = 8) -> None:
        """Verify init.

        Args:
            L: L value.
            B: B value.
            D: D value.
        """
        x = np.random.normal(0, 1, (B, D))
        net = Net(layers=L, units=D, name="net")
        params = nj.init(net)({}, x, seed=0)
        assert set(params.keys()) == {
            "outer1",
            "outer2",
            "outer3",
            "net/linear/kernel",
            "net/linear/inner",
        }, 'Expected keys in params keys() to equal {"outer1", "outer2", "outer3", "net/linear/kernel", "net/linear/inner"}.'
        assert params["net/linear/kernel"].shape == (
            L,
            D,
            D,
        ), "Expected params net/linear/kernel shape to equal (L, D, D)."
        assert params["outer1"] == 1, "Expected params outer1 to equal 1."
        assert params["outer2"] == 1, "Expected params outer2 to equal 1."
        assert params["outer3"] == 0, "Expected params outer3 to equal 0."
        assert params["net/linear/inner"].shape == (
            L,
        ), "Expected params net/linear/inner shape to equal (L,)."
        assert (
            params["net/linear/inner"] == 0
        ).all(), "Expected params net/linear/inner to equal 0."
        for i in range(1, L):
            assert not jnp.allclose(
                params["net/linear/kernel"][0], params["net/linear/kernel"][i]
            ), "Expected jnp allclose(params net/linear/kernel[0], params net/linear/kernel[i]) to be false or empty."

    def test_apply(self, L: int = 4, B: int = 2, D: int = 8) -> None:
        """Verify apply.

        Args:
            L: L value.
            B: B value.
            D: D value.
        """
        x = np.random.normal(0, 1, (B, D))
        net = Net(layers=L, units=D, name="net")
        params = nj.init(net)({}, x, seed=0)
        params, out = nj.pure(net)(params, x)
        assert out.shape == (B, D), "Expected out shape to equal (B, D)."
        assert params["outer1"] == L + 2, "Expected params outer1 to equal L + 2."
        assert params["outer2"] == 1, "Expected params outer2 to equal 1."
        assert params["outer3"] == L, "Expected params outer3 to equal L."
        assert params["net/linear/inner"].shape == (
            L,
        ), "Expected params net/linear/inner shape to equal (L,)."
        assert (
            params["net/linear/inner"] == 1
        ).all(), "Expected params net/linear/inner to equal 1."

    def test_grad(self, L: int = 4, B: int = 2, D: int = 8) -> None:
        """Verify grad.

        Args:
            L: L value.
            B: B value.
            D: D value.

        Returns:
            Result of the operation.
        """
        x = np.random.normal(0, 1, (B, D))
        net = Net(layers=L, units=D, name="net")

        def fn(x: Any) -> Any:
            """Handle function.

            Args:
                x: X value.

            Returns:
                Result of the operation.
            """
            if nj.creating():
                net(x)
            params = {k: v for k, v in net.values.items() if v.dtype == f32}
            params = {net.path + "/" + k: v for k, v in params.items()}
            loss, _, grads = nj.grad(lambda x: net(x).mean(), params.keys())(x)
            params = {k: v - 0.1 * grads[k] for k, v in params.items()}
            nj.context().update(params)
            return loss

        params = nj.init(net)({}, x, seed=0)
        params, loss = nj.pure(fn)(params, x)
        assert loss.shape == (), "Expected loss shape to equal ()."
