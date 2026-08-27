"""Provide heads functionality."""

from __future__ import annotations

from typing import Any, Callable

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import nets
from . import outs

i32 = jnp.int32
f32 = jnp.float32


class MLPHead(nj.Module):
    """Represent mlphead."""

    units: int = 1024
    layers: int = 5
    act: str = "silu"
    norm: str = "rms"
    bias: bool = True
    winit: str | Callable = nets.Initializer("trunc_normal")
    binit: str | Callable = nets.Initializer("zeros")

    def __init__(self, space: Any, output: Any, **hkw: Any) -> None:
        """Initialize the mlphead.

        Args:
            space: Space value.
            output: Output value.
            hkw: Hkw value.
        """
        shared = dict(bias=self.bias, winit=self.winit, binit=self.binit)
        mkw = dict(**shared, act=self.act, norm=self.norm)
        hkw = dict(**shared, **hkw)
        self.mlp = nets.MLP(
            self.layers,
            self.units,
            **mkw,
            name="mlp",  # pyright: ignore[reportCallIssue]
        )
        if isinstance(space, dict):
            self.head = DictHead(space, output, **hkw, name="head")
        else:
            self.head = Head(space, output, **hkw, name="head")

    def __call__(self, x: Any, bdims: Any) -> Any:
        """Apply the mlphead.

        Args:
            x: X to process.
            bdims: Bdims to process.

        Returns:
            Result produced by the operation.
        """
        bshape = jax.tree.leaves(x)[0].shape[:bdims]
        x = x.reshape((*bshape, -1))
        x = self.mlp(x)
        x = self.head(x)
        return x


class DictHead(nj.Module):
    """Represent dict head."""

    def __init__(self, spaces: Any, outputs: Any, **kw: Any) -> None:
        """Initialize the dict head.

        Args:
            spaces: Spaces value.
            outputs: Outputs value.
            kw: Kw value.
        """
        assert spaces, spaces
        if not isinstance(spaces, dict):
            spaces = {"output": spaces}
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
        assert spaces.keys() == outputs.keys(), (spaces, outputs)
        self.spaces = spaces
        self.outputs = outputs
        self.kw = kw

    def __call__(self, x: Any) -> Any:
        """Apply the dict head.

        Args:
            x: X to process.

        Returns:
            Result produced by the operation.
        """
        outputs = {}
        for key, impl in self.outputs.items():
            space = self.spaces[key]
            outputs[key] = self.sub(key, Head, space, impl, **self.kw)(x)
        return outputs


class Head(nj.Module):
    """Represent head."""

    minstd: float = 1.0
    maxstd: float = 1.0
    unimix: float = 0.0
    bins: int = 255
    outscale: float = 1.0

    def __init__(self, space: Any, output: Any, **kw: Any) -> None:
        """Initialize the head.

        Args:
            space: Space value.
            output: Output value.
            kw: Kw value.
        """
        if isinstance(space, tuple):
            space = elements.Space(np.float32, space)
        if output == "onehot":
            classes = np.asarray(space.classes).flatten()
            assert (classes == classes[0]).all(), classes
            shape = (*space.shape, classes[0].item())
            space = elements.Space(f32, shape, 0.0, 1.0)
        self.space = space
        self.impl = output
        self.kw = {**kw, "outscale": self.outscale}

    def __call__(self, x: Any) -> Any:
        """Apply the head.

        Args:
            x: X to process.

        Returns:
            Result produced by the operation.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        if not hasattr(self, self.impl):
            raise NotImplementedError(self.impl)
        x = nets.ensure_dtypes(x)
        output = getattr(self, self.impl)(x)
        if self.space.shape:
            output = outs.Agg(output, len(self.space.shape), jnp.sum)
        assert output.pred().shape[x.ndim - 1 :] == self.space.shape, (
            self.space,
            self.impl,
            x.shape,
            output.pred().shape,
        )
        return output

    def binary(self, x: Any) -> Any:
        """Handle binary.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert np.all(self.space.classes == 2), self.space
        logit = self.sub("logit", nets.Linear, self.space.shape, **self.kw)(x)
        return outs.Binary(logit)

    def categorical(self, x: Any) -> Any:
        """Handle categorical.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert self.space.discrete, "Expected a discrete output space."
        classes = np.asarray(self.space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        shape = (*self.space.shape, classes[0].item())
        logits = self.sub("logits", nets.Linear, shape, **self.kw)(x)
        output = outs.Categorical(logits)
        output.minent = 0  # pyright: ignore[reportAttributeAccessIssue]
        output.maxent = np.log(  # pyright: ignore[reportAttributeAccessIssue]
            logits.shape[-1]
        )
        return output

    def onehot(self, x: Any) -> Any:
        """Handle onehot.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        logits = self.sub("logits", nets.Linear, self.space.shape, **self.kw)(x)
        return outs.OneHot(logits, self.unimix)

    def mse(self, x: Any) -> Any:
        """Handle mse.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        pred = self.sub("pred", nets.Linear, self.space.shape, **self.kw)(x)
        return outs.MSE(pred)

    def huber(self, x: Any) -> Any:
        """Handle huber.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        pred = self.sub("pred", nets.Linear, self.space.shape, **self.kw)(x)
        return outs.Huber(pred)

    def symlog_mse(self, x: Any) -> Any:
        """Handle symlog mse.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        pred = self.sub("pred", nets.Linear, self.space.shape, **self.kw)(x)
        return outs.MSE(pred, nets.symlog)

    def symexp_twohot(self, x: Any) -> Any:
        """Handle symexp twohot.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        shape = (*self.space.shape, self.bins)
        logits = self.sub("logits", nets.Linear, shape, **self.kw)(x)
        if self.bins % 2 == 1:
            half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
            half = nets.symexp(half)
            bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
        else:
            half = jnp.linspace(-20, 0, self.bins // 2, dtype=f32)
            half = nets.symexp(half)
            bins = jnp.concatenate([half, -half[::-1]], 0)
        return outs.TwoHot(logits, bins)

    def bounded_normal(self, x: Any) -> Any:
        """Handle bounded normal.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        mean = self.sub("mean", nets.Linear, self.space.shape, **self.kw)(x)
        stddev = self.sub("stddev", nets.Linear, self.space.shape, **self.kw)(x)
        lo, hi = self.minstd, self.maxstd
        stddev = (hi - lo) * jax.nn.sigmoid(stddev + 2.0) + lo
        output = outs.Normal(jnp.tanh(mean), stddev)
        output.minent = outs.Normal(  # pyright: ignore[reportAttributeAccessIssue]
            jnp.zeros_like(mean), self.minstd
        ).entropy()
        output.maxent = outs.Normal(  # pyright: ignore[reportAttributeAccessIssue]
            jnp.zeros_like(mean), self.maxstd
        ).entropy()
        return output

    def normal_logstd(self, x: Any) -> Any:
        """Handle normal logstd.

        Args:
            x: X value.

        Returns:
            Result of the operation.
        """
        assert (
            not self.space.discrete
        ), "Expected self space discrete to be false or empty."
        mean = self.sub("mean", nets.Linear, self.space.shape, **self.kw)(x)
        stddev = self.sub("stddev", nets.Linear, self.space.shape, **self.kw)(x)
        output = outs.Normal(
            mean, jnp.exp(stddev)  # pyright: ignore[reportArgumentType]
        )
        return output
