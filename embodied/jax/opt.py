"""Provide optimizer functionality."""

from typing import Any


import math

import jax
import jax.numpy as jnp
import ninjax as nj
import optax

from . import internal
from . import nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):
    """Represent optimizer."""

    summary_depth: int = 2

    def __init__(self, modules: Any, opt: Any) -> None:
        """Initialize the optimizer.

        Args:
            modules: Modules value.
            opt: Optimizer value.
        """
        modules = modules if isinstance(modules, (list, tuple)) else (modules,)
        self.modules = modules
        self.opt = opt
        self.step = nj.Variable(jnp.array, 0, i32, name="step")
        self.scaling = nets.COMPUTE_DTYPE == jnp.float16
        if self.scaling:
            self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
            self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name="grad_scale")
            self.good_steps = nj.Variable(jnp.array, 0, i32, name="good_steps")

    def __call__(
        self, lossfn: Any, *args: Any, has_aux: bool = False, **kwargs: Any
    ) -> Any:
        """Apply the optimizer.

        Args:
            lossfn: Lossfn to process.
            has_aux: Has aux to process.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result produced by the operation.
        """
        metrics = {}

        def lossfn2(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            """Handle lossfn2.

            Args:
                args: Positional arguments forwarded to the wrapped callable.
                kwargs: Keyword arguments forwarded to the wrapped callable.

            Returns:
                Result of the operation.
            """
            outs = lossfn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        loss, params, grads, aux = nj.grad(  # pyright: ignore[reportAssignmentType]
            lossfn2, self.modules, has_aux=True
        )(*args, **kwargs)
        if self.scaling:
            loss *= 1 / self.grad_scale.read()

        counts = {
            k: math.prod(v.shape)
            for k, v in params.items()  # pyright: ignore[reportAttributeAccessIssue]
        }
        if nj.creating():
            print(self._summarize_params(counts, self.summary_depth))

        axes = internal.get_data_axes()
        if axes:
            grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

        if self.scaling:
            invscale = 1 / self.grad_scale.read()
            grads = jax.tree.map(lambda x: x * invscale, grads)

        state = self.sub("state", nj.Tree, self.opt.init, params)
        updates, new_state = self.opt.update(grads, state.read(), params)
        nj.context().update(optax.apply_updates(params, updates))
        state.write(new_state)
        grad_norm = optax.global_norm(grads)
        if self.scaling:
            self._update_scale(grads, jnp.isfinite(grad_norm))
            grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
            self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
            metrics["grad_scale"] = self.grad_scale.read()
            metrics["grad_overflow"] = f32(~jnp.isfinite(grad_norm))
        else:
            self.step.write(self.step.read() + 1)
        metrics["loss"] = loss.mean()
        metrics["updates"] = self.step.read()
        metrics["grad_norm"] = grad_norm
        metrics["grad_rms"] = nets.rms(grads)
        metrics["update_rms"] = nets.rms(updates)
        metrics["param_rms"] = nets.rms([x.values for x in self.modules])
        metrics["param_count"] = jnp.array(list(counts.values()), f32).sum()
        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, grads: Any, finite: Any) -> Any:
        keep = finite & (self.good_steps.read() < 1000)
        incr = finite & (self.good_steps.read() >= 1000)
        decr = ~finite
        self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
        self.grad_scale.write(
            jnp.clip(
                f32(keep) * self.grad_scale.read()
                + f32(incr) * self.grad_scale.read() * 2
                + f32(decr) * self.grad_scale.read() / 2,
                1e-4,
                1e5,
            )
        )
        return finite

    def _summarize_params(self, counts: Any, depth: Any) -> Any:
        lines = []
        pfxs = []
        for key in counts:
            parts = key.split("/")
            pfxs += ["/".join(parts[: i + 1]) for i in range(min(len(parts), depth))]
        subcounts = {
            prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
            for prefix in set(pfxs)
        }
        lines = [f"Optimizer {self.name} has {sum(counts.values()):,} params:"]
        for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
            lines.append(f"{count:>14,} {prefix}")
        return "\n".join(lines)


def clip_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> Any:
    """Handle clip by agc.

    Args:
        clip: Clip value.
        pmin: Pmin value.

    Returns:
        Result of the operation.
    """

    def init_fn(params: Any) -> tuple[Any, ...]:
        """Handle init function.

        Args:
            params: Parameters value.

        Returns:
            Result of the operation.
        """
        return ()

    def update_fn(
        updates: Any, state: Any, params: Any | None = None
    ) -> tuple[Any, ...]:
        """Update function.

        Args:
            updates: Updates value.
            state: State value.
            params: Parameters value.

        Returns:
            Result of the operation.
        """

        def fn(param: Any, update: Any) -> Any:
            """Handle function.

            Args:
                param: Param value.
                update: Update value.

            Returns:
                Result of the operation.
            """
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))

        updates = jax.tree.map(fn, params, updates) if clip else updates
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta: float = 0.999, eps: float = 1e-8) -> Any:
    """Handle scale by rms.

    Args:
        beta: Beta value.
        eps: Eps value.

    Returns:
        Result of the operation.
    """

    def init_fn(params: Any) -> tuple[Any, ...]:
        """Handle init function.

        Args:
            params: Parameters value.

        Returns:
            Result of the operation.
        """
        nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return (step, nu)

    def update_fn(
        updates: Any, state: Any, params: Any | None = None
    ) -> tuple[Any, ...]:
        """Update function.

        Args:
            updates: Updates value.
            state: State value.
            params: Parameters value.

        Returns:
            Result of the operation.
        """
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree.map(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree.map(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta: float = 0.9, nesterov: bool = False) -> Any:
    """Handle scale by momentum.

    Args:
        beta: Beta value.
        nesterov: Nesterov value.

    Returns:
        Result of the operation.
    """

    def init_fn(params: Any) -> tuple[Any, ...]:
        """Handle init function.

        Args:
            params: Parameters value.

        Returns:
            Result of the operation.
        """
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return (step, mu)

    def update_fn(
        updates: Any, state: Any, params: Any | None = None
    ) -> tuple[Any, ...]:
        """Update function.

        Args:
            updates: Updates value.
            state: State value.
            params: Parameters value.

        Returns:
            Result of the operation.
        """
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)
