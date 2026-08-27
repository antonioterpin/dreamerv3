"""Provide outs functionality."""

import functools

import jax
import jax.numpy as jnp

i32 = jnp.int32
f32 = jnp.float32
sg = jax.lax.stop_gradient


class Output:
    """Represent output."""

    def __repr__(self):
        name = type(self).__name__
        pred = self.pred()
        return f"{name}({pred.dtype}, shape={pred.shape})"

    def pred(self):
        """Handle pred.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def loss(self, target):
        """Handle loss.

        Args:
            target: Target value.

        Returns:
            Result of the operation.
        """
        return -self.logp(sg(target))

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def prob(self, event):
        """Handle prob.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        return jnp.exp(self.logp(event))

    def entropy(self):
        """Handle entropy.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def kl(self, other):
        """Handle kl.

        Args:
            other: Other value.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError


class Agg(Output):
    """Represent agg."""

    def __init__(self, output, dims, agg=jnp.sum):
        """Initialize the agg.

        Args:
            output: Output value.
            dims: Dims value.
            agg: Agg value.
        """
        self.output = output
        self.axes = [-i for i in range(1, dims + 1)]
        self.agg = agg

    def __repr__(self):
        name = type(self.output).__name__
        pred = self.pred()
        dims = len(self.axes)
        return f"{name}({pred.dtype}, shape={pred.shape}, agg={dims})"

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return self.output.pred()

    def loss(self, target):
        """Handle loss.

        Args:
            target: Target value.

        Returns:
            Result of the operation.
        """
        loss = self.output.loss(target)
        return self.agg(loss, self.axes)

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Returns:
            Result of the operation.
        """
        return self.output.sample(seed, shape)

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        return self.output.logp(event).sum(self.axes)

    def prob(self, event):
        """Handle prob.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        return self.output.prob(event).sum(self.axes)

    def entropy(self):
        """Handle entropy.

        Returns:
            Result of the operation.
        """
        entropy = self.output.entropy()
        return self.agg(entropy, self.axes)

    def kl(self, other):
        """Handle kl.

        Args:
            other: Other value.

        Returns:
            Result of the operation.
        """
        assert isinstance(other, Agg), other
        kl = self.output.kl(other.output)
        return self.agg(kl, self.axes)


class Frozen:
    """Represent frozen."""

    def __init__(self, output):
        """Initialize the frozen.

        Args:
            output: Output value.
        """
        self.output = output

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            fn = getattr(self.output, name)
        except AttributeError:
            raise ValueError(name)
        return functools.partial(self._wrapper, fn)

    def _wrapper(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        result = sg(result)
        return result


class Concat:
    """Represent concat."""

    def __init__(self, outputs, midpoints, axis):
        """Initialize the concat.

        Args:
            outputs: Outputs value.
            midpoints: Midpoints value.
            axis: Axis value.
        """
        assert (
            len(midpoints) == len(outputs) - 1
        ), "Expected number of midpoints to equal len(outputs) - 1."
        self.outputs = outputs
        self.midpoints = tuple(midpoints)
        self.axis = axis

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            fns = [getattr(x, name) for x in self.outputs]
        except AttributeError:
            raise ValueError(name)
        return functools.partial(self._wrapper, fns)

    def _wrapper(self, fns, *args, **kwargs):
        los = (None,) + self.midpoints
        his = self.midpoints + (None,)
        results = []
        for fn, lo, hi in zip(fns, los, his):
            segment = [slice(None, None, None)] * (self.axis + 1)
            segment[self.axis] = slice(lo, hi, None)
            segment = tuple(segment)
            a, kw = jax.tree.map(lambda x: x[segment], (args, kwargs))
            results.append(fn(*a, **kw))
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, self.axis), *results)


class MSE(Output):
    """Represent mse."""

    def __init__(self, mean, squash=None):
        """Initialize the mse.

        Args:
            mean: Mean value.
            squash: Squash value.
        """
        self.mean = f32(mean)
        self.squash = squash or (lambda x: x)

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return self.mean

    def loss(self, target):
        """Handle loss.

        Args:
            target: Target value.

        Returns:
            Result of the operation.
        """
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        return jnp.square(self.mean - sg(self.squash(f32(target))))


class Huber(Output):
    """Represent huber."""

    def __init__(self, mean, eps=1.0):
        # Soft Huber loss or Charbonnier loss.
        """Initialize the huber.

        Args:
            mean: Mean value.
            eps: Eps value.
        """
        self.mean = f32(mean)
        self.eps = eps

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return self.mean

    def loss(self, target):
        """Handle loss.

        Args:
            target: Target value.

        Returns:
            Result of the operation.
        """
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        dist = self.mean - sg(f32(target))
        return jnp.sqrt(jnp.square(dist) + jnp.square(self.eps)) - self.eps


class Normal(Output):
    """Represent normal."""

    def __init__(self, mean, stddev=1.0):
        """Initialize the normal.

        Args:
            mean: Mean value.
            stddev: Stddev value.
        """
        self.mean = f32(mean)
        self.stddev = jnp.broadcast_to(f32(stddev), self.mean.shape)

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return self.mean

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Returns:
            Result of the operation.
        """
        sample = jax.random.normal(seed, shape + self.mean.shape, f32)
        return sample * self.stddev + self.mean

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        assert jnp.issubdtype(event.dtype, jnp.floating), event.dtype
        return jax.scipy.stats.norm.logpdf(f32(event), self.mean, self.stddev)

    def entropy(self):
        """Handle entropy.

        Returns:
            Result of the operation.
        """
        return 0.5 * jnp.log(2 * jnp.pi * jnp.square(self.stddev)) + 0.5

    def kl(self, other):
        """Handle kl.

        Args:
            other: Other value.

        Returns:
            Result of the operation.
        """
        assert isinstance(other, type(self)), (self, other)
        return 0.5 * (
            jnp.square(self.stddev / other.stddev)
            + jnp.square(other.mean - self.mean) / jnp.square(other.stddev)
            + 2 * jnp.log(other.stddev)
            - 2 * jnp.log(self.stddev)
            - 1
        )


class Binary(Output):
    """Represent binary."""

    def __init__(self, logit):
        """Initialize the binary.

        Args:
            logit: Logit value.
        """
        self.logit = f32(logit)

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return self.logit > 0

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        event = f32(event)
        logp = jax.nn.log_sigmoid(self.logit)
        lognotp = jax.nn.log_sigmoid(-self.logit)
        return event * logp + (1 - event) * lognotp

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Returns:
            Result of the operation.
        """
        prob = jax.nn.sigmoid(self.logit)
        return jax.random.bernoulli(seed, prob, -1, shape + self.logit.shape)


class Categorical(Output):
    """Represent categorical."""

    def __init__(self, logits, unimix=0.0):
        """Initialize the categorical.

        Args:
            logits: Logits value.
            unimix: Unimix value.
        """
        logits = f32(logits)
        if unimix:
            probs = jax.nn.softmax(logits, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = jnp.log(probs)
        self.logits = logits

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        return jnp.argmax(self.logits, -1)

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Returns:
            Result of the operation.
        """
        return jax.random.categorical(
            seed, self.logits, -1, shape + self.logits.shape[:-1]
        )

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        onehot = jax.nn.one_hot(event, self.logits.shape[-1])
        return (jax.nn.log_softmax(self.logits, -1) * onehot).sum(-1)

    def entropy(self):
        """Handle entropy.

        Returns:
            Result of the operation.
        """
        logprob = jax.nn.log_softmax(self.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        entropy = -(prob * logprob).sum(-1)
        return entropy

    def kl(self, other):
        """Handle kl.

        Args:
            other: Other value.

        Returns:
            Result of the operation.
        """
        logprob = jax.nn.log_softmax(self.logits, -1)
        logother = jax.nn.log_softmax(other.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        return (prob * (logprob - logother)).sum(-1)


class OneHot(Output):
    """Represent one hot."""

    def __init__(self, logits, unimix=0.0):
        """Initialize the one hot.

        Args:
            logits: Logits value.
            unimix: Unimix value.
        """
        self.dist = Categorical(logits, unimix)

    def pred(self):
        """Handle pred.

        Returns:
            Result of the operation.
        """
        index = self.dist.pred()
        return self._onehot_with_grad(index)

    def sample(self, seed, shape=()):
        """Sample state.

        Args:
            seed: Random seed.
            shape: Shape of the resulting value.

        Returns:
            Result of the operation.
        """
        index = self.dist.sample(seed, shape)
        return self._onehot_with_grad(index)

    def logp(self, event):
        """Handle logp.

        Args:
            event: Event value.

        Returns:
            Result of the operation.
        """
        return (jax.nn.log_softmax(self.dist.logits, -1) * event).sum(-1)

    def entropy(self):
        """Handle entropy.

        Returns:
            Result of the operation.
        """
        return self.dist.entropy()

    def kl(self, other):
        """Handle kl.

        Args:
            other: Other value.

        Returns:
            Result of the operation.
        """
        return self.dist.kl(other.dist)

    def _onehot_with_grad(self, index):
        # Straight through gradients.
        value = jax.nn.one_hot(index, self.dist.logits.shape[-1], dtype=f32)
        probs = jax.nn.softmax(self.dist.logits, -1)
        value = sg(value) + (probs - sg(probs))
        return value


class TwoHot(Output):
    """Represent two hot."""

    def __init__(self, logits, bins, squash=None, unsquash=None):
        """Initialize the two hot.

        Args:
            logits: Logits value.
            bins: Bins value.
            squash: Squash value.
            unsquash: Unsquash value.
        """
        logits = f32(logits)
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert bins.dtype == f32, bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.bins = jnp.array(bins)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)

    def pred(self):
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.unsquash((self.probs * self.bins).sum(-1))
        """Handle pred.

        Returns:
            Result of the operation.
        """
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
            return self.unsquash(wavg)

    def loss(self, target):
        """Handle loss.

        Args:
            target: Target value.

        Returns:
            Result of the operation.
        """
        assert target.dtype == f32, target.dtype
        target = sg(self.squash(target))
        below = (self.bins <= target[..., None]).astype(i32).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).astype(i32).sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - target))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - jax.scipy.special.logsumexp(
            self.logits, -1, keepdims=True
        )
        return -(target * log_pred).sum(-1)
