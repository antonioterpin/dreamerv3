"""Provide streams functionality."""

import functools
import queue
import threading

import elements
import numpy as np
import portal

from . import base


class Stateless(base.Stream):
    """Represent stateless."""

    def __init__(self, nextfn, *args, **kwargs):
        """Initialize the stateless.

        Args:
            nextfn: Nextfn value.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        if not callable(nextfn) and hasattr(nextfn, "__next__"):
            nextfn = nextfn.__next__
        self.nextfn = functools.partial(nextfn, *args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.nextfn()

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return None

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        pass


class Prefetch(base.Stream):
    """Prefetch source items while preserving normal iterator exhaustion."""

    # A unique sentinel distinguishes source exhaustion from data and exceptions.
    _DONE = object()

    def __init__(self, source, transform=None, amount=1):
        """Initialize the prefetch.

        Args:
            source: Source value.
            transform: Transform value.
            amount: Amount value.
        """
        self.source = iter(source) if hasattr(source, "__iter__") else source()
        self.transform = transform or (lambda x: x)
        self.state = self._getstate()
        self.requests = threading.Semaphore(amount)
        self.amount = amount
        self.queue = queue.Queue()
        self.worker = portal.Thread(self._worker)
        self.started = False

    def __iter__(self):
        assert not self.started, "Expected self started to be false or empty."
        self.worker.start()
        self.started = True
        return self

    def __next__(self):
        assert self.started, "The stream must be started before requesting an item."
        result = self.queue.get()
        self.requests.release()
        if result is self._DONE:
            raise StopIteration
        if isinstance(result, str):
            raise RuntimeError(result)
        data, self.state = result
        return data

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return self.state

    def load(self, state):
        """Load state.

        Args:
            state: State value.
        """
        if self.started:
            for _ in range(self.amount):
                self.queue.get()
        self.source.load(state)
        if self.started:
            self.requests.release(self.amount)

    def _worker(self):
        try:
            while True:
                self.requests.acquire()
                data = next(self.source)
                data = self.transform(data)
                state = self._getstate()
                self.queue.put((data, state))
        except StopIteration:
            # Deliver exhaustion to the consumer instead of reporting it as a failed
            # Portal worker during graceful finite-run shutdown.
            self.queue.put(self._DONE)
        except Exception as e:
            self.queue.put(str(e))
            raise

    def _getstate(self):
        if hasattr(self.source, "save"):
            return self.source.save()
        else:
            return None


class Consec(base.Stream):
    """Yield consecutive chunks from a source stream.

    Example:
    length = 3
    consec = 3
    prefix = 2

    source:   0 1 2 3 4 5 6 7 8 9 10
    chunk 1:  p-p-#-#-#
    chunk 2:        p-p-#-#-#
    chunk 3:              p-p-#-#-#
    """

    def __init__(self, source, length, consec, prefix=0, strict=True, contiguous=False):
        """Initialize the consec.

        Args:
            source: Source value.
            length: Length value.
            consec: Consec value.
            prefix: Prefix value.
            strict: Strict value.
            contiguous: Contiguous value.
        """
        self.source = source
        self.length = length
        self.consec = consec
        self.prefix = prefix
        self.strict = strict
        self.contiguous = contiguous
        self.index = 0
        self.current = None
        self.it = None

    def __iter__(self):
        self.it = iter(self.source)
        return self

    def __next__(self):
        if self.index >= self.consec:
            self.index = 0
        if self.index == 0:
            self.current = next(self.it)
            available = self.current["is_first"].shape[-1]
            assert self.length * self.consec + self.prefix <= available, (
                self.length,
                self.consec,
                self.prefix,
                available,
            )
            if self.strict:
                assert self.consec * self.length + self.prefix == available, (
                    self.consec,
                    self.length,
                    self.prefix,
                    available,
                )
        start = self.index * self.length
        stop = start + (self.length + self.prefix)
        chunk = {k: v[:, start:stop] for k, v in self.current.items()}
        chunk["consec"] = np.full(chunk["is_first"].shape, self.index, np.int32)
        if self.contiguous:
            # This is expensive but can speed up following operations, such as
            # sending arrays via networking.
            chunk = {k: np.ascontiguousarray(v) for k, v in chunk.items()}
        self.index += 1
        return chunk

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return {
            "source": self.source.save(),
            "index": self.index,
        }

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        self.source.load(data["source"])
        self.index = data["index"]


class Zip(base.Stream):
    """Represent zip."""

    def __init__(self, sources):
        """Initialize the zip.

        Args:
            sources: Sources value.
        """
        assert len(sources) > 1, len(sources)
        self.sources = sources
        self.iterators = None
        self.started = False

    def __iter__(self):
        assert not self.started, "Expected self started to be false or empty."
        self.started = True
        self.iterators = [iter(x) for x in self.sources]
        return self

    def __next__(self):
        parts = [next(x) for x in self.iterators]
        result = elements.tree.map(lambda *el: np.concatenate(el), *parts)
        return result

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return [x.save() for x in self.iterators]

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        assert len(data) == len(
            self.iterators
        ), "Expected number of data to equal len(self.iterators)."
        [it.load(d) for it, d in zip(self.iterators, data)]


class Map(base.Stream):
    """Represent map."""

    def __init__(self, source, fn, *args, **kwargs):
        """Initialize the map.

        Args:
            source: Source value.
            fn: Function to apply.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        self.source = source
        self.fn = lambda x: fn(x, *args, **kwargs)
        self.iterator = None
        self.started = False

    def __iter__(self):
        assert not self.started, "Expected self started to be false or empty."
        self.started = True
        self.iterator = iter(self.source)
        return self

    def __next__(self):
        assert self.started, "The stream must be started before requesting an item."
        return self.fn(next(self.iterator))

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return self.iterator.save()

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        self.iterator.load(data)


class Mixer(base.Stream):
    """Represent mixer."""

    def __init__(self, sources, weights, seed=0):
        """Initialize the mixer.

        Args:
            sources: Sources value.
            weights: Weights value.
            seed: Random seed.
        """
        assert sources.keys() == weights.keys(), (sources, weights)
        self.keys = sorted(sources.keys())
        self.iterators = [iter(sources[k]) for k in self.keys]
        weights = np.array([weights[k] for k in self.keys], np.float32)
        self.probs = weights / weights.sum()
        self.seed = seed
        self.started = False
        self.step = 0

    def __iter__(self):
        assert not self.started, "Expected self started to be false or empty."
        return self

    def __next__(self):
        assert self.started, "The stream must be started before requesting an item."
        rng = np.random.default_rng(seed=[self.seed, self.step])
        self.step += 1
        index = rng.choice(len(self.keys), p=self.probs)
        return next(self.iterators[index])

    def save(self):
        """Save state.

        Returns:
            Result of the operation.
        """
        return {
            "step": self.step,
            "seed": self.seed,
            "sources": {k: it.save() for k, it in zip(self.keys, self.iterators)},
        }

    def load(self, data):
        """Load state.

        Args:
            data: Data to process.
        """
        self.step = data["step"]
        self.seed = data["seed"]
        assert sorted(data["sources"].keys()) == self.keys, (data["sources"], self.keys)
        for key in self.keys:
            self.iterators[key].load(data["sources"][key])
