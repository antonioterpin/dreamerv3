"""Provide selectors functionality."""

from __future__ import annotations
from typing import Any


import collections
import threading

import numpy as np


class Fifo:
    """Represent fifo."""

    def __init__(self) -> None:
        """Initialize the fifo."""
        self.queue = collections.deque()

    def __call__(self) -> Any:
        """Apply the fifo.

        Returns:
            Result produced by the operation.
        """
        return self.queue[0]

    def __len__(self) -> int:
        return len(self.queue)

    def __setitem__(self, key: Any, stepids: Any) -> None:
        self.queue.append(key)

    def __delitem__(self, key: Any) -> None:
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # This is very slow but typically not used.
            self.queue.remove(key)


class Uniform:
    """Represent uniform."""

    def __init__(self, seed: int = 0) -> None:
        """Initialize the uniform.

        Args:
            seed: Random seed.
        """
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.keys)

    def __call__(self) -> Any:
        """Apply the uniform.

        Returns:
            Result produced by the operation.
        """
        with self.lock:
            index = self.rng.integers(
                0, len(self.keys)
            ).item()  # pyright: ignore[reportAttributeAccessIssue]
            return self.keys[index]

    def __setitem__(self, key: Any, stepids: Any) -> None:
        with self.lock:
            self.indices[key] = len(self.keys)
            self.keys.append(key)

    def __delitem__(self, key: Any) -> None:
        with self.lock:
            assert 2 <= len(self), len(self)
            index = self.indices.pop(key)
            last = self.keys.pop()
            if index != len(self.keys):
                self.keys[index] = last
                self.indices[last] = index


class Recency:
    """Represent recency."""

    def __init__(self, uprobs: Any, seed: int = 0) -> None:
        """Initialize the recency.

        Args:
            uprobs: Uprobs value.
            seed: Random seed.
        """
        assert uprobs[0] >= uprobs[-1], uprobs
        self.uprobs = uprobs
        self.tree = self._build(uprobs)
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.steps = {}
        self.items = {}

    def __len__(self) -> int:
        return len(self.items)

    def __call__(self) -> Any:
        """Apply the recency.

        Returns:
            Result produced by the operation.

        Raises:
            KeyError: If the sampled item is repeatedly removed concurrently.
        """
        for retry in range(10):
            try:
                age = self._sample(self.tree, self.rng)
                if len(self.items) < len(self.uprobs):
                    age = int(age / len(self.uprobs) * len(self.items))
                return self.items[self.step - 1 - age]
            except KeyError:
                # Item might have been deleted very recently.
                if retry < 9:
                    import time

                    time.sleep(0.01)
                else:
                    raise

    def __setitem__(self, key: Any, stepids: Any) -> None:
        self.steps[key] = self.step
        self.items[self.step] = key
        self.step += 1

    def __delitem__(self, key: Any) -> None:
        step = self.steps.pop(key)
        del self.items[step]

    def _sample(self, tree: Any, rng: Any, bfactor: int = 16) -> Any:
        path = []
        for level, prob in enumerate(tree):
            p = prob
            for segment in path:
                p = p[segment]
            index = rng.choice(len(segment), p=p)
            path.append(index)
        index = sum(
            index * bfactor ** (len(tree) - level - 1)
            for level, index in enumerate(path)
        )
        return index

    def _build(self, uprobs: Any, bfactor: int = 16) -> Any:
        assert np.isfinite(uprobs).all(), uprobs
        assert (uprobs >= 0).all(), uprobs
        depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
        size = bfactor**depth
        uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
        tree = [uprobs]
        for level in reversed(range(depth - 1)):
            tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
        for level, prob in enumerate(tree):
            prob = prob.reshape([bfactor] * (1 + level))
            total = prob.sum(-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                tree[level] = np.where(total, prob / total, prob)
        return tree


class Prioritized:
    """Represent prioritized."""

    def __init__(
        self,
        exponent: float = 1.0,
        initial: float = 1.0,
        zero_on_sample: bool = False,
        maxfrac: float = 0.0,
        branching: int = 16,
        seed: int = 0,
    ) -> None:
        """Initialize the prioritized.

        Args:
            exponent: Exponent value.
            initial: Initial value.
            zero_on_sample: Zero on sample value.
            maxfrac: Maxfrac value.
            branching: Branching value.
            seed: Random seed.
        """
        assert 0 <= maxfrac <= 1, maxfrac
        self.exponent = float(exponent)
        self.initial = float(initial)
        self.zero_on_sample = zero_on_sample
        self.maxfrac = maxfrac
        self.tree = SampleTree(branching, seed)
        self.prios = collections.defaultdict(lambda: self.initial)
        self.stepitems = collections.defaultdict(list)
        self.items = {}

    def prioritize(self, stepids: Any, priorities: Any) -> None:
        """Handle prioritize.

        Args:
            stepids: Stepids value.
            priorities: Priorities value.
        """
        if not isinstance(stepids[0], bytes):
            stepids = [x.tobytes() for x in stepids]
        for stepid, priority in zip(stepids, priorities):
            try:
                self.prios[stepid] = priority
            except KeyError:
                print("Ignoring priority update for removed time step.")
        items = []
        for stepid in stepids:
            items += self.stepitems[stepid]
        for key in list(set(items)):
            try:
                self.tree.update(key, self._aggregate(key))
            except KeyError:
                print("Ignoring tree update for removed time step.")

    def __len__(self) -> int:
        return len(self.items)

    def __call__(self) -> Any:
        """Apply the prioritized.

        Returns:
            Result produced by the operation.
        """
        key = self.tree.sample()
        if self.zero_on_sample:
            zeros = [0.0] * len(self.items[key])
            self.prioritize(self.items[key], zeros)
        return key

    def __setitem__(self, key: Any, stepids: Any) -> None:
        if not isinstance(stepids[0], bytes):
            stepids = [x.tobytes() for x in stepids]
        self.items[key] = stepids
        [self.stepitems[stepid].append(key) for stepid in stepids]
        self.tree.insert(key, self._aggregate(key))

    def __delitem__(self, key: Any) -> None:
        self.tree.remove(key)
        stepids = self.items.pop(key)
        for stepid in stepids:
            stepitems = self.stepitems[stepid]
            stepitems.remove(key)
            if not stepitems:
                del self.stepitems[stepid]
                del self.prios[stepid]

    def _aggregate(self, key: Any) -> Any:
        # Both list comprehensions in this function are a performance bottleneck
        # because they are called very often.
        prios = [self.prios[stepid] for stepid in self.items[key]]
        if self.exponent != 1.0:
            prios = [x**self.exponent for x in prios]
        mean = sum(prios) / len(prios)
        if self.maxfrac:
            return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
        else:
            return mean


class Mixture:
    """Represent mixture."""

    def __init__(self, selectors: Any, fractions: Any, seed: int = 0) -> None:
        """Initialize the mixture.

        Args:
            selectors: Selectors value.
            fractions: Fractions value.
            seed: Random seed.
        """
        assert set(selectors.keys()) == set(
            fractions.keys()
        ), "Expected keys in selectors keys() to equal set(fractions.keys())."
        assert sum(fractions.values()) == 1, fractions
        for key, frac in list(fractions.items()):
            if not frac:
                selectors.pop(key)
                fractions.pop(key)
        keys = sorted(selectors.keys())
        self.selectors = [selectors[key] for key in keys]
        self.fractions = np.array([fractions[key] for key in keys], np.float32)
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Any:
        """Apply the mixture.

        Returns:
            Result produced by the operation.
        """
        return self.rng.choice(self.selectors, p=self.fractions)()

    def __setitem__(self, key: Any, stepids: Any) -> None:
        for selector in self.selectors:
            selector[key] = stepids

    def __delitem__(self, key: Any) -> None:
        for selector in self.selectors:
            del selector[key]

    def prioritize(self, stepids: Any, priorities: Any) -> None:
        """Handle prioritize.

        Args:
            stepids: Stepids value.
            priorities: Priorities value.
        """
        for selector in self.selectors:
            if hasattr(selector, "prioritize"):
                selector.prioritize(stepids, priorities)


class SampleTree:
    """Represent sample tree."""

    def __init__(self, branching: int = 16, seed: int = 0) -> None:
        """Initialize the sample tree.

        Args:
            branching: Branching value.
            seed: Random seed.
        """
        assert 2 <= branching, "Expected 2 to be at most branching."
        self.branching = branching
        self.root = SampleTreeNode()
        self.last = None
        self.entries = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def insert(self, key: Any, uprob: Any) -> None:
        """Handle insert.

        Args:
            key: Key identifying the requested value.
            uprob: Uprob value.
        """
        if not self.last:
            node = self.root
        else:
            ups = 0
            node = self.last.parent
            while node and len(node) >= self.branching:
                node = node.parent
                ups += 1
            if not node:
                node = SampleTreeNode()
                node.append(self.root)
                self.root = node
            for _ in range(ups):
                below = SampleTreeNode()
                node.append(below)
                node = below
        entry = SampleTreeEntry(key, uprob)
        node.append(entry)
        self.entries[key] = entry
        self.last = entry

    def remove(self, key: Any) -> None:
        """Handle remove.

        Args:
            key: Key identifying the requested value.
        """
        entry = self.entries.pop(key)
        assert self.last is not None, "A non-empty sample tree must have a last entry."
        assert (
            entry.parent is not None
        ), "A stored sample-tree entry must have a parent."
        assert (
            self.last.parent is not None
        ), "The last sample-tree entry must have a parent."
        entry_parent = entry.parent
        last_parent = self.last.parent
        entry.parent.remove(entry)
        if entry is not self.last:
            entry_parent.append(self.last)
        node = last_parent
        ups = 0
        while node.parent and not len(node):
            above = node.parent
            above.remove(node)
            node = above
            ups += 1
        if not len(node):
            self.last = None
            return
        while isinstance(node, SampleTreeNode):
            node = node.children[-1]
        self.last = node

    def update(self, key: Any, uprob: Any) -> None:
        """Update state.

        Args:
            key: Key identifying the requested value.
            uprob: Uprob value.
        """
        entry = self.entries[key]
        entry.uprob = uprob
        assert (
            entry.parent is not None
        ), "A stored sample-tree entry must have a parent."
        entry.parent.recompute()

    def sample(self) -> Any:
        """Sample state.

        Returns:
            Result of the operation.
        """
        node = self.root
        while isinstance(node, SampleTreeNode):
            uprobs = np.array([x.uprob for x in node.children])
            total = uprobs.sum()
            if not np.isfinite(total):
                finite = np.isinf(uprobs)
                probs = finite / finite.sum()
            elif total == 0:
                probs = np.ones(len(uprobs)) / len(uprobs)
            else:
                probs = uprobs / total
            choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
            node = node.children[choice.item()]
        return node.key


class SampleTreeNode:
    """Represent sample tree node."""

    __slots__ = ("parent", "children", "uprob")

    def __init__(self, parent: Any | None = None) -> None:
        """Initialize the sample tree node.

        Args:
            parent: Parent value.
        """
        self.parent = parent
        self.children = []
        self.uprob = 0

    def __repr__(self) -> str:
        return (
            f"SampleTreeNode(uprob={self.uprob}, "
            f"children={[x.uprob for x in self.children]})"
        )

    def __len__(self) -> int:
        return len(self.children)

    def __bool__(self) -> bool:
        return True

    def append(self, child: Any) -> None:
        """Handle append.

        Args:
            child: Child value.
        """
        if child.parent:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        self.recompute()

    def remove(self, child: Any) -> None:
        """Handle remove.

        Args:
            child: Child value.
        """
        child.parent = None
        self.children.remove(child)
        self.recompute()

    def recompute(self) -> None:
        """Handle recompute."""
        self.uprob = sum(x.uprob for x in self.children)
        self.parent and self.parent.recompute()  # pyright: ignore[reportUnusedExpression]


class SampleTreeEntry:
    """Represent sample tree entry."""

    __slots__ = ("parent", "key", "uprob")

    def __init__(self, key: Any | None = None, uprob: Any | None = None) -> None:
        """Initialize the sample tree entry.

        Args:
            key: Key identifying the requested value.
            uprob: Uprob value.
        """
        self.parent = None
        self.key = key
        self.uprob = uprob
