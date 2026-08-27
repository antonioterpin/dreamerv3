"""Provide test replay functionality."""

from __future__ import annotations
from collections.abc import Iterator
from typing import Any


import collections
import threading
import time

import elements
import embodied
import numpy as np
import pytest

REPLAYS_UNLIMITED = [
    embodied.replay.Replay,
    # embodied.replay.Reverb,
]

REPLAYS_SAVECHUNKS = [
    embodied.replay.Replay,
]

REPLAYS_UNIFORM = [
    embodied.replay.Replay,
]


def unbatched(dataset: Any) -> Iterator[Any]:
    """Handle unbatched.

    Args:
        dataset: Dataset value.
    """
    for batch in dataset:
        yield {k: v[0] for k, v in batch.items()}


@pytest.mark.filterwarnings("ignore:.*Pillow.*")
@pytest.mark.filterwarnings("ignore:.*the imp module.*")
@pytest.mark.filterwarnings("ignore:.*distutils.*")
class TestReplay:
    """Represent test replay."""

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    def test_multiple_keys(self, Replay: Any) -> None:
        """Verify multiple keys.

        Args:
            Replay: Replay value.
        """
        replay = Replay(length=5, capacity=10)
        for step in range(30):
            replay.add({"image": np.zeros((64, 64, 3)), "action": np.zeros(12)})
        seq = next(unbatched(replay.dataset(1)))
        assert set(seq.keys()) == {
            "stepid",
            "image",
            "action",
        }, 'Expected keys in seq keys() to equal {"stepid", "image", "action"}.'
        assert seq["stepid"].shape == (
            5,
            20,
        ), "Expected seq stepid shape to equal (5, 20)."
        assert seq["image"].shape == (
            5,
            64,
            64,
            3,
        ), "Expected seq image shape to equal (5, 64, 64, 3)."
        assert seq["action"].shape == (
            5,
            12,
        ), "Expected seq action shape to equal (5, 12)."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,workers,capacity",
        [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)],
    )
    def test_capacity_exact(
        self, Replay: Any, length: Any, workers: Any, capacity: Any
    ) -> None:
        """Verify capacity exact.

        Args:
            Replay: Replay value.
            length: Length value.
            workers: Workers value.
            capacity: Capacity value.
        """
        replay = Replay(length, capacity)
        for step in range(30):
            for worker in range(workers):
                replay.add({"step": step}, worker)
            target = min(workers * max(0, (step + 1) - length + 1), capacity)
            assert len(replay) == target, "Expected number of replay to equal target."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,workers,capacity,chunksize",
        [
            (1, 1, 1, 128),
            (2, 1, 2, 128),
            (5, 1, 10, 128),
            (1, 2, 2, 128),
            (5, 3, 15, 128),
            (2, 7, 20, 128),
            (7, 2, 27, 4),
        ],
    )
    def test_sample_sequences(
        self, Replay: Any, length: Any, workers: Any, capacity: Any, chunksize: Any
    ) -> None:
        """Verify sample sequences.

        Args:
            Replay: Replay value.
            length: Length value.
            workers: Workers value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        replay = Replay(length, capacity, chunksize=chunksize)
        for step in range(30):
            for worker in range(workers):
                replay.add({"step": step, "worker": worker}, worker)
        dataset = unbatched(replay.dataset(1))
        for _ in range(10):
            seq = next(dataset)
            assert (
                seq["step"] - seq["step"][0] == np.arange(length)
            ).all(), "Expected seq step - seq step[0] to equal np.arange(length)."
            assert (
                seq["worker"] == seq["worker"][0]
            ).all(), 'Expected seq worker to equal seq["worker"][0].'

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,capacity", [(1, 1), (2, 2), (5, 10), (1, 2), (5, 15), (2, 20)]
    )
    def test_sample_single(self, Replay: Any, length: Any, capacity: Any) -> None:
        """Verify sample single.

        Args:
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
        """
        replay = Replay(length, capacity)
        for step in range(length):
            replay.add({"step": step})
        dataset = unbatched(replay.dataset(1))
        for _ in range(10):
            seq = next(dataset)
            assert (
                seq["step"] == np.arange(length)
            ).all(), "Expected seq step to equal np.arange(length)."

    @pytest.mark.parametrize("Replay", REPLAYS_UNIFORM)
    def test_sample_uniform(self, Replay: Any) -> None:
        """Verify sample uniform.

        Args:
            Replay: Replay value.
        """
        replay = Replay(capacity=20, length=5, seed=0)
        for step in range(7):
            replay.add({"step": step})
        assert len(replay) == 3, "Expected number of replay to equal 3."
        histogram = collections.defaultdict(int)
        dataset = unbatched(replay.dataset(1))
        for _ in range(100):
            seq = next(dataset)
            histogram[seq["step"][0]] += 1
        assert len(histogram) == 3, histogram
        histogram = tuple(histogram.values())
        assert histogram[0] > 20, "Expected histogram[0] to be greater than 20."
        assert histogram[1] > 20, "Expected histogram[1] to be greater than 20."
        assert histogram[2] > 20, "Expected histogram[2] to be greater than 20."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    def test_workers_simple(self, Replay: Any) -> None:
        """Verify workers simple.

        Args:
            Replay: Replay value.
        """
        replay = Replay(length=2, capacity=20)
        replay.add({"step": 0}, worker=0)
        replay.add({"step": 1}, worker=1)
        replay.add({"step": 2}, worker=0)
        replay.add({"step": 3}, worker=1)
        dataset = unbatched(replay.dataset(1))
        for _ in range(10):
            seq = next(dataset)
            assert tuple(seq["step"]) in (
                (0, 2),
                (1, 3),
            ), "Expected tuple(seq step) to be present in ((0, 2), (1, 3))."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    def test_workers_random(
        self, Replay: Any, length: int = 4, capacity: int = 30
    ) -> None:
        """Verify workers random.

        Args:
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
        """
        rng = np.random.default_rng(seed=0)
        replay = Replay(length, capacity)
        streams = {i: iter(range(10)) for i in range(3)}
        for _ in range(40):
            worker = int(rng.integers(0, 3, ()))
            try:
                step = {"step": next(streams[worker]), "stream": worker}
                replay.add(step, worker=worker)
            except StopIteration:
                pass
        histogram = collections.defaultdict(int)
        dataset = unbatched(replay.dataset(1))
        for _ in range(10):
            seq = next(dataset)
            assert (
                seq["step"] - seq["step"][0] == np.arange(length)
            ).all(), "Expected seq step - seq step[0] to equal np.arange(length)."
            assert (
                seq["stream"] == seq["stream"][0]
            ).all(), 'Expected seq stream to equal seq["stream"][0].'
            histogram[int(seq["stream"][0])] += 1
        assert all(
            count > 0 for count in histogram.values()
        ), "Expected every element to satisfy that count to be greater than 0."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,workers,capacity",
        [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)],
    )
    def test_worker_delay(
        self, Replay: Any, length: Any, workers: Any, capacity: Any
    ) -> None:
        """Verify worker delay.

        Args:
            Replay: Replay value.
            length: Length value.
            workers: Workers value.
            capacity: Capacity value.
        """
        replay = Replay(length, capacity)
        rng = np.random.default_rng(seed=0)
        streams = [iter(range(10)) for _ in range(workers)]
        while streams:
            try:
                worker = rng.integers(0, len(streams))
                replay.add({"step": next(streams[worker])}, worker)
            except StopIteration:
                del streams[worker]

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,capacity,chunksize",
        [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)],
    )
    def test_restore_exact(
        self, tmpdir: Any, Replay: Any, length: Any, capacity: Any, chunksize: Any
    ) -> None:
        """Verify restore exact.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        elements.UUID.reset(debug=True)
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        for step in range(30):
            replay.add({"step": step})
        num_items = np.clip(30 - length + 1, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        replay = Replay(length, capacity, directory=tmpdir)
        replay.load(data)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        dataset = unbatched(replay.dataset(1))
        for _ in range(len(replay)):
            assert (
                len(next(dataset)["step"]) == length
            ), "Expected number of next(dataset) step to equal length."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,capacity,chunksize",
        [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)],
    )
    def test_restore_noclear(
        self, tmpdir: Any, Replay: Any, length: Any, capacity: Any, chunksize: Any
    ) -> None:
        """Verify restore noclear.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        elements.UUID.reset(debug=True)
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        for _ in range(30):
            replay.add({"foo": 13})
        num_items = np.clip(30 - length + 1, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        for _ in range(30):
            replay.add({"foo": 42})
        replay.load(data)
        dataset = unbatched(replay.dataset(1))
        if capacity < num_items:
            for _ in range(len(replay)):
                assert (
                    next(dataset)["foo"] == 13
                ), "Expected next(dataset) foo to equal 13."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize("workers", [1, 2, 5])
    @pytest.mark.parametrize("length,capacity", [(1, 1), (3, 10), (5, 100)])
    def test_restore_workers(
        self, tmpdir: Any, Replay: Any, workers: Any, length: Any, capacity: Any
    ) -> None:
        """Verify restore workers.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            workers: Workers value.
            length: Length value.
            capacity: Capacity value.
        """
        capacity *= workers
        replay = Replay(length, capacity, directory=tmpdir, save_wait=True)
        for step in range(50):
            for worker in range(workers):
                replay.add({"step": step}, worker)
        num_items = np.clip((50 - length + 1) * workers, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        replay = Replay(length, capacity, directory=tmpdir)
        replay.load(data)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        dataset = unbatched(replay.dataset(1))
        for _ in range(len(replay)):
            assert (
                len(next(dataset)["step"]) == length
            ), "Expected number of next(dataset) step to equal length."

    @pytest.mark.parametrize("Replay", REPLAYS_SAVECHUNKS)
    @pytest.mark.parametrize(
        "length,capacity,chunksize", [(1, 1, 1), (3, 10, 5), (5, 100, 12)]
    )
    def test_restore_chunks_exact(
        self, tmpdir: Any, Replay: Any, length: Any, capacity: Any, chunksize: Any
    ) -> None:
        """Verify restore chunks exact.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        elements.UUID.reset(debug=True)
        assert (
            len(list(elements.Path(tmpdir).glob("*.npz"))) == 0
        ), 'Expected number of list(elements Path(tmpdir) glob("* npz")) to equal 0.'
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        for step in range(30):
            replay.add({"step": step})
        num_items = np.clip(30 - length + 1, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        filenames = list(elements.Path(tmpdir).glob("*.npz"))
        lengths = [int(x.stem.split("-")[3]) for x in filenames]
        stored_steps = min(capacity + length - 1, 30)
        total_chunks = int(np.ceil(30 / chunksize))
        pruned_chunks = int(np.floor((30 - stored_steps) / chunksize))
        assert (
            len(filenames) == total_chunks - pruned_chunks
        ), "Expected number of filenames to equal total_chunks - pruned_chunks."
        last_chunk_empty = total_chunks * chunksize - 30
        saved_steps = (total_chunks - pruned_chunks) * chunksize - last_chunk_empty
        assert (
            sum(lengths) == saved_steps
        ), "Expected sum(lengths) to equal saved_steps."
        assert all(
            1 <= x <= chunksize for x in lengths
        ), "Expected every element to satisfy that 1 <= x <= chunksize to hold."
        replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
        replay.load(data)
        assert sorted(elements.Path(tmpdir).glob("*.npz")) == sorted(
            filenames
        ), 'Expected sorted(elements Path(tmpdir) glob("* npz")) to equal sorted(filenames).'
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        dataset = unbatched(replay.dataset(1))
        for _ in range(len(replay)):
            assert (
                len(next(dataset)["step"]) == length
            ), "Expected number of next(dataset) step to equal length."

    @pytest.mark.parametrize("Replay", REPLAYS_SAVECHUNKS)
    @pytest.mark.parametrize("workers", [1, 2, 5])
    @pytest.mark.parametrize(
        "length,capacity,chunksize", [(1, 1, 1), (3, 10, 5), (5, 100, 12)]
    )
    def test_restore_chunks_workers(
        self,
        tmpdir: Any,
        Replay: Any,
        workers: Any,
        length: Any,
        capacity: Any,
        chunksize: Any,
    ) -> None:
        """Verify restore chunks workers.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            workers: Workers value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        capacity *= workers
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        for step in range(50):
            for worker in range(workers):
                replay.add({"step": step}, worker)
        num_items = np.clip((50 - length + 1) * workers, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        filenames = list(elements.Path(tmpdir).glob("*.npz"))
        lengths = [int(x.stem.split("-")[3]) for x in filenames]
        stored_steps = min(capacity // workers + length - 1, 50)
        total_chunks = int(np.ceil(50 / chunksize))
        pruned_chunks = int(np.floor((50 - stored_steps) / chunksize))
        assert (
            len(filenames) == (total_chunks - pruned_chunks) * workers
        ), "Expected number of filenames to equal (total_chunks - pruned_chunks) * workers."
        last_chunk_empty = total_chunks * chunksize - 50
        saved_steps = (total_chunks - pruned_chunks) * chunksize - last_chunk_empty
        assert (
            sum(lengths) == saved_steps * workers
        ), "Expected sum(lengths) to equal saved_steps * workers."
        replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
        replay.load(data)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        dataset = unbatched(replay.dataset(1))
        for _ in range(len(replay)):
            assert (
                len(next(dataset)["step"]) == length
            ), "Expected number of next(dataset) step to equal length."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    @pytest.mark.parametrize(
        "length,capacity,chunksize",
        [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)],
    )
    def test_restore_insert(
        self, tmpdir: Any, Replay: Any, length: Any, capacity: Any, chunksize: Any
    ) -> None:
        """Verify restore insert.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
        """
        elements.UUID.reset(debug=True)
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        inserts = int(1.5 * chunksize)
        for step in range(inserts):
            replay.add({"step": step})
        num_items = np.clip(inserts - length + 1, 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        data = replay.save()
        replay = Replay(length, capacity, directory=tmpdir)
        replay.load(data)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."
        dataset = unbatched(replay.dataset(1))
        for _ in range(len(replay)):
            assert (
                len(next(dataset)["step"]) == length
            ), "Expected number of next(dataset) step to equal length."
        for step in range(inserts):
            replay.add({"step": step})
        num_items = np.clip(2 * (inserts - length + 1), 0, capacity)
        assert len(replay) == num_items, "Expected number of replay to equal num_items."

    @pytest.mark.parametrize("Replay", REPLAYS_UNLIMITED)
    def test_threading(
        self,
        tmpdir: Any,
        Replay: Any,
        length: int = 5,
        capacity: int = 128,
        chunksize: int = 32,
        adders: int = 8,
        samplers: int = 4,
    ) -> None:
        """Verify threading.

        Args:
            tmpdir: Tmpdir value.
            Replay: Replay value.
            length: Length value.
            capacity: Capacity value.
            chunksize: Chunksize value.
            adders: Adders value.
            samplers: Samplers value.
        """
        elements.UUID.reset(debug=True)
        replay = Replay(
            length, capacity, directory=tmpdir, chunksize=chunksize, save_wait=True
        )
        running = [True]

        def adder() -> None:
            """Handle adder."""
            ident = threading.get_ident()
            step = 0
            while running[0]:
                replay.add({"step": step}, worker=ident)
                step += 1
                time.sleep(0.001)

        def sampler() -> None:
            """Handle sampler."""
            dataset = unbatched(replay.dataset(1))
            while running[0]:
                seq = next(dataset)
                assert (
                    seq["step"] - seq["step"][0] == np.arange(length)
                ).all(), "Expected seq step - seq step[0] to equal np.arange(length)."
                time.sleep(0.001)

        workers = []
        for _ in range(adders):
            workers.append(threading.Thread(target=adder))
        for _ in range(samplers):
            workers.append(threading.Thread(target=sampler))

        try:
            [worker.start() for worker in workers]
            for _ in range(4):

                time.sleep(0.1)
                stats = replay.stats()
                assert (
                    stats["inserts"] > 0
                ), "Expected stats inserts to be greater than 0."
                assert (
                    stats["samples"] > 0
                ), "Expected stats samples to be greater than 0."

                print("SAVING")
                data = replay.save()
                time.sleep(0.1)

                print("LOADING")
                replay.load(data)

        finally:
            running[0] = False
            [worker.join() for worker in workers]

        assert len(replay) == capacity, "Expected number of replay to equal capacity."
