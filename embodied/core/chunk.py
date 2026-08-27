"""Provide chunk functionality."""

from __future__ import annotations
from typing import Any


import io
import sys
import traceback

import elements
import numpy as np


class Chunk:
    """Represent chunk."""

    __slots__ = ("time", "uuid", "succ", "length", "size", "data", "saved")

    def __init__(self, size: int = 1024) -> None:
        """Initialize the chunk.

        Args:
            size: Requested number of elements.
        """
        self.time = elements.timestamp(millis=True)
        self.uuid = elements.UUID()
        self.succ = elements.UUID(0)
        # self.uuid = int(np.random.randint(1, 2 * 63))
        # self.succ = 0
        self.length = 0
        self.size = size
        self.data = None
        self.saved = False

    def __repr__(self) -> str:
        return f"Chunk({self.filename})"

    def __lt__(self, other: Any) -> bool:
        return self.time < other.time

    @property
    def filename(self) -> Any:
        """Handle filename.

        Returns:
            Result of the operation.
        """
        succ = self.succ.uuid if isinstance(self.succ, type(self)) else self.succ
        return f"{self.time}-{str(self.uuid)}-{str(succ)}-{self.length}.npz"

    @property
    def nbytes(self) -> Any:
        """Handle nbytes.

        Returns:
            Result of the operation.
        """
        if not self.data:
            return 0
        return sum(x.nbytes for x in self.data.values())

    def append(self, step: Any) -> None:
        """Handle append.

        Args:
            step: Step value.
        """
        assert (
            self.length < self.size
        ), "Expected self length to be less than self.size."
        if not self.data:
            example = step
            self.data = {
                k: np.empty((self.size, *v.shape), v.dtype) for k, v in example.items()
            }
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1
        # if self.length == self.size:
        #   [x.setflags(write=False) for x in self.data.values()]

    def update(self, index: Any, length: Any, mapping: Any) -> None:
        """Update state.

        Args:
            index: Position of the requested element.
            length: Length value.
            mapping: Mapping value.
        """
        assert 0 <= index <= self.length, (index, self.length)
        assert 0 <= index + length <= self.length, (index, length, self.length)
        assert self.data is not None, "Cannot update a chunk before its first append."
        for key, value in mapping.items():
            self.data[key][index : index + length] = value

    def slice(self, index: Any, length: Any) -> Any:
        """Handle slice.

        Args:
            index: Position of the requested element.
            length: Length value.

        Returns:
            Result of the operation.
        """
        assert (
            0 <= index and index + length <= self.length
        ), "Expected all parts of the 0 <= index and index + length <= self.length invariant to hold."
        assert self.data is not None, "Cannot slice a chunk before its first append."
        return {k: v[index : index + length] for k, v in self.data.items()}

    @elements.timer.section("chunk_save")
    def save(self, directory: Any, log: bool = False) -> None:
        """Save state.

        Args:
            directory: Directory value.
            log: Log value.
        """
        assert not self.saved, "Expected self saved to be false or empty."
        assert self.data is not None, "Cannot save a chunk before its first append."
        self.saved = True
        filename = elements.Path(directory) / self.filename
        data = {k: v[: self.length] for k, v in self.data.items()}
        with io.BytesIO() as stream:
            np.savez_compressed(stream, **data)
            stream.seek(0)
            filename.write(stream.read(), mode="wb")
        log and print(f"Saved chunk: {filename.name}")

    @classmethod
    def load(cls, filename: Any, error: str = "raise") -> Any:
        """Load state.

        Args:
            filename: Filename value.
            error: Error value.

        Returns:
            Result of the operation.
        """
        assert error in (
            "raise",
            "none",
        ), 'Expected error to be present in ("raise", "none").'
        time, uuid, succ, length = filename.stem.split("-")
        length = int(length)
        try:
            with elements.Path(filename).open("rb") as f:
                data = np.load(f)
                data = {k: data[k] for k in data.keys()}
        except Exception:
            tb = "".join(traceback.format_exception(sys.exception()))
            print(f"Error loading chunk {filename}:\n{tb}")
            if error == "raise":
                raise
            else:
                return None
        chunk = cls(length)
        chunk.time = time
        chunk.uuid = elements.UUID(uuid)
        chunk.succ = elements.UUID(succ)
        chunk.length = length
        chunk.data = data
        chunk.saved = True
        return chunk
