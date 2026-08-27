"""Provide pinpad functionality."""

from __future__ import annotations
from typing import Any


import collections

import elements
import embodied
import numpy as np


class PinPad(embodied.Env):
    """Represent pin pad."""

    COLORS = {
        "1": (255, 0, 0),
        "2": (0, 255, 0),
        "3": (0, 0, 255),
        "4": (255, 255, 0),
        "5": (255, 0, 255),
        "6": (0, 255, 255),
        "7": (128, 0, 128),
        "8": (0, 128, 128),
    }

    def __init__(self, task: Any, length: int = 10000) -> None:
        """Initialize the pin pad.

        Args:
            task: Task value.
            length: Length value.
        """
        assert length > 0, "Expected length to be greater than 0."
        layout = {
            "three": LAYOUT_THREE,
            "four": LAYOUT_FOUR,
            "five": LAYOUT_FIVE,
            "six": LAYOUT_SIX,
            "seven": LAYOUT_SEVEN,
            "eight": LAYOUT_EIGHT,
        }[task]
        self.layout = np.array([list(line) for line in layout.split("\n")]).T
        assert self.layout.shape == (16, 14), self.layout.shape
        self.length = length
        self.random = np.random.RandomState()
        self.pads = set(self.layout.flatten().tolist()) - set("* #\n")
        self.target = tuple(sorted(self.pads))
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != "#":
                self.spawns.append((x, y))
        print(f'Created PinPad env with sequence: {"->".join(self.target)}')
        self.sequence = collections.deque(maxlen=len(self.target))
        self.player = None
        self.steps = None
        self.done = None
        self.countdown = None

    @property
    def act_space(self) -> dict[Any, Any]:
        """Handle act space.

        Returns:
            Result of the operation.
        """
        return {
            "action": elements.Space(np.int32, (), 0, 5),
            "reset": elements.Space(bool),
        }

    @property
    def obs_space(self) -> dict[Any, Any]:
        """Handle observation space.

        Returns:
            Result of the operation.
        """
        return {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    def step(self, action: Any) -> Any:
        """Advance state.

        Args:
            action: Action value.

        Returns:
            Result of the operation.
        """
        if self.done or action["reset"]:
            self.player = self.spawns[self.random.randint(len(self.spawns))]
            self.sequence.clear()
            self.steps = 0
            self.done = False
            self.countdown = 0
            return self._obs(reward=0.0, is_first=True)
        if self.countdown:
            self.countdown -= 1
            if self.countdown == 0:
                self.player = self.spawns[self.random.randint(len(self.spawns))]
                self.sequence.clear()
        reward = 0.0
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action["action"]]
        x = np.clip(
            self.player[0] + move[0], 0, 15  # pyright: ignore[reportOptionalSubscript]
        )
        y = np.clip(
            self.player[1] + move[1], 0, 13  # pyright: ignore[reportOptionalSubscript]
        )
        tile = self.layout[x][y]
        if tile != "#":
            self.player = (x, y)
        if tile in self.pads:
            if not self.sequence or self.sequence[-1] != tile:
                self.sequence.append(tile)
        if tuple(self.sequence) == self.target and not self.countdown:
            reward += 10.0
            self.countdown = 10
        self.steps += 1  # pyright: ignore[reportOperatorIssue]
        self.done = self.done or (self.steps >= self.length)
        return self._obs(reward=reward, is_last=self.done)

    def _obs(
        self,
        reward: Any,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> Any:
        return dict(
            image=self._render(),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def _render(self) -> Any:
        grid = np.zeros((16, 16, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if self.countdown:
            grid[:] = (223, 255, 223)
        current = self.layout[
            self.player[0]  # pyright: ignore[reportOptionalSubscript]
        ][
            self.player[1]  # pyright: ignore[reportOptionalSubscript]
        ]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == "#":
                grid[x, y] = (192, 192, 192)
            elif char in self.pads:
                color = np.array(self.COLORS[char])
                color = color if char == current else (10 * color + 90 * white) / 100
                grid[x, y] = color
        grid[self.player] = (0, 0, 0)
        grid[:, -2:] = (192, 192, 192)
        for i, char in enumerate(self.sequence):
            grid[2 * i + 1, -2] = self.COLORS[char]
        image = np.repeat(np.repeat(grid, 4, 0), 4, 1)
        return image.transpose((1, 0, 2))


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip("\n")

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip("\n")

LAYOUT_FIVE = """
################
#          4444#
#111       4444#
#111       4444#
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333       2222#
#333       2222#
#          2222#
################
""".strip("\n")

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#444        222#
#444        222#
#444        222#
################
""".strip("\n")

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip("\n")

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip("\n")
