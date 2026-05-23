"""Grid maze environment for NEAT agents.

A maze is a 2-D integer grid:

- ``0`` = empty (passable)
- ``1`` = wall
- ``2`` = start
- ``3`` = goal

The agent occupies a cell and can move in four cardinal directions
(up, down, left, right).  Wall collisions are penalised and the
agent stays in place.

Network interface
-----------------
**Inputs (8):**

0. Wall sensor — up    (1.0 if wall, 0.0 otherwise)
1. Wall sensor — down
2. Wall sensor — left
3. Wall sensor — right
4. Goal direction dx (normalised)
5. Goal direction dy (normalised)
6. Distance to goal (normalised by maze diagonal)
7. Fraction of time remaining

**Outputs (4):**

0. Move up
1. Move down
2. Move left
3. Move right

The agent picks the action with the highest activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CELL_EMPTY = 0
CELL_WALL = 1
CELL_START = 2
CELL_GOAL = 3

# Cardinal directions: (row_delta, col_delta)
DIRECTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
N_INPUTS = 8
N_OUTPUTS = 4


@dataclass
class StepResult:
    """One step of the agent in the maze."""
    position: Tuple[int, int]
    action: int
    hit_wall: bool
    reached_goal: bool


class GridMaze:
    """An immutable grid maze.

    Parameters
    ----------
    grid : np.ndarray
        2-D integer array with cell types.
    name : str
        Optional human-readable label.
    """

    def __init__(self, grid: np.ndarray, name: str = "maze") -> None:
        self.grid = np.asarray(grid, dtype=np.int32)
        self.name = name
        self._start: Optional[Tuple[int, int]] = None
        self._goal: Optional[Tuple[int, int]] = None
        self._scan()

    def _scan(self) -> None:
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                if self.grid[r, c] == CELL_START:
                    self._start = (r, c)
                elif self.grid[r, c] == CELL_GOAL:
                    self._goal = (r, c)
        if self._start is None:
            raise ValueError(f"Maze {self.name!r} has no start cell (2)")
        if self._goal is None:
            raise ValueError(f"Maze {self.name!r} has no goal cell (3)")

    @property
    def start(self) -> Tuple[int, int]:
        assert self._start is not None
        return self._start

    @property
    def goal(self) -> Tuple[int, int]:
        assert self._goal is not None
        return self._goal

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def diagonal(self) -> float:
        return float(np.sqrt(self.rows ** 2 + self.cols ** 2))

    def is_wall(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True  # out-of-bounds = wall
        return int(self.grid[r, c]) == CELL_WALL

    def is_passable(self, r: int, c: int) -> bool:
        return not self.is_wall(r, c)

    def manhattan_distance(self, pos: Tuple[int, int]) -> int:
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def __repr__(self) -> str:
        return f"GridMaze({self.name!r}, {self.rows}x{self.cols})"


class MazeAgent:
    """An agent running through a :class:`GridMaze`.

    After construction call :meth:`step` with the chosen action index
    (0–3) and inspect the returned :class:`StepResult`.
    """

    def __init__(self, maze: GridMaze, max_steps: int = 100) -> None:
        self.maze = maze
        self.max_steps = max_steps
        self.position: Tuple[int, int] = maze.start
        self.step_count: int = 0
        self.wall_hits: int = 0
        self.reached_goal: bool = False
        self.path: List[Tuple[int, int]] = [maze.start]
        self._visited: Dict[Tuple[int, int], int] = {maze.start: 1}

    # ------------------------------------------------------- sensor input

    def get_inputs(self) -> np.ndarray:
        """Return the 8-element input vector for the neural network."""
        r, c = self.position
        gr, gc = self.maze.goal

        # Wall sensors (up, down, left, right).
        wall_up = 1.0 if self.maze.is_wall(r - 1, c) else 0.0
        wall_down = 1.0 if self.maze.is_wall(r + 1, c) else 0.0
        wall_left = 1.0 if self.maze.is_wall(r, c - 1) else 0.0
        wall_right = 1.0 if self.maze.is_wall(r, c + 1) else 0.0

        # Goal direction (normalised).
        dy = float(gr - r)
        dx = float(gc - c)
        dist = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        norm_dx = dx / dist
        norm_dy = dy / dist

        # Normalised distance.
        norm_dist = dist / max(self.maze.diagonal, 1e-6)

        # Time remaining fraction.
        time_frac = max(0.0, 1.0 - self.step_count / max(self.max_steps, 1))

        return np.array([
            wall_up, wall_down, wall_left, wall_right,
            norm_dx, norm_dy, norm_dist, time_frac,
        ], dtype=np.float64)

    # ------------------------------------------------------- step

    def step(self, action: int) -> StepResult:
        """Execute *action* (0-3) and return the result."""
        dr, dc = DIRECTIONS[action]
        nr, nc = self.position[0] + dr, self.position[1] + dc
        hit_wall = False
        if self.maze.is_wall(nr, nc):
            hit_wall = True
            self.wall_hits += 1
            # Stay in place.
        else:
            self.position = (nr, nc)

        self.step_count += 1
        self.path.append(self.position)
        self._visited[self.position] = self._visited.get(self.position, 0) + 1

        goal_reached = (self.position == self.maze.goal)
        if goal_reached:
            self.reached_goal = True

        return StepResult(
            position=self.position,
            action=action,
            hit_wall=hit_wall,
            reached_goal=goal_reached,
        )

    # ---------------------------------------------------- diagnostics

    @property
    def unique_cells_visited(self) -> int:
        return len(self._visited)

    @property
    def revisit_count(self) -> int:
        """Total extra visits to already-visited cells."""
        return sum(max(0, v - 1) for v in self._visited.values())

    @property
    def done(self) -> bool:
        return self.reached_goal or self.step_count >= self.max_steps


# --------------------------------------------------------- built-in mazes

def _maze_simple() -> GridMaze:
    """5×5 corridor maze."""
    return GridMaze(np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.int32), name="simple")


def _maze_corridor() -> GridMaze:
    """7×7 L-shaped corridor."""
    return GridMaze(np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.int32), name="corridor")


def _maze_open() -> GridMaze:
    """6×6 open room with a central obstacle."""
    return GridMaze(np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int32), name="open")


def _maze_zigzag() -> GridMaze:
    """7×7 zigzag maze requiring multiple turns."""
    return GridMaze(np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1],
        [1, 3, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.int32), name="zigzag")


BUILT_IN_MAZES = {
    "simple": _maze_simple,
    "corridor": _maze_corridor,
    "open": _maze_open,
    "zigzag": _maze_zigzag,
}


__all__ = [
    "CELL_EMPTY", "CELL_WALL", "CELL_START", "CELL_GOAL",
    "DIRECTIONS", "N_INPUTS", "N_OUTPUTS",
    "StepResult", "GridMaze", "MazeAgent",
    "BUILT_IN_MAZES",
]