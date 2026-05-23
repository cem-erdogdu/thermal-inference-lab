"""Maze environment and evaluation for NEAT agents."""

from thermal_inference_lab.neat.maze.environment import GridMaze, MazeAgent
from thermal_inference_lab.neat.maze.evaluation import (
    evaluate_genome,
    evaluate_population,
    MAZE_SUITE,
)

__all__ = [
    "GridMaze",
    "MazeAgent",
    "evaluate_genome",
    "evaluate_population",
    "MAZE_SUITE",
]