"""Interactive viewer for the multi-agent hidden role environment."""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pygame  # noqa: F401  # ensure dependency is installed before opening window
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pygame is required for the GUI. Install it with `pip install pygame` "
        "or `pip install -r requirements.txt`."
    ) from exc

from src.multi_agent_deception.gui import run_gui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a simple GUI for the hidden role environment.")
    parser.add_argument("--grid-size", type=int, default=8, help="Size of the square grid.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents in the environment.")
    parser.add_argument("--tasks-per-agent", type=int, default=3, help="Number of tasks assigned to each agent.")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode truncation horizon.")
    parser.add_argument("--task-duration", type=int, default=3, help="Steps required on a task tile to complete it.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed passed to the environment.")
    parser.add_argument("--fps", type=int, default=8, help="Frames (environment steps) rendered per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_gui(
            grid_size=args.grid_size,
            num_agents=args.num_agents,
            tasks_per_agent=args.tasks_per_agent,
            max_steps=args.max_steps,
            task_duration=args.task_duration,
            seed=args.seed,
            fps=args.fps,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
