"""Interactive viewer showing agents with simple heuristic behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pygame
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pygame is required for the GUI. Install it with `pip install pygame` "
        "or `pip install -r requirements.txt`."
    ) from exc

import numpy as np
from src.multi_agent_deception.gui import (
    GUIState,
    FPS,
    _draw_agents,
    _draw_grid,
    _draw_panel,
    _draw_tasks,
    _init_pygame,
)
from src.multi_agent_deception.environment import parallel_env


def heuristic_action(env, agent):
    """Simple heuristic: move towards the nearest incomplete task."""
    agent_pos = env._agent_positions[agent]
    tasks = env._agent_tasks[agent]
    completed = env._completed_tasks[agent]

    # Find nearest incomplete task
    min_dist = float('inf')
    target = None
    for idx, task_pos in enumerate(tasks):
        if not completed[idx]:
            dist = abs(task_pos[0] - agent_pos[0]) + abs(task_pos[1] - agent_pos[1])
            if dist < min_dist:
                min_dist = dist
                target = task_pos

    if target is None:
        return 0  # Stay if all tasks complete

    # Move towards target
    dx = target[0] - agent_pos[0]
    dy = target[1] - agent_pos[1]

    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 1  # down or up
    elif dy != 0:
        return 4 if dy > 0 else 3  # right or left
    else:
        return 0  # Stay (we're at the task)


def main():
    parser = argparse.ArgumentParser("Demo GUI with heuristic agents.")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--tasks-per-agent", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--task-duration", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--random", action="store_true", help="Use random actions instead of heuristic")
    args = parser.parse_args()

    env = parallel_env(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        tasks_per_agent=args.tasks_per_agent,
        max_steps=args.max_steps,
        task_duration=args.task_duration,
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    surface = _init_pygame(args.grid_size)
    clock = pygame.time.Clock()
    state = GUIState(env=env, last_rewards={}, last_infos={}, running=True, paused=False)

    while state.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset(seed=args.seed)
                    state.last_rewards = {}
                    state.last_infos = {}
                if event.key == pygame.K_p:
                    state.paused = not state.paused

        if not state.paused and env.agents:
            if args.random:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            else:
                actions = {agent: heuristic_action(env, agent) for agent in env.agents}

            observations, rewards, terminations, truncations, infos = env.step(actions)
            state.last_rewards = rewards
            state.last_infos = infos

            if not env.agents:
                state.last_rewards = rewards
                state.last_infos = infos

        _draw_grid(surface, env)
        _draw_tasks(surface, env)
        _draw_agents(surface, env)
        _draw_panel(surface, env, state)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
