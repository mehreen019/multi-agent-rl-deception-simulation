"""Pygame-based visualiser for the hidden role environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pygame

from .environment import AgentID, SimpleHiddenRoleParallelEnv, parallel_env

CELL_SIZE = 64
INFO_PANEL_HEIGHT = 96
FPS = 8

AGENT_COLORS = [
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]

ACTION_KEYS = [
    (pygame.K_UP, 1),
    (pygame.K_DOWN, 2),
    (pygame.K_LEFT, 3),
    (pygame.K_RIGHT, 4),
]


@dataclass
class GUIState:
    env: SimpleHiddenRoleParallelEnv
    last_rewards: Dict[AgentID, float]
    last_infos: Dict[AgentID, dict]
    running: bool = True
    paused: bool = False


def _init_pygame(grid_size: int) -> pygame.Surface:
    pygame.init()
    width = grid_size * CELL_SIZE
    height = grid_size * CELL_SIZE + INFO_PANEL_HEIGHT
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Multi-Agent Hidden Role – Task Visualiser")
    return surface


def _get_agent_color(agent_idx: int) -> tuple[int, int, int]:
    return AGENT_COLORS[agent_idx % len(AGENT_COLORS)]


def _draw_grid(surface: pygame.Surface, env: SimpleHiddenRoleParallelEnv) -> None:
    surface.fill((24, 24, 24))
    grid_size = env.grid_size
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, (44, 62, 80), rect, border_radius=6)
            pygame.draw.rect(surface, (70, 80, 90), rect, width=2, border_radius=6)


def _draw_tasks(surface: pygame.Surface, env: SimpleHiddenRoleParallelEnv) -> None:
    font = pygame.font.SysFont("arial", 16)
    for idx, agent in enumerate(env.possible_agents):
        task_positions = env._agent_tasks.get(agent)
        if task_positions is None:
            continue
        completed = env._completed_tasks[agent]
        progress = env._task_progress[agent]
        color = _get_agent_color(idx)
        for task_idx, pos in enumerate(task_positions):
            x, y = pos.tolist()
            rect = pygame.Rect(y * CELL_SIZE + 8, x * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
            pygame.draw.rect(surface, color, rect, width=2, border_radius=4)
            if completed[task_idx]:
                pygame.draw.rect(surface, color, rect, border_radius=4)
                check_font = pygame.font.SysFont("arial", 18, bold=True)
                check = check_font.render("✔", True, (0, 0, 0))
                surface.blit(check, (rect.x + rect.width / 2 - 6, rect.y + rect.height / 2 - 12))
            elif progress[task_idx] > 0:
                pct = progress[task_idx] / env.task_duration
                bar_width = int((CELL_SIZE - 16) * pct)
                bar_rect = pygame.Rect(rect.x, rect.bottom - 6, bar_width, 6)
                pygame.draw.rect(surface, color, bar_rect, border_radius=3)
            label = font.render(f"{idx}:{task_idx}", True, color)
            surface.blit(label, (rect.x + 2, rect.y + 2))


def _draw_agents(surface: pygame.Surface, env: SimpleHiddenRoleParallelEnv) -> None:
    for idx, agent in enumerate(env.possible_agents):
        if agent not in env._agent_positions:
            continue
        pos = env._agent_positions[agent]
        x, y = pos.tolist()
        center = (int(y * CELL_SIZE + CELL_SIZE / 2), int(x * CELL_SIZE + CELL_SIZE / 2))
        pygame.draw.circle(surface, _get_agent_color(idx), center, CELL_SIZE // 3)
        font = pygame.font.SysFont("arial", 16, bold=True)
        label = font.render(str(idx), True, (0, 0, 0))
        surface.blit(label, (center[0] - 6, center[1] - 8))


def _draw_panel(surface: pygame.Surface, env: SimpleHiddenRoleParallelEnv, state: GUIState) -> None:
    width = env.grid_size * CELL_SIZE
    panel_rect = pygame.Rect(0, env.grid_size * CELL_SIZE, width, INFO_PANEL_HEIGHT)
    pygame.draw.rect(surface, (33, 33, 33), panel_rect)

    font = pygame.font.SysFont("arial", 18)
    small_font = pygame.font.SysFont("arial", 14)

    instructions = "Arrow keys move agent 0. Space: Stay. R: Reset. P: Pause."
    surface.blit(small_font.render(instructions, True, (236, 240, 241)), (12, panel_rect.y + 10))

    if not env.agents:
        msg = "Episode finished. Press R to reset."
        surface.blit(font.render(msg, True, (231, 76, 60)), (12, panel_rect.y + 38))
    else:
        step_text = f"Step: {state.env._step_count}/{state.env.max_steps}"
        surface.blit(font.render(step_text, True, (236, 240, 241)), (12, panel_rect.y + 36))
        for idx, agent in enumerate(env.possible_agents):
            if state.last_rewards and agent in state.last_rewards:
                reward = state.last_rewards[agent]
                completed = state.last_infos.get(agent, {}).get("completed_tasks", 0)
                total = state.last_infos.get(agent, {}).get("total_tasks", 0)
                text = f"A{idx} reward {reward:+.2f} | tasks {completed}/{total}"
                surface.blit(
                    small_font.render(text, True, _get_agent_color(idx)),
                    (12, panel_rect.y + 60 + idx * 18),
                )


def _compute_actions(env: SimpleHiddenRoleParallelEnv) -> Dict[AgentID, int]:
    actions: Dict[AgentID, int] = {}
    pressed = pygame.key.get_pressed()
    action = 0
    for key, mapped in ACTION_KEYS:
        if pressed[key]:
            action = mapped
            break
    if pressed[pygame.K_SPACE]:
        action = 0
    if env.agents:
        actions[env.agents[0]] = action
        for agent in env.agents[1:]:
            actions[agent] = env.action_space(agent).sample()
    return actions


def run_gui(
    *,
    grid_size: int = 8,
    num_agents: int = 4,
    tasks_per_agent: int = 3,
    max_steps: int = 200,
    task_duration: int = 3,
    seed: Optional[int] = None,
    fps: int = FPS,
) -> None:
    """Launch an interactive Pygame window for the environment."""
    env = parallel_env(
        grid_size=grid_size,
        num_agents=num_agents,
        tasks_per_agent=tasks_per_agent,
        max_steps=max_steps,
        task_duration=task_duration,
        seed=seed,
    )
    env.reset(seed=seed)
    surface = _init_pygame(grid_size)
    clock = pygame.time.Clock()
    state = GUIState(env=env, last_rewards={}, last_infos={})

    while state.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset(seed=seed)
                    state.last_rewards = {}
                    state.last_infos = {}
                if event.key == pygame.K_p:
                    state.paused = not state.paused

        if not state.paused and env.agents:
            actions = _compute_actions(env)
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
        clock.tick(fps)

    pygame.quit()
