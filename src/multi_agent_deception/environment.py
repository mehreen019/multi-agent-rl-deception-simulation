"""PettingZoo environment modelling a cooperative task-focused hidden role scenario."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

AgentID = str


class SimpleHiddenRoleParallelEnv(ParallelEnv):
    """Simple hidden role parallel environment focused on cooperative task completion."""

    metadata = {"render_modes": ["human"], "name": "simple_hidden_role_v0"}

    def __init__(
        self,
        *,
        grid_size: int = 8,
        num_agents: int = 4,
        tasks_per_agent: int = 3,
        max_steps: int = 200,
        task_duration: int = 3,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.tasks_per_agent = tasks_per_agent
        self.max_steps = max_steps
        self.task_duration = max(1, task_duration)
        self._step_count = 0
        self._rng = np.random.default_rng(seed)
        self.render_mode: str | None = None

        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents: List[AgentID] = []

        self._action_space = spaces.Discrete(5)  # stay, up, down, left, right
        obs_low = np.zeros(2 + self.tasks_per_agent * 3, dtype=np.int32)
        obs_high = np.zeros_like(obs_low, dtype=np.int32)
        obs_high[:2] = self.grid_size
        for idx in range(self.tasks_per_agent):
            base = 2 + idx * 3
            obs_high[base] = self.grid_size
            obs_high[base + 1] = self.grid_size
            obs_high[base + 2] = self.task_duration
        self._observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._agent_positions: Dict[AgentID, np.ndarray] = {}
        self._agent_tasks: Dict[AgentID, np.ndarray] = {}
        self._completed_tasks: Dict[AgentID, np.ndarray] = {}
        self._task_progress: Dict[AgentID, np.ndarray] = {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        available_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self._rng.shuffle(available_cells)

        # Assign distinct starting positions
        self._agent_positions = {}
        for agent in self.agents:
            self._agent_positions[agent] = np.array(available_cells.pop(), dtype=np.int32)

        # Assign tasks (avoid starting cell reuse when possible)
        self._agent_tasks = {}
        self._completed_tasks = {}
        self._task_progress = {}
        for agent in self.agents:
            task_cells: List[np.ndarray] = []
            while len(task_cells) < self.tasks_per_agent:
                if not available_cells:
                    available_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
                    self._rng.shuffle(available_cells)
                candidate = np.array(available_cells.pop(), dtype=np.int32)
                if not np.array_equal(candidate, self._agent_positions[agent]):
                    task_cells.append(candidate)
            self._agent_tasks[agent] = np.stack(task_cells, axis=0)
            self._completed_tasks[agent] = np.zeros(self.tasks_per_agent, dtype=bool)
            self._task_progress[agent] = np.zeros(self.tasks_per_agent, dtype=np.int32)

        observations = {agent: self._build_observation(agent) for agent in self.agents}
        infos = {agent: self._build_info(agent) for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[AgentID, int]):
        if not self.agents:
            raise RuntimeError("Step called on terminated environment. Call reset().")

        self._step_count += 1

        moves = {
            0: np.array([0, 0], dtype=np.int32),   # stay
            1: np.array([-1, 0], dtype=np.int32),  # up
            2: np.array([1, 0], dtype=np.int32),   # down
            3: np.array([0, -1], dtype=np.int32),  # left
            4: np.array([0, 1], dtype=np.int32),   # right
        }

        rewards: Dict[AgentID, float] = {agent: -0.01 for agent in self.agents}
        terminations: Dict[AgentID, bool] = {agent: False for agent in self.agents}
        truncations: Dict[AgentID, bool] = {agent: False for agent in self.agents}
        infos: Dict[AgentID, dict] = {}

        for agent in self.agents:
            action = int(actions.get(agent, 0))
            move = moves.get(action, moves[0])
            new_position = self._agent_positions[agent] + move
            new_position = np.clip(new_position, 0, self.grid_size - 1)
            self._agent_positions[agent] = new_position

            # Track task progress and completion
            task_positions = self._agent_tasks[agent]
            completion_mask = self._completed_tasks[agent]
            progress = self._task_progress[agent]
            unfinished_indices = np.where(~completion_mask)[0]
            if unfinished_indices.size > 0:
                for idx in unfinished_indices:
                    if np.array_equal(task_positions[idx], new_position):
                        progress[idx] = min(progress[idx] + 1, self.task_duration)
                        if progress[idx] >= self.task_duration and not completion_mask[idx]:
                            completion_mask[idx] = True
                            rewards[agent] += 1.0
                    else:
                        progress[idx] = 0
            # Cap progress for completed tasks
            completed_indices = np.where(completion_mask)[0]
            if completed_indices.size > 0:
                progress[completed_indices] = self.task_duration

            if self._completed_tasks[agent].all():
                rewards[agent] += 0.5  # completion bonus

        all_tasks_done = all(self._completed_tasks[agent].all() for agent in self.agents)
        hit_step_limit = self._step_count >= self.max_steps

        if all_tasks_done:
            for agent in self.agents:
                terminations[agent] = True
        if hit_step_limit and not all_tasks_done:
            for agent in self.agents:
                truncations[agent] = True

        observations = {agent: self._build_observation(agent) for agent in self.agents}
        infos = {agent: self._build_info(agent) for agent in self.agents}

        if all(terminations[agent] or truncations[agent] for agent in self.agents):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if not self._agent_positions:
            print("Environment not initialised. Call reset() first.")
            return
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U3")
        for agent_id, pos in self._agent_positions.items():
            x, y = pos.tolist()
            grid[x, y] = agent_id.split("_")[1]

        for agent_id, tasks in self._agent_tasks.items():
            for idx, task in enumerate(tasks):
                marker = f"T{idx}"
                if self._completed_tasks[agent_id][idx]:
                    marker = "✔"
                x, y = task.tolist()
                # Avoid overwriting agent marker
                if grid[x, y] == ".":
                    grid[x, y] = marker

        print("\n".join(" ".join(row) for row in grid))

    def _build_observation(self, agent: AgentID) -> np.ndarray:
        position = self._agent_positions[agent]
        task_positions = self._agent_tasks[agent].copy()
        for idx, completed in enumerate(self._completed_tasks[agent]):
            if completed:
                task_positions[idx] = np.array([self.grid_size, self.grid_size], dtype=np.int32)
        progress = self._task_progress[agent].copy()
        obs_components: List[int] = list(position.astype(np.int32))
        for idx in range(self.tasks_per_agent):
            task_pos = task_positions[idx]
            obs_components.extend(task_pos.astype(np.int32))
            obs_components.append(int(progress[idx]))
        obs = np.array(obs_components, dtype=np.int32)
        return obs.astype(np.int32)

    def _build_info(self, agent: AgentID) -> dict:
        return {
            "step": self._step_count,
            "completed_tasks": int(self._completed_tasks[agent].sum()),
            "total_tasks": self.tasks_per_agent,
            "task_progress": self._task_progress[agent].copy(),
        }

    def observation_space(self, agent: AgentID):
        return self._observation_space

    def action_space(self, agent: AgentID):
        return self._action_space

    @property
    def num_agents(self) -> int:
        """Return number of agents."""
        return self._num_agents


def parallel_env(**kwargs) -> SimpleHiddenRoleParallelEnv:
    """Factory returning the parallel environment."""
    return SimpleHiddenRoleParallelEnv(**kwargs)


def raw_env(**kwargs) -> ParallelEnv:
    """Alias for compatibility with PettingZoo registration helpers."""
    return parallel_env(**kwargs)


def env(**kwargs) -> ParallelEnv:
    """Legacy alias used by PettingZoo."""
    return raw_env(**kwargs)
