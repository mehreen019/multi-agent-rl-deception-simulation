"""Tier 2 spatial environment with grid-based game mechanics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
from gymnasium import spaces

from .base import DeceptionGameEnvironment
from .models import (
    GameState,
    GameAction,
    GameEvent,
    AgentObservation,
    ScenarioConfig,
    Agent,
    AgentRole,
    ActionType,
    GameStatus,
)

AgentID = str


class Tier2Environment(DeceptionGameEnvironment):
    """
    Spatial game environment with grid-based task completion.

    This is the refactored implementation of SimpleHiddenRoleParallelEnv.
    Preserves all original behavior while providing the abstract interface
    for deception game environments.

    Features:
        - 2D grid world with configurable size
        - Multi-agent support with hidden roles
        - Task locations and completion tracking
        - Partial observability (proximity-based observation radius)
        - Deterministic behavior given seed
        - 100% backward compatible with original implementation
    """

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
        """
        Initialize Tier 2 environment.

        Args:
            grid_size: Size of square grid (grid_size x grid_size)
            num_agents: Number of agents in game
            tasks_per_agent: Tasks assigned to each agent
            max_steps: Maximum game duration (ticks)
            task_duration: Steps required to complete each task
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.tasks_per_agent = tasks_per_agent
        self.max_steps = max_steps
        self.task_duration = max(1, task_duration)
        self._step_count = 0
        self._rng = np.random.default_rng(seed)
        self.render_mode: str | None = None

        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agents: List[AgentID] = []

        # PettingZoo compatibility
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

        # Internal state
        self._agent_positions: Dict[AgentID, np.ndarray] = {}
        self._agent_tasks: Dict[AgentID, np.ndarray] = {}
        self._completed_tasks: Dict[AgentID, np.ndarray] = {}
        self._task_progress: Dict[AgentID, np.ndarray] = {}

        # GameState tracking
        self._game_state: Optional[GameState] = None
        self._scenario_config: Optional[ScenarioConfig] = None
        self._game_id = str(uuid.uuid4())

    def reset(self, scenario_config: Optional[ScenarioConfig] = None, seed: int | None = None) -> GameState:
        """
        Reset environment to initial state.

        Args:
            scenario_config: Optional ScenarioConfig (if not provided, uses default)
            seed: Random seed for reproducibility

        Returns:
            GameState: Initial game state
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Create scenario config if not provided
        if scenario_config is None:
            scenario_config = ScenarioConfig(
                scenario_id="default",
                tier=2,
                num_agents=self.num_agents,
                num_imposters=1,
                grid_size=self.grid_size,
                tasks_per_agent=self.tasks_per_agent,
                max_ticks=self.max_steps,
                task_duration=self.task_duration,
            )

        self._scenario_config = scenario_config
        self._step_count = 0
        self.agents = self.possible_agents[:]

        # Initialize agent positions
        available_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self._rng.shuffle(available_cells)

        self._agent_positions = {}
        for agent in self.agents:
            self._agent_positions[agent] = np.array(available_cells.pop(), dtype=np.int32)

        # Initialize tasks
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

        # Initialize GameState
        self._game_state = GameState(
            game_id=self._game_id,
            scenario_config=scenario_config,
            tick=0,
            agents={agent_id: Agent(agent_id=agent_id) for agent_id in self.agents},
            active_agents=set(self.agents),
            roles={agent_id: AgentRole.UNKNOWN for agent_id in self.agents},
            game_status=GameStatus.RUNNING,
        )

        return self._game_state

    def step(self, game_state: GameState, actions: Dict[AgentID, GameAction]) -> Tuple[GameState, Dict[str, GameEvent]]:
        """
        Execute one game tick with agent actions.

        Args:
            game_state: Current game state
            actions: Dict mapping agent_id -> GameAction

        Returns:
            Tuple of (new_game_state, events_dict)
        """
        if not self.agents:
            raise RuntimeError("Step called on terminated environment. Call reset().")

        self._step_count += 1

        # For backward compatibility, convert GameAction to PettingZoo-style actions
        pettingzoo_actions = {}
        for agent_id, action in actions.items():
            if isinstance(action, GameAction):
                # Extract movement from GameAction parameters
                if action.action_type == ActionType.MOVE:
                    pettingzoo_actions[agent_id] = action.parameters.get("direction", 0)
                else:
                    pettingzoo_actions[agent_id] = 0  # Default to stay
            else:
                pettingzoo_actions[agent_id] = action if isinstance(action, int) else 0

        # Execute movement and task completion (original logic)
        moves = {
            0: np.array([0, 0], dtype=np.int32),   # stay
            1: np.array([-1, 0], dtype=np.int32),  # up
            2: np.array([1, 0], dtype=np.int32),   # down
            3: np.array([0, -1], dtype=np.int32),  # left
            4: np.array([0, 1], dtype=np.int32),   # right
        }

        events_dict: Dict[str, GameEvent] = {}

        for agent in self.agents:
            action_obj = actions.get(agent)
            action_val = pettingzoo_actions.get(agent, 0)
            move = moves.get(action_val, moves[0])
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
                    else:
                        progress[idx] = 0

            # Cap progress for completed tasks
            completed_indices = np.where(completion_mask)[0]
            if completed_indices.size > 0:
                progress[completed_indices] = self.task_duration

            # Create GameEvent
            event = GameEvent(
                tick=self._step_count,
                agent_id=agent,
                action=action_obj if isinstance(action_obj, GameAction) else GameAction(agent_id=agent, action_type=ActionType.MOVE),
                event_type="move_and_task_progress",
                changes={
                    "position": new_position.tolist(),
                    "completed_tasks": int(completion_mask.sum()),
                },
            )
            events_dict[agent] = event

        # Check win conditions
        all_tasks_done = all(self._completed_tasks[agent].all() for agent in self.agents)
        hit_step_limit = self._step_count >= self.max_steps

        if all_tasks_done:
            game_status = GameStatus.COMPLETED
            winner = "crewmates"
        elif hit_step_limit:
            game_status = GameStatus.COMPLETED
            winner = "imposters"
        else:
            game_status = GameStatus.RUNNING
            winner = None

        # Update GameState
        self._game_state = GameState(
            game_id=self._game_id,
            scenario_config=self._scenario_config,
            tick=self._step_count,
            agents=self._game_state.agents,
            active_agents=set(self.agents) if self.agents else set(),
            roles=self._game_state.roles,
            game_status=game_status,
            winner=winner,
        )

        # Clear agents list if game is over
        if all_tasks_done or hit_step_limit:
            self.agents = []

        return self._game_state, events_dict

    def get_observations(self, game_state: GameState) -> Dict[str, AgentObservation]:
        """
        Get partial observability views for all agents.

        For Tier 2: Agents see other agents within observation radius,
        see their own position and tasks, but don't know roles of others.

        Args:
            game_state: Current game state

        Returns:
            Dict mapping agent_id -> AgentObservation
        """
        observations = {}
        observation_radius = game_state.scenario_config.observation_radius

        for agent_id in game_state.active_agents:
            visible_agents = []
            visible_positions = {}

            agent_pos = self._agent_positions.get(agent_id)
            if agent_pos is None:
                continue

            # Find agents within observation radius
            for other_agent_id in game_state.active_agents:
                if other_agent_id == agent_id:
                    continue

                other_pos = self._agent_positions.get(other_agent_id)
                if other_pos is None:
                    continue

                # Euclidean distance
                distance = float(np.linalg.norm(agent_pos - other_pos))
                if distance <= observation_radius:
                    visible_agents.append(other_agent_id)
                    visible_positions[other_agent_id] = tuple(other_pos.tolist())

            # Build observation
            obs = AgentObservation(
                agent_id=agent_id,
                tick=game_state.tick,
                visible_agents=visible_agents,
                visible_positions=visible_positions,
                visible_roles={},  # Roles hidden unless explicitly revealed
                discussion_history=[],  # No discussion in Tier 2 (pure spatial)
                known_roles={agent_id: AgentRole.UNKNOWN},  # Don't know own role initially
            )
            observations[agent_id] = obs

        return observations

    def is_terminal(self, game_state: GameState) -> bool:
        """
        Check if game has reached terminal state.

        Args:
            game_state: Current game state

        Returns:
            bool: True if game is over
        """
        return game_state.game_status in [GameStatus.COMPLETED, GameStatus.TERMINATED]

    def get_state(self) -> GameState:
        """
        Get current complete game state.

        Returns:
            GameState: Current state
        """
        if self._game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._game_state

    def render(self) -> None:
        """Render ASCII grid visualization."""
        if not self._agent_positions:
            print("Environment not initialised. Call reset() first.")
            return

        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U3")

        # Place agents
        for agent_id, pos in self._agent_positions.items():
            x, y = pos.tolist()
            grid[x, y] = agent_id.split("_")[1]

        # Place tasks
        for agent_id, tasks in self._agent_tasks.items():
            for idx, task in enumerate(tasks):
                marker = f"T{idx}"
                if self._completed_tasks[agent_id][idx]:
                    marker = "✔"
                x, y = task.tolist()
                if grid[x, y] == ".":
                    grid[x, y] = marker

        print("\n".join(" ".join(row) for row in grid))

    def _build_observation(self, agent: AgentID) -> np.ndarray:
        """Build numpy observation for PettingZoo compatibility."""
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
        """Build info dict for PettingZoo compatibility."""
        return {
            "step": self._step_count,
            "completed_tasks": int(self._completed_tasks[agent].sum()),
            "total_tasks": self.tasks_per_agent,
            "task_progress": self._task_progress[agent].copy(),
        }

    def observation_space(self, agent: AgentID):
        """Return observation space (PettingZoo compatibility)."""
        return self._observation_space

    def action_space(self, agent: AgentID):
        """Return action space (PettingZoo compatibility)."""
        return self._action_space
