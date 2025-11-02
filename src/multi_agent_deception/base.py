"""Abstract base class for deception game environments."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from .models import (
    GameState,
    GameAction,
    GameEvent,
    AgentObservation,
    ScenarioConfig,
)


class DeceptionGameEnvironment(ABC):
    """
    Abstract game environment interface for deception games.

    Supports both Tier 1 (dialogue-only) and Tier 2 (spatial) game variants.
    All implementations must provide clear contracts for game state management,
    action execution, and partial observability.

    Example usage:
        >>> env = Tier2Environment(scenario_config)
        >>> initial_state = env.reset(scenario_config, seed=42)
        >>> actions = {"agent_0": GameAction(...), "agent_1": GameAction(...)}
        >>> new_state, events = env.step(initial_state, actions)
        >>> observations = env.get_observations(new_state)
        >>> is_over = env.is_terminal(new_state)
    """

    @abstractmethod
    def reset(self, scenario_config: ScenarioConfig, seed: int) -> GameState:
        """
        Initialize game with scenario configuration and seed.

        Args:
            scenario_config: ScenarioConfig with game parameters (agents, roles, etc.)
            seed: Random seed for reproducibility

        Returns:
            GameState: Initial game state after setup

        Raises:
            ValueError: If scenario_config is invalid
        """
        pass

    @abstractmethod
    def step(
        self, game_state: GameState, actions: Dict[str, GameAction]
    ) -> tuple[GameState, Dict[str, GameEvent]]:
        """
        Execute one game tick with actions from all agents.

        Args:
            game_state: Current game state
            actions: Dict mapping agent_id -> GameAction to execute

        Returns:
            Tuple of (new_game_state, events_dict)
            where events_dict maps agent_id -> GameEvent describing what happened

        Raises:
            RuntimeError: If environment not properly initialized
            ValueError: If actions invalid or inconsistent
        """
        pass

    @abstractmethod
    def get_observations(self, game_state: GameState) -> Dict[str, AgentObservation]:
        """
        Get partial observability views for all agents.

        Each agent sees only the information they have access to based on their role,
        position (Tier 2), and game state. Roles are hidden unless revealed.

        Args:
            game_state: Current game state

        Returns:
            Dict mapping agent_id -> AgentObservation
            Contains visible agents, positions, discussion history, known roles, etc.
        """
        pass

    @abstractmethod
    def is_terminal(self, game_state: GameState) -> bool:
        """
        Check if game has reached terminal state.

        A game is terminal if:
        - Win condition reached (imposters win or crewmates win)
        - Max ticks exceeded
        - All agents are dead/ejected

        Args:
            game_state: Current game state

        Returns:
            bool: True if game is over, False if ongoing
        """
        pass

    @abstractmethod
    def get_state(self) -> GameState:
        """
        Export current complete game state.

        Returns the internal game state as a GameState object.
        This is used for serialization and logging.

        Returns:
            GameState: Full internal game state
        """
        pass

    def render(self) -> None:
        """
        Render/visualize game state for debugging.

        Default implementation does nothing. Subclasses may override
        to provide ASCII art grid visualization or other output.
        """
        pass

    def close(self) -> None:
        """
        Clean up environment resources.

        Default implementation does nothing. Subclasses may override
        to clean up files, network connections, etc.
        """
        pass

    # Properties for convenience access
    @property
    def num_agents(self) -> int:
        """Return total number of agents."""
        return self.get_state().scenario_config.num_agents

    @property
    def scenario_config(self) -> ScenarioConfig:
        """Return current scenario configuration."""
        return self.get_state().scenario_config

    @property
    def current_tick(self) -> int:
        """Return current game tick."""
        return self.get_state().tick

    @property
    def current_state(self) -> GameState:
        """Return current complete game state."""
        return self.get_state()
