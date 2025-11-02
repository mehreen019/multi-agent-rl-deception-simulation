"""Utilities and environments for the multi-agent deception simulation project."""

# New refactored interfaces (Story 1) - always available
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
    GameLog,
)

# Original PettingZoo interface (backward compatibility)
try:
    from .environment import parallel_env, raw_env, env
except ImportError:
    # If PettingZoo dependencies not available, set to None
    parallel_env = None
    raw_env = None
    env = None

# Tier2Environment may also have import issues if gymnasium not available
try:
    from .tier2_environment import Tier2Environment
except ImportError:
    Tier2Environment = None

__all__ = [
    # Data models (always available)
    "GameState",
    "GameAction",
    "GameEvent",
    "AgentObservation",
    "ScenarioConfig",
    "Agent",
    "AgentRole",
    "ActionType",
    "GameStatus",
    "GameLog",
    # New abstract interfaces
    "DeceptionGameEnvironment",
    # Original PettingZoo interface (if available)
    "parallel_env",
    "raw_env",
    "env",
    # Tier 2 environment (if available)
    "Tier2Environment",
]
