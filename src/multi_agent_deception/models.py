"""Core data models for deception game simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json

import numpy as np


class GameStatus(Enum):
    """Game status enumeration."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class AgentRole(Enum):
    """Agent role enumeration."""

    IMPOSTER = "imposter"
    CREWMATE = "crewmate"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Action type enumeration."""

    MOVE = "move"
    INTERACT = "interact"
    COMMUNICATE = "communicate"
    VOTE = "vote"
    COMPLETE_TASK = "complete_task"
    OBSERVE = "observe"


@dataclass
class ScenarioConfig:
    """Configuration for a game scenario."""

    scenario_id: str
    tier: int  # 1 or 2
    num_agents: int
    num_imposters: int
    grid_size: Optional[int] = None
    tasks_per_agent: int = 3
    max_ticks: int = 200
    task_duration: int = 3
    meeting_frequency: Optional[int] = None
    observation_radius: int = 5  # For Tier 2
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["tier"] = self.tier
        data["num_agents"] = self.num_agents
        data["num_imposters"] = self.num_imposters
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioConfig:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Agent:
    """Represents an agent in the game."""

    agent_id: str
    role: AgentRole = AgentRole.UNKNOWN
    status: str = "alive"  # "alive" or "dead"
    completed_tasks: int = 0
    total_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.status,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Agent:
        """Create from dictionary."""
        data["role"] = AgentRole(data["role"])
        return cls(**data)


@dataclass
class GameAction:
    """Represents a parsed game action from an agent."""

    agent_id: str
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "validation_error": self.validation_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameAction:
        """Create from dictionary."""
        data["action_type"] = ActionType(data["action_type"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class GameEvent:
    """Represents the outcome of an action (what changed)."""

    tick: int
    agent_id: str
    action: GameAction
    event_type: str  # e.g., "move_success", "task_completed", "vote_cast"
    changes: Dict[str, Any] = field(default_factory=dict)
    metrics_delta: Dict[str, float] = field(default_factory=dict)
    observations_updated: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tick": self.tick,
            "agent_id": self.agent_id,
            "action": self.action.to_dict(),
            "event_type": self.event_type,
            "changes": self.changes,
            "metrics_delta": self.metrics_delta,
            "observations_updated": self.observations_updated,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameEvent:
        """Create from dictionary."""
        data["action"] = GameAction.from_dict(data["action"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AgentObservation:
    """Represents what one agent observes (partial observability)."""

    agent_id: str
    tick: int
    visible_agents: List[str] = field(default_factory=list)
    visible_positions: Dict[str, tuple] = field(default_factory=dict)
    visible_roles: Dict[str, AgentRole] = field(default_factory=dict)
    discussion_history: List[str] = field(default_factory=list)
    known_roles: Dict[str, AgentRole] = field(default_factory=dict)
    own_beliefs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "tick": self.tick,
            "visible_agents": self.visible_agents,
            "visible_positions": self.visible_positions,
            "visible_roles": {k: v.value for k, v in self.visible_roles.items()},
            "discussion_history": self.discussion_history,
            "known_roles": {k: v.value for k, v in self.known_roles.items()},
            "own_beliefs": self.own_beliefs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentObservation:
        """Create from dictionary."""
        data["visible_roles"] = {k: AgentRole(v) for k, v in data.get("visible_roles", {}).items()}
        data["known_roles"] = {k: AgentRole(v) for k, v in data.get("known_roles", {}).items()}
        return cls(**data)


@dataclass
class GameState:
    """Complete game state at any tick."""

    game_id: str
    scenario_config: ScenarioConfig
    tick: int = 0
    agents: Dict[str, Agent] = field(default_factory=dict)
    active_agents: Set[str] = field(default_factory=set)
    roles: Dict[str, AgentRole] = field(default_factory=dict)
    game_status: GameStatus = GameStatus.INITIALIZED
    winner: Optional[str] = None  # "imposters", "crewmates", or None
    internal_state: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "scenario_config": self.scenario_config.to_dict(),
            "tick": self.tick,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "active_agents": list(self.active_agents),
            "roles": {k: v.value for k, v in self.roles.items()},
            "game_status": self.game_status.value,
            "winner": self.winner,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameState:
        """Create from dictionary."""
        data["scenario_config"] = ScenarioConfig.from_dict(data["scenario_config"])
        data["agents"] = {k: Agent.from_dict(v) for k, v in data.get("agents", {}).items()}
        data["active_agents"] = set(data.get("active_agents", []))
        data["roles"] = {k: AgentRole(v) for k, v in data.get("roles", {}).items()}
        data["game_status"] = GameStatus(data["game_status"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class GameLog:
    """Complete game history and event log."""

    game_id: str
    scenario_id: str
    model_name: str
    seed: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ticks: int = 0
    final_state: Optional[GameState] = None
    events: List[GameEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "seed": self.seed,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ticks": self.duration_ticks,
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameLog:
        """Create from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("completed_at"), str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        data["final_state"] = GameState.from_dict(data["final_state"]) if data.get("final_state") else None
        data["events"] = [GameEvent.from_dict(e) for e in data.get("events", [])]
        return cls(**data)

    def get_events_for_agent(self, agent_id: str) -> List[GameEvent]:
        """Get all events for a specific agent."""
        return [e for e in self.events if e.agent_id == agent_id]
