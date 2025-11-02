"""Tests for data models (can run without environment dependencies)."""

import sys
from pathlib import Path

# Add src to path - import models directly to avoid importing environment.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from datetime import datetime

# Import directly from models module to avoid circular import with environment
from multi_agent_deception.models import (
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


def test_scenario_config():
    """Test ScenarioConfig serialization."""
    config = ScenarioConfig(
        scenario_id="test_1",
        tier=2,
        num_agents=4,
        num_imposters=1,
        grid_size=8,
        tasks_per_agent=3,
        max_ticks=200,
        seed=42,
    )

    # Test dict conversion
    config_dict = config.to_dict()
    assert config_dict["scenario_id"] == "test_1"
    assert config_dict["tier"] == 2
    assert config_dict["num_agents"] == 4

    # Test from_dict
    config_restored = ScenarioConfig.from_dict(config_dict)
    assert config_restored.scenario_id == config.scenario_id
    assert config_restored.num_agents == config.num_agents


def test_agent():
    """Test Agent serialization."""
    agent = Agent(
        agent_id="agent_0",
        role=AgentRole.CREWMATE,
        status="alive",
        completed_tasks=2,
        total_tasks=3,
    )

    # Test dict conversion
    agent_dict = agent.to_dict()
    assert agent_dict["agent_id"] == "agent_0"
    assert agent_dict["role"] == "crewmate"
    assert agent_dict["completed_tasks"] == 2

    # Test from_dict
    agent_restored = Agent.from_dict(agent_dict)
    assert agent_restored.agent_id == agent.agent_id
    assert agent_restored.role == AgentRole.CREWMATE


def test_game_action():
    """Test GameAction serialization."""
    action = GameAction(
        agent_id="agent_0",
        action_type=ActionType.MOVE,
        parameters={"direction": 1},
        reasoning="Moving towards task location",
        confidence=0.95,
    )

    # Test dict conversion
    action_dict = action.to_dict()
    assert action_dict["agent_id"] == "agent_0"
    assert action_dict["action_type"] == "move"
    assert action_dict["parameters"]["direction"] == 1

    # Test from_dict
    action_restored = GameAction.from_dict(action_dict)
    assert action_restored.agent_id == action.agent_id
    assert action_restored.action_type == ActionType.MOVE


def test_game_event():
    """Test GameEvent serialization."""
    action = GameAction(
        agent_id="agent_0",
        action_type=ActionType.MOVE,
        parameters={"direction": 1},
    )

    event = GameEvent(
        tick=5,
        agent_id="agent_0",
        action=action,
        event_type="move_success",
        changes={"position": [1, 2]},
    )

    # Test dict conversion
    event_dict = event.to_dict()
    assert event_dict["tick"] == 5
    assert event_dict["agent_id"] == "agent_0"
    assert event_dict["event_type"] == "move_success"

    # Test from_dict
    event_restored = GameEvent.from_dict(event_dict)
    assert event_restored.tick == event.tick
    assert event_restored.event_type == event.event_type


def test_agent_observation():
    """Test AgentObservation serialization."""
    obs = AgentObservation(
        agent_id="agent_0",
        tick=5,
        visible_agents=["agent_1", "agent_2"],
        visible_positions={"agent_1": (2, 3), "agent_2": (4, 5)},
        visible_roles={},
    )

    # Test dict conversion
    obs_dict = obs.to_dict()
    assert obs_dict["agent_id"] == "agent_0"
    assert obs_dict["tick"] == 5
    assert len(obs_dict["visible_agents"]) == 2

    # Test from_dict
    obs_restored = AgentObservation.from_dict(obs_dict)
    assert obs_restored.agent_id == obs.agent_id
    assert len(obs_restored.visible_agents) == 2


def test_game_state():
    """Test GameState serialization."""
    config = ScenarioConfig(
        scenario_id="test_1",
        tier=2,
        num_agents=4,
        num_imposters=1,
    )

    agents = {
        "agent_0": Agent(agent_id="agent_0", role=AgentRole.IMPOSTER),
        "agent_1": Agent(agent_id="agent_1", role=AgentRole.CREWMATE),
    }

    state = GameState(
        game_id="game_123",
        scenario_config=config,
        tick=10,
        agents=agents,
        active_agents={"agent_0", "agent_1"},
        roles={"agent_0": AgentRole.IMPOSTER, "agent_1": AgentRole.CREWMATE},
        game_status=GameStatus.RUNNING,
    )

    # Test dict conversion
    state_dict = state.to_dict()
    assert state_dict["game_id"] == "game_123"
    assert state_dict["tick"] == 10
    assert state_dict["game_status"] == "running"

    # Test JSON conversion
    json_str = state.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["game_id"] == "game_123"

    # Test from_dict
    state_restored = GameState.from_dict(state_dict)
    assert state_restored.game_id == state.game_id
    assert state_restored.tick == state.tick


def test_game_log():
    """Test GameLog serialization."""
    config = ScenarioConfig(
        scenario_id="test_1",
        tier=2,
        num_agents=4,
        num_imposters=1,
    )

    log = GameLog(
        game_id="game_123",
        scenario_id="scenario_001",
        model_name="claude-3-opus",
        seed=42,
        created_at=datetime.now(),
        duration_ticks=50,
    )

    # Test dict conversion
    log_dict = log.to_dict()
    assert log_dict["game_id"] == "game_123"
    assert log_dict["model_name"] == "claude-3-opus"

    # Test JSON conversion
    json_str = log.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["game_id"] == "game_123"

    # Test from_dict
    log_restored = GameLog.from_dict(log_dict)
    assert log_restored.game_id == log.game_id
    assert log_restored.seed == log.seed


def test_agent_role_enum():
    """Test AgentRole enumeration."""
    assert AgentRole.IMPOSTER.value == "imposter"
    assert AgentRole.CREWMATE.value == "crewmate"
    assert AgentRole.UNKNOWN.value == "unknown"


def test_action_type_enum():
    """Test ActionType enumeration."""
    assert ActionType.MOVE.value == "move"
    assert ActionType.VOTE.value == "vote"
    assert ActionType.COMMUNICATE.value == "communicate"


def test_game_status_enum():
    """Test GameStatus enumeration."""
    assert GameStatus.RUNNING.value == "running"
    assert GameStatus.COMPLETED.value == "completed"
    assert GameStatus.TERMINATED.value == "terminated"


def test_round_trip_serialization():
    """Test complete round-trip serialization."""
    # Create complex state
    config = ScenarioConfig(
        scenario_id="complex_test",
        tier=2,
        num_agents=5,
        num_imposters=2,
        grid_size=10,
        observation_radius=5,
    )

    agents = {
        f"agent_{i}": Agent(agent_id=f"agent_{i}", role=AgentRole.CREWMATE if i > 1 else AgentRole.IMPOSTER)
        for i in range(5)
    }

    state = GameState(
        game_id="round_trip_test",
        scenario_config=config,
        tick=25,
        agents=agents,
        active_agents={f"agent_{i}" for i in range(5)},
        roles={k: v.role for k, v in agents.items()},
        game_status=GameStatus.RUNNING,
    )

    # Serialize to JSON
    json_str = state.to_json()

    # Deserialize
    data = json.loads(json_str)
    restored_state = GameState.from_dict(data)

    # Verify
    assert restored_state.game_id == state.game_id
    assert restored_state.tick == state.tick
    assert len(restored_state.agents) == len(state.agents)
    assert restored_state.scenario_config.num_agents == state.scenario_config.num_agents


if __name__ == "__main__":
    print("Testing ScenarioConfig...")
    test_scenario_config()
    print("✓ ScenarioConfig tests passed")

    print("Testing Agent...")
    test_agent()
    print("✓ Agent tests passed")

    print("Testing GameAction...")
    test_game_action()
    print("✓ GameAction tests passed")

    print("Testing GameEvent...")
    test_game_event()
    print("✓ GameEvent tests passed")

    print("Testing AgentObservation...")
    test_agent_observation()
    print("✓ AgentObservation tests passed")

    print("Testing GameState...")
    test_game_state()
    print("✓ GameState tests passed")

    print("Testing GameLog...")
    test_game_log()
    print("✓ GameLog tests passed")

    print("Testing Enums...")
    test_agent_role_enum()
    test_action_type_enum()
    test_game_status_enum()
    print("✓ Enum tests passed")

    print("Testing round-trip serialization...")
    test_round_trip_serialization()
    print("✓ Round-trip serialization tests passed")

    print("\n✅ All model tests passed!")
