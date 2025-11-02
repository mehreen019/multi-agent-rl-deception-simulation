"""Tests for environment refactoring (Story 1)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np

from multi_agent_deception.base import DeceptionGameEnvironment
from multi_agent_deception.tier2_environment import Tier2Environment
from multi_agent_deception.models import (
    GameState,
    GameAction,
    AgentObservation,
    ScenarioConfig,
    ActionType,
    GameStatus,
    AgentRole,
)


class TestAbstractInterface:
    """Test that abstract interface is properly defined."""

    def test_abstract_class_not_instantiable(self):
        """DeceptionGameEnvironment cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DeceptionGameEnvironment()

    def test_abstract_methods_exist(self):
        """All required abstract methods are defined."""
        required_methods = ["reset", "step", "get_observations", "is_terminal", "get_state"]
        for method in required_methods:
            assert hasattr(DeceptionGameEnvironment, method)

    def test_properties_exist(self):
        """Accessor properties are defined."""
        properties = ["num_agents", "scenario_config", "current_tick", "current_state"]
        for prop in properties:
            assert hasattr(DeceptionGameEnvironment, prop)


class TestTier2EnvironmentBasics:
    """Test basic Tier2Environment functionality."""

    def test_initialization(self):
        """Environment initializes with correct parameters."""
        env = Tier2Environment(grid_size=8, num_agents=4, tasks_per_agent=3)
        assert env.grid_size == 8
        assert env.num_agents == 4
        assert env.tasks_per_agent == 3

    def test_reset_returns_game_state(self):
        """Reset returns a GameState object."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        assert isinstance(state, GameState)
        assert state.game_id is not None
        assert state.tick == 0
        assert state.game_status == GameStatus.RUNNING

    def test_reset_initializes_agents(self):
        """Reset properly initializes all agents."""
        env = Tier2Environment(num_agents=4)
        state = env.reset(seed=42)

        assert len(state.agents) == 4
        assert len(state.active_agents) == 4
        assert all(agent_id in state.agents for agent_id in ["agent_0", "agent_1", "agent_2", "agent_3"])

    def test_reset_with_seed_deterministic(self):
        """Same seed produces identical game states."""
        env1 = Tier2Environment(grid_size=8, num_agents=4)
        state1 = env1.reset(seed=42)

        env2 = Tier2Environment(grid_size=8, num_agents=4)
        state2 = env2.reset(seed=42)

        # Check positions are identical
        for agent_id in state1.agents:
            assert np.array_equal(env1._agent_positions[agent_id], env2._agent_positions[agent_id])


class TestTier2EnvironmentStep:
    """Test step execution and game mechanics."""

    def test_step_returns_correct_types(self):
        """Step returns GameState and events dict."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        actions = {
            agent_id: GameAction(
                agent_id=agent_id,
                action_type=ActionType.MOVE,
                parameters={"direction": 0}
            )
            for agent_id in state.active_agents
        }

        new_state, events = env.step(state, actions)

        assert isinstance(new_state, GameState)
        assert isinstance(events, dict)
        assert len(events) == len(state.active_agents)

    def test_step_increments_tick(self):
        """Step increments game tick."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        actions = {
            agent_id: GameAction(agent_id=agent_id, action_type=ActionType.MOVE)
            for agent_id in state.active_agents
        }

        new_state, _ = env.step(state, actions)
        assert new_state.tick == 1

    def test_step_clamps_positions_to_grid(self):
        """Agent positions stay within grid bounds."""
        env = Tier2Environment(grid_size=5, num_agents=1)
        state = env.reset(seed=42)

        # Move agent repeatedly to edge
        for _ in range(100):
            actions = {
                state.active_agents[0]: GameAction(
                    agent_id=state.active_agents[0],
                    action_type=ActionType.MOVE,
                    parameters={"direction": 1}  # Move up
                )
            }
            state, _ = env.step(state, actions)

        # Position should be clamped
        pos = env._agent_positions[state.active_agents[0]]
        assert 0 <= pos[0] < 5
        assert 0 <= pos[1] < 5

    def test_task_completion_logic(self):
        """Tasks are completed when agent reaches location for duration."""
        env = Tier2Environment(grid_size=8, num_agents=1, tasks_per_agent=1, task_duration=3)
        state = env.reset(seed=42)

        agent_id = state.active_agents[0]
        task_pos = env._agent_tasks[agent_id][0]

        # Move agent to task location
        for step in range(20):
            if step < 5:
                # Move towards task
                move_x = 1 if task_pos[0] > env._agent_positions[agent_id][0] else -1 if task_pos[0] < env._agent_positions[agent_id][0] else 0
                move_y = 1 if task_pos[1] > env._agent_positions[agent_id][1] else -1 if task_pos[1] < env._agent_positions[agent_id][1] else 0

                if move_x == -1:
                    direction = 1  # up
                elif move_x == 1:
                    direction = 2  # down
                elif move_y == -1:
                    direction = 3  # left
                elif move_y == 1:
                    direction = 4  # right
                else:
                    direction = 0  # stay
            else:
                direction = 0

            actions = {
                agent_id: GameAction(
                    agent_id=agent_id,
                    action_type=ActionType.MOVE,
                    parameters={"direction": direction}
                )
            }
            state, _ = env.step(state, actions)

        # Task should be completed (or very close)
        assert env._completed_tasks[agent_id][0] == True or env._task_progress[agent_id][0] >= 2


class TestPartialObservability:
    """Test partial observability implementation."""

    def test_get_observations_returns_dict(self):
        """get_observations returns dict for all agents."""
        env = Tier2Environment(num_agents=4)
        state = env.reset(seed=42)

        obs_dict = env.get_observations(state)

        assert isinstance(obs_dict, dict)
        assert len(obs_dict) == 4
        assert all(agent_id in obs_dict for agent_id in state.active_agents)

    def test_observations_are_agent_observation_objects(self):
        """Observations are AgentObservation instances."""
        env = Tier2Environment(num_agents=4)
        state = env.reset(seed=42)

        obs_dict = env.get_observations(state)

        for obs in obs_dict.values():
            assert isinstance(obs, AgentObservation)

    def test_visibility_radius_respected(self):
        """Agents only see nearby agents within radius."""
        env = Tier2Environment(grid_size=20, num_agents=3)

        # Custom config with small radius
        config = ScenarioConfig(
            scenario_id="test",
            tier=2,
            num_agents=3,
            num_imposters=1,
            grid_size=20,
            observation_radius=3,
        )

        state = env.reset(scenario_config=config, seed=42)
        obs_dict = env.get_observations(state)

        # Check that visible agents respect radius
        for agent_id, obs in obs_dict.items():
            agent_pos = np.array(env._agent_positions[agent_id])
            for visible_id, visible_pos in obs.visible_positions.items():
                distance = np.linalg.norm(agent_pos - np.array(visible_pos))
                assert distance <= config.observation_radius + 0.01  # Small tolerance for floating point

    def test_roles_hidden_by_default(self):
        """Agent roles are hidden in observations."""
        env = Tier2Environment(num_agents=4)
        state = env.reset(seed=42)

        obs_dict = env.get_observations(state)

        for obs in obs_dict.values():
            assert len(obs.visible_roles) == 0  # Roles are hidden


class TestBackwardCompatibility:
    """Test backward compatibility with original environment."""

    def test_pettingzoo_compatibility(self):
        """Tier2Environment has PettingZoo-compatible methods."""
        env = Tier2Environment()

        required_attrs = ["observation_space", "action_space", "agents"]
        for attr in required_attrs:
            assert hasattr(env, attr)

    def test_observation_space_unchanged(self):
        """Observation space definition identical to original."""
        env = Tier2Environment(grid_size=8, num_agents=4, tasks_per_agent=3)

        obs_space = env.observation_space("agent_0")

        expected_size = 2 + 3 * 3  # position (2) + tasks (3 * 3)
        assert obs_space.shape == (expected_size,)
        assert obs_space.dtype == np.int32

    def test_action_space_unchanged(self):
        """Action space definition identical to original."""
        env = Tier2Environment()

        action_space = env.action_space("agent_0")

        assert isinstance(action_space, type(env._action_space))
        assert action_space.n == 5  # stay, up, down, left, right

    def test_render_works(self):
        """Render method works without errors."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        # Should not raise exception
        env.render()

    def test_is_terminal_checks_win_conditions(self):
        """Terminal state detection works correctly."""
        env = Tier2Environment(max_steps=5, tasks_per_agent=0)  # No tasks = immediate win
        state = env.reset(seed=42)

        assert not env.is_terminal(state)

        # Play until end
        for _ in range(10):
            actions = {
                agent_id: GameAction(agent_id=agent_id, action_type=ActionType.MOVE)
                for agent_id in state.active_agents
            }
            state, _ = env.step(state, actions)
            if env.is_terminal(state):
                break

        assert env.is_terminal(state)


class TestGameStateIntegration:
    """Test GameState integration."""

    def test_game_state_persistence(self):
        """Game state maintains information across steps."""
        env = Tier2Environment(num_agents=2)
        state = env.reset(seed=42)
        game_id = state.game_id

        actions = {
            agent_id: GameAction(agent_id=agent_id, action_type=ActionType.MOVE)
            for agent_id in state.active_agents
        }
        state, _ = env.step(state, actions)

        assert state.game_id == game_id
        assert state.tick == 1

    def test_get_state_returns_current_state(self):
        """get_state returns the current game state."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        retrieved_state = env.get_state()

        assert retrieved_state.game_id == state.game_id
        assert retrieved_state.tick == state.tick

    def test_state_serialization(self):
        """GameState can be serialized and deserialized."""
        env = Tier2Environment()
        state = env.reset(seed=42)

        # Serialize
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)

        # Deserialize
        state_restored = GameState.from_dict(state_dict)
        assert state_restored.game_id == state.game_id
        assert state_restored.tick == state.tick


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
