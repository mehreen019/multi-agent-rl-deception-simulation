"""Tests for the abstract DeceptionGameEnvironment interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abc import ABC

from multi_agent_deception.base import DeceptionGameEnvironment
from multi_agent_deception.models import GameState, GameAction, AgentObservation


def test_abstract_class_defined():
    """Test that DeceptionGameEnvironment is defined as ABC."""
    assert issubclass(DeceptionGameEnvironment, ABC)


def test_cannot_instantiate_abstract_class():
    """Test that abstract class cannot be instantiated."""
    try:
        env = DeceptionGameEnvironment()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError as e:
        # Expected to fail
        assert "Can't instantiate abstract class" in str(e) or "abstract" in str(e).lower()


def test_abstract_methods_exist():
    """Test that all required abstract methods are defined."""
    required_methods = ["reset", "step", "get_observations", "is_terminal", "get_state"]

    for method in required_methods:
        assert hasattr(DeceptionGameEnvironment, method), f"Missing required method: {method}"
        # Check that it's abstract
        method_obj = getattr(DeceptionGameEnvironment, method)
        assert hasattr(method_obj, "__isabstractmethod__"), f"Method {method} is not abstract"


def test_concrete_methods_exist():
    """Test that non-abstract methods (with implementations) exist."""
    concrete_methods = ["render", "close"]

    for method in concrete_methods:
        assert hasattr(DeceptionGameEnvironment, method), f"Missing concrete method: {method}"


def test_properties_defined():
    """Test that required properties are defined."""
    properties = ["num_agents", "scenario_config", "current_tick", "current_state"]

    for prop in properties:
        assert hasattr(DeceptionGameEnvironment, prop), f"Missing property: {prop}"


def test_abstract_class_docstring():
    """Test that abstract class has documentation."""
    assert DeceptionGameEnvironment.__doc__ is not None
    assert len(DeceptionGameEnvironment.__doc__) > 0


def test_required_methods_have_docstrings():
    """Test that abstract methods have docstrings."""
    required_methods = ["reset", "step", "get_observations", "is_terminal", "get_state"]

    for method_name in required_methods:
        method = getattr(DeceptionGameEnvironment, method_name)
        assert method.__doc__ is not None, f"Method {method_name} missing docstring"
        assert len(method.__doc__) > 0, f"Method {method_name} has empty docstring"


class ConcreteEnvironment(DeceptionGameEnvironment):
    """Minimal concrete implementation for testing."""

    def __init__(self):
        self._state = None

    def reset(self, scenario_config, seed):
        """Minimal reset implementation."""
        self._state = GameState(
            game_id="test",
            scenario_config=scenario_config,
            tick=0,
        )
        return self._state

    def step(self, game_state, actions):
        """Minimal step implementation."""
        return game_state, {}

    def get_observations(self, game_state):
        """Minimal observations implementation."""
        return {}

    def is_terminal(self, game_state):
        """Minimal terminal check."""
        return False

    def get_state(self):
        """Minimal state getter."""
        return self._state


def test_concrete_implementation_possible():
    """Test that concrete implementation of abstract interface is possible."""
    env = ConcreteEnvironment()
    assert isinstance(env, DeceptionGameEnvironment)


def test_interface_method_signatures():
    """Test that interface methods have correct signatures."""
    import inspect

    # Check reset signature
    reset_sig = inspect.signature(DeceptionGameEnvironment.reset)
    assert "scenario_config" in reset_sig.parameters
    assert "seed" in reset_sig.parameters

    # Check step signature
    step_sig = inspect.signature(DeceptionGameEnvironment.step)
    assert "game_state" in step_sig.parameters
    assert "actions" in step_sig.parameters

    # Check get_observations signature
    get_obs_sig = inspect.signature(DeceptionGameEnvironment.get_observations)
    assert "game_state" in get_obs_sig.parameters

    # Check is_terminal signature
    is_term_sig = inspect.signature(DeceptionGameEnvironment.is_terminal)
    assert "game_state" in is_term_sig.parameters

    # Check get_state signature
    get_state_sig = inspect.signature(DeceptionGameEnvironment.get_state)


def test_properties_accessible():
    """Test that properties work on concrete implementation."""
    from multi_agent_deception.models import ScenarioConfig

    config = ScenarioConfig(
        scenario_id="test",
        tier=2,
        num_agents=4,
        num_imposters=1,
    )

    env = ConcreteEnvironment()
    env.reset(config, seed=42)

    # Properties should be accessible
    assert env.num_agents == 4
    assert env.scenario_config.scenario_id == "test"
    assert env.current_tick == 0
    assert env.current_state is not None


def test_default_render_method():
    """Test that default render method works."""
    env = ConcreteEnvironment()
    # Should not raise exception
    env.render()


def test_default_close_method():
    """Test that default close method works."""
    env = ConcreteEnvironment()
    # Should not raise exception
    env.close()


if __name__ == "__main__":
    print("Testing abstract class definition...")
    test_abstract_class_defined()
    print("✓ Abstract class is properly defined")

    print("Testing instantiation prevention...")
    test_cannot_instantiate_abstract_class()
    print("✓ Cannot instantiate abstract class")

    print("Testing abstract methods...")
    test_abstract_methods_exist()
    print("✓ All abstract methods exist")

    print("Testing concrete methods...")
    test_concrete_methods_exist()
    print("✓ Concrete methods exist")

    print("Testing properties...")
    test_properties_defined()
    print("✓ All properties defined")

    print("Testing docstrings...")
    test_abstract_class_docstring()
    test_required_methods_have_docstrings()
    print("✓ Documentation is complete")

    print("Testing concrete implementation...")
    test_concrete_implementation_possible()
    print("✓ Concrete implementations possible")

    print("Testing method signatures...")
    test_interface_method_signatures()
    print("✓ Method signatures correct")

    print("Testing properties on concrete implementation...")
    test_properties_accessible()
    print("✓ Properties work correctly")

    print("Testing render method...")
    test_default_render_method()
    print("✓ Render method works")

    print("Testing close method...")
    test_default_close_method()
    print("✓ Close method works")

    print("\n✅ All abstract interface tests passed!")
