# Multi-Agent Reinforcement Learning: Deception Simulation Framework

A comprehensive research system for studying deception, hidden roles, and strategic reasoning in multi-agent games. This document covers the theoretical foundations, system architecture, and practical usage.

**Table of Contents:**
- [Theoretical Framework](#theoretical-framework)
- [System Architecture](#system-architecture)
- [File Structure & Components](#file-structure--components)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Quick Start](#quick-start)
- [Advanced Configuration](#advanced-configuration)
- [Development Roadmap](#development-roadmap)

---

## Theoretical Framework

### Problem Setting

This project models **multi-agent strategic reasoning in partially-observable games with hidden roles**.

**Core Elements:**

| Aspect | Description |
|--------|-------------|
| **Environment Type** | Cooperative partially observable Markov game (POMG) on an N×N grid |
| **Agents** | M players (`agent_0` to `agent_{M-1}`), each controlling a pawn |
| **Actions** | 5 discrete cardinal moves: {stay, up, down, left, right} |
| **Hidden Information** | Each agent has private task list; roles/tasks hidden from others (asymmetric knowledge) |
| **Objective** | Complete all personal tasks; episode ends on success or max_steps timeout |
| **Deception Mechanism** | Agents can stall, misdirect, or ignore tasks since task ownership is hidden |

### Game-Theoretic Motivation

The framework enables studying **deceptive equilibria** in multi-agent systems:

1. **Cooperative Base:** All agents *could* help each other complete tasks
2. **Hidden Information:** But agents don't know others' actual roles/tasks
3. **Incentive Misalignment:** Can be extended so some agents benefit from *others* failing
4. **Strategic Dilemma:** Leads to deceptive behavior, signaling, and inference

**Real-world parallels:**
- Among Us social deduction game
- Organizational scenarios with hidden roles (whistleblowers, saboteurs, etc.)
- Multi-party negotiations with private information
- Information asymmetry in markets and conflicts

### Mathematical Formulation

#### State Space

**Global state** `s_t` (not directly observable by agents):
```
s_t = {
  p_i ∈ ℤ_{grid}²           # Agent positions (x, y)
  T_i ∈ (ℤ_{grid}²)^K       # Agent i's task locations (K tasks)
  c_i ∈ {0,1}^K             # Completion masks (task done? yes/no)
  d_i ∈ {0,…,D}^K           # Progress counters (how many steps on task)
  t ∈ ℕ                     # Step count
}
```

#### Observation Space (Local)

Agent `i` observes only:
```
o_i = [p_i.x, p_i.y, T_i[0].x, T_i[0].y, d_i[0], …, T_i[K-1].x, T_i[K-1].y, d_i[K-1]]
```

**Completed task encoding:** Position set to `(grid_size, grid_size)`, progress capped at `D`

**Properties:**
- Fully local (no direct observation of other agents)
- Discrete-valued, integer arrays
- Markovian from individual perspective

#### Action Space

Discrete action: `a_i ∈ {0, 1, 2, 3, 4}`
- `0: stay` → no position change
- `1: up` → p_i.x -= 1 (clipped to bounds)
- `2: down` → p_i.x += 1 (clipped to bounds)
- `3: left` → p_i.y -= 1 (clipped to bounds)
- `4: right` → p_i.y += 1 (clipped to bounds)

**Collision Model:** Agents can share tiles; no collision penalty.

#### Reward Function

Per-step individual reward for agent `i`:

```
r_i(t) = -0.01                                    # Time penalty
       + 1.0 × I(task_progress[i] reaches D)     # Task completion reward
       + 0.5 × I(all_tasks[i] complete)         # Completion bonus
```

**Interpretation:**
- **-0.01:** Encourages speed; prevents endless wandering
- **+1.0:** Primary sparse reward signal
- **+0.5:** Terminal bonus for finishing all tasks

#### Transition Dynamics

Given `s_t` and joint action `a_t = (a_1, …, a_M)`:

1. **Compute tentative positions:**
   ```
   p'_i = clip(p_i + move(a_i), 0, grid_size - 1)
   ```

2. **Update progress counters:**
   ```
   For each task k:
     if p'_i == T_i[k] and not c_i[k]:
       d'_i[k] = min(d_i[k] + 1, D)
     else:
       d'_i[k] = 0
   ```

3. **Update completion masks:**
   ```
   c'_i[k] = 1 if d'_i[k] >= D, else c_i[k]
   ```

4. **Check terminal conditions:**
   ```
   terminate = all agents have all_tasks complete
   truncate = (t + 1 >= max_steps) AND NOT terminate
   ```

**Stochasticity:** Only at reset (random position/task sampling).

### Learning Objectives

**Individual Agent Goal:**
```
Max E[∑_{t=0}^{∞} γ^t r_i(s_t, a_t)]
where γ = 0.99 (discount factor)
```

**Multi-Agent Approach:**
- **Policy Sharing:** Single neural network for all agents (parameter tying)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Implementation:** Stable-Baselines3 + SuperSuit vectorization

### Extensions & Deception Dynamics

**To introduce deception, add:**

1. **Role Asymmetry:**
   - Some agents are "Imposters" with negative rewards for fast crewmate completion
   - Others are "Crewmates" with positive rewards only when all tasks done

2. **Partial Observability:**
   - Agents only see nearby agents (observation radius)
   - Must infer others' roles from movement patterns

3. **Communication Phases:**
   - Discussion rounds where agents vote to eject suspected imposters
   - Information revelation as side effect of voting

4. **LLM Integration:**
   - Feed game state to language models for strategic reasoning
   - Capture reasoning traces for mechanistic interpretability

**This framework is Phase 0 for these extensions.**

---

## System Architecture

### Design Philosophy

**Principles:**
1. **Reproducibility First** - Deterministic mechanics, complete logging, seed management
2. **Separation of Concerns** - Simulation, LLM integration, metrics are independent
3. **Tight Integration at Seams** - LLM and simulation tightly coupled at decision points
4. **Type Safety** - Dataclasses for all models, JSON schemas for I/O
5. **Observability** - Complete event logging at every decision point

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Analysis & Visualization                          │
│  • Mechanistic interpretability                             │
│  • Strategy pattern extraction                              │
│  • Comparative metrics                                      │
└─────────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Experiment Orchestration                          │
│  • GameRunner (single game execution)                       │
│  • ExperimentRunner (batch management)                      │
│  • Result aggregation                                       │
└─────────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Scenarios & Configuration                         │
│  • ScenarioConfig (JSON-based scenario definitions)         │
│  • Scenario loading and validation                          │
└─────────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Metrics & Logging (EventLogger)                   │
│  • Complete game event history                              │
│  • Performance metrics calculation                          │
│  • JSON export for reproducibility                          │
└─────────────────────────────────────────────────────────────┘
                          ↑
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│ Simulation   │  │ LLM Integration  │  │ Data Models  │
│ Engine       │◄─┤ • Prompt engines │─►│ • GameState  │
│              │  │ • API clients    │  │ • GameAction │
│              │  │ • Response parse │  │ • GameEvent  │
└──────────────┘  └──────────────────┘  └──────────────┘
       ↑                    ↑                     ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Simulation Engine (Core)                          │
│  • DeceptionGameEnvironment (abstract interface)            │
│  • Tier1Environment (dialogue-only)                         │
│  • Tier2Environment (spatial grid)                          │
│  • GameRules (action validation, win conditions)            │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | DeceptionGameEnvironment | Abstract interface for all game variants |
| 1 | Tier2Environment | Grid-based spatial game (refactored from original) |
| 1 | Tier1Environment | Dialogue-only variant (Story 2) |
| 1 | GameRules | Action validation, win conditions, voting |
| 2 | LLMClient | Abstract API interface (Claude, GPT-4, Llama) |
| 2 | PromptTemplate | Role-specific prompt rendering |
| 2 | ResponseParser | JSON extraction from LLM outputs |
| 2 | SafetyFilter | Action validation before execution |
| 3 | EventLogger | Game event recording |
| 3 | GameState/GameAction/GameEvent | Core data models |
| 4 | ScenarioConfig | Game configuration |
| 5 | GameRunner | Single game orchestration |
| 5 | ExperimentRunner | Batch game execution |
| 6 | MetricsAnalyzer | Post-game analysis |

---

## File Structure & Components

### Source Code Organization

```
src/multi_agent_deception/
├── __init__.py                    # Package exports (models, interfaces, env)
├── models.py                      # Core dataclasses (GameState, GameAction, etc.)
├── base.py                        # DeceptionGameEnvironment abstract interface
├── environment.py                 # Original SimpleHiddenRoleParallelEnv (PettingZoo)
├── tier2_environment.py           # Refactored Tier2Environment
├── gui.py                         # Pygame visualization (legacy)
└── [Future Story Components]
    ├── tier1_environment.py       # Tier1 dialogue-only variant (Story 2)
    ├── rules.py                   # GameRules engine (Story 3)
    ├── llm/
    │   ├── client.py              # LLMClient abstraction (Story 4)
    │   ├── prompts.py             # PromptTemplate system (Story 6)
    │   └── parser.py              # ResponseParser (Story 5)
    ├── logging.py                 # EventLogger (Story 7)
    ├── runner.py                  # GameRunner (Story 9)
    └── analysis/
        └── metrics.py             # Analysis tools (Story 10)
```

### Documentation Organization

```
docs/
├── ARCHITECTURE.md                # Complete system design (1087 lines)
├── IMPLEMENTATION_GUIDE.md        # Practical developer guide
├── IMPLEMENTATION_ROADMAP.md      # Phase timeline
├── epics/
│   └── phase-0-foundation.md      # Phase 0 epic (10 stories)
└── stories/
    ├── story-1-refactor-environment-abstract-interface.md  ✅ DONE
    ├── story-2-implement-tier1-dialogue-only-environment.md
    ├── story-3-build-game-rules-engine.md
    ├── story-4-implement-llm-client-abstraction.md
    ├── story-5-build-response-parser-safety-filter.md
    ├── story-6-create-prompt-templates.md
    ├── story-7-implement-event-logger.md
    ├── story-8-create-data-models.md
    ├── story-9-implement-game-runner.md
    └── story-10-phase-0-integration-testing.md
```

### Key Files Explained

#### `src/multi_agent_deception/models.py` (327 lines)
**What it does:** Defines all core data structures for game state management.

**Key Classes:**
- `GameState` - Complete game snapshot at any tick
- `GameAction` - Parsed LLM decision with validation
- `GameEvent` - Outcome of an action
- `AgentObservation` - Partial view of world (what agent sees)
- `Agent` - Agent metadata (role, status, tasks)
- `ScenarioConfig` - Game configuration
- `GameLog` - Complete game history

**Serialization:** All models support `to_dict()`, `from_dict()`, `to_json()` for reproducibility.

**Enumerations:**
```python
AgentRole: IMPOSTER, CREWMATE, UNKNOWN
ActionType: MOVE, INTERACT, COMMUNICATE, VOTE, COMPLETE_TASK, OBSERVE
GameStatus: INITIALIZED, RUNNING, COMPLETED, TERMINATED
```

#### `src/multi_agent_deception/base.py` (111 lines)
**What it does:** Defines abstract interface that all environments must implement.

**Key Abstract Methods:**
```python
reset(scenario_config, seed) → GameState
  # Initialize game

step(game_state, actions) → (GameState, dict[agent_id → GameEvent])
  # Execute one game tick

get_observations(game_state) → dict[agent_id → AgentObservation]
  # Compute partial observability views

is_terminal(game_state) → bool
  # Check if game has ended

get_state() → GameState
  # Export current state
```

**Properties:**
- `num_agents` - Total agent count
- `scenario_config` - Current configuration
- `current_tick` - Game tick
- `current_state` - Full game state

**Concrete Methods:**
- `render()` - Visualize (default: no-op)
- `close()` - Cleanup (default: no-op)

#### `src/multi_agent_deception/tier2_environment.py` (368 lines)
**What it does:** Refactored grid-based environment with spatial mechanics.

**Key Features:**
- Fully implements `DeceptionGameEnvironment`
- Preserves 100% of original `SimpleHiddenRoleParallelEnv` behavior
- Uses `GameState` for internal tracking
- Implements partial observability via `get_observations()`
- Maintains PettingZoo compatibility

**Partial Observability:**
```python
def get_observations(self, game_state: GameState) → dict[str, AgentObservation]:
  # For each agent:
  #   - Compute distance to all other agents
  #   - If distance <= observation_radius: agent is visible
  #   - Return AgentObservation with visible_agents, visible_positions
  #   - Roles hidden (empty visible_roles dict)
```

#### `src/multi_agent_deception/environment.py` (225 lines)
**What it does:** Original PettingZoo-compatible environment (preserved for backward compatibility).

**Use Cases:**
- Existing RL training scripts
- Legacy code compatibility
- Reference implementation

**Note:** Tier2Environment is the new canonical implementation.

#### `scripts/train_simple.py` (103 lines)
**What it does:** PPO training loop for the environment.

**Usage:**
```bash
PYTHONPATH=src python scripts/train_simple.py \
  --timesteps 20000 \
  --grid-size 8 \
  --num-agents 4 \
  --tasks-per-agent 3 \
  --learning-rate 3e-4 \
  --output-dir artifacts
```

**Output:**
- `artifacts/ppo_simple_hidden_role.zip` - Trained model
- `artifacts/tensorboard/` - Training logs (optional)

**Features:**
- SuperSuit vectorization
- Policy sharing across agents
- PPO hyperparameter tuning support

#### `scripts/play_gui.py` (49 lines)
**What it does:** Interactive Pygame visualization for manual testing.

**Controls:**
- Arrow keys - Move agent_0
- Space - Stay
- R - Reset
- P - Pause
- Close window - Exit

**Usage:**
```bash
PYTHONPATH=src python scripts/play_gui.py \
  --grid-size 8 \
  --num-agents 4 \
  --tasks-per-agent 3
```

#### `tests/test_models.py` (280 lines)
**What it does:** Comprehensive tests for data models.

**Coverage:**
- Serialization (to_dict, from_dict, JSON)
- Type validation
- Round-trip conversion
- Enumeration values

#### `tests/test_base_interface.py` (260 lines)
**What it does:** Tests for abstract interface compliance.

**Coverage:**
- Interface definition
- Abstract method existence
- Concrete implementation possibility
- Property accessibility

---

## Installation & Setup

### Prerequisites

- **Python:** 3.10 or later
- **OS:** Windows, macOS, or Linux
- **Memory:** 2GB minimum (4GB+ recommended)
- **Optional:** GPU support for training (CUDA 11.8+)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/multi-agent-rl-deception-simulation.git
cd multi-agent-rl-deception-simulation
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements:**
- `gymnasium` - RL environment API
- `numpy` - Numerical computing
- `pettingzoo` - Multi-agent RL framework
- `stable-baselines3` - RL algorithms
- `supersuit` - Environment vectorization
- `pygame` - GUI visualization

**Optional for visualization:**
```bash
pip install tensorboard  # Training logs
```

### Step 4: Verify Installation

```bash
PYTHONPATH=src python -c "from multi_agent_deception import *; print('✓ Installation successful')"
```

---

## Usage Guide

### Quick Test: Run Environment

```bash
PYTHONPATH=src python3 - <<'PY'
from multi_agent_deception import Tier2Environment, ScenarioConfig

# Create environment
config = ScenarioConfig(
    scenario_id="test",
    tier=2,
    num_agents=4,
    num_imposters=1,
    grid_size=8,
)

env = Tier2Environment()
state = env.reset(config, seed=42)

print(f"✓ Game initialized: {len(state.active_agents)} agents")
print(f"✓ State shape: tick={state.tick}, agents={len(state.agents)}")

# Get observations (partial observability)
observations = env.get_observations(state)
for agent_id, obs in observations.items():
    print(f"  {agent_id}: sees {len(obs.visible_agents)} agents")
PY
```

**Expected Output:**
```
✓ Game initialized: 4 agents
✓ State shape: tick=0, agents=4
  agent_0: sees 0 agents (initially far apart)
  agent_1: sees 0 agents
  agent_2: sees 0 agents
  agent_3: sees 0 agents
```

### Use Case 1: Interactive Play (GUI)

```bash
PYTHONPATH=src python scripts/play_gui.py \
  --grid-size 8 \
  --num-agents 4 \
  --tasks-per-agent 3 \
  --task-duration 3
```

**What happens:**
1. Pygame window opens with grid visualization
2. You control agent_0 with arrow keys
3. Other agents take random actions
4. Watch task progress bars and termination logic
5. Press R to reset, P to pause

### Use Case 2: Train RL Agent

```bash
PYTHONPATH=src python scripts/train_simple.py \
  --timesteps 50000 \
  --grid-size 8 \
  --num-agents 4 \
  --tasks-per-agent 3 \
  --max-steps 200 \
  --task-duration 3 \
  --learning-rate 3e-4 \
  --output-dir artifacts \
  --seed 42
```

**Output:**
- `artifacts/ppo_simple_hidden_role.zip` (trained weights)
- Training logs (if TensorBoard installed)

**View training progress:**
```bash
tensorboard --logdir artifacts/tensorboard
# Then open http://localhost:6006 in browser
```

### Use Case 3: Evaluate Trained Model

```bash
PYTHONPATH=src python3 - <<'PY'
from stable_baselines3 import PPO
from multi_agent_deception import Tier2Environment, ScenarioConfig

# Load trained model
model = PPO.load("artifacts/ppo_simple_hidden_role")

# Create test environment
config = ScenarioConfig(
    scenario_id="eval",
    tier=2,
    num_agents=4,
    num_imposters=1,
)

env = Tier2Environment()
state = env.reset(config, seed=100)

# Run episode with trained policy
total_reward = 0
for step in range(200):
    # Get observations
    obs_dict = env.get_observations(state)

    # Get actions from trained model
    # (Note: would need wrapper for proper vectorization)
    # ... prediction code ...

    if env.is_terminal(state):
        break

print(f"Episode complete: {step} steps")
PY
```

### Use Case 4: Run Tests

```bash
# Test models
python tests/test_models.py

# Test abstract interface
python tests/test_base_interface.py

# Or use pytest (if installed)
pip install pytest
pytest tests/ -v
```

---

## Quick Start

### 1-Minute Setup

```bash
# Clone
git clone <repo>
cd multi-agent-rl-deception-simulation

# Install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify
PYTHONPATH=src python tests/test_models.py
```

### 5-Minute First Run

```bash
# Interactive gameplay
PYTHONPATH=src python scripts/play_gui.py

# In another terminal, train an agent
PYTHONPATH=src python scripts/train_simple.py --timesteps 5000
```

### 30-Minute Full Test

```bash
# 1. Run unit tests
python tests/test_models.py
python tests/test_base_interface.py

# 2. Run environment smoke test
PYTHONPATH=src python - <<'PY'
from multi_agent_deception import Tier2Environment, ScenarioConfig
env = Tier2Environment()
config = ScenarioConfig(scenario_id="test", tier=2, num_agents=4, num_imposters=1)
state = env.reset(config, seed=42)
print("✓ Environment ready")
PY

# 3. Train for 10k timesteps
PYTHONPATH=src python scripts/train_simple.py --timesteps 10000

# 4. Inspect results
ls -la artifacts/
```

---

## Advanced Configuration

### Scenario Configuration

Create custom scenarios via `ScenarioConfig`:

```python
from multi_agent_deception import ScenarioConfig, Tier2Environment

# Small map, few agents
config = ScenarioConfig(
    scenario_id="small_test",
    tier=2,
    num_agents=2,
    num_imposters=0,
    grid_size=5,
    tasks_per_agent=2,
    max_ticks=100,
    task_duration=2,
    observation_radius=3,
    seed=42,
)

env = Tier2Environment()
state = env.reset(config, seed=42)
```

### Hyperparameter Tuning

Modify `scripts/train_simple.py`:

```python
# Line ~60 - PPO parameters
policy_kwargs = dict(
    net_arch=[128, 128],      # Hidden layer sizes
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,       # Adjust learning rate
    n_steps=256,              # Rollout length
    batch_size=256,           # Batch size
    n_epochs=4,               # PPO epochs per update
    gamma=0.99,               # Discount factor
    gae_lambda=0.95,          # GAE lambda
    clip_range=0.2,           # PPO clip range
)
```

### Custom Environment

Subclass `DeceptionGameEnvironment`:

```python
from multi_agent_deception import DeceptionGameEnvironment, GameState, GameEvent

class MyEnvironment(DeceptionGameEnvironment):
    def reset(self, scenario_config, seed):
        # Initialize
        return GameState(...)

    def step(self, game_state, actions):
        # Execute tick
        return new_state, events_dict

    def get_observations(self, game_state):
        # Partial observability
        return observations_dict

    def is_terminal(self, game_state):
        # Win conditions
        return is_done

    def get_state(self):
        return self._state
```

---

## Development Roadmap

### Phase 0: Foundation (Weeks 1-4) - **IN PROGRESS**

**Completed:**
- ✅ **Story 1:** Refactor Environment & Abstract Interface
  - `DeceptionGameEnvironment` base class
  - `Tier2Environment` refactored implementation
  - Core data models (GameState, GameAction, GameEvent, AgentObservation)
  - Partial observability implementation
  - Comprehensive tests

**Upcoming:**
- 🔄 **Story 2:** Tier 1 Dialogue-Only Environment
- 🔄 **Story 3:** Game Rules Engine
- 🔄 **Story 4-6:** LLM Integration (Clients, Parsing, Prompts)
- 🔄 **Story 7:** Event Logging System
- 🔄 **Story 8:** Data Models Completion
- 🔄 **Story 9:** Game Runner Orchestration
- 🔄 **Story 10:** Integration Testing & Validation

### Phase 1: Tier 1 Gameplay (Weeks 5-14)

- Implement complete dialogue-based deception game
- Voting and meeting mechanics
- LLM integration for agent reasoning
- Strategy pattern analysis

### Phase 2: Tier 2 Spatial Enhancement (Weeks 15-28)

- Add movement logs as alibi evidence
- Proximity-based accusations
- Task completion tracking
- Merged dialogue+spatial gameplay

### Phase 3: Analysis & Research (Weeks 29-48)

- Mechanistic interpretability
- Strategy extraction
- LLM behavior analysis
- Publication-ready metrics

---

## Architecture Decision Records

### Why Abstract Interface?

**Problem:** Different game variants (Tier 1 dialogue, Tier 2 spatial, future hybrids) share core mechanics but need different implementations.

**Solution:** `DeceptionGameEnvironment` abstract base class provides clear contract:
- All implementations handle state transitions identically
- Partial observability computed consistently
- Terminal conditions checked uniformly
- LLM integration layer can work with any variant

**Benefits:**
- ✅ Code reuse across variants
- ✅ Easy to swap implementations
- ✅ Clear extension points
- ✅ Testable in isolation

### Why Dataclasses for State?

**Problem:** Game state needs serialization, validation, and clear structure.

**Solution:** Use `dataclass` models (GameState, GameAction, etc.)
- JSON-serializable
- Type-hints for IDE support
- Immutable-friendly patterns
- Easy to log and replay

### Why Partial Observability?

**Problem:** Hidden role deception requires agents to NOT see everything.

**Solution:** `get_observations()` returns filtered views:
- Agents only see nearby agents (observation_radius)
- Roles remain hidden
- Simulates fog of war / limited perception
- Enables deceptive strategies

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Ensure `PYTHONPATH=src` before running |
| `ImportError: gymnasium` | Run `pip install -r requirements.txt` |
| Pygame not rendering | Install desktop dependencies: `apt-get install libsdl2-2.0-0` (Linux) |
| CUDA errors (training) | Use CPU: modify `train_simple.py` to remove GPU device setting |
| Out of memory | Reduce `grid_size`, `num_agents`, or `timesteps` |
| Tests failing | Check Python version ≥ 3.10 |

---

## Citation

If you use this framework in research, please cite:

```bibtex
@misc{among-us-deception,
  title={Multi-Agent RL for Strategic Deception},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/your-repo}},
}
```

---

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

---

## Contact & Support

- **Issues:** GitHub Issues tab
- **Discussions:** GitHub Discussions
- **Email:** [your-email@example.com]

---

## Further Reading

- **Complete Architecture:** See `docs/ARCHITECTURE.md`
- **Implementation Guide:** See `docs/IMPLEMENTATION_GUIDE.md`
- **Theoretical Background:** See `README_THEORETICAL.md`
- **Story Details:** See `docs/stories/*.md`

---

**Last Updated:** 2025-11-02
**Status:** Phase 0 - Story 1 Complete, Stories 2-10 Planned
