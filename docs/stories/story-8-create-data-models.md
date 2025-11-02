# Story 8: Create Data Models & Serialization

**Document Date:** 2025-11-02
**Status:** Ready for Development
**Story ID:** PHASE0-STORY-8
**Epic:** Phase 0 Foundation: Simulation & LLM Integration
**Priority:** Critical (Foundation for all Phase 0 components)
**Estimated Effort:** 2-4 days
**Complexity:** Medium

---

## Executive Summary

Create unified, serializable dataclass models for game state, actions, events, observations, and complete game logs. These models form the data contract between all Phase 0 components (Simulation, LLM Integration, Metrics, Experiment Orchestration) and enable complete reproducibility through JSON export.

---

## Story Title

Create Serializable Data Models for Game State, Actions, Events & Logging - Brownfield Addition

---

## User Story

As a **developer implementing the Phase 0 architecture**,
I want **unified dataclass models for game state, actions, events, and observations**,
So that **all components can seamlessly share game data with full type safety and JSON serialization**.

---

## Story Context

### Existing System Integration

- **Integrates with:** Phase 0 Foundation across all layers (Simulation, LLM, Metrics, Experiment Orchestration)
- **Technology:** Python dataclasses, JSON serialization, existing PettingZoo observation/action spaces
- **Follows pattern:** Dataclass-based state management (consistent with PettingZoo environments)
- **Touch points:**
  - `sim/environment.py` (will use GameState, AgentObservation)
  - `llm/parser.py` (will parse JSON → GameAction)
  - `metrics/event_logger.py` (will record GameEvent, GameLog)
  - `experiments/game_runner.py` (will orchestrate with GameState)

### Current Gap

Currently, game state is managed inconsistently across different components. This story establishes a single source of truth for all game data, enabling:
- Type-safe data passing between components
- Complete game reproducibility via JSON export
- Mechanistic analysis of LLM decision-making
- Straightforward metrics computation

---

## Acceptance Criteria

### Core Data Models

#### 1. GameState Dataclass

Captures complete game state at any tick.

**Required fields:**
- `game_id: str` - unique identifier for this game run
- `scenario_id: str` - scenario configuration reference
- `tick: int` - current game tick/turn number
- `agents: list[Agent]` - all agents in game with metadata
- `active_agents: list[str]` - list of active agent IDs (may exclude ejected agents)
- `roles: dict[str, str]` - mapping agent_id → role ("imposter"/"crewmate")
- `observations: dict[str, AgentObservation]` - partial observability view per agent
- `game_status: str` - "ongoing" or "completed"
- `winner: Optional[str]` - None, "imposters", or "crewmates"
- `metadata: dict[str, any]` - scenario metadata, difficulty, expected_duration, etc.

**Methods:**
- `to_dict() -> dict` - serialize to dict
- `from_dict(cls, data: dict) -> GameState` - deserialize from dict
- `to_json() -> str` - serialize to JSON string
- `from_json(cls, json_str: str) -> GameState` - deserialize from JSON string

#### 2. GameAction Dataclass

Represents a parsed LLM decision.

**Required fields:**
- `agent_id: str` - which agent took the action
- `action: str` - action type (e.g., "discuss", "vote", "accuse", "move", "perform_task")
- `parameters: dict[str, any]` - action-specific parameters (e.g., {"target": "agent_2"} for vote)
- `reasoning: str` - LLM reasoning chain (complete model output explanation)
- `confidence: Optional[float]` - confidence score [0-1] if provided by LLM
- `timestamp: str` - ISO format timestamp when action was parsed
- `is_valid: bool` - whether action passed safety filter validation
- `validation_error: Optional[str]` - if is_valid=False, explanation of validation failure

**Methods:**
- `to_dict() -> dict` - serialize to dict
- `from_dict(cls, data: dict) -> GameAction` - deserialize from dict
- `__post_init__()` - validate action is in allowed types, parameters is dict

#### 3. GameEvent Dataclass

Represents the outcome of a GameAction.

**Required fields:**
- `tick: int` - game tick when event occurred
- `agent_id: str` - which agent triggered the event
- `action: GameAction` - the action that produced this event
- `event_type: str` - outcome type (e.g., "action_executed", "agent_ejected", "task_completed", "discussion_logged", "vote_recorded")
- `changes: dict[str, any]` - description of state changes (e.g., {"agent_2_role": "ejected"})
- `metrics_delta: dict[str, float]` - metrics updated by this event (e.g., {"suspicion_level": -0.1})
- `observations_updated: dict[str, AgentObservation]` - updated observations for affected agents
- `timestamp: str` - ISO timestamp of event occurrence

**Methods:**
- `to_dict() -> dict` - serialize to dict
- `from_dict(cls, data: dict) -> GameEvent` - deserialize from dict

#### 4. GameLog Dataclass

Represents complete game history for analysis and reproducibility.

**Required fields:**
- `game_id: str` - unique game identifier (matches initial GameState)
- `scenario_id: str` - scenario configuration used
- `model_name: str` - LLM model name (e.g., "claude-3-opus", "gpt-4-turbo")
- `seed: int` - random seed for reproducibility
- `created_at: str` - ISO timestamp of game start
- `completed_at: str` - ISO timestamp of game end
- `duration_ticks: int` - total ticks game ran
- `final_state: GameState` - final game state (winner determined)
- `events: list[GameEvent]` - complete list of all events in order
- `metadata: dict[str, any]` - game metadata (final_winner, reason_ended, total_cost_usd, etc.)

**Methods:**
- `to_dict() -> dict` - serialize to dict (all nested events included)
- `from_dict(cls, data: dict) -> GameLog` - deserialize from dict
- `to_json(filepath: str) -> None` - export complete log to JSON file
- `from_json(cls, filepath: str) -> GameLog` - load complete log from JSON file
- `get_events_for_agent(agent_id: str) -> list[GameEvent]` - filter events by agent

#### 5. AgentObservation Dataclass

Represents what one agent observes (partial observability).

**Required fields:**
- `agent_id: str` - which agent has this observation
- `tick: int` - observation at this tick
- `visible_agents: list[str]` - list of other agents this agent can observe
- `visible_information: dict[str, any]` - observable state (role_of_agent_2, position_agent_3, tasks_seen, etc.)
- `discussion_history: list[DiscussionMessage]` - recent discussion messages (agent may not see all)
- `own_beliefs: dict[str, any]` - this agent's internal beliefs (agent_2_likely_imposter: 0.7, etc.)
- `known_roles: dict[str, str]` - roles confirmed to be true (agent_id → role; only if revealed)

**Methods:**
- `to_dict() -> dict` - serialize to dict
- `from_dict(cls, data: dict) -> AgentObservation` - deserialize from dict

### Supporting Data Models

#### 6. Agent Dataclass

Represents an agent in the game.

**Required fields:**
- `agent_id: str` - unique identifier
- `role: str` - "imposter" or "crewmate"
- `status: str` - "alive", "ejected", "dead" (tier 2)
- `completed_tasks: list[str]` - task IDs completed (crewmates only)

#### 7. DiscussionMessage Dataclass

Represents a discussion phase message.

**Required fields:**
- `tick: int` - when message was sent
- `speaker_id: str` - who sent the message
- `message_text: str` - content of message
- `timestamp: str` - ISO timestamp

#### 8. TaskInfo Dataclass

Represents a task (Tier 2).

**Required fields:**
- `task_id: str` - unique identifier
- `location: Optional[tuple[int, int]]` - grid location (None for Tier 1)
- `completion_status: str` - "pending", "completed"
- `completed_by: Optional[str]` - agent_id who completed it

#### 9. VoteRound Dataclass

Represents a voting round.

**Required fields:**
- `tick: int` - when vote occurred
- `votes: dict[str, str]` - mapping voter_agent_id → voted_agent_id
- `ejected_agent: Optional[str]` - who was ejected (None if tie)
- `vote_counts: dict[str, int]` - vote tallies by target

---

## Integration Requirements

### Serialization & Interoperability

1. **JSON Round-Trip:** All models must serialize to JSON and deserialize back with identical content
   - GameLog → `json.dumps(log.to_dict())` → valid JSON file
   - Read file → `json.loads(...)` → `GameLog.from_dict(...)` → identical GameLog

2. **No Breaking Changes:** Existing PettingZoo environment observation/action space usage unaffected
   - Models are additive; they don't replace existing code
   - GameState can coexist with existing environment state structures

3. **Nested Serialization:** Complex nested models must serialize correctly
   - GameLog contains list of GameEvents
   - Each GameEvent contains GameAction
   - All nested objects must round-trip through JSON

4. **Type Preservation:** JSON deserialization must preserve types
   - floats stay floats (not strings)
   - ints stay ints
   - dict values maintain their types
   - ISO timestamps parse back to strings (for simplicity)

5. **Complete Information:** No information loss during serialization
   - Every field serialized; nothing omitted
   - Reasoning chains fully captured (complete LLM output)
   - All observations preserved for mechanistic analysis

---

## Quality Requirements

### Validation & Type Safety

1. **Type Hints:** All fields have explicit type hints (no `Any` except where necessary)

2. **Field Validation:** `__post_init__` validates critical constraints
   - `role` must be in {"imposter", "crewmate"}
   - `action` must be in allowed action types
   - `status` must be in {"alive", "ejected", "dead"}
   - Numeric fields (tick, confidence) are in expected ranges

3. **No Silent Failures:** Validation errors raise exceptions (ValueError, TypeError)
   - Clear error messages indicating what failed and why

4. **Complete Unit Test Coverage:** 100% coverage for `core/models.py`
   - Test each model's `to_dict()` / `from_dict()` methods
   - Test JSON round-trip for complex models
   - Test validation errors
   - Test nested model serialization

5. **DataFrame Compatibility:** Models can convert to/from pandas DataFrames for analysis
   - Flat models (GameAction) → simple DataFrame
   - Complex models (GameLog) → structured DataFrame with event columns

---

## Technical Notes

### Implementation Approach

**File Structure:**
```
core/
├── models.py          # All dataclass definitions
├── __init__.py        # Export key models
└── validation.py      # Shared validation logic (optional)
```

**Serialization Strategy:**
- Use Python `dataclasses` module (standard library, no dependencies)
- Manual `to_dict()` / `from_dict()` methods (simpler than Pydantic for this use case)
- ISO format strings for timestamps (JSON-compatible)
- Recursive serialization for nested models

**Example Structure:**
```python
@dataclass
class GameAction:
    agent_id: str
    action: str
    parameters: dict
    reasoning: str
    confidence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_valid: bool = False
    validation_error: Optional[str] = None

    def __post_init__(self):
        if self.role not in {"imposter", "crewmate"}:
            raise ValueError(f"Invalid role: {self.role}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'GameAction':
        return cls(**data)
```

### Existing Pattern Reference

Follow patterns from:
- **PettingZoo's Gymnasium Spaces:** Type hints, `field()` for defaults
- **ARCHITECTURE.md ScenarioConfig:** Clean, minimal dataclasses with clear field documentation
- **Python dataclasses module:** Standard library; no external dependencies

### Key Constraints

1. **No Circular References:** Dataclass hierarchy must be acyclic (GameLog → GameEvent → GameAction, but no reverse)

2. **JSON Compatibility:** All field types must be JSON-serializable
   - Use strings for dates/times (not datetime objects)
   - Use dicts and lists (not custom objects at leaf level)
   - No numpy arrays at top level (convert to lists)

3. **Backward Compatibility:** No changes to existing environment interfaces
   - Models are purely additive
   - Environment step/reset methods unchanged

4. **Simplicity Over Features:** Start minimal; expand as needed
   - Only include fields explicitly required by Phase 0
   - Don't anticipate Phase 1+ needs prematurely

---

## Definition of Done

### Implementation Checklist

- [ ] Create `core/models.py` with all dataclass definitions
- [ ] All dataclasses include complete docstrings
- [ ] All fields have explicit type hints
- [ ] `__post_init__()` validation implemented where needed
- [ ] `to_dict()` and `from_dict()` methods working for all models
- [ ] Nested dataclass serialization working (GameLog.events → JSON)
- [ ] JSON round-trip test passes (serialize → deserialize → identical)

### Testing Checklist

- [ ] Unit tests: 100% coverage for `core/models.py`
- [ ] Test each model's `to_dict()` method
- [ ] Test each model's `from_dict()` method
- [ ] Test JSON round-trip: model → JSON file → loaded model (identical)
- [ ] Test validation errors (invalid roles, out-of-range values, etc.)
- [ ] Test nested serialization (GameLog with 50+ events)
- [ ] Test that timestamps parse consistently
- [ ] All tests pass; no warnings

### Documentation Checklist

- [ ] Complete docstring for each dataclass explaining purpose
- [ ] Field-level comments for non-obvious fields
- [ ] Example GameLog object documented in code comments
- [ ] Usage examples in module docstring (how to create, serialize, deserialize)

### Integration Checklist

- [ ] No changes to existing PettingZoo environment code
- [ ] Models importable from `core.models`
- [ ] Can be used in other Phase 0 components (Story 1, 4, 7, 9) without modification
- [ ] JSON export matches example structure in ARCHITECTURE.md

### Example Output Checklist

- [ ] Create example GameLog in JSON format
- [ ] Save to `examples/sample-game-log.json`
- [ ] JSON is valid, human-readable, includes full game
- [ ] Can load example JSON back into GameLog object

---

## Risk & Compatibility

### Minimal Risk Assessment

**Primary Risk:** Type mismatches between model definitions and actual usage in other components
**Mitigation:**
- Define models based on ARCHITECTURE.md (design already validated)
- Use test-driven development (write tests before implementation)
- Other components can use models as they're developed (story dependency management)

**Compatibility Verification:**
- [ ] No breaking changes to existing PettingZoo environment
- [ ] Dataclasses can coexist with legacy state structures
- [ ] JSON serialization doesn't affect simulation execution
- [ ] All existing tests still pass on refactored code

### Rollback Plan
- Models are purely additive; no changes to existing code
- If models need adjustment, simple git revert
- No database migrations or system-wide updates required

---

## Success Criteria

The story is complete when:

1. **Functionality**: All 9 dataclass models defined with complete fields and validation
2. **Type Safety**: Full type hints; no `Any` except where documented
3. **Serialization**: JSON round-trip works for all models; no information loss
4. **Testing**: 100% unit test coverage; all edge cases tested
5. **Integration**: Other Phase 0 stories (1-7, 9) can use models without modification
6. **Reproducibility**: Complete GameLog can be exported, shared, and re-imported identically

---

## Related Documents

- **Phase 0 Epic:** `/docs/epics/phase-0-foundation.md` - Full epic context
- **Architecture Design:** `/docs/ARCHITECTURE.md` - Complete system design
- **Story Dependency:** Enables Stories 1-9; required for Stories 4, 7, 9

---

## Story Workflow

### Development Phases

**Phase 1: Model Definition (Day 1)**
- Define all 9 dataclass structures
- Add field documentation
- Implement `__post_init__` validation

**Phase 2: Serialization (Day 1-2)**
- Implement `to_dict()` / `from_dict()` for all models
- Test JSON round-trip on simple models first
- Handle nested serialization (GameLog → events)

**Phase 3: Testing & Validation (Day 2-4)**
- Write unit tests (100% coverage)
- Test edge cases (empty events, None values, etc.)
- Test with example GameLog (50+ events)
- Verify no regressions in existing code

**Phase 4: Documentation & Examples (Day 4)**
- Complete all docstrings
- Create example JSON output
- Document usage patterns for downstream stories

---

## Story Status

**Status:** ✅ Ready for Development
**Last Updated:** 2025-11-02
**Next Review:** Upon completion; before starting Story 1 (Refactor Environment)

