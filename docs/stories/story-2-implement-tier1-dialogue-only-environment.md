# Story 2: Implement Tier 1 Dialogue-Only Environment

**Document Date:** 2025-11-02
**Status:** Ready for Development
**Story ID:** PHASE0-STORY-2
**Epic:** Phase 0 Foundation: Simulation & LLM Integration
**Priority:** Critical (Required for LLM integration testing)
**Estimated Effort:** 2-4 days
**Complexity:** Medium (New implementation, established patterns)

---

## Executive Summary

Implement `Tier1Environment`, a dialogue-only game variant without spatial mechanics. This simplified environment enables early LLM integration testing and provides a foundation for understanding game dynamics before adding spatial complexity. Tier 1 games focus on discussion, accusation, defense, and voting as the primary mechanics.

---

## Story Title

Implement Tier 1 Dialogue-Only Environment - Brownfield Addition

---

## User Story

As a **Phase 0 developer**,
I want **a simplified dialogue-only game environment (Tier 1)**,
So that **I can test LLM integration with a focus on deception, persuasion, and social reasoning without spatial complexity**.

---

## Story Context

### Existing System Integration

- **Integrates with:** `DeceptionGameEnvironment` abstract interface (Story 1)
- **Technology:** Python dataclasses, GameState/GameAction/GameEvent (Story 8)
- **Follows pattern:** PettingZoo-style reset/step interface, partial observability model
- **Touch points:**
  - Story 1 abstract interface (will implement)
  - Story 3 (Game Rules) - uses for action validation and win condition checking
  - Story 4-6 (LLM Integration) - first environment to test with LLM
  - Story 9 (Game Runner) - orchestrates Tier 1 games
  - Story 10 (Testing) - comprehensive Tier 1 game testing

### Current Gap

Tier 1 is simpler than Tier 2 (no spatial component) but captures the core deception mechanics:
- Hidden roles (Imposters vs Crewmates)
- Information asymmetry (agents don't know others' roles)
- Discussion and voting (primary social mechanic)
- Win conditions based on role objectives

This story enables early LLM testing without spatial complexity, allowing focus on dialogue, deception, and persuasion.

---

## Acceptance Criteria

### Core Game Mechanics

#### 1. Tier1Environment Class

**File:** `src/environments/tier1_environment.py` (or `sim/tier1_environment.py`)

**Relationship:** `class Tier1Environment(DeceptionGameEnvironment)` - new implementation

**Constructor:**
```python
class Tier1Environment(DeceptionGameEnvironment):
    def __init__(self, scenario_config: ScenarioConfig):
        self.scenario_config = scenario_config
        self.num_agents = scenario_config.num_agents
        self.num_imposters = scenario_config.num_imposters
        self.num_crewmates = self.num_agents - self.num_imposters
        self.agents = {}  # agent_id -> Agent
        self.roles = {}   # agent_id -> role
        self.state = None # Current GameState
        self.tick = 0
        self.seed = None
        self.discussion_history = []  # List of DiscussionMessage
        self.voting_history = []  # List of VoteRound
```

**Implementation Requirements:**

1. **Agent Initialization:**
   - Create N agents (num_agents from scenario)
   - Randomly assign roles: M imposters, N-M crewmates (seed-deterministic)
   - Initialize agent status (alive, dead, ejected)
   - Set initial beliefs (can be None; no initial suspicion)

2. **Game State Management:**
   - Use GameState dataclass (Story 8) for all state
   - Track: agents, roles, active_agents, tick, discussion_history, voting_history
   - Support state snapshots at any tick (for logging)

3. **Action Types (Tier 1):**
   - `discuss`: Post a message in discussion phase
     - Parameters: {message: str}
     - Outcome: Message added to discussion history
   - `accuse`: Make explicit accusation
     - Parameters: {target: agent_id}
     - Outcome: Event logged; affects suspicion but doesn't trigger vote
   - `defend`: Defend against suspicion
     - Parameters: {reasoning: str, defense_statement: str}
     - Outcome: Defense logged; may affect perceptions
   - `vote`: Vote to eject a player (voting phase only)
     - Parameters: {target: agent_id}
     - Outcome: Vote recorded; tallied with all votes

**Acceptance:**
- [ ] All action types correctly executed
- [ ] Discussion messages logged with speaker and timestamp
- [ ] Accusations recorded but don't directly eject
- [ ] Votes tallied correctly
- [ ] Invalid actions rejected (handled by GameRules in Story 3)

---

#### 2. Discussion Phase

**Mechanics:**
- Round-robin: Each agent gets turn to speak (or multiple turns)
- Each agent receives prompt (from Story 6 templates)
- LLM returns action (discuss, accuse, defend, or propose vote)
- Messages added to discussion history in order

**Game Flow:**
```
FOR EACH DISCUSSION ROUND:
  FOR EACH AGENT (in round-robin):
    - Get agent observation (AgentObservation from Story 1)
    - Render prompt with: game state, discussion history, agent role
    - Call LLM via API (Story 4-5)
    - Parse action (discuss/accuse/defend/vote_proposal)
    - If "discuss": add message to history
    - If "accuse": record accusation
    - If "defend": record defense
    - If "vote_proposal": trigger voting phase
    - Log event
  END FOR
  CONTINUE FOR configured num_discussion_rounds
END FOR
```

**Implementation:**
- `discussion_phase(game_state: GameState) -> DiscussionResults`
- Tracks all discussion messages
- Identifies when agents want to vote (triggers voting phase)
- Returns DiscussionResults with all messages and proposals

**Acceptance:**
- [ ] Agents speak in round-robin order
- [ ] Discussion history tracked correctly
- [ ] Messages timestamped and attributed to speaker
- [ ] Voting proposals detected
- [ ] Discussion continues until max_rounds or vote proposed

---

#### 3. Voting Phase

**Mechanics:**
- Synchronous voting: All active agents vote simultaneously
- Each agent votes for one target (or passes)
- Majority rules: Agent with most votes is ejected
- Ties handled gracefully (random selection or no eject)

**Game Flow:**
```
VOTING PHASE:
  1. All agents notified: "Voting phase started"
  2. Each agent called by LLM to vote
  3. Prompt includes: game state, discussion history, vote context
  4. LLM returns action: {action: "vote", parameters: {target: agent_id}}
  5. Votes tallied (delegated to GameRules in Story 3)
  6. Majority target ejected (marked as status="ejected")
  7. Event logged: who was ejected, vote counts
  8. Game continues to next tick
```

**Implementation:**
- `voting_phase(game_state: GameState, agent_actions: dict[str, GameAction]) -> VotingResults`
- Collects votes from all active agents
- Delegates validation to GameRules (Story 3)
- Applies eject decision to game state
- Returns VotingResults with vote tallies and ejected agent

**Acceptance:**
- [ ] All active agents can vote
- [ ] Votes tallied correctly
- [ ] Majority target identified
- [ ] Ejected agent removed from active_agents
- [ ] Role revealed upon eject (agents now see ejected agent's role)
- [ ] Ties handled (configurable: random or no eject)
- [ ] Event logged with vote details

---

#### 4. Win Conditions

**Imposter Win:**
- All crewmates are dead/ejected, OR
- Imposters equal or outnumber crewmates (can't be outvoted), OR
- Crewmates fail to complete all tasks within max_ticks

**Crewmate Win:**
- All imposters ejected, OR
- All crewmates complete assigned tasks (if applicable)

**Game Over:**
- One team has won, OR
- Max ticks reached (timeout)

**Implementation:**
- Delegated to GameRules.check_win_condition() (Story 3)
- Tier1Environment calls this after each state change
- Sets game_state.game_status = "completed" when terminal
- Sets game_state.winner = "imposters" or "crewmates"

**Acceptance:**
- [ ] Imposters win when all crewmates ejected
- [ ] Crewmates win when all imposters ejected
- [ ] Game ends at max_ticks
- [ ] Terminal state detected correctly
- [ ] Winner determined accurately

---

### Tier 1 Specific Features

#### 5. Simplified Observability (No Spatial)

**What Agents Observe:**
- All other agents (presence in game)
- Agent IDs/names
- Who is alive vs. ejected (always)
- Discussion history (all messages)
- Their own role (always)
- Revealed roles (agents ejected reveal their role)

**What Agents DO NOT Observe:**
- Roles of living agents (hidden until ejected)
- Private thoughts/reasoning of others
- Their own suspicion level (internal)

**Implementation:**
- `get_observations(game_state)` returns dict[str, AgentObservation]
- For Tier 1: All agents see all agents (no proximity filtering)
- visible_agents: list of all other agents
- visible_information: includes discussion history, known roles
- known_roles: only revealed roles
- own_beliefs: internal model (can be empty for Tier 1 v1)

**Acceptance:**
- [ ] Agents see all discussion history
- [ ] Roles hidden until revealed
- [ ] All agents visible (no spatial proximity)
- [ ] Ejected agents' roles revealed
- [ ] AgentObservation structure correct

---

#### 6. No Spatial Component

**Explicitly NOT implemented (unlike Tier 2):**
- No grid/map
- No positions
- No movement
- No proximity-based observation
- No task locations
- No physical "kill" actions

**This keeps Tier 1 focused on:**
- Discussion and dialogue
- Deception through words
- Persuasion and accusation
- Social reasoning and suspicion

---

### Game Loop Integration

#### 7. Abstract Interface Implementation

**Tier1Environment must implement all abstract methods from Story 1:**

```python
def reset(self, scenario_config: ScenarioConfig, seed: int) -> GameState:
    """Initialize Tier 1 game from scenario."""
    # 1. Set seed for determinism
    # 2. Create agents and assign roles (seed-deterministic)
    # 3. Initialize discussion history (empty)
    # 4. Initialize voting history (empty)
    # 5. Create initial GameState
    # 6. Return initial state

def step(self, game_state: GameState, actions: dict[str, GameAction]) -> tuple[GameState, dict[str, GameEvent]]:
    """Execute one game tick with agent actions."""
    # 1. Validate actions (via GameRules - Story 3)
    # 2. Execute actions (update discussion/voting history)
    # 3. Check for voting phase trigger
    # 4. If voting: run voting phase
    # 5. Check win conditions (via GameRules)
    # 6. Update game state
    # 7. Log events
    # 8. Return (new_state, events_dict)

def get_observations(self, game_state: GameState) -> dict[str, AgentObservation]:
    """Get what each agent observes (Tier 1: all see all)."""
    # For each agent:
    #   - visible_agents: list of all other agents
    #   - visible_information: discussion history, known roles
    #   - discussion_history: all messages
    #   - known_roles: only revealed (ejected) roles
    # Return dict[str, AgentObservation]

def is_terminal(self, game_state: GameState) -> bool:
    """Check if game is over."""
    # Return True if winner determined or max_ticks reached

def get_state(self) -> GameState:
    """Export current game state."""
    # Return copy of self.state (or self._compute_state())
```

**Acceptance:**
- [ ] All abstract methods implemented
- [ ] Correct signatures and return types
- [ ] No errors or NotImplementedError
- [ ] Can be used interchangeably with other DeceptionGameEnvironment implementations

---

### Integration with Other Stories

#### 8. Story 3 (Game Rules) Integration

- Tier1Environment calls `GameRules.validate_action()` before executing
- Calls `GameRules.check_win_condition()` after state updates
- Delegates voting tally to `GameRules` (if provided)
- No duplication of rule logic

**Acceptance:**
- [ ] Invalid actions rejected
- [ ] Win conditions checked correctly
- [ ] Rules engine consulted before action execution

#### 9. Story 4-6 (LLM Integration) Readiness

Tier1Environment designed to work with LLM decision-making:
- Accepts GameAction (parsed LLM output)
- Works with GameState (LLM gets this in prompts)
- Generates AgentObservation (LLM uses for context)
- Returns GameEvent (logged for metrics)

**Acceptance:**
- [ ] Can run complete game with LLM-controlled agents
- [ ] LLM actions executed correctly
- [ ] Observations given to LLM are consistent
- [ ] Game flow works end-to-end

---

## Technical Notes

### Implementation Approach

**Phase 1: Core Environment (Day 1)**
1. Implement Tier1Environment class skeleton
2. Implement reset() - agent/role initialization
3. Implement is_terminal() - win condition checking
4. Implement get_state() - state export

**Phase 2: Discussion & Voting (Days 1-2)**
1. Implement discussion_phase()
2. Implement voting_phase()
3. Implement step() to orchestrate game flow
4. Handle action execution and event generation

**Phase 3: Observability & Integration (Days 2-3)**
1. Implement get_observations()
2. Ensure partial observability working correctly
3. Integrate with GameRules (Story 3) for validation
4. Ensure GameState/GameAction/GameEvent compatibility

**Phase 4: Testing & Validation (Days 3-4)**
1. Unit tests for all methods
2. Integration tests with GameRules
3. End-to-end game tests (multiple complete games)
4. Determinism tests (same seed = same outcomes)
5. Verify backward compatibility with abstract interface

### Existing Pattern Reference

Follow patterns from:
- **Story 1 (DeceptionGameEnvironment):** Implement abstract interface
- **Story 8 (Data Models):** Use GameState, GameAction, GameEvent, AgentObservation
- **ARCHITECTURE.md Section 3.2:** Tier 1 environment design
- **PettingZoo ParallelEnv:** reset/step interface style

### Key Constraints

1. **Determinism:** Given same seed, game must be deterministic
   - All randomness seeded (role assignment, tie-breaking, etc.)
   - No unsynchronized random sources
   - Can reproduce exact same game flow

2. **No Spatial:** Explicitly exclude spatial mechanics
   - Don't implement grid or positions
   - Don't implement movement or proximity
   - Keep focus on discussion/voting

3. **Configurable Parameters:** Via ScenarioConfig
   - num_agents, num_imposters
   - max_ticks, discussion_rounds_per_tick
   - meeting_frequency (if applicable)
   - Anything not in scenario_config should not be hard-coded

4. **Game Rules Delegation:** Don't duplicate rule logic
   - Action validation → GameRules
   - Win condition checking → GameRules
   - Vote tallying → GameRules (optional)
   - Tier1Environment orchestrates; GameRules decides

---

## Definition of Done

### Implementation Checklist

- [ ] `Tier1Environment` class created in `src/environments/tier1_environment.py`
  - [ ] Constructor initializes agents and roles
  - [ ] Uses ScenarioConfig for parameters
  - [ ] Seeds all randomness for determinism

- [ ] Abstract Interface Implementation
  - [ ] `reset(scenario_config, seed)` → GameState
  - [ ] `step(game_state, actions)` → (GameState, events_dict)
  - [ ] `get_observations(game_state)` → dict[str, AgentObservation]
  - [ ] `is_terminal(game_state)` → bool
  - [ ] `get_state()` → GameState

- [ ] Discussion Phase
  - [ ] `discussion_phase()` runs round-robin agent speaking
  - [ ] Agents can discuss, accuse, defend
  - [ ] Messages logged with speaker and timestamp
  - [ ] Discussion history maintained

- [ ] Voting Phase
  - [ ] `voting_phase()` collects votes from all agents
  - [ ] Votes tallied correctly
  - [ ] Majority target ejected (or no eject if tie)
  - [ ] Roles revealed upon eject
  - [ ] Events logged

- [ ] Win Conditions
  - [ ] Imposters win when all crewmates ejected
  - [ ] Crewmates win when all imposters ejected
  - [ ] Game ends at max_ticks
  - [ ] Winner correctly determined

- [ ] Observability
  - [ ] All agents see all other agents
  - [ ] Roles hidden until revealed
  - [ ] Discussion history visible to all
  - [ ] AgentObservation structure correct

### Testing Checklist

- [ ] Unit tests
  - [ ] `reset()` initializes game correctly
  - [ ] Agents created with correct roles
  - [ ] `step()` executes actions without errors
  - [ ] Discussion messages logged
  - [ ] Voting phase works correctly
  - [ ] Win conditions detected
  - [ ] `get_observations()` returns correct structure

- [ ] Integration tests
  - [ ] Works with Story 1 abstract interface
  - [ ] Works with Story 8 data models
  - [ ] Works with Story 3 GameRules (if available)
  - [ ] LLM-compatible action/observation flow

- [ ] End-to-End tests
  - [ ] Complete Tier 1 game runs to completion
  - [ ] Both imposter and crewmate wins tested
  - [ ] Multiple complete games in sequence
  - [ ] No errors or edge case failures

- [ ] Determinism tests
  - [ ] Same seed → identical role assignments
  - [ ] Same seed → identical game flow
  - [ ] Multiple runs with different seeds produce different outcomes

- [ ] Edge cases
  - [ ] Game with 1 imposter, N-1 crewmates
  - [ ] Game with 2+ imposters, 2+ crewmates
  - [ ] Max ticks reached (game timeout)
  - [ ] No agents left to vote
  - [ ] Agents vote out their own role?

### Code Quality Checklist

- [ ] All methods have docstrings with type hints
- [ ] Type hints on all attributes
- [ ] Error handling for invalid inputs
- [ ] Clear variable and method names
- [ ] No unused imports or dead code
- [ ] Code follows project style guide
- [ ] Complexity reasonable (no deeply nested logic)

### Documentation Checklist

- [ ] Tier1Environment docstring explains purpose and differences from Tier 2
- [ ] Each method documented with purpose, inputs, outputs, side effects
- [ ] Architecture decisions documented (why separate discussion/voting phases, etc.)
- [ ] Usage examples in code comments
- [ ] Any deviations from Story 1 interface explained

---

## Risk & Compatibility

### Primary Risks

**Risk 1: Game mechanics inconsistent with design intent**
- Mitigation: Early validation with Story 3 (GameRules)
- Mitigation: Design review before implementation
- Mitigation: Comprehensive edge case testing

**Risk 2: Discussion/voting logic doesn't work for LLM agents**
- Mitigation: Prototype with simple policy-based agents first
- Mitigation: Early integration testing (Story 9)
- Mitigation: Adjust action types if LLM can't generate them

**Risk 3: Determinism broken by hidden randomness**
- Mitigation: Trace all random() calls; ensure seeded
- Mitigation: Seed management comprehensive testing
- Mitigation: Side-by-side comparison: multiple games same seed

### Compatibility Verification

- [ ] Implements all abstract methods from Story 1
- [ ] Uses GameState, GameAction, GameEvent, AgentObservation correctly
- [ ] No import errors or missing dependencies
- [ ] Compatible with Story 3 (GameRules) interface
- [ ] Compatible with Story 8 (Data Models) types

### Rollback Plan

1. Keep changes in feature branch
2. Easy revert if fundamental design issue
3. If major issue, can strip back to Story 1 abstract interface
4. No changes to existing Story 1 code (pure extension)

---

## Success Criteria

The story is complete when:

1. **Functionality:** Tier1Environment fully implements abstract interface; all game mechanics working
2. **Game Flow:** Complete games run without errors; both teams can win
3. **Observability:** All agents see correct partial game state; roles hidden until revealed
4. **Integration:** Works with Story 1, 3, 8; ready for Story 4-6 LLM integration
5. **Testing:** 100% unit test coverage; end-to-end games pass; determinism verified
6. **Quality:** Well-documented, type-safe, no unexpected behaviors

---

## Related Documents

- **Phase 0 Epic:** `/docs/epics/phase-0-foundation.md` - Full epic context
- **Architecture Design:** `/docs/ARCHITECTURE.md` - Tier 1 design (Section 3.2)
- **Story 1:** `docs/stories/story-1-refactor-environment-abstract-interface.md` - Abstract interface
- **Story 3:** `docs/stories/story-3-build-game-rules-engine.md` (upcoming) - Game rules
- **Story 8:** `docs/stories/story-8-create-data-models.md` - Data models

---

## Story Workflow

### Development Timeline

**Day 1: Core Environment**
- [ ] Tier1Environment class skeleton
- [ ] Constructor with agent/role initialization
- [ ] reset() method
- [ ] get_state() method
- [ ] Initial abstract interface methods

**Days 1-2: Discussion & Voting**
- [ ] discussion_phase() implementation
- [ ] voting_phase() implementation
- [ ] step() orchestration
- [ ] Action execution and event generation
- [ ] GameState management

**Days 2-3: Observability & Integration**
- [ ] get_observations() implementation
- [ ] Partial observability verification
- [ ] GameRules integration (delegation for validation)
- [ ] Action type definitions

**Days 3-4: Testing & Validation**
- [ ] Comprehensive unit tests
- [ ] Integration tests with abstract interface
- [ ] End-to-end game tests
- [ ] Determinism verification
- [ ] Documentation and code review

### Dependency Management

**Blocks:** Stories 9 (Game Runner needs Tier 1 environment)
**Blocked by:** Story 1 (needs abstract interface), Story 8 (needs data models)
**Optional prior:** Story 3 (GameRules) - can integrate later if ready

---

## Story Status

**Status:** ✅ Ready for Development
**Last Updated:** 2025-11-02
**Next Review:** Upon completion; before starting Story 4 (LLM Client Abstraction)

**Notes:**
- Second story in critical path (after Story 1)
- Foundation for LLM integration testing (Stories 4-6)
- Simpler than Tier 2; good for validating architecture patterns
- Early testing with mock LLM recommended (before Story 4)

