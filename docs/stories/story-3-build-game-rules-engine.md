# Story 3: Build Game Rules Engine

**Document Date:** 2025-11-02
**Status:** Ready for Development
**Story ID:** PHASE0-STORY-3
**Epic:** Phase 0 Foundation: Simulation & LLM Integration
**Priority:** Critical (Required for Stories 1-2, 9)
**Estimated Effort:** 2-3 days
**Complexity:** Medium (Clear requirements, deterministic logic)

---

## Executive Summary

Implement the `GameRules` engine that enforces all game mechanics, validates actions, checks win conditions, and applies state transitions. This centralizes game rule logic, making it reusable across both Tier 1 and Tier 2 environments and ensuring consistent behavior.

---

## Story Title

Build Game Rules Engine - Brownfield Addition

---

## User Story

As a **Phase 0 developer**,
I want **a centralized GameRules engine that validates actions and determines game outcomes**,
So that **both Tier 1 and Tier 2 environments enforce identical rules and game logic is testable in isolation**.

---

## Story Context

### Existing System Integration

- **Integrates with:** Story 1 (abstract interface), Story 2 (Tier 1), Story 8 (data models)
- **Technology:** Python, dataclasses, pure logic (no I/O or external calls)
- **Follows pattern:** Static utility class with no mutable state (similar to rule engines)
- **Touch points:**
  - Story 1 (Tier2Environment) - calls for action validation and win condition checks
  - Story 2 (Tier1Environment) - calls for action validation and win condition checks
  - Story 4-6 (LLM Integration) - validates LLM-generated actions before execution
  - Story 9 (Game Runner) - orchestrates game loop with rule checks

### Current Gap

Currently, game rules are:
- Partially implemented in environment code (hard to test)
- Not formalized (implicit in logic rather than explicit rules)
- Difficult to reuse across Tier 1 and Tier 2
- No single source of truth for action validation

This story centralizes all rule logic into a testable, reusable engine.

---

## Acceptance Criteria

### Core Rule Engine

#### 1. GameRules Static Class

**File:** `src/game/rules.py` (or `sim/rules.py`)

**Purpose:** Enforce game mechanics; validate actions; compute outcomes

**Design Pattern:** Static utility class (no state; all methods are pure functions)

```python
class GameRules:
    """Enforce game rules; validate actions; compute outcomes."""

    # Static methods only (no __init__, no instance state)

    @staticmethod
    def validate_action(action: GameAction, game_state: GameState, agent_role: str) -> tuple[bool, Optional[str]]:
        """
        Validate if action is legal for this agent at this game state.

        Args:
            action: GameAction to validate
            game_state: Current game state
            agent_role: Agent's role ("imposter" or "crewmate")

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        pass

    @staticmethod
    def check_win_condition(game_state: GameState) -> Optional[str]:
        """
        Check if game has reached terminal state; return winner.

        Args:
            game_state: Current game state

        Returns:
            None if ongoing, "imposters" or "crewmates" if game over
        """
        pass

    @staticmethod
    def apply_action(action: GameAction, game_state: GameState) -> GameEvent:
        """
        Execute action; return event record.

        Args:
            action: GameAction to apply (assumed valid)
            game_state: Current game state

        Returns:
            GameEvent: Description of what changed
        """
        pass

    @staticmethod
    def handle_meeting(game_state: GameState, voting_results: VotingResults) -> GameState:
        """
        Process voting results; eject player; return new state.

        Args:
            game_state: Game state before meeting
            voting_results: VotingResults with vote tallies

        Returns:
            GameState: Updated state with ejected agent removed
        """
        pass
```

---

### Action Validation

#### 2. Action Validation Rules

**Method:** `validate_action(action, game_state, agent_role) -> (bool, Optional[str])`

**Validation Logic:**

**General Rules (all agents, all tiers):**
1. Agent must be alive (status == "alive")
2. Agent must exist in game (agent_id in active_agents)
3. Action type must be valid ("discuss", "vote", "accuse", "defend", "move", "use_vent", "kill", "perform_task")
4. Action parameters must be dict (non-null)
5. If action targets an agent, target must exist in game

**Tier 1 Specific Rules:**
- "discuss": Always valid (post message)
- "accuse": Target must be alive and different from self
- "defend": Always valid (post defense statement)
- "vote": Only valid during voting phase; target must be alive
- "move", "use_vent", "kill", "perform_task": INVALID for Tier 1

**Tier 2 Specific Rules:**
- All Tier 1 actions valid plus:
- "move": Target position must be valid (in bounds, passable)
- "perform_task": Agent must be at task location
- "use_vent": Only valid for imposters; vent must exist
- "kill": Only valid for imposters; target must be in proximity; not self

**Role Specific Rules:**
- Imposters can:
  - discuss, accuse, defend, vote
  - (Tier 2) move, use_vent, kill
  - (Tier 2) fake complete tasks
- Crewmates can:
  - discuss, accuse, defend, vote
  - (Tier 2) move, perform_task

**Acceptance:**
- [ ] All validation rules correctly implemented
- [ ] Invalid actions rejected with clear error messages
- [ ] Valid actions pass validation
- [ ] No false positives (valid rejected) or false negatives (invalid accepted)

---

#### 3. Specific Action Validation

**Discuss Action**
- Always valid (no parameters to check)
- Parameter: {message: str}
- Validation: message is non-empty string

**Vote Action**
- Only during voting phase
- Target must be different from self
- Target must be alive (can't vote for ejected agent)
- Parameter: {target: agent_id}

**Accuse Action**
- Always valid in discussion
- Parameter: {target: agent_id}
- Validation: target exists and is alive

**Defend Action**
- Always valid in discussion
- Parameter: {reasoning: str, defense_statement: str}
- Validation: both strings non-empty

**Move Action (Tier 2)**
- Position must be valid (in bounds)
- Position must be passable (no walls/obstacles)
- Parameter: {target_pos: tuple[int, int]}

**Perform Task Action (Tier 2)**
- Agent must be at task location (within proximity)
- Task must be pending (not already completed)
- Parameter: {task_id: str}

**Use Vent Action (Tier 2)**
- Only imposters
- Vent must exist on map
- Parameter: {vent_id: str}

**Kill Action (Tier 2)**
- Only imposters
- Target must be alive and nearby (proximity check)
- Target must not be self
- Parameter: {target: agent_id}

---

### Win Condition Checking

#### 4. Win Condition Logic

**Method:** `check_win_condition(game_state) -> Optional[str]`

**Imposter Win Conditions:**
1. All crewmates are dead/ejected (imposters remain) → return "imposters"
2. Imposters >= Crewmates (can't be outvoted) → return "imposters"
3. All crewmate tasks completed by imposters? (if applicable) → return "imposters"

**Crewmate Win Conditions:**
1. All imposters ejected (no living imposters) → return "crewmates"
2. All crewmate tasks completed (all tasks done) → return "crewmates"

**Ongoing Conditions:**
- Neither team has won → return None

**Terminal Conditions:**
- game_state.game_status == "completed" → return game_state.winner

**Acceptance:**
- [ ] Imposters win when all crewmates ejected
- [ ] Crewmates win when all imposters ejected
- [ ] Ongoing games return None
- [ ] Completed games return stored winner
- [ ] No false positives (game ends prematurely)

---

### Action Application

#### 5. Action Execution

**Method:** `apply_action(action, game_state) -> GameEvent`

**Requirement:** Assume action is already validated (validate_action returned True)

**Side Effects:**
1. Update game_state based on action type
2. Return GameEvent describing the change

**For Each Action Type:**

**discuss:**
- Add DiscussionMessage to game_state.discussion_history
- Event type: "message_posted"
- Changes: {"discussion_history": new_history}

**accuse:**
- Add accusation to discussion history (implicit)
- Update suspicion levels (optional v1: skip)
- Event type: "accusation_made"
- Changes: {"accused_agent": target, "accuser": agent_id}

**defend:**
- Add defense to discussion history (implicit)
- Event type: "defense_made"
- Changes: {"defending_agent": agent_id}

**vote:**
- Record vote in voting_history
- Event type: "vote_recorded"
- Changes: {"votes": new_vote_tally}

**move (Tier 2):**
- Update agent position
- Add to movement_log
- Event type: "agent_moved"
- Changes: {"agent_position": new_pos, "movement_log": updated}

**perform_task (Tier 2):**
- Mark task as completed
- Mark in crewmate's completed_tasks
- Event type: "task_completed"
- Changes: {"task_status": "completed", "completed_by": agent_id}

**kill (Tier 2):**
- Mark target as dead (status = "dead")
- Remove from active_agents
- Event type: "agent_killed"
- Changes: {"victim": target, "killer": agent_id, "active_agents": updated}

**use_vent (Tier 2):**
- Teleport agent to vent exit
- Add to movement_log (special type: vent)
- Event type: "agent_vented"
- Changes: {"agent_position": vent_exit, "movement_log": updated}

**Acceptance:**
- [ ] All action types correctly applied
- [ ] game_state updated appropriately
- [ ] GameEvent generated with accurate change description
- [ ] No action side effects beyond documented changes

---

### Meeting & Voting

#### 6. Meeting Logic

**Method:** `handle_meeting(game_state, voting_results) -> GameState`

**Input:**
- game_state: Current state
- voting_results: VotingResults with votes dict and ejected agent

**Process:**
1. Validate voting results (all votes are for valid agents)
2. Eject target agent:
   - Set agent status = "ejected"
   - Remove from active_agents
   - Reveal agent's role to all remaining agents
3. Update known_roles in all agents' observations
4. Check win conditions (imposters eliminated?)
5. Return updated game_state

**Win on Eject:**
- If ejected agent was last imposter → crewmates win
- If all crewmates ejected → imposters win
- Otherwise, game continues

**Acceptance:**
- [ ] Ejected agent removed from active_agents
- [ ] Role revealed to all agents
- [ ] Win conditions checked
- [ ] Correct winner determined if applicable

---

### Tier Abstraction

#### 7. Tier-Aware Rules

**Challenge:** Same rules should work for Tier 1 and Tier 2

**Solution:**
- Rules check action type
- Some actions invalid for Tier 1 (move, kill, etc.)
- validate_action() returns False for invalid Tier 1 actions
- Both tiers call same GameRules methods

**Acceptance:**
- [ ] Tier 1 agents can't move (validate_action returns False)
- [ ] Tier 1 agents can't kill (validate_action returns False)
- [ ] Tier 2 agents can move and kill
- [ ] Win conditions work identically for both tiers
- [ ] No tier-specific branching needed in environments

---

## Implementation Requirements

### Pure Logic (No I/O)

- GameRules has no mutable state
- All methods are pure functions
- No external dependencies (no I/O, no API calls)
- No side effects beyond returned GameEvent
- Deterministic: same inputs → same outputs

**Acceptance:**
- [ ] No instance variables in GameRules
- [ ] No global state
- [ ] All methods static
- [ ] No external dependencies

---

### Type Safety

- Full type hints on all methods
- Return types explicit (bool, Optional[str], GameEvent, etc.)
- No `Any` types
- Clear parameter types (GameAction, GameState, agent_role: str)

**Acceptance:**
- [ ] Type checkers (mypy) pass on GameRules
- [ ] All parameters and returns typed
- [ ] No type warnings

---

### Error Handling

- Invalid actions don't raise exceptions (return False from validate_action)
- Invalid game states don't raise exceptions (return None from check_win_condition)
- Invalid method calls (e.g., apply_action on invalid action) may raise ValueError with clear message

**Acceptance:**
- [ ] validate_action returns (False, error_msg) for invalid actions
- [ ] check_win_condition returns None for ongoing games
- [ ] Error messages are clear and actionable

---

## Definition of Done

### Implementation Checklist

- [ ] GameRules class created in `src/game/rules.py`
  - [ ] Static methods only (no instance state)
  - [ ] All required methods implemented

- [ ] Action Validation
  - [ ] `validate_action()` method implemented
  - [ ] All action types validated
  - [ ] All validation rules checked
  - [ ] Clear error messages for invalid actions

- [ ] Win Condition Checking
  - [ ] `check_win_condition()` method implemented
  - [ ] Imposter wins detected correctly
  - [ ] Crewmate wins detected correctly
  - [ ] Ongoing games return None

- [ ] Action Application
  - [ ] `apply_action()` method implemented
  - [ ] All action types applied correctly
  - [ ] GameEvent generated with accurate changes
  - [ ] game_state updated appropriately

- [ ] Meeting Logic
  - [ ] `handle_meeting()` method implemented
  - [ ] Voting results applied correctly
  - [ ] Roles revealed upon eject
  - [ ] Win conditions checked after meeting

- [ ] Type Safety
  - [ ] Full type hints on all methods
  - [ ] Type checkers pass (mypy)
  - [ ] No `Any` types

### Testing Checklist

- [ ] Unit tests for validate_action()
  - [ ] Test each action type for each role
  - [ ] Test valid actions pass
  - [ ] Test invalid actions fail with error messages
  - [ ] Test edge cases (self-targeting, non-existent agents, etc.)

- [ ] Unit tests for check_win_condition()
  - [ ] Test imposter win (all crewmates ejected)
  - [ ] Test crewmate win (all imposters ejected)
  - [ ] Test ongoing game (neither win)
  - [ ] Test terminal game (stored winner)

- [ ] Unit tests for apply_action()
  - [ ] Test each action type
  - [ ] Test game_state updated correctly
  - [ ] Test GameEvent generated accurately
  - [ ] Test multiple actions in sequence

- [ ] Unit tests for handle_meeting()
  - [ ] Test ejection of agent
  - [ ] Test role reveal
  - [ ] Test win condition check after meeting
  - [ ] Test vote tally with ties

- [ ] Integration tests
  - [ ] Works with Story 1 (Tier2Environment)
  - [ ] Works with Story 2 (Tier1Environment)
  - [ ] Works with Story 8 (data models)
  - [ ] Full game flow with validation + application

- [ ] Edge case testing
  - [ ] 1 imposter vs. N-1 crewmates
  - [ ] N imposters vs. 1 crewmate
  - [ ] All agents have same action (multiple votes for same target)
  - [ ] Agents voting for non-existent targets

### Code Quality Checklist

- [ ] All methods have docstrings with examples
- [ ] Type hints on all parameters and returns
- [ ] No unused code or dead branches
- [ ] Error messages are clear and actionable
- [ ] Code follows project style guide
- [ ] Complexity reasonable (no deeply nested conditions)

### Documentation Checklist

- [ ] GameRules class docstring explains purpose
- [ ] Each method documented with purpose, inputs, outputs
- [ ] Validation rules documented (what makes action valid/invalid)
- [ ] Win condition rules documented
- [ ] Examples of valid/invalid actions in comments
- [ ] Architecture decision: why static class vs. instance-based

---

## Technical Notes

### Implementation Approach

**Phase 1: Structure & Validation (Days 1-2)**
1. Create GameRules class skeleton
2. Implement `validate_action()` with all rules
3. Comprehensive validation testing
4. Document all rules clearly

**Phase 2: Logic & Application (Days 2-3)**
1. Implement `check_win_condition()`
2. Implement `apply_action()`
3. Implement `handle_meeting()`
4. Testing and edge case handling

**Phase 3: Integration & Testing (Days 3)**
1. Integration tests with Story 1 & 2 (environments)
2. Edge case and boundary testing
3. Type checking (mypy)
4. Documentation review

### Existing Pattern Reference

- **ARCHITECTURE.md Section 3.4:** GameRules specification
- **Story 1 & 2:** How environments will call GameRules
- **Story 8:** Data model types (GameAction, GameEvent, GameState)

### Key Constraints

1. **Pure Functions:** No mutable state or side effects (except returned GameEvent)
2. **Tier Abstraction:** Work for both Tier 1 and Tier 2 without branching
3. **Clear Contracts:** Input/output types explicit; behavior documented
4. **Deterministic:** Same inputs always produce same outputs
5. **No Duplication:** Single source of truth for all rules

---

## Risk & Compatibility

### Primary Risks

**Risk 1: Validation rules don't match environment expectations**
- Mitigation: Review with Story 1 & 2 developers early
- Mitigation: Prototype with mock environment before finalizing
- Mitigation: Comprehensive edge case testing

**Risk 2: Win condition logic is ambiguous or buggy**
- Mitigation: Formal specification of win conditions before coding
- Mitigation: Extensive testing (both teams winning scenarios)
- Mitigation: Side-by-side comparison with game design doc

**Risk 3: Tier 1 vs. Tier 2 action validation conflicts**
- Mitigation: Clear separation in validate_action() logic
- Mitigation: Test both tiers independently
- Mitigation: Document which actions valid for which tier

### Compatibility Verification

- [ ] Works with Story 1 (Tier2Environment)
- [ ] Works with Story 2 (Tier1Environment)
- [ ] Uses Story 8 data models correctly
- [ ] No import conflicts or missing dependencies
- [ ] Matches ARCHITECTURE.md spec

### Rollback Plan

1. Keep changes in feature branch
2. GameRules is new code (no changes to existing)
3. Easy revert if fundamental design issue
4. Can adjust validation rules without breaking interface

---

## Success Criteria

The story is complete when:

1. **Completeness:** All game rules formalized and implemented
2. **Validation:** Actions correctly validated; invalid actions rejected
3. **Outcomes:** Win conditions accurately determined
4. **Application:** Actions correctly applied; GameEvents generated
5. **Testing:** 100% unit test coverage; edge cases tested; integration verified
6. **Quality:** Type-safe, well-documented, pure functions

---

## Related Documents

- **Phase 0 Epic:** `/docs/epics/phase-0-foundation.md` - Full epic context
- **Architecture Design:** `/docs/ARCHITECTURE.md` - GameRules spec (Section 3.4)
- **Story 1:** `docs/stories/story-1-refactor-environment-abstract-interface.md`
- **Story 2:** `docs/stories/story-2-implement-tier1-dialogue-only-environment.md`
- **Story 8:** `docs/stories/story-8-create-data-models.md`

---

## Story Workflow

### Development Timeline

**Day 1: Validation & Structure**
- [ ] GameRules class created
- [ ] validate_action() implemented with all rules
- [ ] Unit tests for validation
- [ ] Documentation of all validation rules

**Days 1-2: Logic Implementation**
- [ ] check_win_condition() implemented
- [ ] apply_action() implemented
- [ ] handle_meeting() implemented
- [ ] Unit tests for all methods

**Days 2-3: Integration & Testing**
- [ ] Integration tests with Story 1 & 2
- [ ] Edge case testing
- [ ] Type checking (mypy)
- [ ] Documentation review
- [ ] Code review with team

### Dependency Management

**Blocks:** Stories 1, 2, 9 (need GameRules for validation)
**Blocked by:** Story 8 (needs data models: GameAction, GameEvent, GameState)
**Can be parallel:** Story 4-6 (LLM integration doesn't depend on rules)

---

## Story Status

**Status:** ✅ Ready for Development
**Last Updated:** 2025-11-02
**Next Review:** Upon completion; before starting Story 4 (LLM Integration)

**Notes:**
- Third story in critical path (after Stories 1-2)
- Pure logic; no I/O or external calls
- Reusable across all environments and game variations
- Early validation with Story 1 & 2 recommended
- Consider test-driven development (write tests first)

