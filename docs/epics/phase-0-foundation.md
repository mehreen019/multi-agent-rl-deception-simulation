# Phase 0 Foundation: Simulation & LLM Integration - Brownfield Enhancement Epic

**Document Date:** 2025-11-02
**Status:** Ready for Story Development
**Epic ID:** PHASE0-FOUNDATION
**Priority:** Critical (Foundation for all subsequent phases)

---

## Executive Summary

Establish the architectural foundation for the multi-agent deception simulation by refactoring the existing PettingZoo environment into a modular, extensible system with integrated LLM decision-making, comprehensive logging, and reproducible game mechanics.

---

## Epic Title

Phase 0 Foundation: Simulation & LLM Integration - Brownfield Enhancement

### Epic Goal

Establish the architectural foundation for the multi-agent deception simulation by refactoring the existing PettingZoo environment into a modular, extensible system with integrated LLM decision-making, comprehensive logging, and reproducible game mechanics.

---

## Epic Description

### Existing System Context

- **Current Implementation:** PettingZoo-based environment (`SimpleHiddenRoleParallelEnv`) with grid-based task completion
- **Current Capabilities:** Basic multi-agent support with position tracking, task progress, and state management
- **Technology Stack:** Python, PettingZoo, Gymnasium, NumPy
- **Current Gaps:** No LLM integration, no dialogue phase, minimal logging, no abstract architecture

### Enhancement Details

#### What's being added/changed

- Refactor existing environment into abstract `DeceptionGameEnvironment` interface
- Create `Tier1Environment` (dialogue-only variant) alongside existing spatial mechanics
- Implement `GameRules` engine to formalize win conditions, action validation, voting logic
- Build LLM integration layer (`LLMClient` abstraction + Claude/GPT-4 implementations)
- Add JSON response parsing and validation (`ResponseParser` + `SafetyFilter`)
- Create comprehensive event logging system (`EventLogger`) for reproducibility

#### How it integrates

- Preserves existing PettingZoo environment as `Tier2Environment` foundation
- Adds thin abstraction layer (no changes to game loop logic)
- LLM calls inject decisions at game decision points (replaces hardcoded actions)
- Logging hooks into existing step/reset methods
- All components use shared dataclasses (GameState, GameAction, GameEvent, GameLog)

#### Success criteria

1. Abstract environment interface supports both Tier 1 and Tier 2 variants
2. Can run a complete game loop with LLM decisions from Claude or GPT-4
3. All game events logged to JSON-exportable GameLog
4. Game mechanics deterministic given a seed (reproducible)
5. System supports Imposter/Crewmate roles with distinct prompting

---

## Stories

### Story 1: Refactor Environment & Create Abstract Interface

**Title:** Refactor Environment & Create Abstract Interface

**Description:** Extract abstract `DeceptionGameEnvironment` base class and refactor existing `SimpleHiddenRoleParallelEnv` into `Tier2Environment` implementation.

**Acceptance Criteria:**
- Abstract base class `DeceptionGameEnvironment` defined with required methods
- `Tier2Environment` implements interface and preserves all existing behavior
- Partial observability implemented for agents
- All existing unit tests pass without modification
- Game mechanics verified unchanged vs. original implementation

---

### Story 2: Implement Tier 1 Dialogue-Only Environment

**Title:** Implement Tier 1 Dialogue-Only Environment

**Description:** Create `Tier1Environment` (no spatial component) with discussion phase and voting logic.

**Acceptance Criteria:**
- `Tier1Environment` class implements `DeceptionGameEnvironment`
- Discussion phase fully functional with round-robin messaging
- Voting logic correctly tallies votes and ejects majority target
- Win conditions properly evaluated (Imposters survive OR Crewmates complete tasks)
- Tier 1 games complete and generate correct winners

---

### Story 3: Build Game Rules Engine

**Title:** Build Game Rules Engine

**Description:** Implement `GameRules` class to formalize win conditions, action validation, and voting mechanics.

**Acceptance Criteria:**
- `GameRules` class provides static methods for validation and rule checking
- Action validation correctly rejects illegal moves/actions for each agent role
- Win condition checking works for both Tier 1 and Tier 2
- Voting/meeting logic properly handles ties and edge cases
- Rules applied consistently; illegal actions rejected

---

### Story 4: Implement LLM Client Abstraction & API Integration

**Title:** Implement LLM Client Abstraction & API Integration

**Description:** Create `LLMClient` abstract interface and implement concrete clients for Claude and GPT-4 with cost tracking.

**Acceptance Criteria:**
- `LLMClient` abstract class defines interface for all models
- `ClaudeClient` successfully calls Anthropic API with error handling
- `GPT4Client` successfully calls OpenAI API with error handling
- `LLMFactory` correctly instantiates clients by model name
- Cost per token calculated correctly for each model
- API errors handled gracefully with retry logic
- Token estimation works for both models

---

### Story 5: Build Response Parser & Safety Filter

**Title:** Build Response Parser & Safety Filter

**Description:** Implement JSON response parsing, validation, and safety filtering before action execution.

**Acceptance Criteria:**
- `ResponseParser` extracts JSON from raw LLM responses (with reasoning)
- Extracted JSON validated against role/tier-specific schemas
- Malformed responses handled with clear error messages
- `SafetyFilter` validates actions before execution
- Invalid actions trigger retry with corrective prompt
- Retry loop prevents infinite loops (max attempts configurable)

---

### Story 6: Create Prompt Templates & Role-Based Prompting

**Title:** Create Prompt Templates & Role-Based Prompting

**Description:** Implement prompt template system with role-specific (Imposter/Crewmate) Tier 1 prompts.

**Acceptance Criteria:**
- `PromptTemplate` class provides role/tier-specific templates
- Imposter Tier 1 prompt includes strategy guidance and deception tactics
- Crewmate Tier 1 prompt includes task/voting guidance
- Prompts render with full game state context
- All required information included (tick, agents, discussion history, observations)
- Prompts follow JSON response format specification

---

### Story 7: Implement Event Logger & Game Log Export

**Title:** Implement Event Logger & Game Log Export

**Description:** Create comprehensive event logging system and JSON export for complete game reproducibility.

**Acceptance Criteria:**
- `EventLogger` records all game events (actions, observations, votes, discussions)
- Game state serializable at any tick
- Complete game log exportable to JSON
- JSON includes: game_id, scenario_id, model_name, all events, final outcome, timestamps
- Exported JSON valid and importable for analysis
- Logs capture all decisions, observations, and outcomes

---

### Story 8: Create Data Models & Serialization

**Title:** Create Data Models & Serialization

**Description:** Define core dataclasses for game state, actions, events, and complete serialization support.

**Acceptance Criteria:**
- `GameState` captures all game information (agents, roles, positions, tasks, tick)
- `GameAction` represents parsed LLM decision (action, parameters, reasoning)
- `GameEvent` represents action outcome (what changed, metrics)
- `GameLog` represents complete game history
- `AgentObservation` represents partial observability view
- All models JSON serializable/deserializable
- Type validation on all critical fields

---

### Story 9: Implement Game Runner & Single Game Orchestration

**Title:** Implement Game Runner & Single Game Orchestration

**Description:** Create `GameRunner` to execute complete games integrating environment, LLM, logging, and rules.

**Acceptance Criteria:**
- `GameRunner` accepts scenario config and LLM clients
- Game loop: reset → tick loop → event logging → completion
- LLM called for each agent decision; output parsed and validated
- Actions executed through environment; events logged
- Win conditions checked and game terminated correctly
- Complete games run end-to-end without errors
- All events logged with full reproducibility

---

### Story 10: Phase 0 Integration Testing & Validation

**Title:** Phase 0 Integration Testing & Validation

**Description:** Comprehensive testing of all Phase 0 components with focus on reproducibility and backward compatibility.

**Acceptance Criteria:**
- Unit tests for all core modules (target: 80% coverage)
- Tier 1 and Tier 2 environment behavior verified independently
- LLM integration tested with mocked responses
- Reproducibility verified (same seed = same outcomes)
- Backward compatibility verified (existing behavior unchanged)
- All tests pass with no regressions
- Integration tests demonstrate end-to-end game execution

---

## Compatibility Requirements

- ✅ Existing PettingZoo environment remains functional
- ✅ No breaking changes to observation/action spaces
- ✅ Backward compatible with existing task/reward structure
- ✅ Optional LLM integration (can run without API keys for testing)
- ✅ No external dependencies beyond current `requirements.txt`

---

## Risk Mitigation

### Primary Risk

Refactoring existing environment could break existing functionality or change game mechanics.

### Mitigation Strategy

1. Keep existing `SimpleHiddenRoleParallelEnv` code intact during refactoring
2. Create wrapper (`Tier2Environment`) that preserves 100% backward compatibility
3. Comprehensive unit tests on all existing features before/after refactoring
4. Test with existing scenario configs to verify identical behavior
5. Feature flags to toggle new LLM integration on/off during development

### Rollback Plan

- All changes managed in git; easy revert if needed
- Original environment class preserved during refactoring phase
- Version tags for stable points before integration

---

## Definition of Done

### Functional Requirements

- ✅ All 10 stories completed with acceptance criteria met
- ✅ Phase 0 architecture (6 layers) implemented:
  - Layer 1: Simulation Engine (abstract + Tier 1 & 2)
  - Layer 2: LLM Integration (clients, prompts, parsing)
  - Layer 3: Metrics & Logging (event logger, data models)
  - Layer 4: Scenarios (configuration management)
  - Layer 5: Experiment Orchestration (game runner)
  - Layer 6: Analysis (placeholder for Phase 1+)
- ✅ Existing environment behavior verified unchanged
- ✅ LLM integration functional (Claude + GPT-4)
- ✅ Game loop executes without errors (Tier 1 & 2)
- ✅ Complete game logs exportable to JSON

### Quality Requirements

- ✅ 80%+ unit test coverage on core modules
- ✅ No regressions in existing features
- ✅ All tests pass (unit, integration)
- ✅ Code follows project style guide
- ✅ Architecture documentation complete
- ✅ Reproducibility verified (seed management)

---

## Epic Validation

### Scope Validation

- ✅ Epic can be completed in 10 focused stories
- ✅ No architectural documentation required (design provided)
- ✅ Enhancement follows existing patterns (PettingZoo, dataclasses)
- ✅ Integration complexity manageable (thin abstraction layer)

### Risk Assessment

- ✅ Risk to existing system is low (backward compatible)
- ✅ Rollback plan is feasible (git versioning)
- ✅ Testing approach covers existing functionality
- ✅ Team has sufficient knowledge of integration points

### Completeness Check

- ✅ Epic goal is clear and achievable
- ✅ Stories are properly scoped (each 2-4 days)
- ✅ Success criteria are measurable
- ✅ Dependencies clearly identified (Stories 1-3 before others)

---

## Implementation Timeline

**Phase Duration:** 4 weeks (20 business days)

**Week 1:** Stories 1-3 (Abstraction & Environment Variants)
- Refactor environment interface
- Implement Tier 1 environment
- Formalize game rules

**Week 2:** Stories 4-6 (LLM Integration)
- LLM client abstraction + API integration
- Response parsing & safety filtering
- Prompt templates

**Week 3:** Stories 7-9 (Logging & Orchestration)
- Event logging system
- Data models & serialization
- Game runner integration

**Week 4:** Story 10 (Testing & Validation)
- Comprehensive testing
- Reproducibility verification
- Backward compatibility validation

---

## Handoff to Story Manager

**Story Manager Handoff:**

Please develop detailed user stories for the Phase 0 Foundation brownfield epic. Key considerations:

- This is an enhancement to an existing PettingZoo-based multi-agent system
- Integration points: Existing environment → Abstract layer → Tier 1/2 variants
- Existing patterns to follow:
  * NumPy array usage for observations
  * Gymnasium spaces for action/observation definitions
  * PettingZoo ParallelEnv interface patterns
  * Dataclass-based state management

- Critical compatibility requirements:
  * Preserve 100% backward compatibility with existing SimpleHiddenRoleParallelEnv
  * No changes to observation/action spaces
  * Existing task completion logic must remain identical
  * All tests must pass on refactored code

- Each story must include:
  * Verification that existing PettingZoo functionality remains intact
  * Unit tests demonstrating new functionality
  * Integration points clearly specified
  * Code examples showing how new components integrate with existing environment

The epic should maintain system integrity while establishing foundational architecture for LLM-driven multi-agent deception simulation across Tier 1 (dialogue) and Tier 2 (spatial) game variants.

**Critical path:** Stories should proceed sequentially with clear dependencies:
  1. Abstraction (Stories 1-3) enables all following work
  2. Data models (Story 8) enable LLM and logging work
  3. LLM integration (Stories 4-6) enables game runner
  4. Game runner (Story 9) orchestrates everything
  5. Testing (Story 10) validates all dependencies

---

## Related Documents

- **Architecture Design:** `/docs/ARCHITECTURE.md` - Comprehensive system design
- **Requirements:** `/docs/REQUIREMENTS.md` (to be created during story development)
- **Test Plan:** `/docs/TEST_PLAN.md` (to be created during story development)

---

**Epic Status:** ✅ Ready for Story Development
**Last Updated:** 2025-11-02
**Next Review:** Upon story completion
