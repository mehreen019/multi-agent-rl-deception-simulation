================================================================================
MULTI-AGENT REINFORCEMENT LEARNING: DECEPTION SIMULATION FRAMEWORK
Documentation Index & Quick Navigation
================================================================================

PICKING A README TO READ? START HERE:

📖 README_COMPLETE.md
  └─ Main comprehensive guide covering theory, architecture, code, and usage
  └─ Read this if: You want the full picture
  └─ Time: 30-60 minutes for full read; 5 min for Quick Start section
  └─ Contains: Theory, Architecture, File Guide, Installation, Usage Examples

📖 README_THEORETICAL.md
  └─ Game-theoretic formulation and mathematical foundations
  └─ Read this if: You need academic framing or want to cite this work
  └─ Time: 20-30 minutes
  └─ Contains: Problem setting, state/action/reward spaces, transition dynamics

📖 GETTING_STARTED.md
  └─ Navigation guide - helps you pick which readme to read
  └─ Read this if: You're confused about where to start
  └─ Time: 5 minutes to navigate, then read specific sections
  └─ Contains: Role-based paths, common tasks, FAQ, learning paths by background

================================================================================
QUICK NAVIGATION TABLE
================================================================================

PURPOSE                          | READ THIS FILE               | TIME
─────────────────────────────────┼──────────────────────────────┼────────
Just want to run it              | GETTING_STARTED.md → Quick   | 5 min
                                 | Start section                |
─────────────────────────────────┼──────────────────────────────┼────────
Need full picture for thesis     | README_COMPLETE.md +         | 1-2 hrs
                                 | README_THEORETICAL.md        |
─────────────────────────────────┼──────────────────────────────┼────────
Want to implement a feature      | docs/stories/story-*.md      | 2-8 hrs
                                 | + src/base.py               |
─────────────────────────────────┼──────────────────────────────┼────────
Building on top of this system   | README_COMPLETE.md →         | 1-2 hrs
                                 | Advanced Configuration +     |
                                 | docs/ARCHITECTURE.md        |
─────────────────────────────────┼──────────────────────────────┼────────
Game theory & deception deep dive| README_THEORETICAL.md        | 30 min
                                 | + README_COMPLETE.md →      |
                                 | Theoretical Framework       |
─────────────────────────────────┼──────────────────────────────┼────────
Training RL agents on this       | scripts/train_simple.py      | 20 min
                                 | + README_COMPLETE.md →      |
                                 | Usage Guide                 |
─────────────────────────────────┼──────────────────────────────┼────────
Understanding the architecture   | docs/ARCHITECTURE.md         | 45 min
─────────────────────────────────┼──────────────────────────────┼────────
Troubleshooting a problem        | README_COMPLETE.md →         | 10 min
                                 | Troubleshooting             |
─────────────────────────────────┴──────────────────────────────┴────────

================================================================================
FILE QUICK REFERENCE
================================================================================

SOURCE CODE (What does what?)
  src/models.py              - Core dataclasses (GameState, GameAction, etc.)
  src/base.py                - Abstract interface (DeceptionGameEnvironment)
  src/tier2_environment.py   - Spatial environment implementation
  src/environment.py         - Original PettingZoo environment (legacy)
  src/gui.py                 - Pygame visualization

SCRIPTS (How to run it?)
  scripts/train_simple.py    - Train RL agents with PPO
  scripts/play_gui.py        - Interactive gameplay visualization

TESTS (How to validate?)
  tests/test_models.py       - Data model tests (✅ passing)
  tests/test_base_interface.py - Interface compliance tests (✅ passing)

DOCUMENTATION (What to read?)
  README_COMPLETE.md         - MAIN GUIDE (theory + practice)
  README_THEORETICAL.md      - Game theory & math foundations
  GETTING_STARTED.md         - Navigation guide
  docs/ARCHITECTURE.md       - System architecture (6 layers)
  docs/stories/story-*.md    - Implementation specifications

================================================================================
INSTALLATION (Copy & Paste)
================================================================================

LINUX/MACOS:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  PYTHONPATH=src python scripts/play_gui.py

WINDOWS (PowerShell):
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  $env:PYTHONPATH="src"; python scripts/play_gui.py

================================================================================
GETTING STARTED IN 3 STEPS
================================================================================

Step 1: Install
  python -m venv .venv
  source .venv/bin/activate  (Linux/Mac) OR .venv\Scripts\Activate.ps1 (Windows)
  pip install -r requirements.txt

Step 2: Run It
  PYTHONPATH=src python scripts/play_gui.py

Step 3: Learn
  Read: GETTING_STARTED.md (5 min)
  Read: README_COMPLETE.md (30-60 min)

================================================================================
CURRENT PROJECT STATUS
================================================================================

PHASE 0: Foundation (Weeks 1-4)
  Story 1: Refactor & Abstract Interface .................... ✅ COMPLETE
  Story 2: Tier 1 Dialogue Environment ...................... 🔄 TODO
  Story 3: Game Rules Engine ............................... 🔄 TODO
  Story 4-6: LLM Integration (Clients, Parsing, Prompts) .... 🔄 TODO
  Story 7-10: Logging, Runner, Testing, Integration ........ 🔄 TODO

Overall: 25% Complete (1 of 4 weeks done)

================================================================================
WHERE'S THE FULL DOCUMENTATION?
================================================================================

Each major document is in its own file:

README_COMPLETE.md
  • Sections: Theoretical Framework, System Architecture, File Structure,
    Installation, Usage Guide, Advanced Config, Development Roadmap,
    Architecture Decision Records, Troubleshooting, Citation
  • ~1500 lines
  • Best for: Full understanding from first principles

README_THEORETICAL.md
  • Sections: Problem Setting, Game-Theoretic Motivation, Mathematical
    Formulation, Learning Objectives, Extensions & Deception Dynamics
  • ~200 lines
  • Best for: Academic framing and citations

GETTING_STARTED.md
  • Sections: Which README to Read, Quick Paths by Role, File Guide,
    Common Tasks, Learning Paths by Background, FAQ
  • ~400 lines
  • Best for: Navigation and role-specific guidance

docs/ARCHITECTURE.md
  • Sections: Design Philosophy, Layered Architecture (6 layers), Component
    Responsibilities, Data Flow, detailed API specifications for each layer
  • ~1087 lines
  • Best for: Deep technical understanding and integration points

docs/stories/story-*.md (10 files)
  • Each file: Acceptance criteria, task breakdown, testing checklist, file list
  • Story 1: ✅ COMPLETE (~560 lines) - in story-1-refactor-*.md
  • Stories 2-10: 🔄 TODO (will be created)
  • Best for: Implementing specific features

================================================================================
RECOMMENDED READING SEQUENCE
================================================================================

BEGINNER (2-3 hours total):
  1. GETTING_STARTED.md (5 min) - pick your role path
  2. README_COMPLETE.md → Quick Start (5 min) - get it running
  3. Run: PYTHONPATH=src python scripts/play_gui.py (10 min)
  4. README_COMPLETE.md → Theoretical Framework (15 min)
  5. README_COMPLETE.md → System Architecture (20 min)
  6. README_COMPLETE.md → File Structure (15 min)
  7. README_THEORETICAL.md (complete) (30 min)
  8. docs/ARCHITECTURE.md (skim for overview) (15 min)

EXPERIENCED RESEARCHER (1-2 hours):
  1. README_THEORETICAL.md (complete) (30 min)
  2. README_COMPLETE.md → System Architecture (20 min)
  3. docs/ARCHITECTURE.md → Layers 1-3 (20 min)
  4. Pick a story from docs/stories/ and start implementing (varies)

DEVELOPER (1-2 hours):
  1. GETTING_STARTED.md → Developer path (10 min)
  2. README_COMPLETE.md → File Structure (15 min)
  3. src/base.py (read full file) (10 min)
  4. src/tier2_environment.py (read full file) (20 min)
  5. Pick a story and start coding (varies)

VERY BUSY PERSON (15 minutes):
  1. GETTING_STARTED.md → Your role path (5 min)
  2. README_COMPLETE.md → Quick Start (5 min)
  3. Run it: scripts/play_gui.py (5 min)

================================================================================
KEY TAKEAWAYS
================================================================================

WHAT IS THIS PROJECT?
  → Multi-agent reinforcement learning for strategic deception games
  → Grid-based spatial environment + planned dialogue variant
  → Supports LLM integration for reasoning and strategy

CURRENT STATE?
  → Story 1 Complete: Architecture refactored, models created, tests passing
  → Phase 0 (25%): Foundation ready; Stories 2-10 planned

HOW TO RUN IT?
  → Install: `pip install -r requirements.txt`
  → Play: `PYTHONPATH=src python scripts/play_gui.py`
  → Train: `PYTHONPATH=src python scripts/train_simple.py --timesteps 10000`

WHERE TO START?
  → GETTING_STARTED.md (navigation)
  → README_COMPLETE.md (full guide)
  → README_THEORETICAL.md (theory)

HOW TO CONTRIBUTE?
  → Pick a story from docs/epics/phase-0-foundation.md
  → Follow instructions in docs/stories/story-*.md
  → Reference docs/ARCHITECTURE.md for system design
  → Write tests following tests/ pattern

WHAT IF I HAVE QUESTIONS?
  → Check README_COMPLETE.md → Troubleshooting
  → Read GETTING_STARTED.md → FAQ
  → Open GitHub Issue with reproducible example

================================================================================
CONTACT & SUPPORT
================================================================================

Questions/Bugs?
  → GitHub Issues: Describe problem + reproducible steps
  → GitHub Discussions: Share ideas and collaborate

Want to Contribute?
  → See README_COMPLETE.md → Contributing section
  → Pick a story and follow its specification

Need Help Getting Started?
  → GETTING_STARTED.md is your friend (5 min read)
  → README_COMPLETE.md → Quick Start (5 min to run)

================================================================================
LAST UPDATED: 2025-11-02
STATUS: Phase 0, Story 1 Complete ✅ | Stories 2-10 TODO 🔄
NEXT: Story 2 (Tier 1 Dialogue Environment)
================================================================================
