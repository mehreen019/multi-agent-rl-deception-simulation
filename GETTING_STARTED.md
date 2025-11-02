# Getting Started with Multi-Agent Deception Simulation

**Welcome!** This guide helps you navigate the documentation and get up and running quickly.

## 📚 Which README Should I Read?

### **Just want to run it?** → Start here

```bash
# 3 commands to get going
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt && PYTHONPATH=src python scripts/play_gui.py
```

Then come back and read **Quick Start** section in `README_COMPLETE.md`.

---

### **Need the complete picture?** → `README_COMPLETE.md` (THIS IS THE MAIN README)

This document covers:
- ✅ **Theoretical Framework** - Game theory, math, deception dynamics
- ✅ **System Architecture** - Layer design, component responsibilities
- ✅ **File Structure** - What each file does and why
- ✅ **Installation** - Step-by-step setup
- ✅ **Usage Guide** - All use cases with examples
- ✅ **Development Roadmap** - Future directions
- ✅ **Troubleshooting** - Common problems & solutions

**Read this if:** You want to understand the whole system, are starting development, or need to explain this to others.

---

### **Want only theory?** → `README_THEORETICAL.md`

This document covers:
- Game-theoretic formulation
- State/action/observation spaces (mathematical notation)
- Reward function
- Transition dynamics
- Why hidden roles matter
- Suggested extensions

**Read this if:** You're writing a paper, need to cite the framework, or want deep theory.

---

### **Need architecture details?** → `docs/ARCHITECTURE.md`

This document covers:
- Component architecture (6 layers)
- Each component's responsibility & interface
- Data flow & execution model
- Reproducibility & scientific rigor
- Extensibility framework
- Implementation priorities

**Read this if:** You're implementing new features, need to understand integration points, or designing systems on top of this.

---

### **Want development stories?** → `docs/epics/phase-0-foundation.md` + `docs/stories/`

These documents specify:
- Epic goals and acceptance criteria
- Story-by-story breakdown
- Implementation details
- Testing requirements
- Dependency management

**Read this if:** You're implementing a specific story or feature.

---

## 🚀 Quick Paths by Your Role

### **Student / Learner**
1. Read: `README_COMPLETE.md` → Theoretical Framework section
2. Run: Interactive GUI (`scripts/play_gui.py`)
3. Explore: `tests/test_models.py` and `tests/test_base_interface.py`
4. Read: `README_THEORETICAL.md` for deeper understanding

### **RL Researcher**
1. Read: `README_COMPLETE.md` → System Architecture section
2. Review: `docs/ARCHITECTURE.md` → Layer 2 (LLM Integration)
3. Run: Training script (`scripts/train_simple.py`)
4. Explore: Modify hyperparameters in `scripts/train_simple.py`

### **Developer / Contributor**
1. Read: `README_COMPLETE.md` → File Structure section
2. Review: `docs/ARCHITECTURE.md` → complete system overview
3. Check: `docs/epics/phase-0-foundation.md` → see roadmap
4. Pick: A story from `docs/stories/` and implement
5. Run: `pytest tests/` to validate changes

### **Game Theory / Deception Expert**
1. Read: `README_THEORETICAL.md` → complete
2. Review: `README_COMPLETE.md` → Theoretical Framework section
3. Suggest: Extensions in `README_THEORETICAL.md` → Suggested Extensions
4. Contribute: Implementation of your ideas

### **System Integrator**
1. Read: `docs/ARCHITECTURE.md` → complete
2. Review: `README_COMPLETE.md` → System Architecture section
3. Check: `README_COMPLETE.md` → Advanced Configuration
4. Integrate: Your systems via Layer 5+ interfaces

---

## 📁 File Guide at a Glance

### **Readme Files (Documentation)**
| File | Purpose | Read When |
|------|---------|-----------|
| **README_COMPLETE.md** | Complete system guide (theory + practice) | First thing! |
| **README_THEORETICAL.md** | Game theory & math | Need academic context |
| **GETTING_STARTED.md** | This file - quick navigation | Confused which to read? |

### **Source Code**
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/models.py` | Core dataclasses | 327 | ✅ Done (Story 1) |
| `src/base.py` | Abstract interface | 111 | ✅ Done (Story 1) |
| `src/tier2_environment.py` | Spatial environment | 368 | ✅ Done (Story 1) |
| `src/environment.py` | Original PettingZoo env | 225 | ✅ Legacy |
| `src/gui.py` | Pygame visualization | 206 | ✅ Legacy |

### **Scripts (Executable Programs)**
| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/train_simple.py` | Train RL agent | `python scripts/train_simple.py --timesteps 10000` |
| `scripts/play_gui.py` | Interactive gameplay | `python scripts/play_gui.py` |

### **Tests**
| Test File | Purpose | Run With |
|-----------|---------|----------|
| `tests/test_models.py` | Model validation | `python tests/test_models.py` |
| `tests/test_base_interface.py` | Interface compliance | `python tests/test_base_interface.py` |

### **Documentation**
| Doc | Purpose | Read When |
|-----|---------|-----------|
| `docs/ARCHITECTURE.md` | System architecture | Implementing features |
| `docs/IMPLEMENTATION_GUIDE.md` | Developer guide | Starting development |
| `docs/epics/phase-0-foundation.md` | Phase 0 specification | Understanding roadmap |
| `docs/stories/story-*.md` | Individual story specs | Implementing that story |

---

## ⚡ Common Tasks

### Task: Understand what this project does

1. Read `README_COMPLETE.md` → **System Architecture** (15 min)
2. Run `PYTHONPATH=src python scripts/play_gui.py` (5 min)
3. Read `README_COMPLETE.md` → **Theoretical Framework** (15 min)

**Total: 35 minutes**

---

### Task: Run the environment yourself

```bash
# Setup (5 min)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run interactive game (10 min)
PYTHONPATH=src python scripts/play_gui.py

# Run tests (5 min)
python tests/test_models.py && python tests/test_base_interface.py
```

**Total: 20 minutes**

---

### Task: Train an RL agent

```bash
# Setup (5 min) - if not already done
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (varies) - example: 5 min training
PYTHONPATH=src python scripts/train_simple.py --timesteps 5000

# Monitor progress
tensorboard --logdir artifacts/tensorboard
```

**Total: 5+ minutes (depending on timesteps)**

---

### Task: Implement a new feature (Story 2: Tier 1 Environment)

1. Read: `docs/stories/story-2-implement-tier1-dialogue-only-environment.md` (20 min)
2. Review: `src/base.py` - interface you must implement (10 min)
3. Review: `src/tier2_environment.py` - reference implementation (20 min)
4. Create: `src/tier1_environment.py` following interface (2-4 hours)
5. Test: Write tests in `tests/test_tier1_environment.py` (1-2 hours)
6. Validate: Run full test suite (10 min)

**Total: 4-8 hours**

---

### Task: Understand model serialization

1. Read: `README_COMPLETE.md` → **File Structure & Components** → **models.py** (10 min)
2. Review: `src/models.py` → GameState class definition (10 min)
3. Run: `python tests/test_models.py` → watch round-trip test (5 min)
4. Try: Create custom GameState, serialize to JSON (15 min)

**Total: 40 minutes**

---

## 🎓 Learning Path by Background

### Background: No ML/Game Theory Experience

1. **Week 1:** Theory + Basics
   - Read: `README_THEORETICAL.md`
   - Skim: `README_COMPLETE.md` → Theoretical Framework
   - Run: `scripts/play_gui.py` (play a few games)
   - Task: Answer "What is a hidden role?" and "Why does it matter?"

2. **Week 2:** System Understanding
   - Read: `README_COMPLETE.md` → System Architecture
   - Review: `docs/ARCHITECTURE.md` → Layers 1-3
   - Task: Draw a diagram of how Tier2Environment fits in the system

3. **Week 3:** Code Exploration
   - Read: `src/models.py` → understand GameState
   - Read: `src/base.py` → understand interface
   - Read: `src/tier2_environment.py` → understand implementation
   - Task: Modify task_duration in Tier2Environment and see effect

### Background: ML/RL Experience

1. **Day 1:** Architecture & Theory
   - Read: `docs/ARCHITECTURE.md` (complete)
   - Skim: `README_COMPLETE.md` → Theoretical Framework
   - Task: Map PPO training (in story) to this environment

2. **Day 2:** Code & Models
   - Review: `src/models.py` + `src/base.py`
   - Review: `src/tier2_environment.py`
   - Review: `scripts/train_simple.py`
   - Task: Explain data flow from reset → step → observations

3. **Day 3:** Extend It
   - Pick a Story 2-10 from `docs/epics/phase-0-foundation.md`
   - Implement following story specification
   - Write tests following pattern in `tests/`

### Background: Game Theory / Research

1. **Day 1:** Theory Deep-Dive
   - Read: `README_THEORETICAL.md` (complete)
   - Read: `docs/ARCHITECTURE.md` → design principles
   - Task: Identify equilibria in current game

2. **Day 2:** Extension Design
   - Review: `README_THEORETICAL.md` → Suggested Extensions
   - Review: `docs/stories/` → see what's planned
   - Task: Propose 2-3 novel extensions (voting, role asymmetry, etc.)

3. **Day 3:** Implementation Roadmap
   - Review: `docs/epics/phase-0-foundation.md` → timeline
   - Plan: Which stories implement your extensions
   - Task: Write a specification for your extension

---

## 🔗 Related Documents Quick Links

### For Each Story
- Story 1 (Refactor): ✅ DONE - `docs/stories/story-1-refactor-environment-abstract-interface.md`
- Story 2 (Tier 1): 🔄 TODO - `docs/stories/story-2-implement-tier1-dialogue-only-environment.md`
- Story 3 (Rules): 🔄 TODO - `docs/stories/story-3-build-game-rules-engine.md`
- Story 4-6 (LLM): 🔄 TODO - `docs/stories/story-4,5,6*.md`
- Story 7 (Logging): 🔄 TODO - `docs/stories/story-7-implement-event-logger.md`
- Story 8 (Models): 🔄 TODO - `docs/stories/story-8-create-data-models.md`
- Story 9 (Runner): 🔄 TODO - `docs/stories/story-9-implement-game-runner.md`
- Story 10 (Testing): 🔄 TODO - `docs/stories/story-10-phase-0-integration-testing.md`

### By Topic
- **Game Mechanics:** `README_THEORETICAL.md` → sections 2-4
- **Environment Interface:** `src/base.py` + `docs/ARCHITECTURE.md` → Layer 1
- **LLM Integration:** `docs/ARCHITECTURE.md` → Layer 2
- **Data Models:** `src/models.py` + `docs/ARCHITECTURE.md` → Data Models
- **Training:** `scripts/train_simple.py` + `README_COMPLETE.md` → Usage Guide

---

## ❓ FAQ

**Q: Where do I start if I'm completely new?**
A: Run `README_COMPLETE.md` → Quick Start → 1-Minute Setup. Takes 3 minutes!

**Q: What's the difference between Tier 1 and Tier 2?**
A: **Tier 2** = spatial grid (current). **Tier 1** = dialogue-only (planned). See `README_THEORETICAL.md` → Extensions.

**Q: Can I use this for my thesis/paper?**
A: Yes! See `README_COMPLETE.md` → Citation. Also read `README_THEORETICAL.md` for academic framing.

**Q: Is this production-ready?**
A: Phase 0 (Story 1) is complete. Features are solid but LLM integration (Stories 4-6) not yet implemented.

**Q: What's the roadmap?**
A: See `README_COMPLETE.md` → Development Roadmap or `docs/epics/phase-0-foundation.md`.

**Q: Can I contribute?**
A: Yes! See `README_COMPLETE.md` → Contributing. Start with any Story 2-10.

**Q: What if I get stuck?**
A: See `README_COMPLETE.md` → Troubleshooting. If still stuck, open a GitHub issue!

---

## 📞 Support

- **Questions:** GitHub Issues (detailed questions)
- **Bugs:** GitHub Issues (reproducible errors)
- **Discussions:** GitHub Discussions (ideas, extensions)
- **Email:** [your-email] (for private matters)

---

**Last Updated:** 2025-11-02
**Next Step:** Pick a README above and dive in! 🚀
