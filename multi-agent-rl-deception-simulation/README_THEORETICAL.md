# Theoretical Overview – Multi-Agent Hidden Role Learning

This document explains the conceptual background behind the reinforcement-learning prototype in this repository. It is intended for reporting or presentation purposes when you need to describe the underlying theory to a teacher or teammate.

## Problem Setting

- **Environment Type:** Cooperative partially observable Markov game (POMG) on an `N×N` grid.
- **Agents:** `M` players labelled `agent_0 … agent_{M-1}`. Each controls a pawn with cardinal moves `{stay, up, down, left, right}`.
- **Hidden Roles:** Every agent receives a private list of `K` target tiles. Ownership information is hidden from other agents, creating asymmetric knowledge that can support deceptive behaviour (e.g., stalling instead of helping).
- **Episode Objective:** Each agent aims to complete all personal tasks. The episode terminates when all agents succeed or truncates after a fixed horizon `max_steps`.

## State and Observation Spaces

- **Global state** (not directly visible in the code but implicit for analysis):
  - Agent positions `p_i ∈ ℤ_{grid}²`.
  - Task matrices `T_i ∈ (ℤ_{grid}²)^K`.
  - Completion masks `c_i ∈ {0,1}^K`.
  - Progress counters `d_i ∈ {0,…,D}^K`, where `D = task_duration`.
  - Step counter `t`.
- **Local observation** for agent `i`:
  ```
  o_i = [p_i.x, p_i.y, T_i[0].x, T_i[0].y, d_i[0], …, T_i[K-1].x, T_i[K-1].y, d_i[K-1]]
  ```
  Completed tasks substitute their coordinates with `(grid_size, grid_size)` and clamp progress to `D` to keep observations inside the defined bounds.
- **Observation characteristics:** Fully local (no direct view of other agents), discrete-valued, and Markovian from the agent’s perspective once tasks are encoded in the observation vector.

## Action Space

- Each agent chooses from five discrete actions every step. Moves are deterministic and clipped to stay inside the grid. Multiple agents can occupy the same cell; collisions do not introduce stochastic transitions.

## Reward Design

For agent `i`, the per-step reward is:

```
r_i = -0.01
      + 1.0 * I(task progress for agent i reaches D this step)
      + 0.5 * I(all tasks for agent i are complete after the move)
```

- **Time penalty (-0.01):** Encourages agents to reach tasks quickly instead of wandering.
- **Task reward (+1.0):** Primary sparse signal to reinforce visiting assigned tiles.
- **Completion bonus (+0.5):** Provides a small terminal boost once all tasks are done.

The reward is individual, so deceptive strategies (e.g., delaying or shadowing) would need custom modifications to provide adversarial incentives.

## Transition Dynamics

Let `s_t` be the global state and `a_t` the joint action. The transition function is deterministic given actions:

1. Compute tentative positions `p_i' = clip(p_i + move(a_i))`.
2. Update progress counters `d_i'`: increment if `p_i'` matches task coordinates, otherwise reset to zero for unfinished tasks; cap at `D`.
3. Update completion masks `c_i'` where `d_i'` hits `D`.
4. Increment step counter `t' = t + 1` and check terminal conditions: termination if every `c_i'` is all ones; truncation if `t' ≥ max_steps` and not all tasks are complete.

Stochasticity only appears at environment reset when tasks and starting positions are sampled without replacement from the grid.

## Learning Formulation

- **Objective:** For each agent, maximise expected discounted return `E[∑ γ^t r_i]` with PPO’s default discount factor `γ = 0.99`.
- **Policy Sharing:** The provided training script shares a single neural network policy across agents (parameter tying). This mirrors independent learners with shared weights and can be extended to role-specific policies if needed.
- **Algorithm:** Proximal Policy Optimisation (PPO) optimises a clipped surrogate loss:
  ```
  L^(CLIP)(θ) = E_t [ min( r_t(θ)Â_t, clip(r_t(θ), 1 - ε, 1 + ε)Â_t ) ]
  ```
  where `r_t(θ)` is the probability ratio between new and old policies and `Â_t` is the advantage estimate. SuperSuit converts the multi-agent PettingZoo environment into a single-agent vectorised environment so Stable-Baselines3 can apply vanilla PPO updates.

## Why Hidden Roles Matter

- Agents know their own tasks but not others’, so optimal strategies require coordination without full transparency.
- This structure supports extensions such as:
  - **Impostor roles:** assign sabotage rewards to selected agents.
  - **Voting or meetings:** introduce communication phases where agents infer others’ roles based on observed motion.
  - **Dynamic tasks:** allow tasks to respawn or relocate, modelling more complex deception patterns.

## Suggested Extensions for Study

1. **Reward asymmetry:** give one agent a negative reward for fast completion to incentivise deception.
2. **Observation noise:** limit visibility to a neighbourhood to introduce partial observability.
3. **Centralised training, decentralised execution:** replace shared PPO with algorithms like QMIX or MADDPG.
4. **Dialogue integration:** after each episode, feed trajectories into an LLM to simulate social reasoning about hidden roles.

## Summary

The environment captures the core mechanics of task-based hidden-role games in a minimal grid world. The theoretical framing combines cooperative MARL with asymmetric information, making it a suitable sandbox for experimenting with deception-aware learning strategies. A lightweight Pygame visualiser (`scripts/play_gui.py`) is available to illustrate these dynamics interactively. Use this document alongside the main `README.md` to present both the implementation details and the academic motivation of the project.
