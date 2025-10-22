# Multi-Agent Hidden Role Learning – Reinforcement Prototype

This repository contains an executable reinforcement-learning environment and training loop for a simplified social-deception scenario. The focus of this README is the practical implementation so you can set up, run, and extend the code quickly. A complementary theoretical write-up is available in `README_THEORETICAL.md`.

## Repository Layout

- `src/multi_agent_deception/environment.py` – PettingZoo `ParallelEnv` implementation (`SimpleHiddenRoleParallelEnv`) plus helper factories.
- `scripts/train_simple.py` – Stable-Baselines3 PPO training entry point using SuperSuit vectorisation.
- `scripts/play_gui.py` – Pygame viewer for stepping through an episode interactively.
- `requirements.txt` – Python dependency list.
- `README_THEORETICAL.md` – conceptual background and mathematical framing (see separate document).

## Prerequisites

- Python 3.10 or later.
- A POSIX-compatible shell (commands use Bash syntax; adjust activation for Windows as noted below).
- Optional: TensorBoard for logging (`pip install tensorboard`).
- Optional: Pygame-friendly display when running the GUI (a local desktop session; headless servers require a virtual framebuffer).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

All code lives under `src`, and the scripts automatically add it to the Python path.

## Usage

- **Smoke test the environment**

  ```bash
  # On Windows (PowerShell):
  python -c "import sys; sys.path.insert(0, '.'); from src.multi_agent_deception.environment import parallel_env; env = parallel_env(grid_size=8, num_agents=4, tasks_per_agent=3, max_steps=200, task_duration=3); observations, infos = env.reset(seed=42); print('Agents:', list(observations)); print('Agent_0 observation shape:', observations['agent_0'].shape)"

  # On Linux/Mac:
  python3 -c "import sys; sys.path.insert(0, '.'); from src.multi_agent_deception.environment import parallel_env; env = parallel_env(grid_size=8, num_agents=4, tasks_per_agent=3, max_steps=200, task_duration=3); observations, infos = env.reset(seed=42); print('Agents:', list(observations)); print('Agent_0 observation shape:', observations['agent_0'].shape)"
  ```

  This runs the actual environment with random actions and prints live rewards.
  Agents must remain on a task tile for the configured `task_duration` steps to finish it, so random movement rarely completes tasks.

- **Train PPO agents**

  ```bash
  # Windows (PowerShell) or Command Prompt:
  python scripts/train_simple.py --timesteps 20000 --grid-size 8 --num-agents 4 --tasks-per-agent 3 --max-steps 200 --task-duration 3 --learning-rate 3e-4 --output-dir artifacts

  # Linux/Mac:
  python3 scripts/train_simple.py --timesteps 20000 --grid-size 8 --num-agents 4 --tasks-per-agent 3 --max-steps 200 --task-duration 3 --learning-rate 3e-4 --output-dir artifacts
  ```

  The script saves `artifacts/ppo_simple_hidden_role.zip` and (optionally) TensorBoard logs under `artifacts/tensorboard/`.

- **Evaluate a trained model**

  ```bash
  # Windows (PowerShell) or Command Prompt:
  python scripts/run_trained.py --model-path artifacts/ppo_simple_hidden_role.zip --episodes 5 --deterministic

  # Linux/Mac:
  python3 scripts/run_trained.py --model-path artifacts/ppo_simple_hidden_role.zip --episodes 5 --deterministic
  ```

  This will run 5 evaluation episodes and display per-agent rewards. Add `--render-ascii` to see the grid visualization.

## Key Implementation Notes

- `SimpleHiddenRoleParallelEnv` tracks agent positions, per-task progress counters, and completion arrays so that PettingZoo receives vector-friendly observations (`numpy.int32`).
- Tasks now require the agent to remain on the corresponding tile for `task_duration` consecutive steps (default `3`) before they are marked complete; progress resets if the agent leaves early.
- Actions map to cardinal moves with boundary clipping; collisions are allowed (agents can share a tile).
- Rewards are shaped directly inside `step`, and termination/truncation dictionaries follow PettingZoo’s parallel API.
- `train_simple.py` converts to a Stable-Baselines3 compatible vector environment via SuperSuit.
- `scripts/play_gui.py` uses Pygame to visualise the grid, tasks, agent positions, and per-task progress bars while you control one agent locally.
- Hyperparameters (`n_steps=256`, `batch_size=256`) target small cooperative maps; modify in the script if you scale the scenario.

- **Interactive GUI**

  ```bash
  # Windows (PowerShell) or Command Prompt:
  python scripts/play_gui.py --grid-size 8 --num-agents 4 --tasks-per-agent 3 --max-steps 200 --task-duration 3 --seed 42 --fps 8

  # Linux/Mac:
  python3 scripts/play_gui.py --grid-size 8 --num-agents 4 --tasks-per-agent 3 --max-steps 200 --task-duration 3 --seed 42 --fps 8
  ```
  Control agent `0` with the arrow keys (space to stay). Other agents take random actions so you can observe task progress, the hold-to-complete mechanic, and termination behaviour. Use `R` to reset, `P` to pause, and close the window to exit.

- **Run demo GUI with heuristic agents (RECOMMENDED)**
  ```bash
  # Windows (PowerShell) or Command Prompt:
  python scripts/play_demo_gui.py --fps 8

  # Linux/Mac:
  python3 scripts/play_demo_gui.py --fps 8
  ```
  Watch intelligent heuristic agents navigate to their tasks! This demonstrates the environment working properly.
  - Agents use a simple heuristic: move towards the nearest incomplete task
  - Use `--random` flag for random movement instead
  - Use `R` to reset, `P` to pause, and close the window to exit

- **Run GUI with trained model** ⚠️ (Note: Current trained model needs improvement)
  ```bash
  # Windows (PowerShell) or Command Prompt:
  python scripts/play_trained_gui.py --model-path artifacts/ppo_simple_hidden_role.zip --deterministic --fps 8

  # Linux/Mac:
  python3 scripts/play_trained_gui.py --model-path artifacts/ppo_simple_hidden_role.zip --deterministic --fps 8
  ```
  **Known Issue**: The current PPO model learned a degenerate policy (action 0 / stay). The environment works correctly, but the reward structure may need tuning for effective RL training. Use the demo GUI above to see working agents!

## Troubleshooting

- `ModuleNotFoundError`: The scripts now handle imports automatically. If issues persist, ensure you're running from the repository root directory.
- `ImportError: stable-baselines3` or `supersuit`: missing dependencies—rerun `pip install -r requirements.txt`.
- **Agents not moving in trained model**: This is a known issue. The PPO model learned to always output action 0 (stay), likely due to:
  - Sparse rewards making it difficult to learn
  - Small movement penalty (-0.01) discouraging exploration
  - Random task/agent placement making consistent learning difficult

  **Solutions**:
  - Use `play_demo_gui.py` to see working heuristic agents
  - Improve reward shaping (e.g., reward progress towards tasks, distance-based rewards)
  - Increase exploration (higher entropy coefficient)
  - Curriculum learning (start with easier scenarios)
  - Use the manual control GUI (`play_gui.py`) to play yourself

- Want richer logs? Install TensorBoard and launch `tensorboard --logdir artifacts/tensorboard`.

For a detailed explanation of the math and game-theoretic motivation behind this environment, read `README_THEORETICAL.md`.
