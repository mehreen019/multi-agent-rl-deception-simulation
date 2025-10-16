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

All code lives under `src`, so set `PYTHONPATH=src` when running local modules.

## Usage

- **Smoke test the environment**

  ```bash
  PYTHONPATH=src python3 - <<'PY'
  from multi_agent_deception.environment import parallel_env

  env = parallel_env(grid_size=8, num_agents=4, tasks_per_agent=3, max_steps=200, task_duration=3)
  observations, infos = env.reset(seed=42)
  print("Agents:", list(observations))
  print("Agent_0 observation shape:", observations["agent_0"].shape)

  for step in range(3):
      actions = {agent: env.action_space(agent).sample() for agent in env.agents}
      observations, rewards, terminations, truncations, infos = env.step(actions)
      print(f"Step {step} rewards:", rewards)
      if not env.agents:
          break
  PY
  ```

  This runs the actual environment with random actions and prints live rewards.
  Agents must remain on a task tile for the configured `task_duration` steps to finish it, so random movement rarely completes tasks.

- **Train PPO agents**

  ```bash
  PYTHONPATH=src python scripts/train_simple.py \
      --timesteps 20000 \
      --grid-size 8 \
      --num-agents 4 \
      --tasks-per-agent 3 \
      --max-steps 200 \
      --task-duration 3 \
      --learning-rate 3e-4 \
      --output-dir artifacts
  ```

  The script saves `artifacts/ppo_simple_hidden_role.zip` and (optionally) TensorBoard logs under `artifacts/tensorboard/`.

- **Evaluate a trained model**

  ```python
  from stable_baselines3 import PPO
  from supersuit import pettingzoo_env_to_vec_env_v1
  from pettingzoo.utils import parallel_to_aec
  from multi_agent_deception.environment import parallel_env

  env = parallel_env(task_duration=3)
  vec_env = pettingzoo_env_to_vec_env_v1(parallel_to_aec(env))
  model = PPO.load("artifacts/ppo_simple_hidden_role.zip")
  obs = vec_env.reset()
  action, _ = model.predict(obs, deterministic=True)
  ```

  Extend with a rollout loop to gather statistics or render trajectories.

## Key Implementation Notes

- `SimpleHiddenRoleParallelEnv` tracks agent positions, per-task progress counters, and completion arrays so that PettingZoo receives vector-friendly observations (`numpy.int32`).
- Tasks now require the agent to remain on the corresponding tile for `task_duration` consecutive steps (default `3`) before they are marked complete; progress resets if the agent leaves early.
- Actions map to cardinal moves with boundary clipping; collisions are allowed (agents can share a tile).
- Rewards are shaped directly inside `step`, and termination/truncation dictionaries follow PettingZoo’s parallel API.
- `train_simple.py` wraps the environment with `parallel_to_aec`, `AssertOutOfBoundsWrapper`, and `OrderEnforcingWrapper` before converting to a Stable-Baselines3 compatible vector environment via SuperSuit.
- `scripts/play_gui.py` uses Pygame to visualise the grid, tasks, agent positions, and per-task progress bars while you control one agent locally.
- Hyperparameters (`n_steps=256`, `batch_size=256`) target small cooperative maps; modify in the script if you scale the scenario.

## Troubleshooting

- `ModuleNotFoundError`: ensure commands are run with `PYTHONPATH=src`.
- `ImportError: stable-baselines3` or `supersuit`: missing dependencies—rerun `pip install -r requirements.txt`.
- Want richer logs? Install TensorBoard and launch `tensorboard --logdir artifacts/tensorboard`.

For a detailed explanation of the math and game-theoretic motivation behind this environment, read `README_THEORETICAL.md`.
- **Interactive GUI**

  ```bash
  PYTHONPATH=src python scripts/play_gui.py \
      --grid-size 8 \
      --num-agents 4 \
      --tasks-per-agent 3 \
      --task-duration 3
  ```

  Control agent `0` with the arrow keys (space to stay). Other agents take random actions so you can observe task progress, the hold-to-complete mechanic, and termination behaviour. Use `R` to reset, `P` to pause, and close the window to exit.
