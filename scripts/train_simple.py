"""Minimal PPO training loop for the simplified hidden role environment."""

from __future__ import annotations

import argparse
import pathlib

from pettingzoo.utils import parallel_to_aec, wrappers

try:
    from stable_baselines3 import PPO
except ImportError as exe:
    raise ImportError(
        "stable-baselines3 is required for training. "
        "Install it with `pip install stable-baselines3`."
    ) from exe

try:
    from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
except ImportError as exe:
    raise ImportError(
        "supersuit is required to interface PettingZoo with Stable-Baselines3. "
        "Install it with `pip install supersuit`."
    ) from exe

from src.multi_agent_deception.environment import parallel_env


def make_env(grid_size: int, num_agents: int, tasks_per_agent: int, max_steps: int, task_duration: int):
    """Construct the wrapped environment ready for vectorisation."""
    env = parallel_env(
        grid_size=grid_size,
        num_agents=num_agents,
        tasks_per_agent=tasks_per_agent,
        max_steps=max_steps,
        task_duration=task_duration,
    )
    #env = parallel_to_aec(env)

    #env = wrappers.AssertOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    return env


def train(
    total_timesteps: int,
    grid_size: int,
    num_agents: int,
    tasks_per_agent: int,
    max_steps: int,
    task_duration: int,
    learning_rate: float,
    output_dir: pathlib.Path,
) -> None:
    """Train independent PPO policies controlling the agents."""
    env = make_env(grid_size, num_agents, tasks_per_agent, max_steps, task_duration)
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        verbose=1,
        n_steps=256,
        batch_size=256,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "ppo_simple_hidden_role"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agents in the simple hidden role environment.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total PPO timesteps.")
    parser.add_argument("--grid-size", type=int, default=8, help="Size of the square grid.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of controllable agents.")
    parser.add_argument("--tasks-per-agent", type=int, default=3, help="Number of tasks assigned to each agent.")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode length before truncation.")
    parser.add_argument("--task-duration", type=int, default=3, help="Steps required on a task tile to complete it.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("artifacts"), help="Directory for saved models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        total_timesteps=args.timesteps,
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        tasks_per_agent=args.tasks_per_agent,
        max_steps=args.max_steps,
        task_duration=args.task_duration,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
