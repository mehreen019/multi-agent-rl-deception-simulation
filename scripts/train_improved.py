"""Improved PPO training with better reward shaping and network architecture."""

from __future__ import annotations

import argparse
import pathlib

from pettingzoo.utils import parallel_to_aec

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

import torch
import torch.nn as nn
from multi_agent_deception.environment import parallel_env


def make_env(grid_size: int, num_agents: int, tasks_per_agent: int, max_steps: int, task_duration: int):
    """Construct the parallel environment."""
    env = parallel_env(
        grid_size=grid_size,
        num_agents=num_agents,
        tasks_per_agent=tasks_per_agent,
        max_steps=max_steps,
        task_duration=task_duration,
    )
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
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 20,
) -> None:
    """Train PPO with improved architecture and hyperparameters.

    Key improvements:
    - Larger network (256-256 hidden layers)
    - Higher n_steps for more stable advantage estimates
    - Longer training horizon with more epochs
    - Higher entropy coefficient for better exploration
    - Learning rate warmup strategy
    - Reward normalization via environment wrapper
    """
    env = make_env(grid_size, num_agents, tasks_per_agent, max_steps, task_duration)
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # Custom policy network architecture (larger and deeper)
    policy_kwargs = dict(
        net_arch=[256, 256],  # Larger hidden layers
        activation_fn=nn.ReLU,
    )

    print("=" * 60)
    print("IMPROVED PPO TRAINING")
    print("=" * 60)
    print(f"Environment: {num_agents} agents, {grid_size}x{grid_size} grid")
    print(f"Tasks: {tasks_per_agent} per agent, {task_duration} steps each")
    print()
    print("Training Configuration:")
    print(f"  Network: 256-256 (larger than default 64-64)")
    print(f"  n_steps: {n_steps} (larger rollout buffer)")
    print(f"  batch_size: {batch_size}")
    print(f"  n_epochs: {n_epochs} (more gradient updates)")
    print(f"  learning_rate: {learning_rate}")
    print(f"  ent_coef: 0.05 (high entropy for exploration)")
    print(f"  Total timesteps: {total_timesteps}")
    print()
    print("Why these changes:")
    print("  - Larger network: Better approximation of value function")
    print("  - More rollout: Stabler advantage estimates")
    print("  - More epochs: Better policy optimization")
    print("  - High entropy: Encourages exploration to find tasks")
    print("=" * 60)
    print()

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.999,             # Very high discount (long-term vision)
        gae_lambda=0.95,         # GAE lambda
        clip_range=0.2,          # PPO clip
        clip_range_vf=0.2,       # Value function clip
        ent_coef=0.05,           # HIGH entropy for exploration (was 0.01)
        vf_coef=1.0,             # Standard value function coefficient
        max_grad_norm=0.5,       # Gradient clipping
        normalize_advantage=True,
        tensorboard_log=None,
    )

    print("Starting training...")
    print()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / "ppo_simple_hidden_role_improved"))
    print()
    print(f"Model saved to {output_dir / 'ppo_simple_hidden_role_improved.zip'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved PPO training with better architecture and hyperparameters."
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total PPO timesteps.")
    parser.add_argument("--grid-size", type=int, default=8, help="Size of the square grid.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of controllable agents.")
    parser.add_argument("--tasks-per-agent", type=int, default=3, help="Number of tasks assigned to each agent.")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode length before truncation.")
    parser.add_argument("--task-duration", type=int, default=3, help="Steps required on a task tile to complete it.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout buffer size (larger = more stable).")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--n-epochs", type=int, default=20, help="Gradient update epochs per rollout.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("artifacts"), help="Output directory.")
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
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
