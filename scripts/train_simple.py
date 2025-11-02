"""Minimal PPO training loop for the simplified hidden role environment."""

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
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 10,
) -> None:
    """Train independent PPO policies controlling the agents.

    Improved hyperparameters for better stability:
    - Higher n_steps (512) for more stable advantage estimates
    - Smaller batch_size (64) for more frequent updates
    - More n_epochs (10) for better optimization
    - Lower learning rate helps with stability
    """
    env = make_env(grid_size, num_agents, tasks_per_agent, max_steps, task_duration)
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,              # Discount factor (standard)
        gae_lambda=0.98,         # GAE lambda for better advantage estimation
        clip_range=0.2,          # PPO clip range
        clip_range_vf=0.2,       # Clip value function updates too
        ent_coef=0.01,           # Entropy coefficient for exploration
        vf_coef=0.5,             # Value function loss coefficient
        max_grad_norm=0.5,       # Gradient clipping to prevent exploding gradients
        normalize_advantage=True, # Normalize advantages for stability
        tensorboard_log=None,    # Disable TensorBoard logging
    )

    print(f"Training PPO with:")
    print(f"  n_steps={n_steps} (rollout buffer size)")
    print(f"  batch_size={batch_size} (mini-batch size)")
    print(f"  n_epochs={n_epochs} (gradient updates per rollout)")
    print(f"  learning_rate={learning_rate}")
    print(f"  Total timesteps: {total_timesteps}")
    print()

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

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
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate (lower is more stable).")
    parser.add_argument("--n-steps", type=int, default=256, help="Number of steps to collect before PPO update (rollout length).")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size for gradient updates.")
    parser.add_argument("--n-epochs", type=int, default=4, help="Number of PPO epochs per update.")
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
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
