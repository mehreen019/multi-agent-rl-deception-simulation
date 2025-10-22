from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

from src.multi_agent_deception.environment import AgentID, SimpleHiddenRoleParallelEnv, parallel_env
                                                                                                                                          
                                                                                                                                          
def build_vec_env(**env_kwargs) -> tuple[SimpleHiddenRoleParallelEnv, VecEnv]:
    """Create the PettingZoo environment and wrap it for SB3 compatibility."""
    def env_fn():
        env = parallel_env(**env_kwargs)
        return env

    # Use concat_vec_envs which will create envs from the factory
    vec_env = concat_vec_envs_v1(
        pettingzoo_env_to_vec_env_v1(env_fn()),
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    # Create a separate instance for rendering/inspection
    base_env = env_fn()
    base_env.reset()

    return base_env, vec_env
                                                                                                                                          
def format_step_summary(env: SimpleHiddenRoleParallelEnv, rewards: Dict[AgentID, float]) -> str:                                          
    """Return a human-readable summary for the current step."""
    lines: list[str] = []                                                                                                                 
    for agent in env.possible_agents:                                                                                                     
        if agent not in env._agent_positions:                                                                                             
            continue                                                                                                                      
        pos = tuple(int(c) for c in env._agent_positions[agent])                                                                          
        completed = int(env._completed_tasks[agent].sum())
        total = env.tasks_per_agent                                                                                                       
        reward = rewards.get(agent, 0.0)                                                                                                  
        lines.append(f"{agent} pos={pos} tasks={completed}/{total} last_reward={reward:+.2f}")                                            
    return " | ".join(lines)                                                                                                              
                                                                                                                                          
def run_episode(
    base_env: SimpleHiddenRoleParallelEnv,
    vec_env: VecEnv,
    model: PPO,
    *,
    deterministic: bool,
    render_ascii: bool,
    seed: int | None = None,
) -> Dict[str, float]:
    """Roll out a single episode and return aggregate statistics."""
    obs = vec_env.reset()
    base_env.reset(seed=seed)
    episode_rewards = {agent: 0.0 for agent in base_env.possible_agents}
    dones = np.zeros(vec_env.num_envs, dtype=bool)

    step_idx = 0
    while not dones.all():
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = vec_env.step(actions)

        # Step the base_env to keep it synchronized for rendering
        if base_env.agents:
            actions_dict = {agent: int(actions[idx]) for idx, agent in enumerate(base_env.possible_agents)}
            base_env.step(actions_dict)

        for agent_idx, agent in enumerate(base_env.possible_agents):
            episode_rewards[agent] += float(rewards[agent_idx])

        if render_ascii:
            base_env.render()
        step_rewards = {
            agent: float(rewards[idx]) for idx, agent in enumerate(base_env.possible_agents)
        }
        summary = format_step_summary(base_env, step_rewards)
        print(f"step {step_idx:03d}: {summary}")
        step_idx += 1

    return episode_rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run a trained PPO policy in the hidden-role environment.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/ppo_simple_hidden_role.zip"),
        help="Path to the Stable-Baselines3 .zip checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions instead of sampling.")
    parser.add_argument("--render-ascii", action="store_true", help="Print the ASCII grid on every step.")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=4)                                                                              
    parser.add_argument("--tasks-per-agent", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--task-duration", type=int, default=3)                                                                           
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed passed to env reset.")                                  
    return parser.parse_args()                                                                                                            
                                                                                                                                          
                                                                                                                                          
def main() -> None:
    args = parse_args()                                                                                                                   
                                                                                                                                          
    if not args.model_path.exists():                                                                                                      
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")                                                         
                                                                                                                                          
    base_env, vec_env = build_vec_env(                                                                                                    
        grid_size=args.grid_size,                                                                                                         
        num_agents=args.num_agents,                                                                                                       
        tasks_per_agent=args.tasks_per_agent,                                                                                             
        max_steps=args.max_steps,                                                                                                         
        task_duration=args.task_duration,                                                                                                 
        seed=args.seed,                                                                                                                   
    )                                                                                                                                     
    model = PPO.load(str(args.model_path))                                                                                                
                                                                                                                                          
    for episode in range(args.episodes):
        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
        episode_rewards = run_episode(
            base_env,
            vec_env,
            model,
            deterministic=args.deterministic,
            render_ascii=args.render_ascii,
            seed=args.seed,
        )                                                                                                                                 
        reward_summary = ", ".join(                                                                                                       
            f"{agent}:{total:+.2f}" for agent, total in episode_rewards.items()                                                           
        )                                                                                                                                 
        print(f"Episode {episode + 1} finished | total reward per agent -> {reward_summary}")                                             
                                                                                                                                          
if __name__ == "__main__":                                                                                                                
    main()       
