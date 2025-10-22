from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict

import numpy as np
import pygame
from stable_baselines3 import PPO
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

from src.multi_agent_deception.environment import AgentID, SimpleHiddenRoleParallelEnv, parallel_env                                          

from src.multi_agent_deception.gui import (                                                                                                   

    GUIState,                                                                                                                             

    FPS,                                                                                                                                  

    _draw_agents,                                                                                                                         

    _draw_grid,

    _draw_panel,

    _draw_tasks,                                                                                                                          

    _init_pygame,                                                                                                                         

)

def build_vec_env(**env_kwargs):
    def env_fn():
        return parallel_env(**env_kwargs)

    vec_env = concat_vec_envs_v1(
        pettingzoo_env_to_vec_env_v1(env_fn()),
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    # Create a separate instance for rendering
    base_env = env_fn()
    base_env.reset()

    return base_env, vec_env

                                                                                                                                          

                                                                                                                                          

def rewards_to_dict(env: SimpleHiddenRoleParallelEnv, rewards: np.ndarray) -> Dict[AgentID, float]:                                       

    return {agent: float(rewards[idx]) for idx, agent in enumerate(env.possible_agents)}                                                  

                                                                                                                                          

                                                                                                                                          

def collect_infos(env: SimpleHiddenRoleParallelEnv) -> Dict[AgentID, dict]:

    infos: Dict[AgentID, dict] = {}

    for agent in env.possible_agents:

        if agent not in env._completed_tasks:

            continue

        infos[agent] = env._build_info(agent)

    return infos

                                                                                                                                          

                                                                                                                                          

def main():                                                                                                                               

    parser = argparse.ArgumentParser("Render a trained PPO policy inside the Pygame viewer.")                                             

    parser.add_argument("--model-path", type=Path, default=Path("artifacts/ppo_simple_hidden_role.zip"))                                  

    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions.")                                               

    parser.add_argument("--fps", type=int, default=FPS)                                                                                   

    parser.add_argument("--grid-size", type=int, default=8)                                                                               

    parser.add_argument("--num-agents", type=int, default=4)                                                                              

    parser.add_argument("--tasks-per-agent", type=int, default=3)                                                                         

    parser.add_argument("--max-steps", type=int, default=200)                                                                             

    parser.add_argument("--task-duration", type=int, default=3)                                                                           

    parser.add_argument("--seed", type=int, default=None)                                                                                 

    args = parser.parse_args()                                                                                                            

                                                                                                                                          

    if not args.model_path.exists():                                                                                                      

        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")                                                               

                                                                                                                                          

    base_env, vec_env = build_vec_env(                                                                                                    

        grid_size=args.grid_size,                                                                                                         

        num_agents=args.num_agents,                                                                                                       

        tasks_per_agent=args.tasks_per_agent,                                                                                             

        max_steps=args.max_steps,                                                                                                         

        task_duration=args.task_duration,                                                                                                 

        seed=args.seed,                                                                                                                   

    )                                                                                                                                     

    model = PPO.load(str(args.model_path))                                                                                                

    obs = vec_env.reset()                                                                                                                 

                                                                                                                                          

    surface = _init_pygame(args.grid_size)                                                                                                

    clock = pygame.time.Clock()                                                                                                           

    state = GUIState(env=base_env, last_rewards={}, last_infos={}, running=True, paused=False)                                            

                                                                                                                                          

    while state.running:                                                                                                                  

        for event in pygame.event.get():                                                                                                  

            if event.type == pygame.QUIT:

                state.running = False                                                                                                     

            if event.type == pygame.KEYDOWN:                                                                                              

                if event.key == pygame.K_p:                                                                                               

                    state.paused = not state.paused                                                                                       

                if event.key == pygame.K_r:
                    obs = vec_env.reset()
                    base_env.reset()
                    state.last_rewards = {}
                    state.last_infos = collect_infos(base_env)                                                                            

                                                                                                                                          

        if not state.paused:
            actions, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _ = vec_env.step(actions)

            # Step the base_env to keep it synchronized for rendering
            if base_env.agents:
                actions_dict = {agent: int(actions[idx]) for idx, agent in enumerate(base_env.possible_agents)}
                base_env.step(actions_dict)

            state.last_rewards = rewards_to_dict(base_env, rewards)
            state.last_infos = collect_infos(base_env)

            if dones.all():
                # vec_env already reset the environment internally; grab the fresh observation
                obs = vec_env.reset()
                base_env.reset()
                state.last_rewards = {}
                state.last_infos = collect_infos(base_env)                                                                                

                                                                                                                                          

        _draw_grid(surface, base_env)                                                                                                     

        _draw_tasks(surface, base_env)                                                                                                    

        _draw_agents(surface, base_env)                                                                                                   

        _draw_panel(surface, base_env, state)                                                                                             

                                                                                                                                          

        pygame.display.flip()                                                                                                             

        clock.tick(args.fps)                                                                                                              

    pygame.quit()

if __name__ == "__main__":

    main()

