import sys
sys.path.insert(0, '.')

from src.multi_agent_deception.environment import parallel_env
import numpy as np

env = parallel_env(grid_size=4, num_agents=2, tasks_per_agent=2, max_steps=20, task_duration=2)
obs, _ = env.reset(seed=42)

print('Initial positions:')
for agent in env.agents:
    print(f'  {agent}: {env._agent_positions[agent]}')

print('\nTesting random actions:')
for step in range(3):
    actions = {agent: np.random.randint(0, 5) for agent in env.agents}
    print(f'\nStep {step} actions: {actions}')
    obs, rewards, terms, truncs, infos = env.step(actions)
    print('New positions:')
    for agent in env.agents:
        print(f'  {agent}: {env._agent_positions[agent]}')
