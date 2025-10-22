import sys
sys.path.insert(0, '.')

from src.multi_agent_deception.environment import parallel_env
from stable_baselines3 import PPO
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
import numpy as np

# Build environment
def env_fn():
    return parallel_env(grid_size=8, num_agents=4, tasks_per_agent=3, max_steps=200, task_duration=3)

vec_env = concat_vec_envs_v1(
    pettingzoo_env_to_vec_env_v1(env_fn()),
    num_vec_envs=1,
    num_cpus=1,
    base_class="stable_baselines3",
)

# Load model
model = PPO.load("artifacts/ppo_simple_hidden_role.zip")

# Reset and get observations
obs = vec_env.reset()
print(f"Observation shape: {obs.shape}")
print(f"First observation:\n{obs[0]}")

# Predict actions
actions, _ = model.predict(obs, deterministic=True)
print(f"\nActions predicted by model: {actions}")
print(f"Actions shape: {actions.shape}")
print(f"Action values: {[int(a) for a in actions]}")

# Test a few steps
print("\nTesting 10 steps:")
for step in range(10):
    actions, _ = model.predict(obs, deterministic=True)
    print(f"Step {step}: actions = {[int(a) for a in actions]}")
    obs, rewards, dones, _ = vec_env.step(actions)
    if dones.all():
        print("Episode ended")
        break
