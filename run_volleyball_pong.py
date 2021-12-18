import os
from pettingzoo.atari import volleyball_pong_v2
import numpy as np

from agents import Agent
from environment_wrapper import EnvironmentWrapper

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#CUDA_VISIBLE_DEVICES=""

# Create 2 player env
env = volleyball_pong_v2.parallel_env(obs_type='grayscale_image', num_players=2)

env = EnvironmentWrapper(env)

NUM_ACTIONS = 18 # TODO
MAX_EPISODE_LENGTH = 500
OBS_DIM = (84, 84, 5) # TODO

first_0 = Agent(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
second_0 = Agent(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)

agents = {"first_0": first_0, "second_0": second_0}

observations = env.reset()
for _ in range(MAX_EPISODE_LENGTH):
    actions = {}
    logits_dict = {}
    attention_layers = {}
    for agent in env.env.agents:
        action, logits, attention = agents[agent].select_action(observations[agent])

        actions[agent] = action
        logits_dict[agent] = logits
        attention_layers[agent] = attention

    # Step the environment
    next_observations, rewards, dones, infos = env.step(actions, attention_layers)

    # Add transitions to replay buffers
    for agent in env.env.agents:
        agents[agent].observe(
            observations[agent], 
            actions[agent], 
            logits_dict[agent], 
            rewards[agent],
            next_observations[agent],
            dones[agent]
        )

batch = agents["second_0"].buffer.sample()

print(batch[0].shape)