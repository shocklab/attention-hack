import os
import numpy as np
import gym
import slimevolleygym
import wandb
import tensorflow as tf

from agents import Agent
from environment_wrapper import Environment

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = {
    "max_episodes": 1_000_000,
    "train_period": 4,
    "batch_size": 128,
    "eps_min": 0.05,
    "eps_dec_steps": 1e6,
    "buffer_size": int(1e4),
    "lr": 3e-4,
    "eval_period": 25,
    "eval_episodes": 2
}

wandb.init(
    project="Attention Hack",
    config=config,
)

# Create env
env = Environment()

num_actions = env.num_actions
obs_shape = env.obs_shape

# Setup agents
agents = {}
for agent_id in env.agents:
    agents[agent_id] = Agent(
        obs_shape=obs_shape, 
        num_actions=num_actions, 
        **config
    )

timesteps = 0
for ep in range(config["max_episodes"]):

    obs = env.reset()
    ep_done = False

    episode_return = {agent_id: 0 for agent_id in agents.keys()}
    episode_len = 0

    while not ep_done:

        act = {}
        for agent_id, agent in agents.items():
            act[agent_id] = agent.select_action(obs[agent_id])
        wandb.log({"Epsilon": agents["agent_0"].eps}, commit=False)

        next_obs, rew, done = env.step(act)

        for agent_id, agent in agents.items():
            agent.store(
                obs[agent_id],
                act[agent_id],
                rew[agent_id],
                next_obs[agent_id],
                done[agent_id]
            )

        for agent in agents.values():
            wandb.log(agent.train(), commit=False)

        obs = next_obs
        ep_done = all(done.values())

        for agent_id in agents.keys():
            episode_return[agent_id] += rew[agent_id]
        episode_len += 1
        timesteps += 1

    # Bookkeeping
    logs = {f"{agent_id} Episode Return": episode_return[agent_id] for agent_id in agents.keys()}
    logs["Episode Length"] = episode_len
    logs["Episodes"] = ep
    logs["Timesteps"] = timesteps

    # Eval loop
    if ep % config["eval_period"] == 0:
        eval_episode_len = 0
        for eval_ep in range(config["eval_episodes"]):
            ep_done = False
            env.reset()
            while not ep_done:
                act = {}
                for agent_id, agent in agents.items():
                    act[agent_id] = agent.select_action(obs[agent_id], eval=True)

                obs, _, done = env.step(act)

                eval_episode_len += 1

                ep_done = all(done.values())

        avg_eval_episode_len = eval_episode_len / config["eval_episodes"]
        print(f"Episode {ep}  Avg. Eval. Episode Length {avg_eval_episode_len}")
        logs.update({"Avg. Eval. Episode Length": avg_eval_episode_len})

    # Commit Logs
    wandb.log(logs)