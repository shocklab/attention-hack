
class EnvironmentLoop:

    def __init__(self, env, agents, logger):

        self._env = env
        self._agents = agents
        self._logger = logger

    def run(self):
        agents = self._agents
        env = self._env
        logger = self._logger

        ep = 0
        timesteps = 0
        while True:
            ep += 1
            obs = env.reset()
            ep_done = False

            episode_return = {agent_id: 0 for agent_id in agents.keys()}
            episode_len = 0
            while not ep_done:

                act = {}
                for agent_id, agent in agents.items():
                    act[agent_id] = agent.select_action(obs[agent_id])

                next_obs, rew, done = env.step(act)

                for agent_id, agent in agents.items():
                    agent.store(
                        obs[agent_id],
                        act[agent_id],
                        rew[agent_id],
                        next_obs[agent_id],
                        done[agent_id]
                    )

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
            logs["Epsilon"] = agents["agent_0"].eps

            logger.log(logs)