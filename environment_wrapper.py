import supersuit
import numpy as np

class EnvironmentWrapper:

    def __init__(self, env):

        self.env = self._preprocess(env)

    def _preprocess(self, env):
        # Preprocessing
        # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
        # to deal with frame flickering
        env = supersuit.max_observation_v0(env, 2)

        # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
        env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

        # skip frames for faster processing and less control
        # to be compatible with gym, use frame_skip(env, (2,5))
        env = supersuit.frame_skip_v0(env, 4)

        # downscale observation for faster processing
        env = supersuit.resize_v0(env, 84, 84)

        # allow agent to see everything on the screen despite Atari's flickering screen problem
        env = supersuit.frame_stack_v1(env, 4)

        return env

    def _add_zero_attention_layer(self, observations):

        zero_attention_layer = np.zeros((84,84,1), dtype=float) # TODO (Claude) hard coded dims for now.

        for agent in self.env.agents:
            observations[agent] = np.concatenate([observations[agent], zero_attention_layer], axis=-1)

        return observations

    def _add_attention_layers(self, observations, attention_layers):

        for agent in self.env.agents:
            observations[agent] = np.concatenate([observations[agent], attention_layers[agent]], axis=-1)

        return observations

    def reset(self):
        observations = self.env.reset()

        observations = self._add_zero_attention_layer(observations)

        return observations

    def step(self, actions, attention_layers):

        observations, rewards, dones, info = self.env.step(actions)

        observations = self._add_attention_layers(observations, attention_layers)

        # TODO (Claude) adjust reward so that it becomes a cooperative game

        return observations, rewards, dones, info
