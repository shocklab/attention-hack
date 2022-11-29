from collections import deque
import numpy as np
import cv2
import gym
from gym import spaces

class Environment:

    def __init__(self):
        self.env = gym.make("SlimeVolleyNoFrameskip-v0")
        self.env = FrameStack(GreyScale(Reshape(self.env)), n_frames=4)
        self.agents = ["agent_0", "agent_1"]
        self.num_actions = 6
        self.obs_shape = self.env.observation_space.shape

    def reset(self):
        raw_obs = self.env.reset()

        obs = {"agent_0": raw_obs, "agent_1": raw_obs}

        return obs

    def step(self, actions):
        next_obs = {}
        rew = {}
        done = {}

        next_obs["agent_0"], rew["agent_0"], done["agent_0"], info = self.env.step(
            actions["agent_0"], 
            actions["agent_1"]
        )
        next_obs["agent_1"] = info['otherObs']
        rew["agent_1"] = - rew["agent_0"]
        done["agent_1"] = done["agent_0"]

        return next_obs, rew, done

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames."""
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames_0 = deque([], maxlen=n_frames)
        self.frames_1 = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames_0.append(obs)
            self.frames_1.append(obs)
        return self._get_obs()

    def step(self, action_1, action_2):
        obs, reward, done, info = self.env.step(action_1, action_2)
        self.frames_0.append(obs)
        self.frames_1.append(info["otherObs"])
        info["otherObs"] = self._get_otherObs()
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames_0) == self.n_frames
        return np.concatenate(list(self.frames_0), axis=2)

    def _get_otherObs(self):
        assert len(self.frames_1) == self.n_frames
        return np.concatenate(list(self.frames_1), axis=2)


class GreyScale(gym.Wrapper):

    def __init__(self, env):
        """Normalise and grayscale obs."""
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0], shp[1], 1),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        obs = np.mean(obs, axis=-1, keepdims=True) / 255.
        return obs

    def step(self, action_1, action_2):
        obs, reward, done, info = self.env.step(action_1, action_2)
        obs = np.mean(obs, axis=-1, keepdims=True) / 255.
        info["otherObs"] = np.mean(info["otherObs"], axis=-1, keepdims=True) / 255.
        return obs, reward, done, info

class Reshape(gym.Wrapper):

    def __init__(self, env, shape=(84,84)):
        """Normalise and grayscale obs."""
        gym.Wrapper.__init__(self, env)
        num_channels = self.observation_space.shape[-1]
        self.observation_space = spaces.Box(low=0, high=1, shape=(shape[0], shape[1], num_channels),
                                            dtype=env.observation_space.dtype)
        self.reshape_fn = lambda x: cv2.resize(x, shape)

    def reset(self):
        obs = self.env.reset()
        obs = self.reshape_fn(obs)
        return obs

    def step(self, action_1, action_2):
        obs, reward, done, info = self.env.step(action_1, action_2)
        obs = self.reshape_fn(obs)
        info["otherObs"] = self.reshape_fn(info["otherObs"])
        return obs, reward, done, info

