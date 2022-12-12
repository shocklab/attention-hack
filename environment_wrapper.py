from collections import deque
import numpy as np
import cv2
import gym
from gym import spaces

class DoubleIndependentGym:

    def __init__(self, env_name="CartPole-v1"):

        self.env_0 = gym.make(env_name)
        self.env_1 = gym.make(env_name)

        self.agents = ["agent_0", "agent_1"]
        self.num_actions = 2
        self.obs_shape = self.env_0.observation_space.shape

    def reset(self):
        obs_0 = self.env_0.reset()
        obs_1 = self.env_1.reset()

        obs = {"agent_0": obs_0, "agent_1": obs_1}

        return obs

    def step(self, actions):
        next_obs = {}
        rew = {}
        done = {}

        next_obs["agent_0"], rew["agent_0"], done["agent_0"], info = self.env_0.step(actions["agent_0"])
        next_obs["agent_1"], rew["agent_1"], done["agent_1"], info = self.env_1.step(actions["agent_1"])

        if done["agent_0"] or done["agent_1"]:
            done["agent_0"] = True
            done["agent_1"] = True

        return next_obs, rew, done

class Environment:

    def __init__(self, pixel_obs=False, survival_bonus=False, single_agent=False):

        if pixel_obs:
            self.env = gym.make("SlimeVolleyNoFrameskip-v0")
            self.env = FrameStack(GreyScale(Reshape(FrameSkipStickyAction(self.env, n_frames=2))), n_frames=4)
        else:
            self.env = gym.make("SlimeVolley-v0")

        self.single_agent = single_agent
        self.env.survival_bonus = survival_bonus

        if self.single_agent:
            self.agents = ["agent_0"]
        else:
            self.agents = ["agent_0", "agent_1"]

        self.num_actions = 6
        self.obs_shape = self.env.observation_space.shape

        self.action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)

    def reset(self):
        raw_obs = self.env.reset()

        if self.single_agent:
            obs = {"agent_0": raw_obs}
        else:
            obs = {"agent_0": raw_obs, "agent_1": raw_obs}

        return obs

    def step(self, actions):
        next_obs = {}
        rew = {}
        done = {}

        next_obs["agent_0"], rew["agent_0"], done["agent_0"], info = self.env.step(
            self.action_table[actions["agent_0"]], 
            self.action_table[actions["agent_1"]] if not self.single_agent else None
        )

        if not self.single_agent:
            next_obs["agent_1"] = info['otherObs']
            rew["agent_1"] = - rew["agent_0"]
            done["agent_1"] = done["agent_0"]

        return next_obs, rew, done

class FrameSkipStickyAction(gym.Wrapper):

    def __init__(self, env, n_frames):
        """Skip n_frams by repeating same action."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space
        self.n_frames = n_frames

    def reset(self):
        return self.env.reset()

    def step(self, action_1, action_2):
        tot_rew = 0
        for _ in range(self.n_frames):
            next_obs, rew, done, info = self.env.step(action_1, action_2)

            tot_rew += rew

            if done:
                break
        
        return next_obs, tot_rew, done, info

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

