import numpy as np
import tensorflow as tf

class ReplayBuffer:

    def __init__(self, obs_dim, num_actions, buffer_size=100_000, batch_size=32):
        obs_buffer_shape = (buffer_size, *obs_dim)
        self.obs_buffer = np.zeros(shape=obs_buffer_shape, dtype=float)
        self.next_obs_buffer = np.zeros(shape=obs_buffer_shape, dtype=float)

        act_buffer_shape =  (buffer_size, 1)
        self.act_buffer = np.zeros(shape=act_buffer_shape, dtype=int)

        rew_buffer_shape = (buffer_size, 1)
        self.rew_buffer = np.zeros(shape=rew_buffer_shape, dtype=float)

        logits_buffer_shape = (buffer_size, num_actions)
        self.logits_buffer = np.zeros(shape=logits_buffer_shape, dtype=float)

        done_buffer_shape =  (buffer_size, 1)
        self.done_buffer = np.zeros(shape=done_buffer_shape, dtype=int)

        self.ctr = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def store_transition(self, obs, act, logits, rew, next_obs, done):

        idx = self.ctr % self.buffer_size

        self.obs_buffer[idx] = obs
        self.act_buffer[idx] = act
        self.logits_buffer[idx] = logits
        self.rew_buffer[idx] = rew
        self.next_obs_buffer[idx] = next_obs
        self.done_buffer[idx] = done

        self.ctr += 1

    def sample(self):
        max_idx = min(self.ctr, self.buffer_size)
        idxs = np.random.choice(max_idx, self.batch_size, replace=True)

        obs_batch = self.obs_buffer[idxs]
        act_batch = self.act_buffer[idxs]
        logits_batch = self.logits_buffer[idxs]
        rew_batch = self.rew_buffer[idxs]
        next_obs_batch = self.next_obs_buffer[idxs]
        done_batch = self.done_buffer[idxs]

        return obs_batch, act_batch, logits_batch, rew_batch, next_obs_batch, done_batch