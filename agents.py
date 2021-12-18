import random
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from model import Model

class Agent:

    def __init__(self, obs_dim, num_actions, buffer_size=100_000, batch_size=32):

        self.buffer = ReplayBuffer(obs_dim, num_actions, buffer_size=buffer_size, batch_size=batch_size)
        self.actor_model = Model(output_size=num_actions)
        self.critic_model = Model(output_size=1)

        self.num_actions = num_actions
        self.attention_dim = (*obs_dim[:-1], 1)

    def _random_action(self):
        action = random.randint(0, self.num_actions-1)
        logits = np.zeros(self.num_actions, dtype=float)
        attention = np.zeros(self.attention_dim, dtype=float)

        return action, logits, attention

    def observe(self, observation, action, logits, reward, next_observation, done):
        self.buffer.store_transition(observation, action, logits, reward, next_observation, done)

    def select_action(self, observation):

        # action, logits, attention = self._random_action() # TODO insert model (network) here! 

        # Convert numpy array into tensor and add dummy batch dim
        observation = tf.convert_to_tensor(observation)
        observation = tf.expand_dims(observation, axis=0)

        action, logits, attention = self.actor_model.forward(observation)

        action = action.numpy()[0]
        logits = logits.numpy()
        attention = attention.numpy()

        return action, logits, attention

    def learn(self):
        obs_batch, act_batch, logits_batch, rew_batch, next_obs_batch, done_batch = self.buffer.sample()

        with tf.GradientTape() as tape:
            

    def policy_loss(log_prob_prev, log_prob, advantage, epsilon):
        ratio = tf.exp(log_prob - log_prob_prev)
        clip = tf.clip_by_value(ratio, 1-epsilon, 1+epsilon)
        return tf.minimum(ratio*advantage, clip)