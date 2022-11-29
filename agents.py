import random
import copy
import numpy as np
import tensorflow as tf
import tree
import trfl
import sonnet as snt

from replay_buffer import NumpyReplayBuffer, ReverbReplayBuffer
from model import QNetwork

class Agent:

    def __init__(self, obs_shape, num_actions, buffer_size=100_000, batch_size=32, eps_min=0.05,
        eps_dec_steps=1e4, train_period=4, lr=3e-4, gamma=0.99, target_update_rate=0.01, **kwargs):

        # self.buffer = NumpyReplayBuffer(obs_shape, buffer_size=buffer_size, batch_size=batch_size)
        self.buffer = ReverbReplayBuffer(obs_shape, buffer_size=buffer_size, batch_size=batch_size)
        self.Q_net = QNetwork(obs_shape=obs_shape, num_actions=num_actions)
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.optim = snt.optimizers.Adam(lr)

        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_rate = target_update_rate

        self.train_period = train_period
        self.train_call_ctr = 0
        self.train_ctr = 0

        # Epsilon for exploration
        self.eps = 1.0
        self.eps_min = eps_min
        self.eps_dec_steps = eps_dec_steps
        self.eps_dec = eps_min ** (1/eps_dec_steps) # exponential decay

    # TODO: make this function faster. Add tf.function.
    def select_action(self, obs, eval=False):
        if not eval:
            # Decrement epsilon
            self.eps = max(self.eps * self.eps_dec, self.eps_min) # exponential decay
            eps = self.eps
        else:
            eps=0.0

        if random.random() < eps:
            act = random.randint(0, self.num_actions-1)
        else:
            obs = tf.expand_dims(tf.convert_to_tensor(obs, "float32"), axis=0)
            act = int(tf.argmax(self.Q_net(obs), axis=-1).numpy()[0])

        return act


    def store(self, obs, act, rew, next_obs, done):
        self.buffer.store_transition(obs, act, rew, next_obs, done)

    def train(self):
        if not self.buffer.is_ready():
            return {}

        self.train_call_ctr += 1

        if self.train_call_ctr % self.train_period == 0:
            self.train_ctr += 1
            batch = self.buffer.sample()
            logs = self._train(batch)
            logs = tree.map_structure(np.array, logs)
            logs["Train Steps"] = self.train_ctr
            return logs
        else:
            return {}

    @tf.function
    def _train(self, batch):
        obs, act, rew, next_obs, done = batch

        # Double Q-learning target
        target_q_values = self.target_Q_net(next_obs)
        target_q_select = self.Q_net(next_obs)
        next_actions = tf.argmax(target_q_select, axis=-1)
        target_qs = trfl.batched_index(target_q_values, next_actions)

        targets = rew + self.gamma * (1.0 - done) * target_qs

        with tf.GradientTape() as tape:
            qs = self.Q_net(obs)
            qs = trfl.batched_index(qs, act)
            loss = 0.5 * (targets - qs) ** 2
            loss = tf.reduce_mean(loss)

        vars = self.Q_net.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optim.apply(grads, vars)

        # Soft target update
        tau = self.target_update_rate
        for src, dest in zip(self.Q_net.variables, self.target_Q_net.variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

        return {"Loss": loss, "Mean Q-value": tf.reduce_mean(qs)}