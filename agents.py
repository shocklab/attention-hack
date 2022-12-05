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
        eps_dec_steps=1e4, train_period=4, lr=3e-4, gamma=0.99, target_update_rate=0.01, 
        importance_sampling_exponent=0.2, huber_loss_parameter=1., **kwargs):

        # self.buffer = NumpyReplayBuffer(obs_shape, buffer_size=buffer_size, batch_size=batch_size)
        self.buffer = ReverbReplayBuffer(obs_shape, buffer_size=buffer_size, batch_size=batch_size)
        self.Q_net = QNetwork(obs_shape=obs_shape, num_actions=num_actions)
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.optim = snt.optimizers.Adam(lr)

        self.num_actions = num_actions
        self.gamma = gamma
        self.importance_sampling_exponent = importance_sampling_exponent
        self.huber_loss_parameter = huber_loss_parameter
        self.target_update_rate = target_update_rate

        self.train_period = train_period
        self.train_call_ctr = 0
        self.train_ctr = 0

        # Epsilon for exploration
        self.eps = 1.0
        self.eps_min = eps_min
        self.eps_dec_steps = eps_dec_steps
        self.eps_dec = eps_min ** (1/eps_dec_steps) # exponential decay

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
            act = self._select_greedy_action(obs)
            act = int(act.numpy()[0])

        return act

    @tf.function
    def _select_greedy_action(self, obs):
        return tf.argmax(self.Q_net(obs), axis=-1)

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.store_transition(obs, act, rew, next_obs, done)

    def train(self):
        if not self.buffer.is_ready():
            return {}

        self.train_call_ctr += 1

        if self.train_call_ctr % self.train_period == 0:
            self.train_ctr += 1
            batch = self.buffer.sample()
            logs, reverb_priorities = self._train(batch)
            logs = tree.map_structure(np.array, logs)
            logs["Train Steps"] = self.train_ctr

            # Mutate reverb priorities
            client = self.buffer._server.localhost_client()
            reverb_priorities = tree.map_structure(np.array, reverb_priorities)
            updates = dict(zip(*reverb_priorities))

            client.mutate_priorities(
                table="my_table",
                updates=updates
            )

            return logs
        else:
            return {}

    @tf.function
    def _train(self, batch):
        keys, probs = batch.info[:2]
        obs, act, rew, next_obs, done = batch.data["obs"], batch.data["act"], batch.data["rew"], batch.data["next_obs"], batch.data["done"]

        # Double Q-learning target
        target_q_values = self.target_Q_net(next_obs)
        target_q_select = self.Q_net(next_obs)
        next_actions = tf.argmax(target_q_select, axis=-1)
        target_qs = trfl.batched_index(target_q_values, next_actions)

        targets = rew + self.gamma * (1.0 - done) * target_qs

        with tf.GradientTape() as tape:
            qs = self.Q_net(obs)
            qs = trfl.batched_index(qs, act)

            # TD-error
            td_error = (targets - qs)

            # Huber loss
            loss = huber(td_error, self.huber_loss_parameter)

            # Get the importance weights.
            importance_weights = 1. / tf.cast(probs, "float32")  # [B]
            importance_weights **= self.importance_sampling_exponent
            importance_weights /= tf.reduce_max(importance_weights)

            # Importance weighted loss
            loss *= importance_weights

            # Mean
            loss = tf.reduce_mean(loss)

        vars = self.Q_net.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optim.apply(grads, vars)

        # Soft target update
        tau = self.target_update_rate
        for src, dest in zip(self.Q_net.variables, self.target_Q_net.variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

        # Reverb priorities
        priorities = tf.abs(td_error)
        keys_priorities = (keys, priorities)

        return {"Loss": loss, "Mean Q-value": tf.reduce_mean(qs), "Max Reward": tf.reduce_max(rew), "Min Reward": tf.reduce_min(rew)}, keys_priorities

def huber(inputs, quadratic_linear_boundary):
    """Calculates huber loss of `inputs`."""
    if quadratic_linear_boundary < 0:
        raise ValueError("quadratic_linear_boundary must be >= 0.")

    abs_x = tf.abs(inputs)
    delta = tf.constant(quadratic_linear_boundary)
    quad = tf.minimum(abs_x, delta)
    lin = (abs_x - quad)
    return 0.5 * quad**2 + delta * lin