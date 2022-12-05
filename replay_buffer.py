import numpy as np
import tensorflow as tf
import tree
import reverb
from acme.datasets import make_reverb_dataset

class NumpyReplayBuffer:

    def __init__(self, obs_shape, buffer_size=100_000, batch_size=32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.obs_buffer_shape = (buffer_size, *obs_shape)
        self.act_buffer_shape =  (buffer_size,)
        self.rew_buffer_shape = self.act_buffer_shape
        self.done_buffer_shape = self.act_buffer_shape

        self.obs_buffer = np.zeros(shape=self.obs_buffer_shape, dtype="float32")
        self.act_buffer = np.zeros(shape=self.act_buffer_shape, dtype="int32")
        self.rew_buffer = np.zeros(shape=self.rew_buffer_shape, dtype="float32")
        self.next_obs_buffer = np.zeros(shape=self.obs_buffer_shape, dtype="float32")
        self.done_buffer = np.zeros(shape=self.done_buffer_shape, dtype="float32")

        self.ctr = 0
        

    def store_transition(self, obs, act, rew, next_obs, done):
        idx = self.ctr % self.buffer_size

        self.obs_buffer[idx] = obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self.next_obs_buffer[idx] = next_obs
        self.done_buffer[idx] = done

        self.ctr += 1

    def sample(self):
        max_idx = min(self.ctr, self.buffer_size)
        idxs = np.random.choice(max_idx, self.batch_size, replace=True)

        obs_batch = self.obs_buffer[idxs]
        act_batch = self.act_buffer[idxs]
        rew_batch = self.rew_buffer[idxs]
        next_obs_batch = self.next_obs_buffer[idxs]
        done_batch = self.done_buffer[idxs]

        batch = (obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)

        batch = tree.map_structure(tf.convert_to_tensor, batch)

        return batch

    def is_ready(self):
        return self.ctr >= self.batch_size

class ReverbReplayBuffer:

    def __init__(self, obs_shape, buffer_size=100_000, batch_size=32, samples_per_insert=None, priority_exponent=0.6):
        self._server = reverb.Server(tables=[
            reverb.Table(
                name='my_table',
                sampler=reverb.selectors.Prioritized(priority_exponent),
                remover=reverb.selectors.Fifo(),
                max_size=int(buffer_size),
                rate_limiter=reverb.rate_limiters.MinSize(1) if samples_per_insert is None else \
                    reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=samples_per_insert),
                signature={
                    'obs':
                        tf.TensorSpec([*obs_shape], "float32"),
                    'act':
                        tf.TensorSpec([], "int32"),
                    'rew':
                        tf.TensorSpec([], "float32"),
                    'next_obs':
                        tf.TensorSpec([*obs_shape], "float32"),
                    'done':
                        tf.TensorSpec([], "float32"),
                },
            ),
        ])

        self._dataset = make_reverb_dataset(
                server_address = f"localhost:{self._server._port}",
                batch_size = batch_size,
                prefetch_size = 4,
                table = "my_table",
                num_parallel_calls = 12,
        )
        self._dataset = iter(self._dataset)

        self.batch_size = batch_size
        self.ctr = 0
        

    def store_transition(self, obs, act, rew, next_obs, done):
        obs = obs.astype("float32")
        act = np.array(act).astype("int32")
        rew = np.array(rew).astype("float32")
        next_obs = next_obs.astype("float32")
        done = np.array(done).astype("float32")

        with self._server.localhost_client().trajectory_writer(num_keep_alive_refs=1) as writer:
            writer.append({
                'obs': obs,
                'act': act,
                'rew': rew,
                'next_obs': next_obs,
                'done': done,
            })

            writer.create_item(
                table='my_table',
                priority=1.,
                trajectory={
                    'obs': writer.history['obs'][0],
                    'act': writer.history['act'][0],
                    'rew': writer.history['rew'][0],
                    'next_obs': writer.history['next_obs'][0],
                    'done': writer.history['done'][0],
            })

            writer.end_episode(timeout_ms=1000)

            self.ctr += 1

    def sample(self):
        batch = next(self._dataset)
        # batch = (batch.data["obs"], batch.data["act"], batch.data["rew"], batch.data["next_obs"], batch.data["done"])
        return batch

    def is_ready(self):
        return self.ctr >= self.batch_size