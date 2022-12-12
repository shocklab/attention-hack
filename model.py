import tensorflow as tf
import sonnet as snt

class QNetwork(snt.Module):

  def __init__(self, obs_shape, num_actions):

    if len(obs_shape) > 1:
      self.conv_net = snt.Sequential([
          snt.Conv2D(8, 8, 4),
          tf.nn.relu,
          snt.Conv2D(16, 4, 2),
          tf.nn.relu,
          snt.Conv2D(16, 3, 1),
          tf.nn.relu,
          tf.keras.layers.Flatten()
      ])
    else:
      self.conv_net = snt.Sequential([
        tf.identity
      ])

    self.mlp = snt.Sequential([
        snt.Linear(256),
        tf.nn.relu,
        snt.Linear(128),
        tf.nn.relu,
        snt.Linear(num_actions)
    ])

    super(QNetwork, self).__init__()

  def __call__(self, obs):
    embed = self.conv_net(obs)
    q_vals = self.mlp(embed)
    return q_vals
