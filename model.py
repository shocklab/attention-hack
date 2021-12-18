import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class Model:

  def __init__(self, output_size):

    self.network = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=[8, 8], strides=[4, 4]),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, kernel_size=[4, 4], strides=[2, 2]),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, kernel_size=[3, 3], strides=[1, 1]),
        keras.layers.ReLU(),
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.Dense(output_size)
    ])

  def forward(self, observations):
    logits = self.network(observations)

    distribution = tfp.distributions.Categorical(logits=logits) 
    action = distribution.sample()

    attention = tf.zeros((84,84,1), dtype="float32") # TODO compute attention here.

    return action, logits, attention
