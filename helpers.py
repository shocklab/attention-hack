import tensorflow as tf
from tensorflow import keras

from networks import Linear, MLPBlock

def build_model(input_shape = (28,28), layers = [128,10], activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape), 
        for layer_size in layers:
            tf.keras.layers.Dense(layer_size, activation=activation), 
    ])