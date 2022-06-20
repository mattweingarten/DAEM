import tensorflow as tf


def inverse_frequency_weights(mask, axis):
    return mask / tf.reduce_sum(mask, axis=axis, keepdims=True)