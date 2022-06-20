import tensorflow as tf

def sign(A):
    return tf.cast(tf.math.greater(A, 0), tf.float32) - tf.cast(tf.math.less(A, 0), tf.float32)

def power_transform(data, zeta):
    return sign(data) * tf.abs(data)**zeta

def power_transform_inverse(data, zeta):
    return sign(data) * tf.abs(data)**(1./zeta)