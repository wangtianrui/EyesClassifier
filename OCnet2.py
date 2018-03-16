import tensorflow as tf


def get_weights(shape):
    weights = tf.get_variable('weights',
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    return weights


def get_biases(shape, value):
    biases = tf.get_variable("biases",
                             shape=shape,
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
    return biases


def inference(image_batch, batch_size, num_class):
    with tf.variable_scope('conv1') as scope:
        weights = get_weights(shape=[3, 3, 3, 64])
        biases = get_biases([64], 0.0)
        conv = tf.nn.conv2d(image_batch, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        bn1 = tf.layers.batch_normalization(pre_activation)
        conv1 = tf.nn.relu(bn1, name=scope.name)
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool1")
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    with tf.variable_scope("conv2") as scope:
        weights = get_weights([3, 3, 64, 32])
        biases = get_biases([32], 0.1)
        conv = tf.nn.conv2d(pool1, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        bn2 = tf.layers.batch_normalization(pre_activation)
        conv2 = tf.nn.relu(bn2, name=scope.name)
    with tf.variable_scope('pool2') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name="pool2")
    with tf.variable_scope("local1") as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = get_weights([dim, 192])
        biases = get_biases([192], 0.1)
        local1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(reshape, weights) + biases, axis=1),
                            name=scope.name)
    with tf.variable_scope('linear_out') as scope:
        weights = get_weights([192, num_class])
        biases = get_biases([2], 0.1)
        linear_out = tf.matmul(local1, weights) + biases

    return linear_out
