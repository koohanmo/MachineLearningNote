import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act= tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            W1 = weight_variable([input_dim, output_dim])
            variable_summaries(W1)
        with tf.name_scope('bias'):
            B1 = bias_variable([output_dim])
            variable_summaries(B1)
        with tf.name_scope('preActivate'):
            preActivate =tf.matmul(input_tensor, W1) + B1
            tf.summary.histogram('pre_activations',preActivate)
        activations = act(preActivate)
        tf.summary.histogram('activations',activations)
        return activations

def cnn_layer(input_tensor, kernel_dim, stride_dim, pool_dim, pool_stride, layer_name, act= tf.nn.relu):

    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            W = weight_variable(kernel_dim)
            variable_summaries(W)

        with tf.name_scope('bias'):
            B = bias_variable([kernel_dim[-1]])
            variable_summaries(B)

        with tf.name_scope('preActivate'):
            preActivate = tf.nn.conv2d(input_tensor, W, strides=stride_dim, padding='SAME') + B
            #tf.summary.image('pre_activations', tf.reshape(preActivate, [-1,28,28,1]))

        with tf.name_scope('preActivate'):
            activations = act(preActivate)
            #tf.summary.image('activations', tf.reshape(activations, [-1,28,28,1]))

        with tf.name_scope('max_pool'):
            pooledImage = tf.nn.max_pool(activations, ksize=pool_dim, strides=pool_stride, padding='SAME')
            #tf.summary.image('max_pool', pooledImage)

    return pooledImage

def cnn_to_fc_layer(input_tensor, flat_dim, output_dim, layer_name, act=tf.nn.relu):

    with tf.name_scope(layer_name):
        with tf.name_scope('flat_input'):
            poolToFlat = tf.reshape(input_tensor, [-1, flat_dim])
            variable_summaries(poolToFlat)

        with tf.name_scope('weight'):
            W = weight_variable([flat_dim, output_dim])
            variable_summaries(W)

        with tf.name_scope('bias'):
            B = bias_variable([output_dim])
            variable_summaries(B)

        with tf.name_scope('preActivate'):
            preActivate = tf.matmul(poolToFlat, W) + B
            tf.summary.histogram('pre_activations', preActivate)

        activations = act(preActivate)
        tf.summary.histogram('activations', activations)

    return activations



def nn_layer(input_tensor, input_dim, output_dim, layer_name, act= tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            W1 = weight_variable([input_dim, output_dim])
            variable_summaries(W1)
        with tf.name_scope('bias'):
            B1 = bias_variable([output_dim])
            variable_summaries(B1)
        with tf.name_scope('preActivate'):
            preActivate =tf.matmul(input_tensor, W1) + B1
            tf.summary.histogram('pre_activations',preActivate)
        activations = act(preActivate)
        tf.summary.histogram('activations',activations)
    return activations