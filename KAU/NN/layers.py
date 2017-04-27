import math
import tensorflow as tf

def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


def weight_variable_xavier(name,shape):
    W=tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    return W

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

        with tf.name_scope('Activate'):
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




class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def cnn_layer_with_BN(input_tensor,
              kernel_dim,
              stride_dim,
              pool_dim,
              pool_stride,
              layer_name,
              act= tf.nn.relu):

    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            W = weight_variable_xavier(name=layer_name+'_weight',shape=kernel_dim)
            variable_summaries(W)

        with tf.name_scope('preActivate'):
            preActivate = tf.nn.conv2d(input_tensor, W, strides=stride_dim, padding='SAME')
            #tf.summary.image('pre_activations', tf.reshape(preActivate, [-1,28,28,1]))

        with tf.name_scope('Activation'):
            bN = batch_norm(name=layer_name+'_BN')
            activations = act(bN(preActivate))
            #tf.summary.image('activations', tf.reshape(activations, [-1,28,28,1]))

        with tf.name_scope('max_pool'):
            pooledImage = tf.nn.max_pool(activations, ksize=pool_dim, strides=pool_stride, padding='SAME')
            #tf.summary.image('max_pool', pooledImage)

    return pooledImage

def nn_layer_with_BN(input_tensor, input_dim, output_dim, layer_name, act= tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            W1 = weight_variable_xavier(name=layer_name + '_weight', shape=[input_dim,output_dim])
            variable_summaries(W1)
        with tf.name_scope('bias'):
            B1 = weight_variable_xavier(name=layer_name + '_bias', shape=[output_dim])
            variable_summaries(B1)
        with tf.name_scope('preActivate'):
            preActivate =tf.matmul(input_tensor, W1) + B1
            tf.summary.histogram('pre_activations',preActivate)
        with tf.name_scope('Activate'):
            if(act != tf.identity):
                bN = batch_norm(name=layer_name+'_BN')
                activations = act(bN(preActivate))
            else : activations = act(preActivate)
            tf.summary.histogram('activations',activations)
        return activations
