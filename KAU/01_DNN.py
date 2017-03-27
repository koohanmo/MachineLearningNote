import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summaries_dir','/tmp/dnn',"""Summary directory path for tensorboard""")
tf.app.flags.DEFINE_integer('learning_rate',0.015,"""Learning rate for Adam Optimizor""")


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape)
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

with tf.name_scope('input'):
    X= tf.placeholder(tf.float32, [None, 784])
    variable_summaries(X)

inputLayer = nn_layer(X, 784, 342, 'inputLayer', act=tf.nn.relu)
hiddenLayer1 = nn_layer(inputLayer, 342, 171, 'HiddenLayer1', act=tf.nn.relu)
hiddenLayer2 = nn_layer(hiddenLayer1, 171, 85, 'HiddenLayer2', act=tf.nn.relu)
hiddenLayer3 = nn_layer(hiddenLayer2, 85, 42, 'HiddenLayer3', act=tf.nn.relu)
hiddenLayer4 = nn_layer(hiddenLayer3, 42, 21, 'HiddenLayer4', act=tf.nn.relu)
outputLayer = nn_layer(hiddenLayer4, 21, 10, 'OutputLayer', act=tf.identity)


Y_ = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=outputLayer)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(Y)
    train_step = tf.train.AdamOptimizer(FLAG.learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

sess = tf.Session()

merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/train',sess.graph)
test_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/test')


sess.run(tf.global_variables_initializer())


for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X:batch_xs, Y_: batch_ys})
    # Test trained model
    if i%100==0:
        summary, acc = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        test_writer.add_summary(summary, i)
        print("Accuracy at step %s: %s"%(i,acc))






