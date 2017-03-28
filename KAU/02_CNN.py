import tensorflow as tf
from NN import layers

from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summaries_dir','/tmp/cnn',"""Summary directory path for tensorboard""")
tf.app.flags.DEFINE_integer('learning_rate',1e-4,"""Learning rate for Adam Optimizor""")




X = tf.placeholder(tf.float32, shape=[None,784])
X_image = tf.reshape(X,[-1,28,28,1])
Y_Label = tf.placeholder(tf.float32, shape=[None,10])

inputLayer = layers.cnn_layer(input_tensor = X_image,
                              kernel_dim = [5,5,1,32], # 5X5 * ch1 * 32
                              stride_dim = [1,1,1,1],  # right 2, down 2
                              pool_dim = [1,2,2,1],    # 2 X 2
                              pool_stride = [1,2,2,1], # 2 X 2
                              layer_name = 'inputLayer',
                              act= tf.nn.relu)

convLayer1 = layers.cnn_layer(input_tensor = inputLayer,
                              kernel_dim = [5,5,32,64], # 5X5 * ch1 * 32
                              stride_dim = [1,1,1,1],  # right 2, down 2
                              pool_dim = [1,2,2,1],    # 2 X 2
                              pool_stride = [1,2,2,1], # 2 X 2
                              layer_name = 'convLayer1',
                              act= tf.nn.relu)

fullConnectedLayer1 = layers.cnn_to_fc_layer(input_tensor = convLayer1,
                                             flat_dim = 7*7*64,
                                             output_dim = 1024,
                                             layer_name = 'fullConnectedLayer1',
                                             act=tf.nn.relu)

outputLayer = layers.nn_layer(input_tensor = fullConnectedLayer1,
                              input_dim = 1024,
                              output_dim = 10,
                              layer_name = 'outputLayer',
                              act= tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=outputLayer)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(Y)
    train_step = tf.train.AdamOptimizer(FLAG.learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(Y_Label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

sess = tf.Session()

merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/test')
train_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/train')


sess.run(tf.global_variables_initializer())


for i in range(10000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={X:batch[0], Y_Label: batch[1]})
    # Test trained model
    if i%100==0:
        summary, acc = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels})
        test_writer.add_summary(summary, i)
        print("Accuracy at step %s: %s"%(i,acc))
    else :
        summary, acc = sess.run([merged, accuracy], feed_dict={X: batch[0], Y_Label: batch[1]})
        train_writer.add_summary(summary, i)
        print("Accuracy at step %s: %s" % (i, acc))


