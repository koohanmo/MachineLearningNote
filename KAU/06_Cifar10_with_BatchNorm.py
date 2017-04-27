#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

print (tf.__version__)

# Load Data
from NN import cifar10
from NN import layers

FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summaries_dir','/tmp/cifar10_BN',"""Summary directory path for tensorboard""")
tf.app.flags.DEFINE_integer('learning_rate',0.0015,"""Learning rate for Adam Optimizor""")

X= tf.placeholder(tf.float32,shape=[None,32,32,3])
Y_Label = tf.placeholder(tf.float32,shape=[None,10])

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
print(class_names)

lr=0.0


for x in range(100):
    lr = lr+0.00001
    x=str(x)

    # X= [Batch,32,32,3]
    CNN1 = layers.cnn_layer_with_BN(input_tensor=X,
                             kernel_dim = [16,16,3,9],
                             stride_dim = [1,1,1,1],
                             pool_dim=[1,4,4,1],
                             pool_stride=[1,2,2,1],
                             layer_name="CNN_Layer1"+"_"+x,
                             )
    # CNN1= [Batch,16,16,9]
    CNN2 = layers.cnn_layer_with_BN(input_tensor=CNN1,
                             kernel_dim = [8,8,9,18],
                             stride_dim = [1,1,1,1],
                                pool_dim=[1, 4, 4, 1],
                                pool_stride=[1, 2, 2, 1],
                             layer_name="CNN_Layer2"+"_"+x,
                             )
    # CNN2= [Batch,8,8,18]
    CNN3 = layers.cnn_layer_with_BN(input_tensor=CNN2,
                             kernel_dim = [4,4,18,36],
                             stride_dim = [1,1,1,1],
                             pool_dim=[1,2,2,1],
                             pool_stride=[1,1,1,1],
                             layer_name="CNN_Layer3"+"_"+x,
                             )
    # CNN3= [Batch,8,8,36]
    CNN4 = layers.cnn_layer_with_BN(input_tensor=CNN3,
                             kernel_dim = [4,4,36,48],
                             stride_dim = [1,1,1,1],
                             pool_dim=[1,2,2,1],
                             pool_stride=[1,1,1,1],
                             layer_name="CNN_Layer4"+"_"+x,
                             )
    # CNN4= [Batch,8,8,48]
    CNN5 = layers.cnn_layer_with_BN(input_tensor=CNN4,
                             kernel_dim = [4,4,48,96],
                             stride_dim = [1,1,1,1],
                             pool_dim=[1,2,2,1],
                             pool_stride=[1,1,1,1],
                             layer_name="CNN_Layer5"+"_"+x,
                             )
    # CNN5= [Batch,8,8,96]
    CNN6 = layers.cnn_layer_with_BN(input_tensor=CNN5,
                             kernel_dim = [4,4,96,192],
                             stride_dim = [1,2,2,1],
                             pool_dim=[1,2,2,1],
                             pool_stride=[1,2,2,1],
                             layer_name="CNN_Layer6"+"_"+x,
                             )
    # CNN6= [Batch,2,2,192]
    CNN6_flat = tf.reshape(CNN6,[-1,2*2*192])

    outputLayer =layers.nn_layer_with_BN(input_tensor=CNN6_flat,
                           input_dim=2*2*192,
                           output_dim=10,
                           layer_name='OutputLayer'+"_"+x,
                            act=tf.identity)


    with tf.name_scope('cross_entropy'+"_"+x):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=outputLayer)
        with tf.name_scope('total'+"_"+x):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy'+"_"+x, cross_entropy)


    with tf.name_scope('train'+"_"+x):
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    with tf.name_scope('accuracy'+"_"+x):
        correct_prediction = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(Y_Label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy'+"_"+x,accuracy)


    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/test')
        train_writer = tf.summary.FileWriter(FLAG.summaries_dir + '/train')

        print("Start....")
        images_train, cls_train, labels_train = cifar10.load_training_data()
        images_test, cls_test, labels_test = cifar10.load_test_data()

        batch_size =100
        batch_start_idx =0
        batch_end_idx = batch_size
        maxidx = images_train.shape[0]
        print(maxidx)

        sess.run(tf.global_variables_initializer())
        for i in range(30000):
            trainingData = images_train[batch_start_idx:batch_end_idx,:]
            Y = labels_train[batch_start_idx:batch_end_idx,:]
            sess.run(train_step,feed_dict={X:trainingData,Y_Label:Y})


            if i%100==0:
                summary, acc = sess.run([merged,accuracy],feed_dict={X:images_test[:2000,],Y_Label:labels_test[:2000,]})
                test_writer.add_summary(summary,i)
                print("Test Accuracy at step %s : %s"%(i,acc))

                summary, acc = sess.run([merged, accuracy], feed_dict={X: trainingData, Y_Label: Y})
                train_writer.add_summary(summary, i)
                print("Train Accuracy at step %s : %s" % (i, acc))

            batch_start_idx = (batch_start_idx+batch_size)%maxidx
            batch_end_idx = batch_start_idx+ batch_size



