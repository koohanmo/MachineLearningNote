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
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print(class_names)

X= tf.placeholder(tf.float32,shape=[None,32,32,3])
Y_Label = tf.placeholder(tf.float32,shape=[None,10])

Kernel1 = tf.get_variable(name='kerner1',shape=[4,4,3,4],initializer=tf.contrib.layers.xavier_initializer())
Bias1 = tf.get_variable(name='Bias1',shape=[4],initializer=tf.contrib.layers.xavier_initializer())
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1,1,1,1], padding='SAME') + Bias1
Activation1 = tf.nn.relu(Conv1)
Pool1 = tf.nn.max_pool(Activation1, ksize=[1,2,2,1] , strides=[1,2,2,1], padding='SAME')

Kernel2 = tf.get_variable(name='kerner2',shape=[4,4,4,8],initializer=tf.contrib.layers.xavier_initializer())
Bias2 = tf.get_variable(name='Bias2',shape=[8],initializer=tf.contrib.layers.xavier_initializer())
Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides=[1,1,1,1], padding='SAME') + Bias2
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2, ksize=[1,2,2,1] , strides=[1,2,2,1], padding='SAME')

Kernel3 = tf.get_variable(name='kerner3',shape=[4,4,8,16],initializer=tf.contrib.layers.xavier_initializer())
Bias3 = tf.get_variable(name='Bias3',shape=[16],initializer=tf.contrib.layers.xavier_initializer())
Conv3 = tf.nn.conv2d(Pool2, Kernel3, strides=[1,1,1,1], padding='SAME') + Bias3
Activation3 = tf.nn.relu(Conv3)
Pool3 = tf.nn.max_pool(Activation3, ksize=[1,2,2,1] , strides=[1,1,1,1], padding='SAME')

W1 = tf.get_variable(name='w1',shape=[8*8*8,8*8],initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.get_variable(name='b1',shape=[8*8],initializer=tf.contrib.layers.xavier_initializer())
Pool2_flat = tf.reshape(Pool2,[-1,8*8*8])
Activation4 = tf.nn.relu(tf.matmul(Pool2_flat,W1)+B1)

W2 = tf.get_variable(name='w2',shape=[8*8,30],initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.get_variable(name='b2',shape=[30],initializer=tf.contrib.layers.xavier_initializer())
Activation5 = tf.nn.relu(tf.matmul(Activation4,W2)+B2)

W3 = tf.get_variable(name='w3',shape=[30,10],initializer=tf.contrib.layers.xavier_initializer())
B3 = tf.get_variable(name='b3',shape=[10],initializer=tf.contrib.layers.xavier_initializer())
OutputLayer = tf.matmul(Activation5,W3)+B3

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.0005).minimize(Loss)

correct_prediction = tf.equal(tf.argmax(OutputLayer, 1), tf.argmax(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    print("Start....")
    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    batch_size =100
    batch_start_idx =0
    batch_end_idx = batch_size
    maxidx = images_train.shape[0]
    print(maxidx)

    sess.run(tf.global_variables_initializer())
    for i in range(100000000):
        trainingData = images_train[batch_start_idx:batch_end_idx,:]
        Y = labels_train[batch_start_idx:batch_end_idx,:]
        sess.run(train_step,feed_dict={X:trainingData,Y_Label:Y})
        if i%100 :
            print(sess.run(accuracy,feed_dict={X:images_test,Y_Label:labels_test}))
        batch_start_idx = (batch_start_idx+batch_size)%maxidx
        batch_end_idx = batch_start_idx+ batch_size


