import tensorflow as tf

#Tensorflow Test
hello=tf.constant('Hello, TensorFlow!')

#Start tf session
sess=tf.Session()

print (hello)

print (sess.run(hello))