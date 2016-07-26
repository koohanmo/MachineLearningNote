import tensorflow as tf

a = tf.placeholder("float")
b=tf.placeholder("float")

y=tf.mul(a,b)

#session 생성
sess= tf.Session()

#실제실행
print(sess.run(y,feed_dict={a:3,b:3}))