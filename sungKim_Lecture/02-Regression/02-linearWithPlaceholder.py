import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#X Y 는 실행 시킬때 값이 정해진다.
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis = W*X+b
cost = tf.reduce_mean(tf.square(hypothesis-Y))

a=tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in range(2001):
    #실행 시점에서 값이 들어간다.
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%20 ==0:
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

#Learn best fit is W: {1}, b: {0}
#학습을 통해서 최적의 값을 찾았다.
#즉 모델을 다시 작성할 필요가 없이 우리가 학습한 것에 값을 넣어서 확인할 수 있다.
print (sess.run(hypothesis, feed_dict={X:5}))
print (sess.run(hypothesis, feed_dict={X:2.5}))