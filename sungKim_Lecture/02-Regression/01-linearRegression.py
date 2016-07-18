import tensorflow as tf

#트레이닝 데이터
x_data=[1,2,3]
y_data=[1,2,3]

#Try to find values for W and b that compute y_data = W*x_data +b
#(We know that W should be 1 and b 0, but Tensorflow will figure that out for us)

#변수로 지정해야지 업데이트가 가능하기 때문에 반드시 변수로 지정
#0~1 사이의 랜덤한 값으로 시작
W= tf.Variable(tf.random_uniform([1],-1.0,1.0))
b= tf.Variable(tf.random_uniform([1],-1.0,1.0))

#Our hypothesis
hypothesis = W * x_data + b

#Simplified cost functin
#우리의 가설모델과 실제값의 차이의 제곱의 평균 - 즉 코스트 펑션
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
#그라디언트 디센트를 통해 미니마이즈 한다.
a=tf.Variable(0.05) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables. We will 'run' this first
#변수를 초기화 , 초기화를 안시키면 에러가 난다.
init = tf.initialize_all_variables()

#Launch the graph
sess=tf.Session()
sess.run(init)

#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 ==0:
        print (step,sess.run(cost),sess.run(W),sess.run(b))