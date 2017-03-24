# Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

"""
Tensorflow는 고효율의 C++ 백앤드의 계산에 의존한다.
백앤드와의 연결을 Session이라고 한다.
Tensorflow의 프로그램들은 주로 그래프를 정의하고 session을  실행하는 것으로 시작한다.
InteractiveSession은 계산 그래프를 1번 실행하면서 인터리브연산을 가능하게 해준다.
이것은 IPython과 같이 interactive context에서 작업할 때 유용하다.
만약 InteractiveSession을 사용하지 않는다면, 전체 계산그래프를 시작하기 전에 모두 작성해야한다.
"""
import tensorflow as tf
sess =tf.InteractiveSession()


"""
Computation Graph
파이썬에서 효율적인 산술연산을 하기위해서, 주로 ,NumPy같은 고효율의 다른 언어의 라이브러리를 사용한다.(매트릭스 곱..)
불행히도, 모든 연산은 여전히 파이썬의 연산으로 넘어오는데 오버헤드가 많다.(Ex C++ -> Python)
특히, 이 오버헤드는, GPU를 사용하거나, 데이터를 전송하는데 high cost가 드는 분산 환경에서 좋지않다.
따라서 Tensorflow는 하나의 연산을 파이썬에서 독립적으로 실행 시키지 않고, 그래프를 정의하고 전부를 Python 밖에서 실행한다.
이러한 접근은 Theano 나 Torch와 유사하다.
따라서 파이썬코드의 역할은 외부의 computation graph를 만드는 것이고, 어떤 graph가 실행되야 하는지 지정하는 것이다.
"""

"""
Build a Softmax Regression Model
이번 장에서 softmax regression model(단일 선형레이어)를 만들고, 다음 섹션에서 CNN으로 확장할 것이다.
"""
"""
Placeholders
우리는 input image와 output classes를 위한 노드를 만듬으로써 계산 그래프를 만들어 볼것이다.
"""
x = tf.placeholder(tf.float32, shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 각각의 값들은 실제 tensorflow의 계산이 시작되어 질 때 입력이 될 것이다.
"""
x : input image
float32 타입의 2차원 텐서로 이루어져 있다.
None = 1번째 차원의 수 : batch size와 동일, 어떤 size라도 가능하다.
784 =  2번째 차원의 수 : 28*28 pixel MNIST image
"""
"""
y_ : targer output classes
float32 타입의 2차원 텐서로 이루어져 있다.
10개 종류의 one-hot 인코딩 형태의 값을 출력할 것이다.(0~9)
"""

"""
Variables
우리는 모델의 가중치 W와 편향 b를 정의할 것이다.
Variable은 tensorflow의 계산그래의 안에 있는 값들이다.
이것들은 tensorflow의 연산으로 사용되거나, 수정되어진다.
머신러닝 어플리케이션에서는, 일반적으로 모델의 파라미터들이 Variable이 된다.
"""

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

"""
W : 784*10 Matrix(784개의 input features 와 10개의 출력)
b : 10 vector (10개의 classes)
Variable들이 session에서 사용되기 전에 반드시 session을 통해서 변수를 초기화 시켜줘야 한다.
"""
sess.run(tf.global_variables_initializer())

"""
Predicted Class and Loss Function
이제 우리는 우리의 회귀 모델을 구현할 수 있다.
또한, 간단하게 loss funciton을 설정 할 수 있다.
loss function 은 얼마나 모델이 나쁜 예측을 하였는지를 알려준다.
우리는 이 loss function을 최소화 하는 것이 목적이다.
"""
y= tf.matmul(x,W)+b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

"""
Train the Model
텐서플로우는 다양한 built-in 최적화 알고리즘이 있다.
예를들어, 우리는 gradient descent를 사용해서 cross entropy를 줄여볼 것이다.
"""

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]}) #feed dict를 통해 Placeholder에 값을 전달

"""
Evaluate the Model
우리의 모델이 얼마나 잘하였는가?
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


"""
Build a Multilayer Convolutional Network
 MNIST의 92%의 정확도는 좋지 않다.
 우리는 이것을 고쳐서, 작은 Convolutional neural network를 만들것이다.
 이것은 99.2%의 결과를 보일 것이다.
"""

"""
Weight Initialization
모델을 만들기 위해서, 우리는 다수의 가중치와 편향을 만들고 초기화 하여야 한다.
gradient가 0이되는 것을 막기 위해서 우리는 랜덤하게 작은 noise로 초기화를 해야한다.
간단한 함수르 만들어서 변수를 쉽게 만들어 보자.
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


print ("합성곱")
"""
Convolution and Pooling
또한 TensorFlow는 convolution과 pooling 연산같은 연산을 제공해준다.
"""

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

"""
First Convolutional Layer
첫번째 층을 만들어 보자.
max pooling을 사용하는 합성곱으로 이루어져 있다.
"""
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

"""
W_conv1 : 5 X 5 : Patch size
          1 : 채널 수
          32 : output channels
b_conv1 : 32 : output channels
"""

"""
x를 4차원 텐서로 바꾸어서, 이미지 모양으로 사용할 수 있게 해야한다.
"""
x_image = tf.reshape(x,[-1,28,28,1])

"""
Conv -> RELU -> max_pool
"""

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
deep network를 만들기 위해서,
2번째 층에서는 5x5 사이즈의 patchf르 64개의 features로 확장한다.
"""
W_conv2 = weight_variable([5, 5, 32, 64]) # 5x5 사이즈, 32채널의 필터를 64개
b_conv2 = bias_variable([64])             # 64개의 아웃풋 : 64개의 편향

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
Densely Connected Layer
이미지의 사이즈가 7x7로 줄어들었을 것이다, 완전연결 레이어를 연결해서
1024 뉴런들이 전체 이미지를 연산 할 수 있도록 해야 한다.
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""
Dropout
오버피팅을 억제하기 위해서, 우리는 레이어를 읽기 전에 드랍아웃을 사용해야한다.
"""

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
Readout Layer
"""
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2)+ b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ("초기화")
sess.run(tf.global_variables_initializer())

print ("학습")
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))