import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

# 3.0, 4.0이 아닌 Tensor의 정보가 출력된다.
print (node1, node2)

sess = tf.Session()
print(sess.run([node1,node2]))

node3 = tf.add(node1,node2)
print("node3: ",node3)
print("sess.run(node3) : ", sess.run(node3))

# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

# Variable
W= tf.Variable([.3], tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b

# Variable은 init이 꼭 필요함 -> 호출시에 초기화가 안되기 떄문에
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model,{x:[1,2,3,4]}))

y= tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
print (sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

