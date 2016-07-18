import tensorflow as tf

#tf graph input
X=[1.,2.,3.]
Y=[1.,2.,3.]
m=n_samples=len(X)

#Set model wights
W = tf.placeholder(tf.float32)

#Construct a linear model
hypothesis = tf.mul(X,W)

#Cost Function
cost = tf.reduce_sum(tf.pow(hypothesis-Y,2))/(m)

#Initializing the variables
init = tf.initialize_all_variables()

#For graphs
W_val = []
cost_val = []

#Launch the graph
sess = tf.Session()
sess.run(init)
# -3.0에서 5.0 까지 스탭
for i in range(-30,50):
    print (i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost,feed_dict={W: i*0.1}))

#Graphic display
plt.plot(W_val,cost_val,'ro')
plt.ylabel('Cost')
plt.xlable('W')
plt.show()