import tensorflow as tf
import numpy as np

num_puntos = 2000
conjunto_puntos=[]

#Make sample dataset
for i in range(num_puntos):
    if np.random.random()>0.5:
        conjunto_puntos.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Show sample dataset
df = pd.DataFrame({"x" : [v[0] for v in conjunto_puntos],
                   "y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x","y",data=df, fit_reg=False,size=6)
plt.show()


#Sample data => Tensor
vectors = tf.constant(conjunto_puntos)

#무작위의 k개 센트로이드 선택
k=4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))


expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in xrange(k)])

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(101):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])


data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()