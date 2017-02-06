import tensorflow as tf 
import numpy as np 

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))
y  = Weight * x_data + bias

# Losses
loss = tf.reduce_mean(tf.square(y  - y_data))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	for step in xrange(101):
		sess.run(train)
		if step % 20 == 0:
			print "Step number: {} , Weight : {} , bias : {}".format(step,sess.run(Weight),sess.run(bias))
