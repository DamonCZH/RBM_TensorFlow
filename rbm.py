import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tfrbm import BBRBM

mnist = input_data.read_data_sets("D:/MNIST/MNIST_data",one_hot=True)

learning_rate = 0.01

x = tf.placeholder(shape=[None,784],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax( tf.matmul(x,W) + b )

loss = -tf.reduce_sum(y*tf.log(pred))

update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    values = {
        x:mnist.train.images,
        y:mnist.train.labels
    }
    sess.run(update,feed_dict=values)