import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tfrbm import BBRBM
import layer

mnist = input_data.read_data_sets("D:/MNIST/MNIST_data",one_hot=True)

learning_rate = 0.01

x = tf.placeholder(shape=[None,784],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

r = layer.train_rbm(784,200)
hidden_layer = layer.rbm_layer(x,r,"hidden_layer")
output_layer = layer.nn_layer(hidden_layer,200,10,"output_layer",tf.nn.softmax)

pred = output_layer
loss = -tf.reduce_sum(y*tf.log(pred))
update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(50)
        train_values = {
            x:batch_xs,
            y:batch_ys
        }
        sess.run(update,feed_dict=train_values)

    test_values = {
        x:mnist.test.images,
        y:mnist.test.labels
    }
    print(sess.run(accuracy,feed_dict=test_values))
