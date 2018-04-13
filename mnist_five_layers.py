import tensorflow as tf 
from   tensorflow.examples.tutorials.mnist import input_data
import layer


mnist = input_data.read_data_sets('D:/MNIST/MNIST_data/',one_hot=True)

N,h1,h2,h3,h4,D = 784,200,100,60,30,10
learning_date = 0.004

x = tf.placeholder(tf.float32,[None,N],name='x_input')
y = tf.placeholder(tf.float32,[None,D],name='y_input')

hidden_layer_1 = layer.nn_layer(x,N,h1,'Layer_1',tf.nn.relu)
hidden_layer_2 = layer.nn_layer(hidden_layer_1,h1,h2,'Layer_2',tf.nn.relu)
hidden_layer_3 = layer.nn_layer(hidden_layer_2,h2,h3,'Layer_3',tf.nn.relu)
hidden_layer_4 = layer.nn_layer(hidden_layer_3,h3,h4,'Layer_4',tf.nn.relu)

prediction = layer.nn_layer(hidden_layer_4,h4,D,'Layer_5',tf.nn.softmax)
loss = -tf.reduce_sum(y*tf.log(prediction))
update = tf.train.AdamOptimizer(learning_date).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(50)
		values = {x:batch_xs,y:batch_ys}
		sess.run(update,feed_dict=values)

	print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))