import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tfrbm import BBRBM

mnist = input_data.read_data_sets("D:/MNIST/MNIST_data",one_hot=True)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,activation):
	W = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
	b = tf.Variable(tf.constant(0.1,shape=[output_dim]))
	output = activation(tf.add(tf.matmul(input_tensor,W),b))
	return output

def rbm_layer(input_tensor,rbm,layer_name,activation=tf.nn.sigmoid):
    W,a,b = rbm.get_weights()
    output = activation(tf.add(tf.matmul(input_tensor,W),b))
    return output


def train_rbm(n_vis, n_hid,learning_rate=0.01,momentum=0.95,n_epoches=30,batch_size=100):
    rbm = BBRBM(n_vis, n_hid, learning_rate=learning_rate, momentum=momentum, use_tqdm=True)
    rbm.fit(mnist.train.images, n_epoches=n_epoches, batch_size=batch_size)
    return rbm
