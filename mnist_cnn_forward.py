import tensorflow as tf 

# INPUT_NODE = 784 # 28x28 = 784.
# OUTPUT_NODE = 10 # 0~9 => 10 numbers.

KERNEL_SIZE = 5
IMAGE_SIZE = 28

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)


def conv2d(x, W):
	# stride [1, x_movement,y_movement,1]
	# Must have strides[0] = strides[3]=1
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	# Must have strides[0] = strides[3]=1
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



# build the neural network
def forward(xs, keep_prob): 
	x_image = tf.reshape(xs, [-1,28,28,1])

	## conv1 layer ##
	W_conv1 = weight_variable([KERNEL_SIZE,KERNEL_SIZE,1,32]) # patch/kernel 5x5, in size 1, out size 32
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
	h_pool1 = max_pool_2x2(h_conv1) 					     # output size 14x14x32

	## conv2 layer ##
	W_conv2 = weight_variable([KERNEL_SIZE,KERNEL_SIZE,32,64]) # patch/kernel 5x5, in size 32, out size 64
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
	h_pool2 = max_pool_2x2(h_conv2) 					     # output size 7x7x64


	## func1 layer ##
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])
	# [n_samples, 7, 7, 64] ->> [n_samples, 7, 7, 64]
	h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	## func2 layer ##
	W_fc2 = weight_variable([1024,10])
	b_fc2 = bias_variable([10])
	prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	return prediction