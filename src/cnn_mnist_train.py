# coding: utf-8

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取图片
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 50

# 计算训练集中有多少个批次 
n_batch = mnist.train.num_examples // batch_size

max_steps = 50000

MODEL_SAVE_PATH = './model/'
MODEL_NAME='cnn_mnist_model' 


fc_node = 1024 

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置
def bias_variablle(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x,W):
    # x是一个4D的tensor: [batch, in_height, in_width, in_channels]
    # W是卷积核的属性: [filter_height, filter_width, in_channels, out_channels]
    # strides: 必须要第一个和最后一个相同 `strides[0] = strides[3] = 1`. 
    # 大多数情况下，水平和垂直方向的strides取一样的值，结构就像下面这样
    # `strides = [1, stride, stride, 1]`.
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 池化层
def max_pool_2x2(x):
    # 核的形状：[ 1,x,y,1] 第一个 和 最后一个元素为1， x,y为核的大小
    # strides: 必须要第一个和最后一个相同 `strides[0] = strides[3] = 1`. 
    # 大多数情况下，水平和垂直方向的strides取一样的值，结构就像下面这样
    # `strides = [1, stride, stride, 1]`.
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None, 784]) # 第一个数字代表行，784代表有784列
y = tf.placeholder(tf.float32,[None, 10])  # 输出：标签

# 改变x为4D: [batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1,28,28,1]) # -1：现在不关心，后续会变成100

# 初始化第一个卷积层
# 卷积核的形态：5*5*1
# 卷积核的个数：32 (这个数字是可以尝试出来的是吧？)
# 用32个卷积核去对一个平面/通道采样，最后会得到32个卷积特征平面
W_conv1 = weight_variable([5,5,1,32]) 
b_conv1 = bias_variablle([32]) 

# 把x_image和卷积向量进行卷积，再加上偏置，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积层
# 卷积核的形态：5*5*32 
# 卷积核的个数：64
# 使用64个卷积核对32个平面提取特征；得到64x32个特征平面 (他说是64) ??
# 回答：的确是64个，卷的时候是考虑了深度的，卷积核在这里考虑成一个cube(立方体)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variablle([64]) # 一个卷积核要一个偏置值

# 把h_pool1和卷积向量进行卷积，再加上偏置，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28x28的图片第一次卷积后还是28x28(same padding)，第一次池化后变14x14(池化窗口2x2)
# 第二次卷积后为14x14,第二次池化后变为7x7
# 通过上面的操作得到64张7x7的平面

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,fc_node]) # 上一层有7x7x64个神经元，定义全连接层有1024个神经元
b_fc1 = bias_variablle([fc_node]) # 1024个节点

# 把池化层的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # 4D->2D
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
# 也就是一次训练中只使用百分之多少的神经元
# 用来避免过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([fc_node,10])
b_fc2 = bias_variablle([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 交叉熵代价函数
cross_enropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# 使用优化器优化
# train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_enropy)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_enropy)

# 结果存放在一个布尔列表中
# argmax返回一维张量中最大值所在位置
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# saver 用来保存训练模型
saver = tf.train.Saver()


# 这种方式有点训练感觉起来有点慢
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(21):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        
#         acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
#         print('Iter ' + str(epoch) + ", accuracy = " + str(acc))
# 这个结果有问题啊才0.2哪里错了(为什么收敛得这么慢)
# Iter 0, accuracy = 0.1015
# Iter 1, accuracy = 0.1021
# Iter 2, accuracy = 0.1031
# Iter 3, accuracy = 0.1043
# Iter 4, accuracy = 0.1053
# Iter 5, accuracy = 0.108
# Iter 6, accuracy = 0.1105
# Iter 7, accuracy = 0.1142
# Iter 8, accuracy = 0.117
# Iter 9, accuracy = 0.1203
# Iter 10, accuracy = 0.1253
# Iter 11, accuracy = 0.1287
# Iter 12, accuracy = 0.132
# Iter 13, accuracy = 0.1384
# Iter 14, accuracy = 0.1467
# Iter 15, accuracy = 0.1542
# Iter 16, accuracy = 0.1626
# Iter 17, accuracy = 0.1721
# Iter 18, accuracy = 0.1808
# Iter 19, accuracy = 0.1921
# Iter 20, accuracy = 0.2



with tf.Session() as sess:   
    sess.run(tf.global_variables_initializer())
    # 断点续训的功能
    # 非常实用啊
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    for i in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.95})
        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            print("Iter " + str(i) + ", Testing Accuracy " + str(acc))
            
            # 保存训练模型
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))

# 下面是这个网络最后的训练结果
# Iter 2180, Testing Accuracy 0.989
# Iter 2190, Testing Accuracy 0.9892
# Iter 2200, Testing Accuracy 0.9896
# Iter 2210, Testing Accuracy 0.9895
# Iter 2220, Testing Accuracy 0.9893



