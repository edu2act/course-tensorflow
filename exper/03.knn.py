import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

k = 7
test_num = 5000

train_images, train_labels = mnist.train.next_batch(55000)
test_images, test_labels = mnist.test.next_batch(test_num)

x_train = tf.placeholder(tf.float32)
x_test = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

# 欧式距离
euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(x_train - x_test), 1))
# 计算最相近的k个样本的索引
_, nearest_index = tf.nn.top_k(-euclidean_distance, k)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    predicted_num = 0
    # 对每个图片进行预测
    for i in range(test_images.shape[0]):
        # 最近k个样本的标记索引
        nearest_index_res = sess.run(
            nearest_index, 
            feed_dict={
                x_train: train_images,
                y_train: train_labels,
                x_test: test_images[i]})
        # 最近k个样本的标记
        nearest_label = []
        for j in range(k):
            nearest_label.append(list(train_labels[nearest_index_res[j]]))

        predicted_class = sess.run(tf.argmax(tf.reduce_sum(nearest_label, 0), 0))
        true_class = sess.run(tf.argmax(test_labels[i]))
        if predicted_class == true_class:
            predicted_num += 1
        
        if i % 100 == 0:
            print('step is %d accuracy is %.4f' % (i, predicted_num / (i+1)))
    print('accuracy is %.4f' % predicted_num / test_num)