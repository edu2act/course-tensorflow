"""
本例子中写了两个图。
g1 描述了两个矩阵的乘法，其中乘法的部分在GPU上执行。
g2 描述了两个子图构成的一个图
"""

import tensorflow as tf


# 获取tensorflow中的默认图
g1 = tf.get_default_graph()
with g1.as_default():
    with tf.device('/cpu:0'):
        a = tf.constant([[1, 2, 3]])
        a = a + a
        b = tf.constant([[3], [2], [1]])
        b = b + b
    with tf.device('/gpu:0'):
        c = tf.matmul(a, b)

config = tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=True)
with tf.Session(graph=g1, config=config) as sess:
    print(sess.run([c]))
    print(sess.run([c], feed_dict={a: [[1, 1, 1]]}))
    

g2 = tf.Graph()
with g2.as_default():
    x = tf.constant(1)
    x = tf.add(x, tf.constant(2))
    x = tf.multiply(x, tf.constant(5))

    y1 = tf.constant(3)
    y2 = tf.constant(1)
    y = y1 + y2
    y = x % y
    y = x // y

with tf.Session(graph=g2) as sess:
    print(sess.run(y))

