import tensorflow as tf 


g = tf.get_default_graph()
with g.as_default():
    a = tf.constant([[1.0, 2.0, 3.0]], name='a')
    b = tf.constant([[3.0], [4.0], [5.0]], name='b')
    c = tf.matmul(a, b)
    # tf.Print可以查看数据流过c时对应的a、b、c的值
    c = tf.Print(c, [a, b, c], 'a、b、c is :')
    c = tf.add(c, tf.constant(5.0))
    c = tf.Print(c, [a, b, c], 'a、b、c is :')
    with g.control_dependencies([c]):
        no_op = tf.no_op()

with tf.Session(graph=g) as sess:
    sess.run(no_op) 