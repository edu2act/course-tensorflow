"""
本例用以强化张量阶、形状、维度等概念
以及掌握张量切片的方法
"""
import tensorflow as tf 


# img代表用1填充的shape为[10, 28, 28, 3]的张量
# 可以理解为10张大小为28*28像素的jpg图片
img = tf.ones(shape=[10, 28, 28, 3])


# img的阶是多少？
img_rank = tf.rank(img)


# tensor切片之后的shape是多少？
print(img[:].shape)
print(img[5:].shape)  # 从第5开始包括第5个到末尾
print(img[:6].shape)  # 从第0开始到第6个，不靠看第6个
print(img[0:1].shape)
print(img[5:10:2].shape)  # 步长为2


# tensor索引
print(img[0].shape)
print(img[0, 1].shape)
print(img[0, 1, 7].shape)

# 切片与索引混用
print(img[5, 3:7:2].shape)
print(img[:, 2].shape)  # 第一个维度上不变，第二个维度上选取第2个元素
print(img[:, 2, 5:10].shape)


# 再次索引与切片
tmp = img[5]
tmp = tmp[3]
assert img[5][3].shape == tmp.shape

print(img[:, 3][5])


# 使用tf.slice进行切片也很方便
res1 = tf.slice(img, [1, 0, 0, 0], [1, 28, 28, 1])

res2 = tf.slice(img, [5, 0, 0, 0], [1, 28, 28, 1])

# 连接的时候连接的维度可以有不同的长度，但其它维度必须相同
res = tf.concat([res1, res2], 0)
res = tf.concat([res1, res2], 1)
res = tf.concat([res1, res2], 2)
res = tf.concat([res1, res2], 3)