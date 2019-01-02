图包含了节点与边。边可以看做是流动着的数据以及相关依赖关系。而节点表示了一种操作，即对当前流动来的数据的运算。

需要说明的是，诸如`tf.add`这样的操作，我们可以看做是一个节点，然而其返回值往往并不代表这个节点，而只代表这个节点输出的张量。当我们使用如`c = tf.add(a, b)`的代码时需要知道`c`一般代表的是张量，而不是操作（也有极少数例外），当然也有一些方法可以获得操作，后面的教程会加以说明。

## 1. 边（edge）

Tensorflow的边有两种连接关系：**数据依赖**和**控制依赖**。其中，实线边表示数据依赖，代表数据，即张量。虚线边表示控制依赖（control dependency），可用于控制操作的运行，这被用来确保happens-before关系，这类边上没有数据流过，但源节点必须在目的节点开始执行前完成执行。

### 1.1 数据依赖

数据依赖很容易理解，某个节点会依赖于其它节点的数据，如下所示，矩阵乘法操作这个节点依赖于`a、b`的数据才能执行：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)
```

当节点关系比较复杂时，如下：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)
d = tf.add(c, 5)
```

此时如果要执行并得到`d`中的数值，则只需在会话中执行`d`即可，TensorFlow会根据数据的依赖关系执行得到`c`的操作。

### 1.2 控制依赖

控制依赖是在某些操作之间没有数值上的依赖关系但执行时又需要使这些操作按照一定顺序执行，这时候我们可以声明执行顺序。这在TensorFlow包含变量相关操作时非常常用。

控制依赖使用**图对象的方法**`tf.Graph.control_dependencies(control_inputs)`，返回一个上下文管理器对象，用法如下：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)

g = tf.get_default_graph()
with g.control_dependencies([c]):
    d = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='d')
    e = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='e')
    f = tf.matmul(d, e)
with tf.Session() as sess:
    sess.run(f)
```

上面的例子中，我们在会话中执行了`f`这个节点，可以看到其与`c`这个节点并无任何数据依赖关系，然而`f`这个节点必须等待`c`这个节点执行完成才能够执行`f`。最终的结果是`c`先执行，`f`再执行。

**注意**：`control_dependencies`方法传入的是一个列表作为参数，列表中包含所有被依赖的操作或张量，被依赖的所有节点可以看做是同时执行的。

控制依赖除了上面的写法以外还拥有简便的写法（推荐使用）：`tf.control_dependencies(control_inputs)`。其调用默认图的`tf.Graph.control_dependencies(control_inputs)`方法。上面的写法等价于：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)

with tf.control_dependencies([c]):
    d = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='d')
    e = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='e')
    f = tf.matmul(d, e)
with tf.Session() as sess:
    sess.run(f)
```

**注意**：有依赖的op必须写在`tf.control_dependencies`上下文中，否则不属于有依赖的op。**如下写法是错误的**：

```python
def my_fun():
    a = tf.constant(1)
    b = tf.constant(2)
    c = a + b

    d = tf.constant(3)
    e = tf.constant(4)
    f = d + e
    # 此处 f 不依赖于 c
	with tf.control_dependencies([c]):
        return f
    
result = my_fun()
```

### 1.3 张量的阶、形状、数据类型

Tensorflow数据流图中的边用于数据传输时，数据是以张量的形式传递的。张量有阶、形状和数据类型等属性。

#### Tensor的阶

在TensorFlow系统中，张量的维数被描述为**阶**。但是张量的阶和矩阵的阶并不是同一个概念。张量的阶是张量维数的一个数量描述。比如，下面的张量（使用Python中list定义的）就是2阶.

```python
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

你可以认为一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.对于一个二阶张量你可以用语句`t[i, j]`来访问其中的任何元素.而对于三阶张量你可以用`t[i, j, k]`来访问其中的任何元素。

| 阶   | 数学实例                | Python 例子                                                  |
| ---- | ----------------------- | ------------------------------------------------------------ |
| 0    | 纯量 (或标量。只有大小) | `s = 483`                                                    |
| 1    | 向量(大小和方向)        | `v = [1.1, 2.2, 3.3]`                                        |
| 2    | 矩阵(数据表)            | `m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`                      |
| 3    | 3阶张量 (数据立体)      | `t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]` |
| n    | n阶                     | `....`                                                       |

#### Tensor的形状

TensorFlow文档中使用了三种记号来方便地描述张量的维度：阶，形状以及维数.下表展示了他们之间的关系：

| 阶   | 形状             | 维数 | 实例                                |
| ---- | ---------------- | ---- | ----------------------------------- |
| 0    | [ ]              | 0-D  | 一个 0维张量. 一个纯量.             |
| 1    | [D0]             | 1-D  | 一个1维张量的形式[5].               |
| 2    | [D0, D1]         | 2-D  | 一个2维张量的形式[3, 4].            |
| 3    | [D0, D1, D2]     | 3-D  | 一个3维张量的形式 [1, 4, 3].        |
| n    | [D0, D1, … Dn-1] | n-D  | 一个n维张量的形式 [D0, D1, … Dn-1]. |

张量的阶可以使用`tf.rank()`获取到：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tf.rank(a)  # <tf.Tensor 'Rank:0' shape=() dtype=int32> => 2
```

张量的形状可以通过Python中的列表或元祖（list或tuples）来表示，或者也可用`TensorShape`对象来表示。如下：

```python
# 指定shape是[2, 3]的常量,这里使用了list指定了shape，也可以使用ndarray和TensorShape对象来指定shape
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=[2, 3])

# 获取shape 方法一：利用tensor的shape属性
a.shape  # TensorShape([Dimension(2), Dimension(3)])

# 获取shape 方法二：利用Tensor的方法get_shape()
a.get_shape()  # TensorShape([Dimension(2), Dimension(3)])

# 获取shape 方法三：利用tf.shape()，返回一个Tensor对象
tf.shape(a) # <tf.Tensor 'Shape:0' shape=(2,) dtype=int32>
```

注意：在动态图中要是有方法三获取`shape`。

`TensorShape`对象有一个方法`as_list()`，可以将`TensorShape`对象转化为python的list对象。

```python
a.get_shape().as_list() # [2, 3]
```

同样的我们也可以使用list构建一个TensorShape的对象：

```python
ts = tf.TensorShape([2, 3])
```

------

**小练习**：

1. 写出如下张量的阶与形状，并使用TensorFlow编程验证：

- `[]`
- `12`
- `[[1], [2], [3]]`
- `[[1, 2, 3], [1, 2, 3]]`
- `[[[1]], [[2]], [[3]]]`
- `[[[1, 2], [1, 2]]]`

1. 设计一个函数，要求实现：可以根据输入张量输出shape完成一样的元素为全1的张量。提示：使用`tf.ones`函数可根据形状生成全1张量。

小技巧：使用`tf.ones_like`、`tf.zeros_like`可以快速创建与输入张量形状一样的全1或全0张量。

#### Tensor的数据类型

每个Tensor均有一个数据类型属性，用来描述其数据类型。合法的数据类型包括：

| 数据类型        | Python 类型     | 描述                                                        |
| --------------- | --------------- | ----------------------------------------------------------- |
| `DT_FLOAT`      | `tf.float32`    | 32 位浮点数.                                                |
| `DT_DOUBLE`     | `tf.float64`    | 64 位浮点数.                                                |
| `DT_INT64`      | `tf.int64`      | 64 位有符号整型.                                            |
| `DT_INT32`      | `tf.int32`      | 32 位有符号整型.                                            |
| `DT_INT16`      | `tf.int16`      | 16 位有符号整型.                                            |
| `DT_INT8`       | `tf.int8`       | 8 位有符号整型.(此处符号位不算在数值位当中)                 |
| `DT_UINT8`      | `tf.uint8`      | 8 位无符号整型.                                             |
| `DT_STRING`     | `tf.string`     | 可变长度的字节数组.每一个张量元素都是一个字节数组.          |
| `DT_BOOL`       | `tf.bool`       | 布尔型.(不能使用number类型表示bool类型，但可转换为bool类型) |
| `DT_COMPLEX64`  | `tf.complex64`  | 由两个32位浮点数组成的复数:实部和虚部。                     |
| `DT_COMPLEX128` | `tf.complex128` | 由两个64位浮点数组成的复数:实部和虚部。                     |
| `DT_QINT32`     | `tf.qint32`     | 用于量化Ops的32位有符号整型.                                |
| `DT_QINT8`      | `tf.qint8`      | 用于量化Ops的8位有符号整型.                                 |
| `DT_QUINT8`     | `tf.quint8`     | 用于量化Ops的8位无符号整型.                                 |

Tensor的数据类型类似于Numpy中的数据类型，但其加入了对string的支持。

##### 设置与获取Tensor的数据类型

设置Tensor的数据类型： 

```python
# 方法一
# Tensorflow会推断出类型为tf.float32
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 方法二
# 手动设置
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)

# 方法三 (不推荐)
# 设置numpy类型 未来可能会不兼容 
# tf.int32 == np.int32  -> True
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
```

获取Tensor的数据类型，可以使用如下方法：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
a.dtype  # tf.float32
print(a.dtype)  # >> <dtype: 'float32'>

b = tf.constant(2+3j)  # tf.complex128 等价于 tf.complex(2., 3.)
print(b.dtype)  # >> <dtype: 'complex128'>

c = tf.constant([True, False], tf.bool)
print(c.dtype)  # <dtype: 'bool'>
```

这里需要注意的是一个张量仅允许一种dtype存在，也就是一个张量中每一个数据的数据类型必须一致。

##### 数据类型转化

如果我们需要将一种数据类型转化为另一种数据类型，需要使用`tf.cast()`进行：

```python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
# tf.cast(x, dtype, name=None) 通常用来在两种数值类型之间互转
b = tf.cast(a, tf.int16)
print(b.dtype)  # >> <dtype: 'int16'>
```

有些类型利用`tf.cast()`是无法互转的，比如string无法转化成为number类型，这时候可以使用以下方法：

```python
# 将string转化为number类型 注意：数字字符可以转化为数字
# tf.string_to_number(string_tensor, out_type = None, name = None)
a = tf.constant([['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0']], name='a')
num = tf.string_to_number(a)
```

实数数值类型可以使用`tf.cast`方法转化为bool类型。

------

**小练习**

判断以下哪个张量是合法的：

- `tf.constant([1, 2, 3], dtype=tf.float64)`
- `tf.constant([1, 2, 3], dtype=tf.complex64)`
- `tf.constant([1, 2, 3], dtype=tf.string)`
- `tf.consant([1, '2', '3'])`
- `tf.constant([1, [2, 3]])`

## 2. 节点

图中的节点也可以称之为**算子**，它代表一个操作(operation, OP)，一般用来表示数学运算，也可以表示数据输入（feed in）的起点以及输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。常见的节点包括以下几种类型：变量、张量逐元素运算、张量变形、张量索引与切片、张量运算、检查点操作、队列和同步操作、张量控制等。

当OP表示数学运算时，每一个运算都会创建一个`tf.Operation`对象。常见的操作，例如生成一个变量或者常量、数值计算均创建`tf.Operation`对象

### 2.1 变量

变量用于存储张量，可以使用list、Tensor等来进行初始化，例如：

```python
# 使用纯量0进行初始化一个变量
var = tf.Variable(0)
```

### 2.2 张量元素运算

常见张量元素运算有很多种，比如张量对应元素的相加、相乘等，这里我们介绍以下几种运算：

- `tf.add()` 两个张量对应元素相加。等价于`A + B`。

  ```python
  tf.add(1, 2)  # 3
  tf.add([1, 2], [3, 4])  # [4, 6]
  tf.constant([1, 2]) + tf.constant([3, 4])  # [4, 6]
  ```

- `tf.subtract()` 两个张量对应元素相减。等价于`A - B`。

  ```python
  tf.subtract(1, 2)  # -1
  tf.subtract([1, 2], [3, 4])  # [-2, -2]
  tf.constant([1, 2]) - tf.constant([3, 4])  # [-2, -2]
  ```

- `tf.multiply()` 两个张量对应元素相乘。等价于`A * B`。

  ```python
  tf.multiply(1, 2)  # 2
  tf.multiply([1, 2], [3, 4])  # [3, 8]
  tf.constant([1, 2]) * tf.constant([3, 4])  # [3, 8]
  ```

- `tf.scalar_mul() `一个纯量分别与张量中每一个元素相乘。等价于 `a * B`

  ```python
  sess.run(tf.scalar_mul(10., tf.constant([1., 2.])))  # [10., 20.]
  ```

- `tf.divide()` 两个张量对应元素相除。等价于`A / B`。这个除法操作是Tensorflow推荐使用的方法。此方法不接受Python自身的数据结构，例如常量或list等。

  ```python
  tf.divide(1, 2)  # 0.5
  tf.divide(tf.constant([1, 2]), tf.constant([3, 4]))  # [0.33333333, 0.5]
  tf.constant([1, 2]) / tf.constant([3, 4])  # [0.33333333, 0.5]
  ```

- `tf.div()`两个张量对应元素相除，得到的结果。不推荐使用此方法。`tf.divide`与`tf.div`相比，`tf.divide`符合Python的语义。例如：

  ```python
  1/2  # 0.5
  tf.divide(tf.constant(1), tf.constant(2))  # 0.5
  tf.div(1/2)  # 0
  ```

- `tf.floordiv()` shape相同的两个张量对应元素相除取整数部分。等价于`A // B`。

  ```python
  tf.floordiv(1, 2)  # 0
  tf.floordiv([4, 3], [2, 5])  # [2, 0]
  tf.constant([4, 3]) // tf.constant([2, 5])  # [2, 0]
  ```

- `tf.mod()` shape相同的两个张量对应元素进行模运算。等价于`A % B`。

  ```python
  tf.mod([4, 3], [2, 5])  # [0, 3]
  tf.constant([4, 3]) % tf.constant([2, 5])  # [0, 3]
  ```

上述运算也支持满足一定条件的`shape`不同的两个张量进行运算，即广播运算。与`numpy`类似在此不做过多演示。

除此以外还有很多的逐元素操作的函数，例如求平方`tf.square()`、开平方`tf.sqrt`、指数运算、三角函数运算、对数运算等等。

### 2.3 张量常用运算

`tf.matmul()` 通常用来做矩阵乘法。

`tf.transpose()` 转置张量。

```python
a = tf.constant([[1., 2., 3.], [4., 5., 6.0]])
# tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
# tf.transpose(a, perm=None, name='transpose')
tf.matmul(a, tf.transpose(a))  # 等价于 tf.matmul(a, a, transpose_b=True)
```

更多线性代数操作在`tf.linalg`模块下。

### 2.4 张量切片与索引

张量变形：

```python
# 将张量变为指定shape的新张量
# tf.reshape(tensor, shape, name=None)
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
new_t = tf.reshape(t, [3, 3]) 
# new_t	==> [[1, 2, 3],
#            [4, 5, 6],
#            [7, 8, 9]]
new_t = tf.reshape(new_t, [-1]) # 这里需要注意shape是一阶张量，此处不能直接使用 -1
# tensor 'new_t' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

张量的拼接：

```python
# 沿着某个维度对二个或多个张量进行连接
# tf.concat(values, axis, name='concat')
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
```

张量的切割：

```python
# 对输入的张量进行切片
# tf.slice(input_, begin, size, name=None)
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                            [4, 4, 4]]]
tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                           [[5, 5, 5]]]

# 将张量分裂成子张量
# tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]
```

------

**小练习**：

已知一张28\*28像素图片存储在一阶张量中（从左到右、从上到下逐行展开的），请问：

1. 一阶张量的形状是多少？
2. 将一阶张量还原回图片格式的二阶张量。
3. 取图片张量中第5-10行、第3列（索引坐标从0开始）的数据。

## 作业

有一4阶张量`img`其`shape=[10, 28, 28, 3])`，代表10张28*28像素的3通道RGB图像，问：

1. 如何利用索引取出第2张图片？（注意：索引均从0开始，第二张则索引为1，下同）
2. 如何利用切片取出第2张图片？
3. 使用切片与使用索引取出的一张图片有何不同？
4. 如何取出其中的第1、3、5、7张图片？
5. 如何取出第6-8张（包括6不包括8）图片中中心区域（14*14）的部分？
6. 如何将图片根据通道拆分成三份单通道图片？
7. 写出`tf.shape(img)`返回的张量的阶数以及`shape`属性的值。

