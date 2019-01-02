TensorFlow用图来完成运算流程的描述。一个图是由OP与Tensor构成，即通过OP生成、消费或改变Tensor。TensorFlow虽然并非是一门编程语言，而是一种符号式编程库，但是因为需要将各种各样的操作符号化，并在C++环境中执行，这时就会出现很多类似于编程语言属性的符号函数与类，事实上TensorFlow包含了常量、变量、数学操作符、流程操作符等看起来很像编程语言属性的内容。

## 1. 常量

常量是一块只读的内存区域，常量在**初始化时就必须赋值**，并且之后值将不能被改变。Python并无内置常量关键字，需要用到时往往需要我们去实现，而Tensorflow内置了常量方法 `tf.constant()`。

### 1.1 普通常量

普通常量使用`tf.constant()`初始化得到，其有5个参数。

```python
constant(
    value, 
    dtype=None, 
    shape=None, 
    name="Const", 
    verify_shape=False):
```

- `value`是必填参数，即常量的初识值。这里需要注意，这个`value`可以是Python中的`list`、`tuple`以及`Numpy`中的`ndarray`对象，但**不可以是Tensor对象**，因为这样没有意义。
- `dtype`可选参数，表示数据类型，`value`中的数据类型应与与`dtype`中的类型一致，如果不填写则会根据`value`中值的类型进行推断。
- `shape` 可选参数，表示value的形状。如果参数`verify_shape=False`，`shape`在与`value`形状不一致时会修改`value`的形状。如果参数`verify_shape=True`，则要求`shape`必须与`value`的`shape`一致。当`shape`不填写时，默认为`value`的`shape`。

**注意**：`tf.constant()`生成的是一个张量。其类型是`tf.Tensor`。

### 1.2 常量存储位置

**常量存储在图的定义当中**，可以将图序列化后进行查看：

```python
const_a = tf.constant([1, 2])
with tf.Session() as sess:
    # tf.Graph.as_graph_def 返回一个代表当前图的序列化的`GraphDef`
    print(sess.graph.as_graph_def()) # 你将能够看到const_a的值
```

当常量包含的数据量较大时，会影响图的加载速度。通常较大的数据使用变量或者在图加载完成之后读取。

### 1.3 序列常量

除了使用`tf.constant()`生成任意常量以外，我们还可以使用一些方法快捷的生成**序列常量**：

```python
# 在指定区间内生成均匀间隔的数字
tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0] 

# 在指定区间内生成均匀间隔的数字 类似于python中的range
tf.range(start, limit=None, delta=1, dtype=None, name='range')
# 例如：'start' is 3, 'limit' is 18, 'delta' is 3
tf.range(start=3, limit=18, delta=3) ==> [3, 6, 9, 12, 15]
# 例如：'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
```

**注意**：`tf.range`中，当传入一个默认参数时，代表的是设置`limit`，此时`start`默认为0。

### 1.4 随机数常量

类似于Python中的`random`模块，Tensorflow也拥有自己的随机数生成方法。可以生成**随机数常量**：

```python
# 生成服从正态分布的随机数
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

# 生成服从截断的正态分布的随机数
# 只保留了两个标准差以内的值，超出的值会被丢掉重新生成
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
name=None)

# 生成服从均匀分布的随机值
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
name=None)

# 将输入张量在第一个维度上进行随机打乱
tf.random_shuffle(value, seed=None, name=None)

# 随机的将张量收缩到给定的尺寸
# 注意：不是打乱，是随机的在某个位置开始裁剪指定大小的样本
# 可以利用样本生成子样本
tf.random_crop(value, size, seed=None, name=None)
```

#### 1.4.1 随机数种子

随机数常量的生成依赖于两个随机数种子，一个是图级别的种子，另一个是操作级别的种子。上述所有操作当中，每个操作均可以接收一个`seed`参数，这个参数可以是任意一个整数，即为操作级种子。图级种子使用`tf.set_random_seed()`进行设置。

**注意**：每个随机数种子可以确定一个随机数序列，而不是一个随机数。

设置随机数种子，可以使得图或其中的一部分操作在不同的会话中出现一样的随机数。具体来讲就是如果设置了某一个操作的随机数种子，则在不同的会话中，这个操作生成的随机数序列是完全一样的；如果设置了图级别的随机数种子，则这个图在不同的会话中所有生成的随机数序列都是完全一样的。

**注意**：如果既设置了图级种子也设置了部分或全部随机数生成操作的种子，那么也会在不同会话中表现一样，只不过最终随机数的种子与默认的不同，其取决于二者。

当不设置随机数种子时，会话与会话之间的随机数生成没有关系，如下代码打印出`a`在不同会话中的结果不同：

```python
with tf.Graph().as_default():
    a = tf.random_uniform([])

    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(a))  

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(a))  
```

当设置了随机数种子时，两个会话中生成的随机数序列是完全一样的，如下：

```python
with tf.Graph().as_default():
    a = tf.random_uniform([], seed=1)

    with tf.Session() as sess1:
        res1 = sess1.run(a)

    with tf.Session() as sess2:
        res2 = sess2.run(a)
    
    print(res1 == res2)  # >>> True
```

使用图级随机数种子，如下：

```python
with tf.Graph().as_default():
    tf.set_random_seed(1)
    a = tf.random_uniform([])
    b = tf.random_normal([])

    with tf.Session() as sess1:
        res1_a, res1_b = sess1.run([a, b])

    with tf.Session() as sess2:
        res2_a, res2_b = sess2.run([a, b])
    
    print(res1_a == res2_a)  # >>> True
    print(res1_b == res2_b)  # >>> True
```

### 1.5 特殊常量

```python
# 生成指定shape的全0张量
tf.zeros(shape, dtype=tf.float32, name=None)
# 生成与输入的tensor相同shape的全0张量
tf.zeros_like(tensor, dtype=None, name=None,optimize=True)
# 生成指定shape的全1张量
tf.ones(shape, dtype=tf.float32, name=None)
# 生成与输入的tensor相同shap的全1张量
tf.ones_like(tensor, dtype=None, name=None, optimize=True)
# 生成一个使用value填充的shape是dims的张量
tf.fill(dims, value, name=None)
```

**小练习**：

> 1. 使用`tf.constant`生成常量时，能够使用其他常量张量初始化吗？为什么？
> 2. 总结各种常量类型的特点与用法，掌握不同种常量的使用场景。

## 2. 变量

变量用于存取张量，在Tensorflow中主要使用类`tf.Variable()`来实例化一个变量对象，作用类似于Python中的变量。

```python
tf.Variable(
    initial_value=None, 
    trainable=True, 
    collections=None, 
    validate_shape=True, 
    caching_device=None, 
    name=None, 
    variable_def=None, 
    dtype=None, 
    expected_shape=None, 
    import_scope=None)
```

`initial_value`是必填参数，即变量的初始值。可以使用Python中的`list`、`tuple`、Numpy中的`ndarray`、`Tensor`对象或者其他变量进行初始化。

```python
# 使用list初始化
var1 = tf.Variable([1, 2, 3])

# 使用ndarray初始化
var2 = tf.Variable(np.array([1, 2, 3]))

# 使用Tensor初始化
var3 = tf.Variable(tf.constant([1, 2, 3]))

# 使用服从正态分布的随机数Tensor初始化
var4 = tf.Variable(tf.random_normal([3, ]))

# 使用变量var1初始化
var5 = tf.Variable(var1)
```

这里需要注意的是：使用`tf.Variable()`得到的对象不是Tensor对象，而是承载了`Tensor`对象的`Variable`对象。`Tensor`对象就行是一个“流动对象”，可以存在于各种操作中，包括存在于`Variable`中。所以这也涉及到了如何给变量对象赋值、取值问题。

#### 使用`tf.get_variable()`创建变量

除了使用`tf.Variable()`类实例化一个变量对象以外，还有一种常用的方法来产生一个变量对象：`tf.get_variable()`这是一个生成或获取变量对象的函数。需要注意的是，使用`tf.get_variable()`方法生成一个变量时，其**name不能与已有的name重名**。

```python
# 生成一个shape为[3, ]的变量，变量的初值是随机的。
tf.get_variable(name='get_var', shape=[3, ])
# <tf.Variable 'get_var:0' shape=(3,) dtype=float32_ref>
```

一般的，`tf.get_variable()`与`variable_scope`配合使用会非常方便，在之后的内容中会详细介绍其用法，这里，我们首先知道其可以创建变量对象即可。

### 2.1变量初始化

变量作为操作的一种，是可以参与图的构建与运行的，但变量在会话中运行时必须在之前进行初始化操作。通常的变量初始化在会话中较早初始化，这可以使之后变量可以正常运行。变量的初始化也是一个操作。直接初始化变量的方法有很多种，例如：

1. 使用变量的属性`initializer`进行初始化：

```python
var = tf.Variable([1, 2, 3])

with tf.Session() as sess:
    # 如果没有初始化操作，则抛出异常
    sess.run(var.initializer)
    print(sess.run(var))  # >>> [1, 2, 3]
```

1. 单独初始化每一个变量较为繁琐，可以使用`tf.variables_initializer()`初始化一批变量。

```python
var1 = tf.Variable([1, 2, 3])
var2 = tf.Variable([1, 2, 3])

with tf.Session() as sess:
    sess.run(tf.variables_initializer([var1, var2]))
```

1. 上述方法仍然较为繁琐，一般的，所有的变量均需要初始化，这时候就不再需要特别申明。直接使用`tf.global_variables_initialize()`与`tf.local_variables_initializer()`即可初始化全部变量。

```python
var1 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
var2 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initialize())
    # 此处并不存在局部变量，所以不需要`tf.local_variables_initializer()`初始化也可以
    sess.run(tf.local_variables_initializer())
```

一般的，我们主动创建的变量均添加进入了一个列表`tf.GraphKeys.GLOBAL_VARIABLES`当中，可以通过`tf.global_variables`获取到所有主动创建的变量。`tf.global_variables_initialize`则是根据`tf.global_variables`中的变量而得到的。除了这些变量之外，有时候某些操作也会附带创建一些变量，但并非我们主动创建的，这其中的一部分变量就添加到了`tf.GraphKeys.LOCAL_VARIABLES`列表当中，可以通过`tf.local_variables`获得全部局部变量，`tf.local_variables_initializer`也是通过此获得全部局部变量的。

通常我们使用第三种方法初始化变量。

特殊的，在不初始化变量的情况下，也可以使用`tf.Variable.initialized_value()`方法获得其中存储的张量并参与运算，但我们在运行图时，依然需要初始化变量，否则使用到变量的地方依然会出错。

直接获取变量中的张量：

```python
var1 = tf.Variable([1, 2, 3])
tensor1 = var1.initialized_value()
```

### 2.2 变量赋值

变量赋值包含两种情况，第一种情况是定义时进行赋值，第二种是在图运行时修改变量的值：

```python
# 定义时赋予变量一个值
A = tf.Variable([1, 2, 3])  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A))  # >> [1, 2, 3]
    
    # 赋值方法一
    sess.run(tf.assign(A, [2, 3, 4]))  
    print(sess.run(A))  # >> [2, 3, 4]
    
    # 赋值方法二
    sess.run(A.assign([2, 3, 4]))
    print(sess.run(A))  # >> [2, 3, 4]
```

**注意**：使用`tf.Variable.assign()`或`tf.assign()`进行赋值时，必须要求所赋的值的`shape`与`Variable`对象中张量的`shape`一样、`dtype`一样。

除了使用`tf.assign()`以外还可以使用`tf.assign_add()`、`tf.assign_sub()`进行增量赋值。

```python
A = tf.Variable(tf.constant([1, 2, 3]))
# 将ref指代的Tensor加上value
# tf.assign_add(ref, value, use_locking=None, name=None)
# 等价于 ref.assign_add(value)
A.assign_add(A, [1, 1, 3])  # >> [2, 3, 6]

# 将ref指代的Tensor减去value
# tf.assign_sub(ref, value, use_locking=None, name=None)
# 等价于 ref.assign_sub(value)
A.assign_sub(A, [1, 1, 3])  # >> [0, 1, 0]
```

### 2.3 变量操作注意事项

- 注意事项一：

当我们在会话中运行并输出一个初始化并再次复制的变量时，输出是多少？如下：

```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(W) 
```

上面的代码将输出`10`而不是`100`，原因是`w`并不依赖于其赋值操作`W.assign(100)`，`W.assign(100)`产生了一个OP，然而在`sess.run()`的时候并没有执行这个OP，所以并没有赋值。需要改为：

```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    sess.run(W) 
```

- 注意事项二：

重复运行变量赋值语句会发生什么？

```python
var = tf.Variable(1)
assign_op = var.assign(2 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run(assign_op)
    sess.run(var)  # > 2
    
    sess.run(assign_op)  
    sess.run(var)  # > ???
```

这里第二次会输出`4`，因为运行了两次赋值op。第一次运行完成时，`var`被赋值为`2`，第二次运行时被赋值为`4`。

那么改为如下情况呢？

```python
var = tf.Variable(1)
assign_op = var.assign(2 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run([assign_op, assign_op])  
    sess.run(var)  # ???
```

这里，会输出`2`。会话`run`一次，图执行一次，而`sess.run([assign_op, assign_op])`仅仅相当于查看了两次执行结果，并不是执行了两次。

那么改为如下情况呢？

```python
var = tf.Variable(1)
assign_op_1 = var.assign(2 * var)
assign_op_2 = var.assign(3 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run([assign_op_1, assign_op_2])
    sess.run(var)  # >> ??
```

这里两次赋值的Op相当于一个图中的两个子图，其执行顺序不分先后，由于两个子图的执行结果会对公共的变量产生影响，当子图A的执行速度快于子图B时，可能是一种结果，反之是另一种结果，所以这样的写法是不安全的写法，执行的结果是不可预知的，所以要避免此种情况出现。但可以通过控制依赖来强制控制两个子图的执行顺序。

- 注意事项三：

在多个图中给一个变量赋值：

```python
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))  # ???

sess1.close()
sess2.close()
```

第二个会打印出`8`。因为在两个图中的OP是互不相干的。**每个会话都保留自己的变量副本**，它们分别执行得到结果。

- 注意事项四：

使用一个变量初始化另一个变量时：

```python
a = tf.Variable(1)
b = tf.Variable(a)

with tf.Session() as sess:
    sess.run(b.initializer)  # 抛出异常
```

出错的原因是`a`没有初始化，`b`就无法初始化。所以使用一个变量初始化另一个变量时，会带来不安全因素。为了确保初始化时不会出错，可以使用如下方法：

```python
a = tf.Variable(1)
b = tf.Variable(a.initialized_value())

with tf.Session() as sess:
    sess.run(b.initializer)
```

- 注意事项五：

变量初始化操作应该置于其他变量操作之前。事实上所有在一次会话执行中的操作与张量，都应该考虑其执行顺序是否会有关联，如有关联，则可能出现不可预知的错误。例如：

```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run([W.initializer, assign_op])
    print(sess.run(W))  # W 可能为10或100
```

正确的写法：

```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(sess.run(W))
```

- 注意事项六：

重新初始化变量之后，变量的值变为初始值：

```python
W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(W.assign(100))
    sess.run(W.initializer)
    print(sess.run(W))  # >>> 10
```

**小练习**：

> 练习上述注意事项代码。

## 3. Tensor、ndarray、原生数据之间的相互转化

使用Tensorflow的Python API仅仅是构建图，图的执行由TensorFlow的**内核**（操作在不同设备上的C++实现）来完成。也就是说图的定义以及部分数据的传入是在Python环境中完成的，图的执行是在C++环境中完成的，最后执行图所得到的结果往往也需要返回给Python环境。所以使用TensorFlow就不可避免的需要了解不同环境中数据的对应关系以及相互转化方法。

**注意**：不同设备上的数据在不同执行环境中频繁的交互会降低性能，通常的我们会尽量避免频繁交互。

### 3.1 原生数据、ndarray转化为Tensor

一般的，构建常量、变量等图结构时经常会使用Python原生数据与数据结构作为输入，例如构建一个常量：

```python
tf.constant([1, 2])
```

这时等价于将Python环境中的数据转化为了TensorFlow中的张量，默认的，整形转化为`DT_INT32`类型，浮点型转化为`DT_FLOAT`（32位）类型，复数转化为`DT_COMPLEX128`类型。

类似的，我们也可以将ndarray类型的数据输入到图中，例如：

```python
tf.constant(np.array([1, 2], dtype=np.float32))
```

与直接输入Python数据与数据结构不同的是，输入ndarray数据，TensorFlow则会继承其数据类型，上述代码中，得到的Tensor数据类型为`DT_FLOAT`。

注意：有些Python原生数据不属于张量，并不能转化为张量，例如：`t = [1, [2, 3]]`。

### 3.2 Tensor 转化为原生数据、ndarray

当完成图的构建之后，在会话中运行图则可以得到结果，这时候会话返回给Python环境中的结果也是包可能是Python原生数据或ndarray数据。一般的张量执行的结果均返回ndarray数据（这是因为TensorFlow内核使用了ndarray数据结构）。

```python
a = tf.constant([1, 2, 3])

with tf.Session() as sess:
    res = sess.run(a)
    print(res)   # >>> [1 2 3]
    print(type(res))  # >>> <class 'numpy.ndarray'>
```

## 4. 占位符

使用TensorFlow定义图，类似于定义了一个算式。但这是不方便的，为了能够在复杂问题中能够流畅构建运算逻辑，我们还需要引入“未知数”来参与构建图，就像代数中的方程中引入未知数一样。在TensorFlow中，我们称之为**占位符（placehoders）**。

引入占位符在TensorFlow中是很重要的，它可以把图的构建与数据的输入关系解耦，这意味着构建一个图只需要知道数据的格式即可，而不需要将庞大的数据输入图。仍然以方程为例$z=x*2+y$，当我们在需要用到这个方程时，再给$x,y$赋值，就能立马得到结果，方程$z=x*2+y$就相当于图，$x,y$就相当于占位符。在TensorFlow中使用`tf.placeholder`构建占位符。占位符也是一个节点。

```python
tf.placeholder(dtype, shape=None, name=None)
```

例如，我们需要将上述方程使用带占位符的图描述：

```python
x = tf.placeholder(dtype=tf.float32, shape=[])
y = tf.placeholder(dtype=tf.float32, shape=[])

z = tf.multiply(x, 2) + y
```

**注意**：占位符必须包含数据类型，但可以没有`shape`，即可以将`shape`设置为`None`或者将`shape`中的某一些维度设置为`None`。一般的，如果`shape`可知，则尽量输入到占位符中，可以避免出现很多不必要的错误。

### 4.1 feed_dict

图构建好之后，运行图时，需要使用张量替代占位符，否则这个图无法运行，如下：

```python
x = tf.placeholder(dtype=tf.float32, shape=[])
y = tf.placeholder(dtype=tf.float32, shape=[])

z = tf.multiply(x, 2) + y

with tf.Session() as sess:
    sess.run(z, feed_dict={x: 5, y: 10})  # >>> 20.0
```

可以看到使用占位符，需要输入占位数据的类型即可，即不需要填入具体数据。占位符可以不设确定的`shape`意味着可以使用不同`shape`但运算规则一致的数据。例如：

```python
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

z = tf.multiply(x, 2) + y

with tf.Session() as sess:
    sess.run(z, feed_dict={x: 5, y: 10})  # >>> 20.0
    sess.run(z, feed_dict={x: [5, 4], y: [10, 18]})  # >>> [20., 26.]
```

### 4.2 feed_dict的更多用法

除了`tf.placeholder`可以并且必须使用张量替代以外，很多张量均可以使用`feed_dict`替代，例如：

```python
a = multiply(1, 2)

with tf.Session() as sess:
    sess.run(a, feed_dict={a: 10})  # >> 10
```

为了保证张量的替代是合法的，可以使用`tf.Graph.is_feedable(tensor)`检查`tensor`是否可以替代：

```python
a = multiply(1, 2)
tf.get_default_graph().is_feedable(a)  # True
```

**注意**：变量、占位符是两类完全不同功能的节点。变量在图的运行中负责保存当前时刻一个张量的具体数值，在运行的不同阶段、时刻可能是不同的，职责是存储运行时的临时张量；而占位符是在图运行之前的充当某一个张量的替身，在运行时必须使用一个张量替代，职责是建构时代替具体的张量值。

**小练习**：

> 回顾上述内容并练习编写上述代码。

## 作业

利用所学知识，完成以下任务：

1. 构建二元线性回归模型，其中模型中的参数使用`tf.Variable()`构建，模型的样本输入使用`tf.placeholder`代替。写出模型结构。
2. 使用`tf.placeholder`代替上述样本中的标记，写出对于一个样本的代价。