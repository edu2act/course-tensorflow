TensorFlow下的eager模式是一个命令式编程模式。无需构建图，所有的操作会立即返回结果，所以也不需要会话。eager模型的优点包括：直观的界面、更容易调试、简单的流程控制。eager模式支持大部分的TensorFlow操作与GPU加速。缺点是执行效率尚不够高。在TensorFlow 1.4版本中，第一次引入了Eager Execution，在TensorFlow 1.5版本中将其加入`tf.contrib`模块下，在TensorFlow 1.7版本中加入核心API。

## 开启eager模式

在程序最开始添加`tf.enable_eager_execution()`语句即可开启eager模式。需要注意的是，`tf.enable_eager_execution()`语句之前不能有任何TensorFlow操作与张量的构建语句。也不要将此语句添加到程序调用的模块中。

用法如下：

~~~python
import tensorflow as tf

tf.enable_eager_execution()
~~~

可以使用`tf.executing_eagerly()`语句查看当前是否开启了eager模式，开启则返回`True`，否则返回`False`。

现在就可以运行TensorFlow操作，之后将立即返回结果：

~~~python
const = tf.constant(5)
print(const)  # >>> tf.Tensor(5, shape=(), dtype=int32)

res = tf.multiply(const, const)
print('result is %d' % res)  # >>> result is 25
~~~

由于所有的执行结果会立即呈现在Python环境中，所以调试起来会很方便。

注意：eager模式一旦开启就无法关闭。

## 基本用法

开启eager模式之后，TensorFlow中的大部分操作都可以使用，但操作也变得有一些不同，主要是Tensor对象的操作与流程控制部分有较大变化。

### Tensor

对于Tensor对象，现在它可以与Python、Numpy很好的进行协作（并不能与其它科学计算库很好的协作）。例如我们不仅可以将Numpy数组作为输入传入TensorFlow的操作中，现在也可以将Tensor传入Numpy的操作中，事实上传入Numpy操作是调用了`tf.Tensor.numpy()`方法将值转换为了`ndarray`对象。例如：

~~~python
a = tf.constant(5)
b = np.array(10)

tf_res = tf.multiply(a, b)  # >>> tf.Tensor(50, shape=(), dtype=int32)
np_res = np.multiply(a, b)  # >>> 50
~~~

显式的调用`tf.Tensor.numpy()`可以得到Tensor对应的ndarray对象，用于更多其他操作例如：

~~~python
import tensorflow as tf
from PIL import Image

tf.enable_eager_execution()

img = tf.random_uniform([28, 28, 3], maxval=255, dtype=tf.int32)
img = tf.cast(img, tf.uint8)

Image.fromarray(img.numpy())  # 调用tf.Tensor.numpy()获取ndarray对象并转化为PIL的Image对象
~~~

**注意**：在Eager模式下，所有操作的`name`属性都是没有意义的。也就是说虽然你仍然可以给操作设置`name`，但实际上毫无意义。

### 变量

Eager下创建变量的用法也有一些不同，最简单的创建变量的方法是使用`tf.get_variable`创建变量，这时候需要注意变量是拥有`name`属性的，但意义不大。用法如下：

~~~python
var = tf.get_variable('var', shape=[])

print(var)  # >>> <tf.Variable 'var:0' shape=() dtype=float32, numpy=-0.08887327>
~~~

Eager模式下不需要初始化变量了。使用`tf.get_variable`创建变量的用法与传统用法并不完全一致，在eager模式下`tf.get_variable`只能用来创建变量，不能获取已经创建好的变量。

除此之外还可以使用如下方法创建变量：

~~~python
import tensorflow.contrib.eager as tfe

var = tfe.Variable(10)
print(var)  # >>> <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=10>
~~~

### 运算

在eager模式下，基本的算术运算、逻辑运算、位运算规则与Python下基本是相同的，但在执行比较运算与赋值运算时需要注意，有些运算符没有重载，有些运算结果与我们想象的不太一样：

在比较运算中，需要注意：

```python
tf.constant(0) == 0  # False

tf.constant(0) <= 0  # True
tf.constant(0) >= 0  # True
```

为了不出现这种问题，通常的我们首先将比较运算的输入数据的类型转化一致之后再进行比较如下：

~~~python
# 方法一：将Tensor转化为Python数据类型
int(tf.constant(0)) == 0  # True

# 方法二：利用比较函数进行比较
tf.equal(tf.constant(0), 0)

# 方法三：利用ndarray进行比较
tf.constant(0).numpy() == 0
~~~

TensorFlow变量在赋值运算中需要注意：

* TensorFlow变量不能直接使用`=`赋值
* TensorFlow变量不支持使用`+=`等赋值运算符重载

如下：

~~~python
var = tfe.Variable(10)

var.assign(5)  # 正确用法：赋值
var.assign_add(5)  # 正确用法：增量赋值

var += 5 # 错误用法，不支持此运算符重载
var = 5  # 错误用法，此时相关于给Python变量var赋值为5，并没有使用TensorFlow中的变量
~~~

但在Tensor中是支持增量赋值运算符的（不支持直接赋值运算符），如下：

~~~python
var = tf.constant(10)
var += 5 
print(var)  # >>>  tf.Tensor(15, shape=(), dtype=int32)

var -= 5 
print(var)  # >>>  tf.Tensor(10, shape=(), dtype=int32)
~~~

### 流程控制

在符号式编程模式下，流程控制是较为复杂的操作，往往需要使用不太直观的`tf.cond`、`tf.case`、`tf.while_loop`流程控制函数进行操作。而在eager模式下，直接使用Python的流程控制语句即可，如下使用TensorFlow实现一个fizzbuzz游戏（规则：输入一系列整数，当数字为3的倍数的时候，使用“Fizz”替代数字，当为5的倍数用“Buzz”代替，既是3的倍数又是5的倍数时用“FizzBuzz”替代。）：

~~~python
def fizzbuzz(max_num):
    max_num = tf.convert_to_tensor(max_num)
    for i in tf.range(max_num):
        if int(i % 3) == 0 and int(i % 5) == 0:
            print('fizzbuzz')
        elif int(i % 3) == 0:
            print('fizz')
        elif int(i % 5) == 0:
            print('buzz')
        else:
            print(int(i))
~~~

### 变量存取

使用 Graph Execution 时，程序状态（如变量）存储在全局集合中，它们的生命周期由 [`tf.Session`](https://tensorflow.google.cn/api_docs/python/tf/Session) 对象管理。相反，在 Eager Execution 期间，状态对象的生命周期由其对应的 Python 对象的生命周期决定。

在Eager模式下，变量的保存使用`tf.train.Checkpoint`的对象来完成，而无法使用`tf.train.Saver`，同时`tf.train.Checkpoint`即支持Graph Execution，也支持Eager Execution，是TensorFlow最新推荐的模型变量保存方法。

二者区别：

* `tf.train.Saver`存取变量时，实际上是依据`variable.name` 进行存取与匹配的方法，只能在图模式下使用；
* `tf.train.Checkpoint`则是存取Python对象与边、节点之间的依赖关系，可以在两种模式下使用。

`tf.train.Checkpoint`用法简单，首先使用`tf.train.Checkpoint`实例化一个对象，同时传入需要保存的变量，在完成一系列操作之后，可以调用`tf.train.Checkpoint.save`方法保存模型。恢复模型只需要调用`tf.train.Checkpoint.restore`即可，如下：

~~~python
x = tf.get_variable('x', shape=[])

checkpoint = tf.train.Checkpoint(x=x)  # 声明将变量x保存为"x"，可以保存为任意合法的字符串

x.assign(2.)   # 给变量x重新复制
save_path = checkpoint.save('./ckpt/')  # 保存变量到指定目录，此时x的值2就保存下来了

x.assign(11.)  # 再次改变x的值

checkpoint.restore(save_path)  # 恢复变量

print(x)  # >>> 2.0
~~~

### 自动求微分

 自动微分对于机器学习算法来讲是很有用的，使用 Graph Execution 时，我们可以借助图本身的性质对变量集合中的所有变量进行自动微分，而在 Eager Execution 期间，则需要使用另一个工具——梯度磁带。当我们定义了一个梯度磁带时，正向传播操作都会记录到磁带中（所以会降低性能，如无需要请勿开启磁带）。要计算梯度时，反向播放磁带即可，默认的磁带只能反向播放一次（求一次梯度）。用法如下：

~~~python
w = tf.get_variable('w', shape=[])
w.assign(5.)

with tf.GradientTape() as gt:
    loss = w * w
    print(gt.gradient(loss, w))  # >>> tf.Tensor(10.0, shape=(), dtype=float32)
~~~

默认的磁带只能反向使用一次，如果需要多次使用则需要设置`persistent=True`，如下：

~~~python
w = tf.get_variable('w', shape=[])
w.assign(5.)

with tf.GradientTape(persistent=True) as gt:
    loss = w * w

print(gt.gradient(loss, w))
print(gt.gradient(loss, w))
~~~

磁带也可以嵌套，求二阶或高阶梯度，如下：

~~~python
x = tf.constant(3.0)
with tf.GradientTape() as g:
    with tf.GradientTape() as gg:
        gg.watch(x)  # watch张量之后，可对张量求梯度
        y = x * x
        
    dy_dx = gg.gradient(y, x)     
    print(dy_dx)  # >>> 6
    d2y_dx2 = g.gradient(dy_dx, x)  
    print(d2y_dx2)  # >>> 2
~~~

注意事项：

* 只有磁带上下文管理器中操作会被记录，以外的部分不会被记录，所以也无法对以外的操作求得梯度，结果会是`None`。
* 默认的磁带只能求一次梯度。
* 在磁带的上下文管理器中连续调用求梯度的方法会存在性能问题，这会使得磁带记录上每次只需的操作，不断增加内存和CPU。
* 默认的求梯度时只能对变量求梯度，也可以对张量求梯度，需要调用`tf.GradientTape.watch`方法加入求梯度的量中即可。

### TensorBoard

在eager模式下，TensorBoard创建事件文件并获取摘要的方法稍有不同。也就是说与图模式下的用法不兼容，但过程类似，创建摘要的一般步骤为：

* 创建一个或多个事件文件，使用`tf.contrib.summary.create_file_writer`。
* 将某一个事件文件置为默认，使用`writer.as_default`。
* 在训练模型的代码处获取之后设置摘要获取频次，可使用`tf.contrib.summary.record_summaries_every_n_global_steps`、`tf.contrib.summary.always_record_summaries`分别设置摘要频次与每次摘要。
* 给指定张量设置摘要。

具体如下：

~~~python
# 创建global_step
gs = tf.train.get_or_create_global_step()
# 创建事件文件
writer = tf.contrib.summary.create_file_writer('/tmp/log')

# 将writer设置为默认writer(当存在多个writer时可用此方法切换writer)
with writer.as_default():
    # 执行训练循环
    for i in range(train_steps):
        # 每100步执行一次提取摘要，默认的此方法会调用
        # `tf.train.get_or_create_global_step`查看当前global_step数
        with tf.contrib.summary.record_summaries_every_n_global_steps(100):
            # 在此处写训练代码
            ...
            ...
            ...
            # 添加摘要
            tf.contrib.summary.scalar('loss', loss)
            # global_step增加1
            gs.assign_add(1)
~~~



### 其它

* 在eager模式下，`tf.placeholder`无法使用。
* 在eager模式下， 由于所有语句均顺序执行，所以也不再需要控制依赖`tf.control_dependencies`进行强制执行顺序。
* `name_scope`与`variable_scope`的也几乎没作用了，除了可以给变量的name添加前缀以外。
* 在eager模式下，变量集合不被支持了，执行`tf.trainable_variables()`等语句会出错。
* ...

