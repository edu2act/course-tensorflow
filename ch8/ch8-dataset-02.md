在TensorFlow 1.3之后，便使用`tf.data`模块替代了原有的数据集读取以及处理方式，其操作更为简便，性能更好。此API按照功能进行划分，主要包含三部分内容：**数据读取**、**数据集与样本处理**以及**输出**。这里以一个简单的例子进行说明。

~~~python
# 制作假数据样本，共5个样本
inputs = tf.random_normal([5, 3])
labels = tf.constant([1, 0, 1, 0, 1])

# 步骤一：数据读取，生成一个dataset对象
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

# 步骤二：样本处理
dataset = dataset.shuffle(2)  # 进行打乱，打乱时cache为2
dataset = dataset.batch(3)  # 设置批量大小，这里为3

# 步骤三：批量输出
iterator = dataset.make_initializable_iterator()  # 生成迭代器对象
init_op = iterator.initializer
next_batch = iterator.get_next()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(next_batch)
~~~

这个例子中，我们创建了5个“假样本”Tensor，并使用`tf.data.Dataset.from_tensor_slices()`方法将Tensor转化为了dataset对象。我们还可以使用`tf.data.Dataset.from_tensors()`通过内存中的张量构建dataset。更一般的，我们还可以读取磁盘上的文件，例如使用`tf.data.TFRecordDataset`读取TFRecord文件。在有了dataset之后，我们又对dataset做了一些处理，即打乱样本，并设置了批量大小。最后我们构建了迭代器对象读取数据。可以看到这里使用了`Dataset`类与`Iterator`类。

## 1. Dataset

`Dataset`类包含了数据集读取、样本处理等功能。

### 1.1 数据集结构

一个数据集包含多个**元素**（此处可理解为广义上的样本），一个元素包含一个或多个Tensor对象，这些对象被称为**组件**（即样本中的特征以及标记）。`Dataset`对象包含`output_types`和`output_shapes`属性，可查看其中所有组件的推理类型与形状。例如：

~~~python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # >>> <dtype: 'float32'>
print(dataset1.output_shapes)  # >>> (10,)


dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # >>> (tf.float32, tf.int32)
print(dataset2.output_shapes)  # >>> (TensorShape([]), TensorShape([Dimension(100)]))


dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # >>> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # >>> "(10, ((), (100,)))"
~~~

为每个元素的每个组件命名通常会带来便利性，例如，如果它们表示训练样本的不同特征。除元组之外，还可以使用`collections.namedtuple`或字符串映射到张量来表示`Dataset`的单个元素。例如：

~~~python
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # >>> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # >>> "{'a': (), 'b': (100,)}"
~~~

### 1.2 样本处理

我们读取到的数据集往往是无法直接使用的，这时候需要对每一个样本进行处理，例如类型转化、过滤、格式转化等。这时候主要使用 `Dataset.map()`、`Dataset.flat_map()` 和 `Dataset.filter()` 等方法。这些方法可以应用于每一个元素中的每一个样本。

**map**

`Dataset.map()`可以对每个数据集中元素中的每个样本进行操作，用法如下：

~~~python
inputs = tf.range(3)
labels = tf.zeros(3)

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.map(lambda x, y: (tf.not_equal(x, 0), y + 1))
~~~

这里为每一个样本输入判断是否非0，给每个label加1。

`Dataset.map()`可以设置参数`num_parallel_calls`，来设置样本处理的线程数。这样在面对需要做复杂处理的数据时，可以大大提高速度。

**filter**

`Dataset.filter()` 可用于过滤某些样本，例如我们需要保留数据集中元素值小于5的样本，去除其它样本，我们可以进行如下操作：

~~~python
inputs = tf.constant([1, 0, 6, 4, 5, 2, 7, 3])
dataset = tf.data.Dataset.from_tensor_slices(inputs)

# 过滤部分样本
dataset = dataset.filter(lambda example: tf.less(example, 5))

iterator = dataset.make_initializable_iterator()
...
~~~

**skip**

使用`Dataset.skip()`方法可以跳过`Dataset`中的前n个组件，通常用于去除样本中的表头，用法如下：

```python
inputs = tf.constant(['feature', '1', '2', '3', '4', '5'])

dataset = tf.data.Dataset.from_tensor_slices(inputs)

# 跳过第0个样本'feature',之后生成的样本从'1'开始
dataset = dataset.skip(0)

iterator = dataset.make_initializable_iterator()
...
```

**flat_map**

有时候我们的数据集分散在多个文件中，这时候我们可以为每个文件创建一个`Dataset`对象，当我们需要对多个`Dataset`对象分别操作时，这时候可以使用`Dataset.flat_map()` ，用法如下：

```python
inputs_0 = tf.constant([1, 2, 3, 4, 5])
inputs_1 = tf.constant([7, 8, 9, 10, 6])

dataset = tf.data.Dataset.from_tensor_slices([inputs_0, inputs_1])
dataset = dataset.flat_map(
    lambda t: tf.data.Dataset.from_tensor_slices(t))

iterator = dataset.make_initializable_iterator()
...
```

上面的例子中我们首先创建一个两个元素的数据集，然后将这个数据集的每个元素又分别作为了一个数据集， 最终的结果是使得两个1阶常量转化为了一个数据集。这个例子可能看起来是没有意义，这里我们再举一个例子。就是从两个text文件中读取数据，其每一行为一个样本，但是这两个text文件的第一行是标题，我们需要分别跳过每个文件的第一行，这时候我们可以为每个文件创建一个`Dataset`对象，然后分别操作`Dataset`对象去除第一行数据，如下：

~~~python
filenames = ["/tmp/file1.csv", "/tmp/file2.csv"]

# 生成文件名`Dataset`对象，其每个元素为一个文件路径
dataset = tf.data.Dataset.from_tensor_slices(filenames)

# 根据每个文件路径创建一个`Dataset`，并为这个`Dataset`去除第一行
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)))
~~~

`Dataset.flat_map()` 与`Dataset.map()`相比，最大的不同点在于`Dataset.flat_map()` 中传入的函数返回值必须是`Dataset`类型。



### 1.3 批量样本处理

除了对单个样本处理以外，我们还需要对每一个批次数据进行处理，例如打乱数据集、设置批量大小等操作。

#### shuffle

使用`Dataset.shuffle()` 打乱数据集，类似于 `tf.RandomShuffleQueue` ，它保留一个固定大小的缓冲区，用于打乱样本。例如：

~~~python
inputs = tf.constant([1, 2, 3, 4, 5])

dataset = tf.data.Dataset.from_tensor_slices(inputs)
# 打乱数据集样本，每次从3个buffer中选择样本
dataset = dataset.shuffle(buffer_size=3)

iterator = dataset.make_initializable_iterator()
...
~~~

#### batch

使用`Dataset.batch()`设置读取数据即出队时，每次出队的样本数，用法如下：

~~~python
inputs = tf.constant([1, 2, 3, 4, 5])

dataset = tf.data.Dataset.from_tensor_slices(inputs_0)
# 每次读取到2个样本
dataset = dataset.batch(2)

iterator = dataset.make_initializable_iterator()
...
~~~

**注意**：通常的，我们首先打乱数据，然后在生成批次样本，如果顺序颠倒，则会出现一个批次内样本没有打乱的情况。

使用batch时，要求batch中每一个元素与其它元素的每一个组件都拥有相同的`shape`，此时才能合成一个batch。当batch中不同元素的组件`shape`不相同时，我们也可以使用`padded_batch`对组件进行填充，使其`shape`相同，用法如下：

~~~python
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
~~~

#### repeat

使用`Dataset.repeat()`方法通常用来设置样本重复使用代数，默认的所有样本只能使用1一次，即默认为1代，当所有元素均出队完成再去获取元素时，会出现`OutOfRangeError`的错误，当设置为`None`时表示不限制样本使用次数。如下：

~~~python
inputs = tf.constant([1, 2, 3, 4, 5])

dataset = tf.data.Dataset.from_tensor_slices(inputs_0)
# 不限制代数
dataset = dataset.repeat(None)

iterator = dataset.make_initializable_iterator()
...
~~~

### 1.4 其它操作

~~~python
# 预取操作，类似于缓存，一般用在map或batch之后
tf.data.Dataset.prefetch()

# 将dataset缓存在内存或硬盘上，默认的不指定路径即为内存(慎用)
tf.data.Dataset.cache()
~~~



## 2. Iterator

创建了表示输入数据的`Dataset`之后，下一步需要创建`Iterator`来访问数据集中的元素。迭代器的种类有很多种，由简单到复杂的迭代器如下：

* 单次迭代器
* 可初始化迭代器
* 可重新初始化迭代器
* 可馈送迭代器

### 2.1 单次Iterator

单次迭代器是最近单的迭代器形式，用法简单，如下：

~~~python
dataset = tf.data.Dataset.range(100)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value
~~~

这里，我们使用`Dataset`对象的`make_one_shot_iterator()`方法创建了单次`Iterator`对象，然后调用其`get_next()`方法即可获得数据，循环调用时，每次都会出队一个数据。

**注意**：单次迭代器是目前唯一可轻松与`Estimator`配合使用的类型。

### 2.2 可初始化Iterator

单次迭代器用法简便，但这也限制其部分功能的使用，例如有时候无法使用占位符，如果要使用占位符，那么必须使用可初始化的迭代器。用法如下：

~~~python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value
~~~

可以看到可初始化迭代器可以方便的加入占位符，但相应的，我们必须对迭代器进行显式的初始化，然后在初始化时把占位符使用具体的张量进行替代。

事实上，可初始化迭代器还可以进行多次初始化，用法如下：

~~~python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# 第一次初始化，max_value使用10替代
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

# 第二次初始化，max_value使用100替代
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
    value = sess.run(next_element)
    assert i == value
~~~

### 2.3 可重新初始化迭代器 

有时候，我们需要构建多个数据集对象，而且这些数据集对象都是相似的。例如我们在训练模型时往往既需要构建训练输入管道，又需要构建验证输入管道，这时候可以构建可重新初始化迭代器。例如：

~~~python
# 构建结构类似（组件shape与dtype相同）的训练集与验证集dataset对象
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# 从一个“结构”中创建迭代器，这里使用的是训练集的dtypes与shapes，验证集的也可以。
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

# 训练集与验证集的初始化op
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
    # 运行20代，每代首先出队100个训练集可用于训练，然后在出队50个验证集，可用于验证
    for _ in range(20):
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)

        sess.run(validation_init_op)
        for _ in range(50):
            sess.run(next_element)
~~~

**注意**：使用可重新初始化迭代器时，每次初始化迭代器之后，都会重新从数据集的开头遍历数据，这在有时候是不可取。

### 2.4 可馈送迭代器

为了避免上述可重新初始化迭代器每次初始化之后从头遍历数据的情况，我们可以使用可馈送迭代器，其功能类似于可重新初始化迭代器。用法如下：

~~~python
# 构建结构类似（组件shape与dtype相同）的训练集与验证集dataset对象
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# 获取训练集与验证集的对应的迭代器
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# 构建可馈送迭代器，这里使用`placeholder`构建，后面我们可使用训练集与验证集的
# handle替换此处的handle，从而切换数据集对应迭代器
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

with tf.Session() as sess:
    # 获得训练集与验证集的handle
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    
    for _ in range(20):
        for _ in range(100):
            sess.run(next_element, feed_dict={handle: training_handle})

        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})
~~~

上面的例子中，每一代的训练集出队时都是从上次出队的位置继续出队，可以保证训练集样本是连续输出的，而验证集每次都从头获取，这是我们常见的训练集与验证集的用法。

**注意：**由于`tf.placeholder`不支持TensorFlow中的`Tensor`传入，所以必须在获取数据集之前首先在会话中得到训练集与验证集的`handle`。

事实上，我们除了借助`tf.placeholder`切换`handle`以外，也可以使用变量进行切换，如下：

~~~python
# 构建结构类似（组件shape与dtype相同）的训练集与验证集dataset对象
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# 获取训练集与验证集的对应的迭代器
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# 构建可馈送迭代器，这里使用`Variable`构建，后面我们可使用训练集与验证集的
# handle张量赋值给此处的handle变量，从而切换数据集对应迭代器
handle = tf.Variable('')
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

with tf.Session() as sess:
    for _ in range(20):
        # handle赋值为训练集的handle
        sess.run(handle.assign(training_iterator.string_handle()))
        for _ in range(100):
            sess.run(next_element)

        sess.run(validation_iterator.initializer)
        # handle赋值为验证集的handle
        sess.run(handle.assign(validation_iterator.string_handle()))
        for _ in range(50):
            sess.run(next_element)
~~~

上述的两种用法都是等价的。可以看到使用可馈送迭代器可以方便灵活的切换我们想要使用的数据集。

## 3. 高效读取数据

直接从原始文件中读取数据可能存在问题，例如当文件数量较多，且解码不够迅速时，就会影响模型训练速度。在TensorFlow中，我们可以使用TFRecord对数据进行存取。TFRecord是一种二进制文件。可以更快速的操作文件。

通常我们得到的数据集并不是TFRecord格式，例如MNIST数据集也是一个二进制文件，每一个字节都代表一个像素值（除去开始的几个字节外）或标记，这与TFRecord文件的数据表示方法（TFRecord的二进制数据中还包含了校验值等数据）并不一样。所以，有时候我们需要将数据转化为TFRecord文件。这里需要注意并不是每一个数据集均需要转化为TFRecord文件，建议将文件数量较多，直接读取效率低下的数据集转化为TFRecord文件格式。

### 3.1 写入TFRecord文件

TFRecord文件的存取，本质上是对生成的包含样本数据的ProtoBuf数据的存取，TFRecord文件只适合用来存储数据样本。一个TFRecord文件存储了一个或多个example对象，example.proto文件描述了一个样本数据遵循的格式。每个样本example包含了多个特征feature，feature.proto文件描述了特征数据遵循的格式。

在了解如何写入TFRecord文件前，我们首先了解一下其对应的消息定义文件。通过这个文件，我们可以知道消息的格式。

**feature.proto**的内容如下（删除了大部分注释内容）：

```protobuf
syntax = "proto3";
option cc_enable_arenas = true;
option java_outer_classname = "FeatureProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.example";

package tensorflow;

// Containers to hold repeated fundamental values.
message BytesList {
  repeated bytes value = 1;
}
message FloatList {
  repeated float value = 1 [packed = true];
}
message Int64List {
  repeated int64 value = 1 [packed = true];
}

// Containers for non-sequential data.
message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};

message FeatureList {
  repeated Feature feature = 1;
};

message FeatureLists {
  // Map from feature name to feature list.
  map<string, FeatureList> feature_list = 1;
};
```

可以看到一个特征`Feature`可以是3中数据类型（`BytesList`，`FloatList`，`Int64List`）之一。多个特征`Feature`组成一个组合特征`Features`，多个特征列表组成特征列表组`FeatureLists`。

**example.proto**的内容如下（删除了大部分注释内容）：

```protobuf
syntax = "proto3";

import "tensorflow/core/example/feature.proto";
option cc_enable_arenas = true;
option java_outer_classname = "ExampleProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.example";

package tensorflow;

message Example {
  Features features = 1;
};

message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};
```

可以看到一个样本`Example`包含一个特征组合。序列样本`SequenceExample`包含一个类型是特征组合的上下文`context`与一个特征列表组`feature_lists`。

可以看到：**TFRecord存储的样本数据是以样本为单位的。**

了解了TFRecord读写样本的数据结构之后，我们就可以使用相关API进行操作。

#### 写入数据

TensorFlow已经为我们封装好了操作protobuf的方法以及文件写入的方法。写入数据的第一步是打开文件并创建writer对象，Tensorflow使用`tf.python_io.TFRecordWriter`来完成，具体如下：

```python
# 传入一个路径，返回一个writer上下文管理器
tf.python_io.TFRecordWriter(path, options=None)
```

TFRecordWriter拥有`write`，`flush`，`close`方法，分别用于写入数据到缓冲区，将缓冲区数据写入文件并清空缓冲区，关闭文件流。

开启文件之后，需要创建样本对象并将example数据写入。根据上面的proto中定义的数据结构，我们知道一个样本对象包含多个特征。所以我们首先需要创建特征对象，然后再创建样本对象。如下为序列化一个样本的例子：

```python
with tf.python_io.TFRecordWriter('./test.tfrecord') as writer:
    f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
    f2 = tf.train.Feature(float_list=tf.train.FloatList(value=[1. , 2.]))
    b = np.ones([i]).tobytes()  # 此处默认为float64类型
    f3 = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

    features = tf.train.Features(feature={'f1': f1, 'f2': f2, 'f3': f3})
    example = tf.train.Example(features=features)

    writer.write(example.SerializeToString())
```

序列化多个样本只需要重复上述的写入过程即可。如下：

```python
with tf.python_io.TFRecordWriter('./test.tfrecord') as writer:
    # 多个样本多次写入
    for i in range(1, 6):
        f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
        f2 = tf.train.Feature(float_list=tf.train.FloatList(value=[1. , 2.]))
        b = np.ones([i]).tobytes()
        f3 = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
        
        features = tf.train.Features(feature={'f1': f1, 'f2': f2, 'f3': f3})
        example = tf.train.Example(features=features)
        
        writer.write(example.SerializeToString())
```

**注意事项：**

- `tf.train.Int64List`、`tf.train.FloatList`、`tf.train.BytesList`均要求输入的是python中的list类型的数据，而且list中的元素分别只能是int、float、bytes这三种类型。
- 由于生成protobuf数据对象的类中，只接受关键字参数，所以参数必须写出参数名。
- protobuf数据对象类需要遵循proto文件中定义的数据结构来使用。

TFRecord文件的数据写入是在Python环境中完成的，不需要启动会话。写入数据的过程可以看做是原结构数据转换为python数据结构，再转换为proto数据结构的过程。完整的数据写入过程如下：

1. 读取文件中的数据。
2. 组织读取到的数据，使其成为以“样本”为单位的数据结构。
3. 将“样本”数据转化为Python中的数据结构格式（int64\_list，float\_list，bytes\_list三种之一）。
4. 将转化后的数据按照proto文件定义的格式写出“Example”对象。
5. 将“Example”对象中存储的数据序列化成为二进制的数据。
6. 将二进制数据存储在TFRecord文件中。

### 3.2 读取TFRecord文件

读取TFRecord文件类似于读取csv文件。只不过使用的是`tf.TFRecordReader`进行读取。除此以外，还需要对读取到的数据进行解码。`tf.TFRecordReader`用法如下：

```python
reader = tf.TFRecordReader(name=None, options=None)
key, value = reader.read(queue, name=None)
```

此处读取到的value是序列化之后的proto样本数据，我们还需要对数据进行解析，这里可以使用方法`tf.parse_single_example`。解析同样需要说明解析的数据格式，这里可以使用`tf.FixedLenFeature`与`tf.VarLenFeature`进行描述。

解析单个样本的方法如下：

```python
# 解析单个样本
tf.parse_single_example(serialized, features, name=None, example_names=None)
```

解析设置数据格式的方法如下

```python
# 定长样本
tf.FixedLenFeature(shape, dtype, default_value=None)
# 不定长样本
tf.VarLenFeature(dtype)
```

完整例子如下（解析的是上文中写入TFRecord文件的例子中生成的文件）：

```python
reader = tf.TFRecordReader()
key, value = reader.read(filename_queue)

example = tf.parse_single_example(value, features={
    'f1': tf.FixedLenFeature([], tf.int64),
    'f2': tf.FixedLenFeature([2], tf.float32),
    'f3': tf.FixedLenFeature([], tf.string)})

feature_1 = example['f1']
feature_2 = example['f2']
feature_3 = tf.io.decode_raw(example['f3'], out_type=tf.float64)
```

这里还用到了`tf.io.decode_raw`用来解析bytes数据，其输出type应该等于输入时的type，否则解析出的数据会有问题。

```python
tf.io.decode_raw(
    bytes,  # string类型的tensor
    out_type,  # 输出类型，可以是`tf.half, tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`
    little_endian=True,  # 字节顺序是否为小端序
    name=None)
```

无论使用哪一数据读取的方法，其过程都是一样的。过程如下：

1. 打开文件，读取数据流。
2. 将数据流解码成为指定格式。在TFRecord中，需要首先从proto数据解码成包含Example字典数据，再把其中bytes类型的数据解析成对应的张量。其它的比如csv则可以直接解析成张量。
3. 关闭文件。

## 4. data模块下的多种数据集读取

除了上述的从内存中的张量中读取数据集以外，还可以从文本、TFRecord文件等不同数据集文件中读取数据集。

**从内存中的张量中读取数据：**

~~~python
features = np.random.normal(size=[100, 3])
labels = np.random.binomial(1, .5, [100])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
~~~

**从文本文件中读取数据：**

~~~python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
~~~

这时候每一行都会作为一个样本，此时dataset中只有一个元素，元素的类型为`tf.string`。这样的数据往往并不能直接使用，还需进行处理，可以使用`map`等方法逐元素处理。

**从TFRecord文件中读取数据：**

~~~python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
~~~

从TFRecord文件中读取到的数据也需要使用`map`等方法对每条数据进行解析。例如：

~~~python
# 将string类型的`example_proto`转化为一个string类型组件和int类型组件，分别代表图片与标记
def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["label"]

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
~~~

