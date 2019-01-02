## 1. 图存取

图是算法过程的描述工具，当我们在某个文件中定义了一个图的时候，也就定义了一个算法，当我们需要运行这个算法时，可以直接找到定义此图的Python文件，就能操作它。但为了方便，我们也可以将图序列化。

图是由一系列Op与Tensor构成的，我们可以通过某种方法对这些Op与Tensor进行描述，在Tensorflow中这就是'图定义'`GraphDef`。图的存取本质上就是`GraphDef`的存取。

### 1.1 图的保存

图的保存方法很简单，只需要将图的定义保存即可。所以：

**第一步，需要获取图定义。**

可以使用`tf.Graph.as_graph_def`方法来获取序列化后的图定义`GraphDef`。

例如：

```python
with tf.Graph().as_default() as graph:
    v = tf.constant([1, 2])
    print(graph.as_graph_def())
```

输出内容：

```python
输入内容：
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
versions {
  producer: 24
}
```

还可以使用绑定图的会话的`graph_def`属性来获取图的序列化后的定义，例如：

```python
with tf.Graph().as_default() as graph:
    v = tf.constant([1, 2])
    print(graph.as_graph_def())
    
with tf.Session(graph=graph) as sess:
    sess.graph_def == graph.as_graph_def()  # True
```

**注意：**当会话中加入Op时，`sess.graph_def == graph.as_graph_def()`不再成立。在会话中graph_def会随着Op的改变而改变。

获取了图的定义之后，便可以去保存图。

**第二步：保存图的定义**

保存图的定义有两种方法，第一种为直接将图存为文本文件。第二种为使用Tensorflow提供的专门的保存图的方法，这种方法更加便捷。

- 方法一：

  直接创建一个文件保存图定义。如下：

  ```python
  with tf.Graph().as_default() as g:
      tf.Variable([1, 2], name='var')
  
      with tf.gfile.FastGFile('test_model.pb', 'wb') as f:
          f.write(g.as_graph_def().SerializeToString())
  ```

  `SerializeToString`是将str类型的图定义转化为二进制的proto数据。

- 方法二：

  使用Tensorflow提供的`tf.train.write_graph`进行保存。使用此方法还有一个好处，就是可以直接将图传入即可。用法如下：

  ```python
  tf.train.write_graph(
      graph_or_graph_def, # 图或者图定义
      logdir,  # 存储的文件路径
      name,   # 存储的文件名
      as_text=True)  # 是否作为文本存储
  ```

  例如，'方法一'中的图也可以这样保存：

  ```python
  with tf.Graph().as_default() as g:
      tf.Variable([1, 2], name='var')
      tf.train.write_graph(g, '', 'test_model.pb', as_text=False)
  ```

  这些参数`as_text`的值为`False`，即保存为二进制的proto数据。此方法等价于'方法一'。

  当`as_text`值为`True`时，保存的是str类型的数据。通常推荐为`False`。

### 1.2 图的读取

图的读取，即将保存的图的节点加载到当前的图中。当我们保存一个图之后，这个图可以再次被获取到。

图的获取步骤如下：

1. 从序列化的二进制文件中读取数据
2. 从读取到数据中创建`GraphDef`对象
3. 导入`GraphDef`对象到当前图中，创建出对应的图结构

具体如下：

```python
with tf.Graph().as_default() as new_graph:
    with tf.gfile.FastGFile('test_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
```

这里`ParseFromString`是protocal message的方法，用于将二进制的proto数据读取成`GraphDef`数据。`tf.import_graph_def`用于将一个图定义导入到当前的默认图中。

这里有一个问题，那就是导入图中的Op与Tensor如何获取到呢？`tf.import_graph_def`都已经帮我们想到这些问题了。这里，我们可以了解下`tf.import_graph_def`的用法：

```python
tf.import_graph_def(
    graph_def,  # 将要导入的图的图定义
    input_map=None,  # 替代导入的图中的Tensor
    return_elements=None,  # 返回指定的OP或Tensor(可以使用新的变量绑定)
    name=None,  # 被导入的图中Op与Tensor的name前缀 默认是'import'
    op_dict=None, 
    producer_op_list=None):
```

**注意**：当`input_map`不为None时，`name`必须不为空。

**注意**：当`return_elements`返回Op时，在会话中执行返回为`None`。

当然了，我们也可以使用`tf.Graph.get_tensor_by_name`与`tf.Graph.get_operation_by_name`来获取Tensor与Op，但要**注意加上name前缀**。

如果图在保存时，存为文本类型的proto数据，即`tf.train.write_graph`中的参数`as_text`为`True`时，获取图的操作稍有不同。即解码时不能使用`graph_def.ParseFromString`进行解码，而需要使用`protobuf`中的`text_format`进行操作，如下：

```python
from google.protobuf import text_format

with tf.Graph().as_default() as new_graph:
    with tf.gfile.FastGFile('test_model.pb', 'r') as f:
        graph_def = tf.GraphDef()
        # graph_def.ParseFromString(f.read())
        text_format.Merge(f.read(), graph_def)
        tf.import_graph_def(graph_def)
```

## 2. 变量存取

变量存储是把模型中定义的变量存储起来，不包含图结构。另一个程序使用时，首先需要重新创建图，然后将存储的变量导入进来，即模型加载。变量存储可以脱离图存储而存在。

变量的存储与读取，在Tensorflow中叫做检查点存取，变量保存的文件是检查点文件(checkpoint file)，扩展名一般为.ckpt。使用`tf.train.Saver()`类来操作检查点。

### 2.1 变量存储

变量是在图中定义的，但实际上是会话中存储了变量，即我们在运行图的时候，变量才会真正存在，且变量在图的运行过程中，值会发生变化，所以我们需要**在会话中保存变量**。保存变量的方法是`tf.train.Saver.save()`。

这里需要注意，通常，我们可以在图定义完成之后初始化`tf.train.Saver()`。`tf.train.Saver()`在图中的位置很重要，在其之后的变量不会被存储在当前的`Save`对象控制。

创建Saver对象之后，此时并不会保存变量，我们还需要指定会话运行到什么时候时再去保存变量，需要使用`tf.train.Saver.save()`进行保存。

例如：

```python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1)
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(var1.assign_add(1))
    saver.save(sess, './model.cpkt')
```

`tf.train.Saver.save()`有两个必填参数，第一个是会话，第二个是存储路径。执行保存变量之后，会在指定目录下生成四个文件。分别是checkpoint文件、model.ckpt.data-00000-of-00001文件、model.ckpt.index文件、model.ckpt.meta文件。这四个文件的作用分别是：

- checkpoint：为文本文件。记录当前模型最近的5次变量保存的路径信息。这里我们只保存了一个模型，所有只有一次模型保存信息，也是当前模型的信息。
- model.ckpt.data-00000-of-00001：保存了当前模型变量的数据。
- model.ckpt.meta：保存了`MetaGraphDef`，可以用于恢复saver对象。
- model.ckpt.index：辅助model.ckpt.meta的数据文件。

#### 循环迭代算法时存储变量

实际中，我们训练一个模型，通常需要迭代较多的次数，迭代的过程会用去很多的时间，为了避免出现意外情况（例如断电、死机），我们可以每迭代一定次数，就保存一次模型，如果出现了意外情况，就可以快速恢复模型，不至于重新训练。

如下，我们需要迭代1000次模型，每100次迭代保存一次：

```python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1, 1001):
        sess.run(var1.assign_add(1))
        if i % 100 == 0:
            saver.save(sess, './model2.cpkt')
            
    saver.save(sess, './model2.cpkt')   
```

这时候每次存储都会覆盖上次的存储信息。但我们存储的模型并没有与训练的次数关联起来，我们并不知道当前存储的模型是第几次训练后保存的结果，如果中途出现了意外，我们并不知道当前保存的模型是什么时候保存下的。所以通常的，我们还需要将训练的迭代的步数进行标注。在Tensorflow中只需要给save方法加一个参数即可，如下：

```python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1, 1001):
        sess.run(var1.assign_add(1))
        if i % 100 == 0:
            saver.save(sess, './model2.cpkt', global_step=i)  # 
            
    saver.save(sess, './model2.cpkt', 1000)
```

这里，我们增加了给`saver.save`增加了`global_step`的参数，这个参数可以是一个0阶Tensor或者一个整数。之后，我们生成的保存变量文件的文件名会加上训练次数，并同时保存最近5次的训练结果，即产生了16个文件。包括这五次结果中每次训练结果对应的data、meta、index三个文件，与一个checkpoint文件。

#### 检查保存的变量信息 

TensorFlow提供了很多种方法查看检查点文件中张量，例如可以使用`tf.train.NewCheckpointReader`。用法如下：

~~~python
reader = tf.train.NewCheckpointReader('./model.ckpt')

tensor = reader.get_tensor('TensorName')
~~~

其中，使用`has_tensor`可以查看是否存在某个张量，使用`get_tensor`可以获取某个张量

或者也可以利用包`inspect_checkpoint`中的方法对检查点文件操作，如下：

~~~python
# 导入 inspect_checkpoint 库
from tensorflow.python.tools import inspect_checkpoint as chkp

#  打印checkpoint文件中所有的tensor
chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# 打印checkpoint文件中，tensor_name为v1的张量
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]
~~~

### 2.2 变量读取

变量读取，即加载模型数据，为了保证数据能够正确加载，必须首先将图定义好，而且必须与保存时的图定义一致。这里“一致”的意思是图相同，对于Python句柄等Python环境中的内容可以不同。

下面我们恢复上文中保存的图：

```python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    saver.restore(sess, './model.ckpt')
    print(sess.run(var))  # >> 1001
```

**注意**：当使用restore方法恢复变量时，可以不用初始化方法初始化变量，但这仅仅在所有变量均会被恢复时可用，当存在部分变量没有恢复时，必须使用变量初始化方法进行初始化。

除了使用这种方法恢复变量以外，还可以借助meta graph，例如：

```python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./log/model.ckpt.meta')
    saver.restore(sess, './model.ckpt')
    print(sess.run(var))  # >> 1001
```

可以看到meta graph包含了saver对象的信息，不仅如此，使用meta数据进行恢复时，meta数据中也包含了图定义。所以在我们没有图定义的情况下，也可以使用meta数据进行恢复，例如：

```python
with tf.Graph().as_default() as g:
    op = tf.no_op()

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./log/model.ckpt.meta')
    saver.restore(sess, './model.ckpt')
    print(sess.run(g.get_tensor_by_name('var:0')))  # >> 1001
```

所以，通过meta数据不仅能够恢复变量，也能够恢复图定义。

训练模型时，往往每过一段时间就会保存一次模型，这时候checkpoint文件会记录最近五次的ckpt文件名记录，所以在恢复数据时，当存在多个模型保存的文件，为了简便可以使用`tf.train.get_checkpoint_state()`来读取checkpoint文件并获取最新训练的模型。其返回一个`CheckpointState`对象，可以使用这个对象`model_checkpoint_path`属性来获得最新模型的路径。例如：

```python
...
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state('./')
    saver.restore(sess1, ckpt.model_checkpoint_path)
```

### 2.3 注意事项

**注意事项1**：

当一个图对应多个会话时，在不同的会话中使用图的Saver对象保存变量时，并不会互相影响。

例如，对于两个会话`sess1`、`sess2`分别操作一个图`g`中的变量，并存储在不同文件中：

```python
with tf.Graph().as_default() as g:
    var = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
    
with tf.Session(graph=g) as sess1:
    sess1.run(tf.global_variables_initializer())
    sess1.run(var.assign_add(1))
    
    saver.save(sess1, './model1/model.ckpt')
    
    
with tf.Session(graph=g) as sess2:
    sess2.run(tf.global_variables_initializer())
    sess2.run(var.assign_sub(1))
    
    saver.save(sess2, './model2/model.ckpt')
```

当我们分别加载变量时，可以发现，并没有互相影响。如下：

```python
with tf.Session(graph=g) as sess3:
    sess3.run(tf.global_variables_initializer())
    saver.restore(sess3, './model1/model.ckpt')
    print(sess3.run(var))  # >> 2
    
with tf.Session(graph=g) as sess4:
    sess4.run(tf.global_variables_initializer())
    saver.restore(sess4, './model1/model.ckpt')
    print(sess4.run(var))  # >> 0
```

**注意事项2**：

当我们在会话中恢复变量时，必须要求会话所绑定的图与所要恢复的变量所代表的图一致，这里我们需要知道什么样的图时一致。

例如：

```python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    
with tf.Graph().as_default() as g2:
    b = tf.Variable([1, 2, 3])
```

上面这两个图是一致的，虽然其绑定的Python变量不同。

```python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    
with tf.Graph().as_default() as g2:
    b = tf.Variable([1, 2, 3], name='var')
```

上面这两个图是不一样的，因为使用了不同的`name`。

```python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3], name='a')
    b = tf.Variable([4, 5], name='b')
    
with tf.Graph().as_default() as g2:
    c = tf.Variable([4, 5], name='b')
    d = tf.Variable([1, 2, 3], name='a')
```

这两个图是一致的。

```python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    b = tf.Variable([4, 5])
    
with tf.Graph().as_default() as g2:
    c = tf.Variable([4, 5])
    d = tf.Variable([1, 2, 3])
```

这两个图是不一致。看起来，两个图一模一样，然而Tensorflow会给每个没有`name`的Op进行命名，两个图由于均使用了2个Variable，并且都没有主动命名，所以在`g1`中a的`name`与`g2`中c的`name`不同，`g1`中b的`name`与`g2`中d的`name`不同。

**注意事项3**：

当两个图只有部分结构一样时，可以恢复部分变量，如下：

~~~python
# g1 图如下，此时我们如果保存了 g1 中的变量那么在其他结构相似的图中也可以恢复
with tf.Graph().as_default() as g1:
    a = tf.Variable(10.)
    b = tf.Variable(20.)
    c = a + b
    saver1 = tf.train.Saver()
    
# g2 图中变量a、b与 g1 中完成一样，此时我们在`tf.train.Saver`中传入需要
# 管理的变量a、b，那么我们就可以在会话中恢复a、b
with tf.Graph().as_default() as g2:
    a = tf.Variable(10.)
    b = tf.Variable(10.)
    d = tf.Variable(5.)
    c = a + b
    saver2 = tf.train.Saver(var_list=[a, b])    
~~~

## 3. 变量的值冻结到图中

当我们训练好了一个模型时，就意味着模型中的变量的值确定了，不在需要改变了。上述方法中我们需要分别将图与变量保存（或者保存变量时也保存了meta graph），这里我们也可以使用更简单的方法完成。即可以将所有变量固化成常量，随图一起保存。这样使用起来也更加简便，我们只需要导入图即可完成模型的完整导入。利用固化后的图参与构建新的图也变得容易了（不需要单独开启会话并加载变量了）。

实现上述功能，需要操作GraphDef中的节点，TensorFlow为我们提供了相关的API：`convert_variables_to_constants`。

`convert_variables_to_constants`在`tensorflow.python.framework.graph_util`中，用法如下：

```python
# 返回一个新的图
convert_variables_to_constants(
    sess,  # 会话
    input_graph_def,  # 图定义
    output_node_names,  # 输出节点(不需要的节点在生成的新图将被排除)(注意是节点而不是Tensor)
    variable_names_whitelist=None,  # 指定的variable转成constant
    variable_names_blacklist=None)  # 指定的variable不转成constant
```

例如：

将变量转化为常量，并存成图。

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Graph().as_default() as g:
    my_input = tf.placeholder(dtype=tf.int32, shape=[], name='input')
    var = tf.get_variable(name='var', shape=[], dtype=tf.int32)
    output = tf.add(my_input, var, name='output')

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(var.assign(10))
    
    # 将变量转化为常量，返回一个新的图
    new_graph = graph_util.convert_variables_to_constants(
        sess, 
        sess.graph_def, 
        output_node_names=['output'])
    # 将新的图保存
    tf.train.write_graph(new_graph, '', 'graph.pb', as_text=False)
```

导入刚刚序列化的图：

```python
with tf.Graph().as_default() as new_graph:
    x = tf.placeholder(dtype=tf.int32, shape=[], name='x')
    with tf.gfile.FastGFile('graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def, 
            input_map={'input:0': x}, 
            return_elements=['output:0'])
        
with tf.Session(graph=new_graph) as sess:
    print(sess.run(g_out[0], feed_dict={x: 5}))  # >> 15
```

## 4. SavedModel

上述模型存取的方法是有局限性的，仅仅适合于Python环境下的TensorFlow模型存取，而且模型的图与变量的保存是分开进行的，这在训练模型的时候是比较方便的，但适用范围较窄且麻烦。而SavedModel的适用范围更广，它是一种与语言无关，可恢复的密封式序列化格式。它可以同时保存图与变量，操作也更加方便。SavedModel 可让较高级别的系统和工具创建、使用和变换 TensorFlow 模型。TensorFlow 提供了多种与 SavedModel 交互的机制，如 tf.saved_model API、Estimator API 和 CLI，但通常只在模型训练完成之后才使用SaveModel保存模型。

### 4.1 模型保存

SavedModel的用法比较简单，其功能模块在TensorFlow的`saved_model`下，主要功能包含在`builder`和、`loader`两个子模块中，分别用于保存和加载SavedModel。保存模型的用法如下：

~~~python
...
builder = tf.saved_model.builder.SavedModelBuilder('/tmp/save_path')
with tf.Session() as sess:
    ...
    builder.add_meta_graph_and_variables(sess, ['train'])
    builder.save()
~~~

可以看到用法很简单，首先构建`SavedModelBuilder`实例对象，构建时需要传入保存路径（注意这个路径文件夹中必须是没有文件的），然后调用`add_meta_graph_and_variables`方法将元图与变量加入`SavedModelBuilder`实例对象中，此时需要传入`Session`以及`tags`标记，最后调用`save`方法完成序列化。这里需要注意的是`tags`是必须的，用来标记存储的元图和变量，因为我们也可以使用一个saved_model多次存储模型，这时候需使用`tag`加以区分。例如，我们需要存两个元图，如下：

~~~python
...
builder = tf.saved_model.builder.SavedModelBuilder('/tmp/save_path')
with tf.Session() as sess:
    ...
    builder.add_meta_graph_and_variables(sess, ['train'])
    
with tf.Session() as sess:
    ...
    builder.add_meta_graph(['serve'])

builder.save()
~~~

这里需要注意，`add_meta_graph_and_variables`方法只能调用一次，即变量只能存储一次，但元图可以存储很多次。

TensorFlow的saved_model模块下的`tag_constants`内置了常用`tag`常量，通常的我们使用这些常量即可，一般不需要自己自定义，这样可以使我们的程序设计更加一致。如下：

~~~python
tf.saved_model.tag_constants.SERVING  # 对应字符串 `serve`
tf.saved_model.tag_constants.GPU  # 对应字符串 `gpu`
tf.saved_model.tag_constants.TPU  # 对应字符串`tpu`
tf.saved_model.tag_constants.TRAINING  # 对应字符串 `train`
~~~

**注意**：当保存模型时使用了多个`tag`，那么在恢复模型时也需要多个一样的`tag`。

### 4.2 模型恢复

模型的恢复不依赖于编程语言，在Python中的用法如下：

~~~python
...
with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ['serve'], '/tmp/save_path')
	...
~~~

即调用`loader`模块下的`load`方法，传入`Session`对象，需要恢复的模型的`tags`，以及模型的保存路径即可。恢复完成之后，可以通过图的一些方法获取到`op`与`tensor`。											

通常的，我们一般使用Python语言构建与训练模型，完成之后需要将模型部署到服务器，这时候我们可以使用TensorFlow Serving通过CLI来恢复元图与变量，如下：

~~~shell
tensorflow_model_server 
	--port=port-numbers 
	--model_name=your-model-name 
	--model_base_path=your_model_base_path
~~~

### 4.3 Signature

上述使用SavedModel存取模型时存在一个问题就是模型的入口与出口部分我们没有显式的标记出来，虽然我们可以在加载完成模型之后使用诸如`get_operation_by_name`的方法获取到模型中节点或张量，但这往往不太方便，也使得模型的设计与部署的耦合性较高。如何使得模型的训练时与应用时在模型的使用上解耦呢？SavedModel提供了Signature来解决这一问题。

具体来讲就是在保存模型时，对模型的输入、输出张量签名。类似于给张量一个别名，用法如下：

~~~python
# 构建一个图，图中的op使用的name如x、y、z不宜在部署时使用
with tf.Graph().as_default() as g:
    x = tf.placeholder(dtype=tf.int32, shape=[], name='x')
    y = tf.placeholder(dtype=tf.int32, shape=[], name='y')
    z = tf.add(x, y, name='z')
    
builder = tf.saved_model.builder.SavedModelBuilder('/tmp/save_path')
# 给图中输入输出张量构建一个签名
inputs = {'inputs_x': tf.saved_model.utils.build_tensor_info(x), 
          'inputs_y': tf.saved_model.utils.build_tensor_info(y)}
outputs = {'outputs': tf.saved_model.utils.build_tensor_info(z)}
signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    builder.add_meta_graph_and_variables(
        sess, 
        [tf.saved_model.tag_constants.SERVING],
        {'my_signature': signature})  # 保存模型时，加上签名的部分
    builder.save()
~~~

上面的代码是模型保存的部分，模型加载的时候就可以使用签名方便的获取输入、输出张量了。如下：

~~~python
with tf.Graph().as_default() as new_graph:
    with tf.Session(graph=new_graph) as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess, 
            [tf.saved_model.tag_constants.SERVING], 
            '/tmp/save_path')
        # 获取到签名
        signature = meta_graph_def.signature_def['my_signature']
        # 根据签名可获取TensorInfo，然后利用相关方法可以得到对应的Tensor
        inputs_x = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature.inputs['inputs_x'])
        inputs_y = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature.inputs['inputs_y'])
        outputs = tf.saved_model.utils.get_tensor_from_tensor_info(
            signature.outputs['outputs'])
        
        print(sess.run(outputs, feed_dict={inputs_x: 5, inputs_y: 10}))  # >>> 15
~~~

上面的代码看起来有些复杂，这里简单的解释一下，首先使用`tf.saved_model.loader.load`载入元图与变量，这时候返回元图的定义。元图定义中包含了签名信息，所以访问元图的`signature_def`属性可以得到所有签名，然后取出对应的我们定义的签名即可。签名信息中包含了`inputs`与`outputs`等属性，存储了`TensorInfo`对象，这时候使用`tf.saved_model.utils.get_tensor_from_tensor_info`方法可以根据`TensorInfo`取出对应的张量。

**注意**：签名可以设置多个，代表一个图中不同的输入输出。这也是为什么需要给签名一个名字的原因。

就像`tags`一样，SavedModel签名也有一些内置的值，推荐使用这些值，如下：

~~~python
tf.saved_model.signature_constants.CLASSIFY_INPUTS  # `inputs`
tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME  # `tensorflow/serving/classify`
...
~~~

#### 简化版的带Signature的模型保存

上述代码写起来比较复杂，事实上，在模型保存部分，还有更简单的写法，但这种写法只支持保存一个签名。用法如下：

~~~python
# 假设x、y与z分别为输入输出张量
...
with tf.Session() as sess:
	...
    tf.saved_model.simple_save(
        sess,
        '/tmp/save_path', 
        inputs={'inputs_x': x, 'inputs_y': y}, outputs={'output': z})  
~~~

可以看到上面的写法屏蔽了很多细节，要更加清晰、明了，但要注意的是，其在保存模型的时候也是使用了`tags`的，其`tag`为`tf.saved_model.tag_constants.SERVING`。签名的名字默认使用的是`tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`。这些信息在模型读取时依然会用到。

## 5. 使用 CLI 检查并执行 SavedModel

一般的成功安装TensorFlow之后，也同样安装好了`saved_model_cli`，这是一个终端中检查并执行SavedModel的工具。主要功能包括查看SavedModel签名以及执行SavedModel中的图。目前支持以下三个命令：

* `show`：显示所有关于指定的SavedModel的全部信息。
* `run`：输入数据到元图并执行得到结果
* `scan`：浏览

### 5.1 show 命令

使用show命令可以显示指定的SavedModel所有的`tag-set`，用法如下：

~~~shell
$ saved_model_cli show --dir savedmodel_path
~~~

`savedmodel_path`是指SavedModel所在的文件夹路径。执行之后显示如下结果：

~~~txt
The given SavedModel contains the following tag-sets:
serve
serve, gpu
~~~

代表模型中有两个`tag-set`，一个是`serve`，另一个是`serve, gpu`。

我们可以根据`tag-set`获取到指定的`tag-set`下的`MetaGraphDef`，这是非常方便的，因为提供这样的工具可以在我们不知道模型详细结构的情况下方便的调用模型。

使用第二个`tag-set`获取`MetaGraphDef`的命令如下：

~~~shell
$ saved_model_cli show --dir savedmodel_path --tag_set serve,gpu
~~~

执行后显示结果如下：

~~~tex
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
~~~

这代表保存的某一个元图中包含了6个`SignatureDef`，也就是说我们可以在这个图中使用6个输入输出。

如果想查看某一个`SignatureDef`中输入、输出的签名，则可以执行如下命令：

~~~shell
$ saved_model_cli show --dir savedmodel_path --tag_set serve,gpu --signature_def serving_default
~~~

执行后的结果如下：

~~~tex
MetaGraphDef with tag-set: 'serve, gpu' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs_x'] tensor_info:
        dtype: DT_INT32
        shape: ()
        name: x:0
    inputs['inputs_y'] tensor_info:
        dtype: DT_INT32
        shape: ()
        name: y:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_INT32
        shape: ()
        name: z:0
  Method name is: tensorflow/serving/predict
~~~

这个结果详细展示了输入、输出签名等信息。可以方便我们调用。

**提示**：可以使用`saved_model_cli show --dir /tmp/saved_model_dir --all`命令查看所有信息。

### 5.2 run命令

调用`run`命令可以运行计算图，并可以将结果显示或保存。同时，为了使模型运行，我们也需要传递输入数据进入模型，`run`命令提供了以下三种方式将输入传递给模型：

* `--inputs`：接受numpy ndarray文件数据作为输入。
* `--input_exprs`：接受Python表达式作为输入，即将命令行的输入解析为了Python表达式。
* `--input_examples`：接受`tf.train.Example`作为输入。

**--inputs**

INPUT 采用以下格式之一：

- `<input_key>=<filename>`
- `<input_key>=<filename>[<variable_name>]`

 您可能会传递多个 INPUT。如果您确实要传递多个输入，请使用分号分隔每个 INPUT。

`saved_model_cli` 使用 `numpy.load` 加载文件名。文件名可以是以下任何一种格式：

- `.npy`
- `.npz`
- pickle 格式

**--inputs_exprs**

要通过 Python 表达式传递输入，请指定 `--input_exprs` 选项。这对于您目前没有数据文件的情形而言非常有用，但最好还是用一些与模型的 `SignatureDef` 的 dtype 和形状匹配的简单输入来检查模型。例如：

```shell
`<input_key>=[[1],[2],[3]]`
```

除了 Python 表达式之外，您还可以传递 numpy 函数。例如：

```shell
`<input_key>=np.ones((32,32,3))`
```

（请注意，`numpy` 模块已可作为 `np` 提供。）

**--inputs_examples**

要将 `tf.train.Example` 作为输入传递，请指定 `--input_examples` 选项。对于每个输入键，它都基于一个字典列表，其中每个字典都是 `tf.train.Example` 的一个实例。不同的字典键代表不同的特征，而相应的值则是每个特征的值列表。例如：

```shell
`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
```

例如：

您的模型只需添加 `x1` 和 `x2` 即可获得输出 `y`。模型中的所有张量都具有形状 `(-1, 1)`。您有两个 `npy` 文件：`/tmp/my_data1.npy`，其中包含多维数组 `[[1], [2], [3]]`。`/tmp/my_data2.npy`，其中包含另一个多维数组 `[[0.5], [0.5], [0.5]]`。：

~~~shell
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve --signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy --outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
~~~

以下命令用 Python 表达式替换输入 `x2`：

~~~shell
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve --signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npz[x] --input_exprs 'x2=np.ones((3,1))'
Result for output key y:
[[ 2]
 [ 3]
 [ 4]]
~~~

**保存输出**

默认情况下，SavedModel CLI 将输出写入 stdout。如果目录传递给 `--outdir` 选项，则输出将被保存为在指定目录下以输出张量键命名的 npy 文件。

使用 `--overwrite` 覆盖现有的输出文件。

#### SavedModel 目录结构

当您以 SavedModel 格式保存模型时，TensorFlow 会创建一个由以下子目录和文件组成的 SavedModel 目录：

```
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```

其中：

- `assets` 是包含辅助（外部）文件（如词汇表）的子文件夹。资源被复制到 SavedModel 的位置，并且可以在加载特定的 `MetaGraphDef` 时读取。
- `assets.extra` 是一个子文件夹，其中较高级别的库和用户可以添加自己的资源，该资源与模型共存，但不会被图加载。此子文件夹不由 SavedModel 库管理。
- `variables` 是包含 `tf.train.Saver` 的输出的子文件夹。
- `saved_model.pb` 或 `saved_model.pbtxt` 是 SavedModel 协议缓冲区。它包含作为 `MetaGraphDef` 协议缓冲区的图定义。

单个 SavedModel 可以表示多个图。在这种情况下，SavedModel 中所有图共享一组检查点（变量）和资源。例如，下图显示了一个包含三个 `MetaGraphDef` 的 SavedModel，它们三个都共享同一组检查点和资源：

![SavedModel 代表检查点、资源以及一个或多个 MetaGraphDef](./images/SavedModel.svg)

每个图都与一组特定的标签相关联，可在加载或恢复操作期间方便您进行识别。r