TensorFlow中所有的`Tensor`、`Operation`均有`name`属性，是其唯一标识符。在Python中，一个变量可以绑定一个对象，同样的也可以绑定一个`Tensor`或`Operation`对象，但这个变量并不是`Tensor`或`Operation`的唯一标识符。

例如，下面的代码中，Python变量`a`首先指向`tf.constant([1, 2, 3])`，之后又指向了`tf.Variable([4, 5, 6])`，那么最终图中仍然包含两块内容，但是使用`a`只能获取到后定义的`tf.Variable([4, 5, 6])`，：

```python
a = tf.constant([1, 2, 3])
a = tf.Variable([4, 5, 6])
```

这时候就只能使用TensorFlow中的`name`解决问题。除此以外，使用`name`可以使得TensorFlow使用某一个编程语言编写的图在其他语言中可以方便操作，实现跨语言环境的兼容。使用`name`也有助于在TensorBoard中可视化。在之前的操作中，我们一般不指定`Tensor`、`Operation`的`name`属性，但这并不代表其没有`name`属性，而是使用了默认的命名。

除此以外，Tensorflow也有作用域(scope)，用来管理`Tensor`、`Operation`的`name`以及完成一些可复用的操作。Tensorflow的作用域分为两种，一种是`variable_scope`，另一种是`name_scope`。

## 1. name

`Tensor`与`Operation`均有`name`属性，但我们只能给`Operation`进行主动命名，`Tensor`的`name`由`Operation`根据自己的`name`与输出数量进行命名（所有的`Tensor`均由`Operation`产生）。

例如，我们定义一个常量，并赋予其`name`：

```python
a = tf.constat([1, 2, 3], name='const')
```

这里我们给常量`Op`定义了一个`name`为`const`。`a`是常量`Op`的返回值（输出），是一个张量`Tensor`对象，所以`a`也有自己的`name`，为`const:0`。

可以看到：`Operation`的`name`是我们进行命名的，其输出的张量在其后增加了冒号与索引，是TensorFlow根据`Operation`的`name`进行命名的。

### 1.1 Op的name命名规范

首先`Tensor`对象的`name`我们并不能直接进行操作，我们只能给`Op`设置`name`。`Op`的命令规范规范是：**由数字、字母、下划线组成，不能以下划线开头，且不区分大小写**。

**正确**的命名方式如下：

```python
a1 = tf.constant([1, 2, 3], name='const')
a2 = tf.Variable([1, 2, 3], name='123')
a3 = tf.add(1, 2, name='const_')
```

**错误**的命名方式如下：

```python
a1 = tf.constant([1, 2, 3], name='_const')
a2 = tf.Variable([1, 2, 3], name='/123')
a3 = tf.add(1, 2, name='const:0')
```

每个`Op`都有`name`属性，可以通过属性查看`name`值，例如：

```python
# 返回一个什么都不做的Op
op = tf.no_op(name='hello')

print(op.name)  # hello
```

这里我们列举了一个空`Op`，而没有使用常用的诸如`tf.add`这样的`Op`，是因为默认的大部分`Op`都返回对应的张量，而不是`Op`对象，但`tf.no_op`函数返回的是`Op`对象，是一个特例。

### 1.2 Tensor的name构成

大部分的`Op`会有返回值，其返回值一般是一个或多个`Tensor`。`Tensor`的`name`并不来源于我们设置，只能来源于生成它的`Op`，所以`Tensor`的`name`是由`Op`的`name`所决定的。

`Tensor`的`name`构成很简单，即在对应的`Op`的`name`之后加上输出索引。即由以下三部分构成：

1. 生成此`Tensor`的`op`的`name`；
2. 冒号；
3. `op`输出内容的索引，索引默认从`0`开始。

例如：

```python
a = tf.constant([1, 2, 3], name='const')

print(a.name)  # const:0
```

这里，我们设置了常量`Op`的`name`为`const`，这个`Op`会返回一个`Tensor`，所以返回的`Tensor`的`name`就是在其后加上冒号与索引。由于只有一个输出，所以这个输出的索引就是`0`。

对于两个或多个的输出，其索引依次增加：如下：

```python
key, value = tf.ReaderBase.read(..., name='read')

print(key.name)  # read:0
print(value.name)  # read:1
```

### 1.3 Op与Tensor的默认name

当我们不去设置`Op`的`name`时，TensorFlow也会默认设置一个`name`，这也正是`name`为可选参数的原因。默认`name`往往与`Op`的类型相同（默认的`name`并无严格的命名规律）。

例如：

```python
a = tf.add(1, 2)  
# op name为 `Add`
# Tensor name 为 `Add:0`

b = tf.constant(1)
# op name 为 `Const`
# Tensor name 为 `Const:0`

c = tf.divide(tf.constant(1), tf.constant(2))
# op name 为 `truediv`
# Tensor name 为 `truediv:0`
```

**注意**：还有一些特殊的`Op`，我们没法指定其`name`，只能使用默认的`name`，例如：

```python
init = tf.global_variables_initializer()
print(init.name)  # init
```

### 1.4 重复name的处理方式

虽然`name`作为唯一性的标识符，但TensorFlow并不会强制要求我们必须设置完全不同的`name`，这并非说明`name`可以重复，而是TensorFlow通过一些方法避免了`name`重复。

当出现了两个`Op`设置相同的`name`时，TensorFlow会自动给后面的`op`的`name`加一个后缀。如下：

```python
a1 = tf.add(1, 2, name='my_add')
a2 = tf.add(3, 4, name='my_add')

print(a1.name)  # my_add:0
print(a2.name)  # my_add_1:0
```

后缀由下划线与索引组成（注意与`Tensor`的`name`属性后的缀冒号索引区分）。从重复的第二个`name`开始加后缀，后缀的索引从`1`开始。

当我们不指定`name`时，使用默认的`name`也是相同的处理方式：

```python
a1 = tf.add(1, 2)
a2 = tf.add(3, 4)

print(a1.name)  # Add:0
print(a2.name)  # Add_1:0
```

不同操作之间是有相同的`name`也是如此：

```python
a1 = tf.add(1, 2, name='my_name')
a2 = tf.subtract(1, 2, name='my_name')

print(a1.name)  # >>> my_name:0
print(a2.name)  # >>> my_name_1:0
```

**注意**：设置`name`时，如果重复设置了一样的`name`，并不会抛出异常，也不会有任何提示，TensorFlow会自动添加后缀。为了避免出现意外情况，通常的`name`设置必须不重复。

### 1.5 不同图中相同操作name

当我们构建了两个或多个图的时候，如果这些图中有相同的操作或者相同的`name`时，并不会互相影响。如下：

```python
g1 = tf.Graph()
with g1.as_default():
    a1 = tf.add(1, 2, name='add')
    print(a1.name)  # add:0

g2 = tf.Graph()
with g2.as_default():
    a2 = tf.add(1, 2, name='add')
    print(a2.name)  # add:0
```

可以看到两个图中的`name`互不影响。并没有关系。

**小练习**：

> 以下操作均为一个图中的`op`，请写出以下操作对应中的`Op`与对应生成的`Tensor`的`name`：
>
> - `tf.constant([1, 2])`
> - `tf.add([1, 2], [3, 4], name='op_1')`
> - `tf.add([2, 3], [4, 5], name='op_1')`
> - `tf.mod([1, 3], [2, 4], name='op_1')`
> - `tf.slice([1, 2], [0], [1], name='123')`

## 2. 通过name获取Op与Tensor

上文，我们介绍了`name`可以看做是`Op`与`Tensor`的唯一标识符。所以可以通过一些方法利用`name`获取到`Op`与`Tensor`。

例如，一个计算过程如下：

```python
g1 = tf.Graph()
with g1.as_default():
	a = tf.add(3, 5)
	b = tf.multiply(a, 10)
    
with tf.Session(graph=g1) as sess:
    sess.run(b)  # 80
```

上述图，我们使用了Python变量`b`获取对应的操作，我们也可以使用如下方式获取，两种方式结果一样：

```python
g1 = tf.Graph()
with g1.as_default():
    tf.add(3, 5, name='add')
    tf.multiply(g1.get_tensor_by_name('add:0'), 10, name='mul')
    
with tf.Session(graph=g1) as sess:
    sess.run(g1.get_tensor_by_name('mul:0'))  # 80
```

这里使用了`tf.Graph.get_tensor_by_name`方法。可以根据`name`获取`Tensor`。其返回值是一个`Tensor`对象。这里要注意`Tensor`的`name`必须写完整，请勿将`Op`的`name`当做是`Tensor`的`name`。

同样的，利用`name`也可以获取到相应的`Op`，这里需要使用`tf.Graph.get_operation_by_name`方法。上述例子中，我们在会话中运行的是乘法操作的返回值`b`。运行`b`的时候，与其相关的依赖，包括乘法`Op`也运行了，当我们不需要返回值时，我们在会话中可以直接运行`Op`，而无需运行`Tensor`。

例如：

```python
g1 = tf.Graph()
with g1.as_default():
    tf.add(3, 5, name='add')
    tf.multiply(g1.get_tensor_by_name('add:0'), 10, name='mul')
    
with tf.Session(graph=g1) as sess:
    sess.run(g1.get_operation_by_name('mul'))  # None
```

在会话中，`fetch`一个`Tensor`，会返回一个`Tensor`，`fetch`一个`Op`，返回`None`。

**小练习**：

> 请自己尝试实现上述代码。

## 3. name_scope

随着构建的图越来越复杂，直接使用`name`对图中的节点命名会出现一些问题。比如功能近似的节点`name`可能命名重复，也难以通过`name`对不同功能的节点加以区分，这时候如果可视化图会发现将全部节点展示出来是杂乱无章的。为了解决这些问题，可以使用`name_scope`。

`name_scope`可以为其作用域中的节点的`name`添加一个或多个前缀，并使用这些前缀作为划分内部与外部`op`范围的标记。同时在TensorBoard可视化时可以作为一个整体出现（也可以展开）。并且`name_scope`可以嵌套使用，代表不同层级的功能区域的划分。

`name_scope`使用`tf.name_scope()`创建，返回一个上下文管理器。`name_scope`的参数`name`可以是字母、数字、下划线，不能以下划线开头。类似于`Op`的`name`的命名方式。

`tf.name_scope()`的详情如下：

```python
tf.name_scope(
    name,  # 传递给Op name的前缀部分
    default_name=None,  # 默认name
    values=None)  # 检测values中的tensor是否与下文中的Op在一个图中
```

**注意**：`values`参数可以不填。当存在多个图时，可能会出现在当前图中使用了在别的图中的`Tensor`的错误写法，此时如果不在`Session`中运行图，并不会抛出异常，而填写到了`values`参数的中的`Tensor`都会检测其所在图是否为当前图，提高安全性。

使用`tf.name_scope()`的例子：

```python
a = tf.constant(1, name='const')
print(a.name)  # >> const:0

with tf.name_scope('scope_name') as name:
  	print(name)  # >> scope_name/
  	b = tf.constant(1, name='const')
    print(b.name)  # >> scope_name/const:0
```

在一个`name_scope`的作用域中，可以填写`name`相同的`Op`，但TensorFlow会自动加后缀，如下：

```python
with tf.name_scope('scope_name') as name:
    a1 = tf.constant(1, name='const')
    print(b.name)  # scope_name/const:0
    a2 = tf.constant(1, name='const')
    print(c.name)  # scope_name/const_1:0
```

#### 多个name_scope

我们可以指定任意多个`name_scope`，并且可以填写相同`name`的两个或多个`name_scope`，但TensorFlow会自动给`name_scope`的`name`加上后缀：

如下：

```python
with tf.name_scope('my_name') as name1:
  	print(name1)  # >> my_name/
    
with tf.name_scope('my_name') as name2:
  	print(name2)  #>> my_name_1/
```

### 3.1 多级name_scope

`name_scope`可以嵌套，嵌套之后的`name`包含上级`name_scope`的`name`。通过嵌套，可以实现多样的命名，如下：

```python
with tf.name_scope('name1'):
  	with tf.name_scope('name2') as name2:
      	print(name2)  # >> name1/name2/
```

不同级的`name_scope`可以填入相同的`name`（不同级的`name_scope`不存在同名），如下：

```python
with tf.name_scope('name1') as name1:
    print(name1)  # >> name1/
  	with tf.name_scope('name1') as name2:
      	print(name2)  # >> name1/name1/
```

在多级`name_scope`中，`Op`的`name`会被累加各级前缀，这个前缀取决于所在的`name_scope`的层级。不同级中的`name`因为其前缀不同，所以不可能重名，如下：

```python
with tf.name_scope('name1'):
  	a = tf.constant(1, name='const')
    print(a.name)  # >> name1/const:0
  	with tf.name_scope('name2'):
      	b = tf.constant(1, name='const')
    	print(b.name)  # >> name1/name2/const:0
```

### 3.2 name_scope的作用范围

使用`name_scope`可以给`Op`的`name`加前缀，但不包括`tf.get_variable()`创建的变量，如下所示：

```python
with tf.name_scope('name'):
  	var = tf.Variable([1, 2], name='var')
    print(var.name)  # >> name/var:0
    var2 = tf.get_variable(name='var2', shape=[2, ])
    print(var2.name)  # >> var2:0
```

这是因为`tf.get_variable`是一种特殊的操作，其只能与`variable_scope`配合完成相应功能。

### 3.3 注意事项

1. 从外部传入的`Tensor`，并不会在`name_scope`中加上前缀。例如：

   ```python
   a = tf.constant(1, name='const')
   with tf.name_scope('my_name', values=[a]):
       print(a.name)  # >> const:0
   ```

2. `Op`与`name_scope`的`name`中可以使用`/`，但`/`并不是`name`的构成，而是区分命名空间的符号，不推荐直接使用`/`。

3. `name_scope`的`default_name`参数可以在函数中使用。`name_scope`返回的`str`类型的`scope`可以作为`name`传给函数中返回`Op`的`name`，这样做的好处是返回的`Tensor`的`name`反映了其所在的模块。例如：

   ```python
   def my_op(a, b, c, name=None):
       with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
           a = tf.convert_to_tensor(a, name="a")
           b = tf.convert_to_tensor(b, name="b")
           c = tf.convert_to_tensor(c, name="c")
           # Define some computation that uses `a`, `b`, and `c`.
           return foo_op(..., name=scope)
   ```

**小练习**：

> 以下说法正确的是：
>
> - `name_scope`可以给所有在其作用域中创建的`Op`的`name`添加前缀。
> - 在多级`name_scope`中的不同层级作用域下创建的`Op`（除去`tf.get_variable`以外），不存在`name`重名。
> - `name_scope`可以通过划分操作范围来组织图结构，并能服务于得可视化。

## 4. variable_scope

`variable_scope`主要用于管理变量作用域以及与变量相关的操作，同时`variable_scope`也可以像`name_scope`一样用来给不同操作区域划分范围（添加`name`前缀）。`variable_scope`功能也要更丰富，最重要的是可以与`tf.get_variable()`等配合使用完成对变量的重复使用。

`variable_scope`使用`tf.variable_scope()`创建，返回一个上下文管理器。

`tf.variable_scope`的详情如下:

```python
variable_scope(name_or_scope,  # 可以是name或者别的variable_scope
               default_name=None,
               values=None,
               initializer=None,  # 作用域中的变量默认初始化方法
               regularizer=None,  # 作用域中的变量默认正则化方法
               caching_device=None,  # 默认缓存变量的device
               partitioner=None,  # 用于应用在被优化之后的投影变量操作
               custom_getter=None,  # 默认的自定义的变量getter方法
               reuse=None,  # 变量重用状态
               dtype=None,  # 默认的创建变量的类型
               use_resource=None):  # 是否使用ResourceVariables代替默认的Variables
```

### 4.1 给Op的name加上name_scope

`variable_scope`包含了`name_scope`的全部功能，即在`variable_scope`下也可以给`Op`与`Tensor`加上`name_scope`：

```python
with tf.variable_scope('abc') as scope:
    a = tf.constant(1, name='const')
    print(a.name)  # >> abc/const:0
```

**注意**：默认的`variable_scope`的`name`等于其对应的`name_scope`的`name`，但并不总是这样。我们可以通过如下方法查看其`variable_scope`的`scope_name`与`name_scope`的`scope_name`：

```python
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('abc') as scope:
      	# 输出variable_scope的`name`
        print(scope.name)  # >>> abc
        
        n_scope = g.get_name_scope()
        # 输出name_scope的`name`
        print(n_scope)  # >>> abc
```

### 4.2 同名variable_scope

创建两个或多个`variable_scope`时可以填入相同的`name`，此时相当于创建了一个`variable_scope`与两个或多个`name_scope`。

```python
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('abc') as scope:
        print(scope.name)  # >> abc
        n_scope = g.get_name_scope()
        print(n_scope)  # >> abc
        
    with tf.variable_scope('abc') as scope:
        print(scope.name)  # >> abc
        n_scope = g.get_name_scope()
        print(n_scope)  # >> abc_1
```

### 4.3 与get_variable()的用法

`variable_scope`的最佳搭档是`tf.get_variable()`函数。一般的，我们会在`variable_scope`中使用`tf.get_variable()`创建与获取模型的变量，并且`variable_scope`为此提供了更多便利。

#### 4.3.1 独立使用get_variable()

与使用`tf.Variable()`不同，独立的使用（不在变量作用域中时）`tf.get_variable()`创建变量时不需要提供初始化的值，但必须提供`name`、`shape`、`dtype`，这是确定一个变量的基本要素。使用`tf.get_variable`创建变量的方法如下：

```python
tf.get_variable('abc', dtype=tf.float32, shape=[])
```

**说明**：没有初始化的值，并不意味着没有值，事实上它的值是随机的。使用`tf.Variable()`创建的变量，一般不需要提供`shape`、`dtype`，这是因为可以从初始化的值中推断出来，也不需要`name`，是因为默认的TensorFlow提供了自动生成`name`的方法。

`tf.get_variable()`还有一个特点是必须提供独一无二的`name`在当前变量作用域下，如果提供了重名的`name`才会抛出异常，如下：

```python
a = tf.get_variable('abcd', shape=[1])
b = tf.get_variable('abcd', shape=[1])  # ValueError
```

#### 4.3.2 在变量作用域中使用get_variable()

`variable_scope`对象包含一个`reuse`属性，默认的值为`None`，在这种情况下，代表`variable_scope`不是可重用的，此时，在`variable_scope`中的`tf.get_variable()`用法与上述独立使用`tf.get_variable()`用法完全一致。`tf.get_variable()`在`variable_scope`中创建的变量会被添加上`variable_scope`的`scope_name`前缀。当`variable_scope`的`reuse`属性值为`True`时，代表此`variable_scope`是可重用的，此时在`variable_scope`中的`tf.get_variable()`用法变成了利用`name`获取已存在的变量，而无法创建变量。也就是说`tf.get_variable()`的用法是随着`reuse`的状态而改变的，例如：

```python
with tf.variable_scope('scope', reuse=None) as scope:
    # 此时reuse=None，可以用来创建变量
    tf.get_variable('var', dtype=tf.float32, shape=[])
	# 改修reuse=True
    scope.reuse_variables()
	# 此时reuse=True，可以用来获得已有变量
    var = tf.get_variable('var')
```

上述的写法也可以写成下面类似形式：

```python
with tf.variable_scope('scope', reuse=None) as scope:
    tf.get_variable('var', dtype=tf.float32, shape=[])

with tf.variable_scope(scope, reuse=True) as scope:
    var = tf.get_variable('var')
```

可以看到下面的`scope`直接使用上面生成的`scope`而生成，也就是说`variable_scope`只要是`name`一样或者使用同一个`scope`生成，那么这些`variable_scope`都是同一个`variable_scope`。

**注意**：以上两种写法从`variable_scope`的角度看是等价的，但每创建一个`variable_scope`都创建了一个`name_scope`，所以上面的写法只包含一个`name_scope`，而下面的写法包含两个`name_scope`。这也是上文提到的`variable_scope`的`scope_name`与其包含的`name_scope`的`scope_name`不完全一样的原因。

为了便捷，`reuse`属性也可以设置为`tf.AUTO_REUSE`，这样`variable_scope`会根据情况自动判断变量的生成与获取，如下：

```python
with tf.Graph().as_default():
    with tf.variable_scope('scope', reuse=None):
        tf.get_variable('my_var_a', shape=[], dtype=tf.float32)

    with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
        a = tf.get_variable('my_var_a')  # 获取变量
        b = tf.get_variable('my_var_b', shape=[],  dtype=tf.float32)  # 生成一个变量
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([a, b]))
```

`variable_scope`与`tf.get_variable`配合使用这样的写法在较大型模型中是非常有用的，可以使得模型变量的复用变得容易。

**小结**：

- 在变量作用域中，如其属性`reuse=None`时，`tf.get_variable`不能获得变量；
- 在变量作用域中，如其属性`reuse=True`时，`tf.get_variable`不能创建变量；
- 在变量作用域中，`scope.reuse_variables()`可以改变下文的`reuse`属性值为`True`；
- 同名的多个变量作用域所处的上下文中的名字作用域不同。

**小练习**：

> 尝试实验验证上文“注意”中提到一个`variable_scope`与两个同名`variable_scope`中`name_scope`的情况。

### 4.4 多级变量作用域

变量作用域是可以嵌套使用的，这时候其`name`前缀也会嵌套。

如下：

```python
with tf.variable_scope('first_scope') as first_scope:
  	print(first_scope.name)  # >> first_scope
  	with tf.variable_scope('second_scope') as second_scope:
      	print(second_scope.name)  # >> first_scope/second_scope
        print(tf.get_variable('var', shape=[1, 2]).name)  
        # >> first_scope/second_scope/var:0
```

#### 4.4.1 跳过作用域

如果在嵌套的一个变量作用域里使用之前预定义的一个作用域，则会跳过当前变量的作用域，保持预先存在的作用域不变：

```python
with tf.variable_scope('outside_scope') as outside_scope:
  	print(outside_scope.name)  # >> outside_scope

with tf.variable_scope('first_scope') as first_scope:
  	print(first_scope.name)  # >> first_scope
  	print(tf.get_variable('var', shape=[1, 2]).name)   # >> first_scope/var:0
   
  	with tf.variable_scope(outside_scope) as second_scope:
      	print(second_scope.name)  # >> outside_scope
      	print(tf.get_variable('var', shape=[1, 2]).name)  # >> outside_scope/var:0
```

#### 4.4.2 多级变量作用域中的reuse

在多级变量作用域中，规定外层的变量作用域设置了`reuse=True`，内层的所有作用域的`reuse`必须设置为`True`（设置为其它无用）。通常的，要尽量避免出现嵌套不同种`reuse`属性的作用域出现，这是难以管理的。

多级变量作用域中，使用`tf.get_variable()`的方法如下：

```python
# 定义
with tf.variable_scope('s1') as s1:
    tf.get_variable('var', shape=[1,2])
    with tf.variable_scope('s2') as s2:
        tf.get_variable('var', shape=[1,2])
        with tf.variable_scope('s3') as s3:
            tf.get_variable('var', shape=[1,2])

# 使用
with tf.variable_scope('s1', reuse=True) as s1:
    v1 = tf.get_variable('var')
    with tf.variable_scope('s2', reuse=None) as s2:
        v2 = tf.get_variable('var')
        with tf.variable_scope('s3', reuse=None) as s3:
            v3 = tf.get_variable('var')
```

### 4.5 其它功能

`variable_scope`可以设置其作用域内的所有变量的一系列默认操作，比如初始化方法与正则化方法等。通过设置默认值可以使得上下文中的代码简化，增强可读性。

例如，这里我们将初始化、正则化、变量数据类型等均使用默认值：

```python
with tf.variable_scope('my_scope', 
                       initializer=tf.ones_initializer,
                       regularizer=tf.keras.regularizers.l1(0.1),
                       dtype=tf.float32):
    var = tf.get_variable('var', shape=[])
    reg = tf.losses.get_regularization_losses()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([var, reg]))  # >>> [1.0, [0.1]]
```

常见的初始化器有：

```python
# 常数初始化器
tf.constant_initializer(value=0, dtype=tf.float32)
# 服从正态分布的随机数初始化器
tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# 服从截断正态分布的随机数初始化器
tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# 服从均匀分布的随机数初始化器
tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32)
# 全0初始化器
tf.zeros_initializer(dtype=tf.float32)
# 全1初始化器
tf.ones_initializer(dtype=tf.float32)
```

常用的范数正则化方法有（此处使用`tf.keras`模块下的方法，也可以使用`tf.contrib.layers`模块中的方法，但不推荐）：

```python
# 1范数正则化
tf.keras.regularizers.l1(l=0.01)
# 2范数正则化
tf.keras.regularizers.l2(l=0.01)
# 1范数2范数正则化
tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

**小练习**：

> 复现最后一小节代码，掌握`variable_scope`的一般用法。

## 作业

1. 总结`name_scope`与`variable_scope`的作用以及异同点。
2. 构建逻辑回归模型（只有模型部分，不包括训练部分），使用`get_variable`与`variable_scope`将变量的创建与使用分开。提示：使用`tf.nn.sigmoid`实现`logistic`函数。