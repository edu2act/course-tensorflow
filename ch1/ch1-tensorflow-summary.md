本节主要内容：

- 理解Tensorflow是什么。
- 理解Tensorflow所使用的编程模式。
- 了解Tensorflow高层库，以及高层库与其关系。
- Tensorflow的作用。

# Tensorflow概述

Tensorflow是由Google Brain Team开发的使用**数据流图**进行数值计算的开源机器学习库。Tensorflow的一大亮点是支持异构设备分布式计算(heterogeneous distributed computing)。这里的异构设备是指使用CPU、GPU等计算设备进行有效地协同合作。

*Google Brain Team与DeepMind是独立运行相互合作的关系。*

Tensorflow拥有众多的用户，除了Alphabet内部使用外，ARM、Uber、Twitter、京东、小米等众多企业均使用Tensorflow作为机器学习的工具。

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/3.png)



常见机器学习库包括Tensorflow、MXNet、Torch、Theano、Caffe、CNTK、scikit-learn等。

|                      库                      | 维护人员或机构                        |              支持语言              |             支持操作系统             |
| :------------------------------------------: | :------------------------------------ | :--------------------------------: | :----------------------------------: |
|  [Tensorflow](https://www.tensorflow.org/)   | google                                |          Python、C++、Go           | Linux、mac os、Android、iOS、Windows |
| [MXNet](https://mxnet.incubator.apache.org/) | 分布式机器学习社区(DMLC)              | Python、Scala、R、Julia、C++、Perl | Linux、mac os、Android、iOS、Windows |
|      [Torch/PyTorch](http://torch.ch/)       | Ronan Collobert等人                   |       Lua、LuaJIT、C/Python        | Linux、mac os、Android、iOS、Windows |
|                    Theano                    | 蒙特利尔大学( Université de Montréal) |               Python               |        Linux、mac os、Winodws        |
|     Computational Network Toolkit(CNTK)      | 微软研究院                            |      Python、C++、BrainScript      |            Linux、Windows            |
|                    Caffe                     | 加州大学伯克利分校视觉与学习中心      |        Python、C++、MATLAB         |        Linux、mac os、Windows        |
|                 PaddlePaddle                 | 百度                                  |            Python、C++             |            Linux、mac os             |



各个框架对比https://github.com/zer0n/deepframeworks

2016-2017年框架热度对比（来源于谷歌趋势）：

![2017年框架热度对比](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/2.png)



2017-2018年框架热度对比（来源于谷歌趋势）：

![2017年框架热度对比](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/4.png)

## 什么是Tensorflow

### Tensor

Tensor是张量的意思，原本在物理学中用来描述大于等于2维的量进行量纲分析的工具。我们早已熟知如何处理0维的量（纯量）、1维的量（向量）、2维的量（矩阵）。对于高维的数据，我们也需要一个工具来表述，这个工具正是张量。

张量类似于编程语言中的多维数组（或列表）。广义的张量包括了常量、向量、矩阵以及高维数据。在处理机器学习问题时，经常会遇到大规模样本与大规模计算的情况，这时候往往需要用到张量来进行计算。Tensorflow中张量是最重要与基础的概念。

### Flow

flow是“流”的意思，这里可以可以理解为数据的流动。Tensorflow所表达的意思就是“张量流”。

### 编程模式

编程模式通常分为**命令式编程（imperative style programs）**和**符号式编程（symbolic style programs）**。命令式编程，直接执行逻辑语句完成相应任务，容易理解和调试；符号式编程涉及较多的嵌入和优化，很多任务中的逻辑需要使用图进行表示，并在其他语言环境中执行完成，不容易理解和调试，但运行速度有同比提升。机器学习中，大部分现代的框架使用符号式编程，其原因是编写程序容易且运行速度快。

命令式编程较为常见，例如直接使用C++、Python进行编程。例如下面的代码：

```python
import numpy as np
a = np.ones([10,])
b = np.ones([10,]) * 5
c = a + b
```

当程序执行到最后一句时，a、b、c三个变量有了值。程序执行的是真正的计算。

符号式编程不太一样，仍然是完成上述功能，使用符号式编程的写法如下（伪代码）：

```python
a = Ones_Variables('A', shape=[10,])
b = Ones_Variables('B', shape=[10,])
c = Add(a, b)

# 计算
Run(c)
```

上述代码执行到c=Add(a, b)时，并不会真正的执行加法运算，同样的a、b也并没有对应的数值，a、b、c均是一个符号，符号定义了执行运算的结构，我们称之为**计算图**，计算图没有执行真正的运算。当执行Run(c)时，计算图开始真正的执行计算，计算的环境通常不是当前的语音环境，而是C++等效率更高的语言环境。

机器学习库中，Tensorflow、theano使用了符号式编程（tensorflow借鉴了theano的很多优点）；Torch使用了命令式编程；caffe、mxnet采用了两种编程模式混合的方式。

### 数据流图

当我们使用计算图来表示计算过程时，事实上可以看做是一个**推断**过程。在推断时，我们输入一些数据，并使用符号来表示各种计算过程，最终得到一个或多个推断结果。所以使用计算图可以在一定程度上对计算结果进行预测。

计算图在推断的过程中也是数据流转的过程，所以我们也可以称之为**数据流图**。举个例子，假如我们计算$(a+b)*(b+1)$的值，那么我们画出其数据流图，如下：

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/tree-def.png)

输入a与b，通过不同通路进行计算并传入下一个节点。这一过程就是数据流动的过程。有了数据流图，我们还可以进行更多的操作，例如自动求微分等，在此不做赘述。

### Tensorflow高层库

Tensorflow本质上是数值计算库，在数值处理与计算方面比较方便、灵活。虽然Tensorflow为机器学习尤其是深度学习提供了很多便捷的API，但在构建算法模型时，仍然较为复杂。为此Tensorflow官方以及众多第三方机构与个人开发了很多的使用简便的高层库，这些库与Tensorflow完全兼容，但可以极大简化模型构建、训练、部署等操作。其中较为常用工具包与高层库为：

1. TF Learn(tf.contrib.learn)：类似于scikit-learn的使用极少代码量即可构建机器学习算法的工具包。

   TF Learn可以尽量使用最少的代码构建我们想要的模型，如下，我们构建一个神经网络模型只需要一行代码：

   ```python
   # 使用 TF Learn 定义一个神经网络
   tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                  hidden_units=[10, 200, 10],
                                  n_classes=3,
                                  model_dir="./my_model")
   ```

2. TF Slim(tf.contrib.slim)：一个轻量级的用于定义、训练、评估深度学习算法的Tensorflow工具包。

   TF Slim只能用于深度学习算法，相比于 TF Learn 。TF Slim既可以方便的设计算法，又可以较大程度的定制模型。例如我们定制一个神经网络的层：

   ```python
   padding = 'SAME'
   initializer = tf.truncated_normal_initializer(stddev=0.01)
   regularizer = slim.l2_regularizer(0.0005)
   net = slim.conv2d(inputs, 64, [11, 11], 4,
                     padding=padding,
                     weights_initializer=initializer,
                     weights_regularizer=regularizer,
                     scope='conv1')
   ```

   可以看到相比 TF Learn，要复杂的多，但与Tensorflow相比仍然要简单的多。

3. 高级API：Keras，TFLearn，Pretty Tensor

   高级API是建立在Tensorflow API之上的API，其尽量抽象了Tensorflow的底层设计，并拥有一套自己完整、简洁、高效的API。可以以极简的语法设计算法模型。最重要的是高级API相比其他API，其功能更丰富，可以独立的不借助Tensorflow的API运行，同时也可以兼容Tensorflow的API。

   在Tensorflow所有的高级API中，Keras是发展最好的一个。Keras主要用在设计神经网络模型之上，可用于快速将想法变为结果。Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端。

   Keras的设计原则是（来自于官网）：

   - 用户友好：Keras是为人类而不是天顶星人设计的API。用户的使用体验始终是我们考虑的首要和中心内容。Keras遵循减少认知困难的最佳实践：Keras提供一致而简洁的API， 能够极大减少一般应用下用户的工作量，同时，Keras提供清晰和具有实践意义的bug反馈。
   - 模块性：模型可理解为一个层的序列或数据的运算图，完全可配置的模块可以用最少的代价自由组合在一起。具体而言，网络层、损失函数、优化器、初始化策略、激活函数、正则化方法都是独立的模块，你可以使用它们来构建自己的模型。
   - 易扩展性：添加新模块超级容易，只需要仿照现有的模块编写新的类或函数即可。创建新模块的便利性使得Keras更适合于先进的研究工作。
   - 与Python协作：Keras没有单独的模型配置文件类型（作为对比，caffe有），模型由python代码描述，使其更紧凑和更易debug，并提供了扩展的便利性。

   ​

   我们可以把 TF Learn 看做是开箱即用的工具，我们几乎不需要了解工具有关的理论，也不需要对工具进行任何修改就能使用；TF Slim 可以看做小巧的多功能改锥，可以快速的更换零件，但功能仅限于改锥，不能做别的；Keras就像是瑞士军刀，有了它既可以快速完成很多任务，又拥有丰富的拓展性。

### Tensorflow的发展

2015年11月9日，Tensorflow的0.5的版本发布并开源。起初Tensorflow的运行效率低下，不支持分布式、异构设备；

2016年4月，经过不到半年的时间发布了0.8版本，开始支持分布式、多GPU运算等，同期大多数机器学习框架对这些重要功能仍然不支持或支持有限；

2016年6月，0.9的版本改进了对移动设备的支持，到此，Tensorflow已经成为了为数不多的支持分布式、异构设备的开源机器学习库，并极大的改善了运算效率问题，成为运算效率最高的机器学习算法库之一。

2017年2月，Tensorflow的1.0正式版发布，增加了专用的编译器XLA、调试工具Debugger和`tf.transform`用来做数据预处理，并开创性的设计了`Tensorflow Fold`用于弥补符号编程在数据预处理时的缺陷。

2018年3月，TensorFlow1.7发布，将Eager模式加入核心API，从此TensorFlow具备了使用命令式编程模式进行编程的方法，极大的提高了TensorFlow的易用性。在2017年到2018年这一年中，PyTorch诞生并快速发展，PyTorch使用更加灵活的命令式编程模式给予了TensorFlow重要启示。

2018年7月，TensorFlow1.9发布，提出了`AutoGraph`模式，将符号式编程与命令式编程的优点相结合。这也标志着TensorFlow从探索阶段进入到了成熟阶段，在编程模式之争中逐步确立了独特的、便捷的、高效的新方法。

TensorFlow在不到三年的发展中，因其快速迭代，适应发展要求而被广大用户所喜爱，目前成为了领先的机器学习框架，然而TensorFlow也有众多缺点为人所诟病，其中较高的学习成本、较快的API变动、复杂重复的高级API等为突出问题，好在TensorFlow社区称在2018年底会发布2.0版本，届时会剥离高层API、简化TensorFlow，并提高API的稳定性。

## Tensorflow能干什么？

设计、训练、部署机器算法。例如可以轻松设计很多有意思的算法，例如：

**图像风格转换**： [**neural-style**](https://github.com/anishathalye/neural-style)

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/n.png)

**游戏内的自动驾驶**：[**TensorKart**](https://github.com/kevinhughes27/TensorKart)：

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/p.gif)

**目标检测**：[**SSD-Tensorflow**](https://github.com/balancap/SSD-Tensorflow)

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/5.png)

TensorFlow支持部署算法到多种设备之上。

![](/Users/wangqi/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/TensorFlow/%E5%9F%BA%E7%A1%80/images/b.png)

