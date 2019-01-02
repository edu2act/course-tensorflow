当我们已经训练好一个模型时，需要把模型保存下来，这样训练的模型才能够随时随地的加载与使用。模型的保存和加载分为两部分，一部分是模型定义的存取即**图的存取**，另一部分中图中的**变量的存取**（常量存储在图中）。TensorFlow可以分别完成两部分的内容的存取，也可以同时完成。除此以外TensorFlow还封装了很多path相关的操作，其类似于Python中的标准库`os.path`的功能。本章主要内容为基础文件操作，接下来的一章内容为模型存取操作。

# 1. io模块

io模块是简单的文件输入输出操作的模块，主要功能是读取文件中的内容并转化为张量以及解码数据。这里我们主要介绍一些常用的方法。

## 1.1 文件读写

使用`tf.io.read_file()`可根据`filename`将数据读取出来，其格式为TensorFlow中的`DT_STRING`，类似于Python中的`Bytes`，使用`tf.io.read_file`读取文件之后可使用相关的解码操作解码出数据来，例如读取一张图片，并解码：

```python
f = tf.io.read_file('test.png')
decode_img = tf.image.decode_png(f, channels=4)
```

此时`decode_img`即为图片对应的张量。

使用`tf.io.write_file()`可将`DT_STRING`写入文件中，例如将上述`f`重新写入文件：

```python
tf.io.write_file('new.png', f)
```

## 1.2 解码数据

解码数据即将`DT_STRING`类型的数据转化为包含真实信息的原始数据的过程。不同类型的数据解码需要使用不同的解码方法，TensorFlow中解码数据的API分布在不同的模块中，`tf.io`模块包主要含以下几种解码方法：

```python
tf.io.decode_base64  # 解码base64编码的数据，可解码使用tf.encode_base64编码的数据
tf.io.decode_compressed  # 解码压缩文件
to.io.decode_json_example  # 解码json数据
tf.io.decode_raw  # 解码张量数据
```

# 2. gfile模块

`tf.gfile`模块也是文件操作模块，其类似于Python中对file对象操作的API，但其与`tf.io`不同，`tf.gfile`模块中的API返回值并不是Tensor，而是Python原生数据。但TensorFlow的文件IO会调用C++的接口，实现文件操作。同时TensorFlow的`gfile`模块也支持更多的功能，包括操作本地文件、Hadoop分布式文件系统(HDFS)、谷歌云存储等。

`tf.gfile`提供了丰富的文件相关操作的API，例如文件与文件夹的创建、修改、复制、删除、查找、统计等。这里我们简单介绍几种常用的文件操作方法。

## 2.1 打开文件流

打开并操作文件首先需要创建一个文件对象。这里有两种方法进行操作：

- `tf.gfile.GFile`。也可以使用`tf.gfile.Open`，两者是等价的。此方法创建一个文件对象，返回一个上下文管理器。

  用法如下：

  ```python
  tf.gfile.GFile(name, mode='r')
  ```

  输入一个文件名进行操作。参数`mode`是操作文件的类型，有`"r", "w", "a", "r+", "w+", "a+"`这几种。分别代表只读、只写、增量写、读写、读写（包括创建）、可读可增量写。默认情况下读写操作都是操作的文本类型，如果需要写入或读取bytes类型的数据，就需要在类型后再加一个`b`。这里要注意的是，其与Python中文件读取的`mode`类似，但不存在`"t","U"`的类型（不加`b`就等价于Python中的`t`类型）。

- `tf.gfile.FastGFile`。与`tf.gfile.GFile`用法、功能都一样(旧版本中`tf.gfile.FastGFile`支持无阻塞的读取，`tf.gfile.GFile`不支持。目前的版本都支持无阻塞读取)。一般的，使用此方法即可。

例如：

```python
# 可读、写、创建文件
with tf.gfile.GFile('test.txt', 'w+') as f:
    ...
    
# 可以给test.txt追加内容
with tf.gfile.Open('test.txt', 'a') as f:
    ...
    
# 只读test.txt
with tf.gfile.FastGFile('test.txt', 'r') as f:
    ...
    
# 操作二进制格式的文件
with tf.gfile.FastGFile('test.txt', 'wb+') as f:
    ...
```

## 2.2 数据读取

文件读取使用文件对象的`read`方法。（这里我们以`FastGFile`为例，与`GFile`一样）。文件读取时，会有一个指针指向读取的位置，当调用`read`方法时，就从这个指针指向的位置开始读取，调用之后，指针的位置修改到新的未读取的位置。`read`的用法如下：

```python
# 返回str类型的内容
tf.gfile.FastGFile.read(n=-1)
```

当参数`n=-1`时，代表读取整个文件。`n!=-1`时，代表读取`n`个bytes长度。

例如：

```python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.read(3)  # 读取前3个bytes
    f.read()  # 读取剩下的所有内容
```

如果我们需要修改文件指针，只读部分内容或跳过部分内容，可以使用`seek`方法。用法如下：

```python
tf.gfile.FastGFile.seek(
    offset=None,  # 偏移量 以字节为单位
    whence=0,  # 偏移其实位置 0表示从文件头开始(正向) 1表示从当前位置开始(正向) 2表示从文件末尾开始(反向)
    position=None  # 废弃参数 使用`offset`参数替代
)
```

例如：

```python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.seed(3)  # 跳过前3个bytes
    f.read()  # 读取剩下的所有内容
```

**注意**：读取文件时，默认的（不加`b`的模式）会对文件进行解码。会将bytes类型转换为UTF-8类型，如果读入的数据编码格式不是UTF-8类型，则在解码时会出错，这时需要使用二进制读取方法。

除此以外，还可以使用`readline`方法对文件进行读取。其可以读取以`\n`为换行符的文件的一行内容。例如：

```python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.readline()  # 读取一行内容(包括行末的换行符)
    f.readlines()  # 读取所有行，返回一个list，list中的每一个元素都是一行
```

以行为单位读取内容时，还可以使用`next`方法或是使用生成器来读取。如下：

```python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.next()  # 读取下一行内容
    
with tf.gfile.FastGFile('test.txt', 'r') as f:
  	# 二进制数据首先会将其中的代表`\n`的字符转换为\n，然后会以\n作为分隔符生成list
    lines = [line for line in f]  
```

**注意**：如果没有使用with承载上下文管理器，文件读取完毕之后，需要显示的使用`close`方法关闭文件IO。

## 2.3 其它文件操作

- **文件复制**：`tf.gfile.Copy(oldpath, newpath, overwrite=False)`
- **删除文件**：`tf.gfile.Remove(filename)`
- **递归删除**：`tf.gfile.DeleteRecursively(dirname)`
- **判断路径是否存在**：`tf.gfile.Exists(filename)`  # filename可指代路径
- **判断路径是否为目录**：`tf.gfile.IsDirectory(dirname)`
- **返回当前目录下的内容**：`tf.gfile.ListDirectory(dirname)`  # 不递归 不显示'.'与'..'
- **创建目录**：`tf.gfile.MkDir(dirname)`  # 其父目录必须存在
- **创建目录**：`tf.gfile.MakeDirs(dirname)`  # 任何一级目录不存在都会进行创建
- **文件改名**：`tf.gfile.Rename(oldname, newname, overwrite=False)`
- **统计信息**：`tf.gfile.Stat(filename)`
- **文件夹遍历**：`tf.gfile.Walk(top, in_order=True)`  # 默认广度优先
- **文件查找**：`tf.gfile.Glob(filename)`  # 支持pattern查找

**小练习：**

> 练习上述文件相关操作，主要是使用`tf.gfile.FastGFile` API 读取数据。

