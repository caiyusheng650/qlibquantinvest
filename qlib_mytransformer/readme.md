# qlib上运行 自定义模型MyTransformer

## 创新点
- 在Transformer 上删除了失活丢弃层

## 效果
- 指标
  - ic 越高，表示模型预测能力越强，更能预测哪个股票亏钱或者赚钱。

  - arr 越高，表示模型的风险调整后的收益越高。

  - 参数量越小，表示模型越简单，占用计算资源少。

- 分数
  - qlib 示例的transformer 的ic 为0.0300， arr为0.03200,  参数量为约800k

  - 此自定义的transformer 的ic 为0.0114， arr为0.01724，参数量为约1200k



## 安装 

- 第一步：克隆qlib代码库
- 第二步：把my_pytorch_transformer.py文件放到qlib/contrib/model目录
- 第三步：可能要在qlib/contrib/model目录下的__init__.py文件中添加一行代码：from .my_pytorch_transformer import MyTransformer
- 第四步：pip install .


## 运行
花4分钟加载数据，每34秒训练一轮

在第13轮停止，因为验证分数已经达到最低且开始反弹
```shell
qrun workflow_config_my_transformer_Alpha158.yaml
```

## 自定义模型步骤
1. 复制examples
   1.  如果像要实现新回归算法，复制`qlib/contrib/model`中其中一个文件，如`pytorch_transformer.py`，重命名为`my_pytorch_transformer.py`，它实现了一个模型的具体的算法。
   2.  如果像要实现新交易策略，在`qlib/crontrib/strategy/signal_strategy.py`中添加一个新的类，继承自`SignalStrategy`，在这个类上实现新策略。
   3.  复制 `qlib/examples/benchmarks` 中其中一个文件，改名成`mymodel.ymal`，一个它是配置文件。

2. 实现自己的自定义模型
   1. 修改那个重要的类中的model成员变量，换成自己的模型
   2. 在配置文件中修改模型的名字，修改`task.model.module_path` 和`task.model.class`两个字段

3. 安装 ```pip install -e .```
4. 运行 
```
 qrun mymodel.ymal
 ```
   或者
   ```python
   python  qlib/workflow/cli.py mymodel.yaml
   ```


## 其他文件

- `experiments`文件夹保存了`qlib示例的transformers`和`我们自定义transformers`的运行日志 
- my_transformer_debug.py用来调试，可以用来看神经网络中的中间输出的形状。调试完数据网络后可以删除它了