# 四个工作
- `qlib_lgbm文件夹`，
  - 介绍：用qlib运行lgbm，并且用qlib默认交易策略交易，按照qlib的示例和教程做
  - 工具：qlib
  - 结果：达到 ARR 0.195
- `qlib_rl_order_execurion文件夹`，
  - 介绍：运行qlib的强化学习示例，
  - 结果：失败，可能因为我跟作者的pandas版本不对
- `qlib_my_transformer文件夹`，
  - 介绍：实现`自定义神经网络模型`和`自定义交易策略`
  - 工具：torch，Transformer，高级的激活函数和归一化方法
  - 工作量：比较神经网络配置，比如神经网络层数和激活方法和归一化方法
  - 结果：达到 ARR 0.01724
- `mydrl文件夹`，
  - 实现`自定义强化学习交易`， 
  - 工具：用qlib下载数据，用openai gym和stable_baseline3 来强化学习
  - 工作量：认真写交易策略和认真尝试各种交易策略，最大的工作量是在设置奖励
  - 结果：
    - 尝试了很多模型和交易策略了
    - 可能有点过拟合，智能体总是认为sh600016就是最强的股票，然后完全不买其他股票
    - 过拟合原因分析：因为我们只用到10个股票，很少，并且sh600016就是优异的股票
