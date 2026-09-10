# BC 宽 TCN32 网络与训练口径

## 1. 设计目标

当前版本不再在时序结构上保留两套路线。网络只保留一条简单、宽且可直接解释的路径：

- 当前状态不再先压缩为 256D，而是由 878D 单层映射到 1024D。
- GRU 和旧 Memory Fusion 完全删除。
- TCN 位于状态输入顶部，独立读取连续 32 帧 878D 状态。
- 两侧对象仍各自编码为 128D，每侧最多 3 个对象。
- 四路特征拼成 1536D，经一层 1024D 融合后直接输出 432 类 logits。

网络版本为 soku_bc_wide_tcn32_joint432_v2。

## 2. 每帧 878D 状态

878D 是经过必要类别 embedding 和资源展开后的实际网络输入宽度，不是 878 个原始标量字段：

| 组成 | 维度 |
|---|---:|
| 双方基础连续状态 | 18 |
| 战术语义状态 | 9 |
| 双方 action / action block 与天气 embedding | 88 |
| 双方 Skill、手牌、卡槽和灵力资源编码 | 722 |
| 上一帧真实 Joint Action embedding | 32 |
| 上一帧方向组合持续时间 | 1 |
| 可选状态数值 | 4 |
| 可选状态有效 mask | 4 |
| 合计 | 878 |

CurrentStateEncoder 先调用 features() 形成该 878D 向量。当前帧分支与 TCN 历史分支复用同一份 878D 表示，避免两套字段定义漂移。

上一帧动作仍使用 433 项 embedding：0～431 是真实 Joint Action，432 是 START/PAD。当前帧标签 action[t] 不会进入 observation[t]。

## 3. 当前帧宽编码

当前帧分支只有一层：

~~~text
878 → Linear(878,1024) → LayerNorm(1024) → SiLU → 1024D
~~~

这里的 1024D 是学习得到的隐藏表示宽度。它不会增加原始观测信息，也不是状态字段从 878 项扩展到 1024 项；它只给 BC 更多通道组合已有状态。

## 4. 独立 TCN32 历史分支

TCN 输入为：

~~~text
[batch, time, 878]
~~~

先逐帧投影到 256D，然后执行：

1. 一个 kernel=2 的因果 stem。
2. 四个因果残差块，dilation 为 1、2、4、8。
3. 每个残差块包含两层 kernel=2 的因果卷积。
4. 每层只左侧补零，不读取未来帧。
5. LayerNorm 只作用于同一帧的 256 个通道，不跨时间统计。
6. padding 帧在每层后重新乘 mask，防止卷积偏置制造伪历史。

感受野按历史间隔计算：

~~~text
stem:                         1
双卷积残差块: 2 × (1+2+4+8) = 30
总历史间隔:                   31
覆盖帧数: 当前帧 + 前 31 帧 = 32
~~~

因此每个输出严格只依赖 32 帧窗口。TCN 输出为 256D，不进行跨整段平均池化，也不会跨 episode、终局或断帧拼接。

## 5. 对象分支

对象结构保持不变。每侧最多读取离角色最近的 3 个对象；每个对象由 8 个连续字段、action embedding 和 action block embedding 组成，经共享对象 MLP 得到 64D：

~~~text
每对象 48D → 64D → 64D
masked mean 64D + masked max 64D = 每侧 128D
~~~

己方与敌方的类别 embedding 按配置使用 separate 模式。空集合返回 128D 零向量。

## 6. 最终融合与输出

~~~text
当前状态             1024D
TCN32 历史            256D
己方对象              128D
敌方对象              128D
                     ─────
拼接                  1536D

1536 → Linear(1536,1024) → LayerNorm → SiLU
1024 → Linear(1024,432) → 原始 logits
~~~

输出端没有 softmax、sigmoid 或 temperature。训练时 CrossEntropy 直接接收 logits；实战推理时对 432 类取 argmax，再 decode 为完整 Controller State。

## 7. 序列训练

默认每个样本读取 31 帧前导上下文和 32 帧监督片段：

~~~text
[31 帧历史] + [32 帧带标签片段]
~~~

- 前导帧不单独产生 CrossEntropy。
- 后续监督帧的梯度可以通过因果卷积回传到其真实历史输入及 embedding。
- 短 episode 的前导先按真实长度左对齐，缺失位置使用 mask。
- CE 只统计 Dataset mask 标记的有效监督位置。
- horizontal mirror augmentation、固定 8:2 split、normalization、Joint432 标签和 best checkpoint 规则保持不变。

训练目标仍为：

~~~text
CrossEntropy(logits[t], expert_joint_action_id[t])
~~~

不存在奖励、TD、N-step、Q value、目标网络、PPO、GAE 或额外辅助损失。

## 8. 冻结与诊断

可冻结模块现在只有：

- current_encoder
- object_encoder
- tcn
- fusion
- policy_head

冻结通过 requires_grad=False 实现，并清除残留梯度。模块梯度范数和单次更新参数变化量继续写入诊断日志。不存在 gru 或 memory_fusion 模块，也不存在“冻结但仍旁路”的特殊模式。

## 9. Checkpoint 兼容性

当前 checkpoint 必须同时满足：

- algorithm 为 bc；
- network_version 为 soku_bc_wide_tcn32_joint432_v2；
- observation manifest 与 Joint432 schema 一致；
- current、temporal 和 fusion 结构清单一致；
- state_dict 严格加载。

旧 GRU 与旧 256D TCN 的输入宽度、参数名和前向语义均不同，因此明确拒绝续训和推理，不进行部分迁移。

## 10. 实战推理

实战端从 LiveFrames.v1 队列按游戏帧积累连续观测：

- 窗口不足 32 帧时不执行模型、不发键。
- 每个新游戏帧只构造并缓存 878D 状态表示。
- TCN 读取完整 32 帧；对象编码器只读取当前帧对象。
- 换局、帧号回退、真实缺帧或协议断开时清空窗口。
- 输出仍为 [1,432] logits，完整方向和 ABCD/切卡/用卡按钮一起解码执行。

本次仅交付源码与静态文档，没有编译、运行测试、启动训练或进行实战。
