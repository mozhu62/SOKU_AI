# DQfD 模型网络架构说明

> 版本提示：本文记录旧 `suika_observation_v1` 网络，供旧模型对照。2026-09-11 新增的当前 NPZ 数据版已删除缺失字段，维度和输入语义见 [resources-v4 模型及训练说明](dqfd_resources_v4.md)。旧模型参数不能直接载入新版。

## 1. 文档范围与实现基线

| 项目 | 内容 |
|---|---|
| 项目 | Touhou Hisoutensoku / TH123 萃香 AI，`soku_ai` |
| 文档版本 | V1.0 |
| 核对日期 | 2026-09-10 |
| 文档性质 | 当前源码实现说明，不是模型实战效果验收报告 |
| 算法名称 | DQfD：Deep Q-learning from Demonstrations；交流中的 DFDQ 指本项目 DQfD |
| 网络类 | `SokuDuelingQNetwork` |
| 架构版本 | `tcn_entity_dueling_dqn_v1` |
| Observation Schema | `suika_observation_v1` |
| 动作映射版本 | `egocentric_3x3x2x2x2x2_v1` |
| 默认维度依据 | `configs/dqfd_suika_v1.yaml`；`dqfd_suika_aggressive_v2.yaml` 的网络维度相同 |
| 本次工作 | 只编写文档；未编译、未运行测试、未启动训练或对局 |

本文说明模型接收什么、各分支怎样计算、时序信息怎样进入决策，以及输出如何转成控制动作。训练损失、数据集和实战采集只介绍与网络理解直接相关的接口，不修改算法、配置或已有模型。

所有维度均为上述本地配置及源码的静态推导，不是性能实测。具体 `.pt` 应以其内嵌 `config`、`normalization` 和权重为准；不能仅根据模型文件名推断其训练阶段。

阅读导航：

- [架构摘要](#2-架构摘要)
- [完整模型输入](#3-模型输入契约)
- [归一化与参数共享](#4-归一化embedding-与参数共享)
- [状态编码器](#5-当前状态编码器)、[历史 TCN](#6-历史-tcn-编码器)、[对象编码器](#7-双方对象编码器)
- [融合与 Q Head](#8-特征融合与-dueling-q-head)、[144 动作编码](#9-144-维输出对应什么动作)
- [轨迹与训练接口](#10-网络与轨迹训练的连接)、[实战历史](#11-实战推理的历史与网络调用)
- [Checkpoint](#12-checkpoint-与兼容性)、[BC 对比](#13-与当前-bc-tcn32-的关键区别)、[限制与核对清单](#14-限制与后续人工核对清单)

## 2. 架构摘要

当前网络是“当前状态编码器 + 独立历史 TCN + 双方共享对象编码器 + 融合 MLP + Dueling Q 输出头”。

一次前向计算输出 `[B, 144]`，即每个样本的 144 个完整控制组合的 Q 值。不是概率，不是一次输出 32 个动作，也不是输出一套预设连招。

图 1 展示四路特征的融合关系。`B` 为批量大小，历史长度默认 `L=32`，每侧对象容量默认 `M=64`。

```text
当前人物/环境状态
  30 数值 + 8 类别 -> 150D -> State Encoder ---------------- 256D --+
                                                                  |
32 帧操作历史                                                     |
  每帧 10 数值 + 4 类别 -> 90D -> TCN -> 时间 mean/max ------ 128D --+
                                                                  +-> concat 640D
当前己方对象                                                      |      |
  每对象 18 数值 + 2 类别 -> 58D -> 共享 Object Encoder ---- 128D --+      v
                                                                  |   Fusion MLP
当前对方对象                                                      |   640 -> 512 -> 256
  每对象 18 数值 + 2 类别 -> 58D -> 同一 Object Encoder ---- 128D --+      |
                                                                         v
                                                               Dueling Q Head
                                                                /          \
                                                        V: 256->128->1   A: 256->128->144
                                                                \          /
                                                          Q = V + A - mean(A)
                                                                         |
                                                                  [B, 144]
```

图 1：DQfD 当前网络的数据流。图中的“类别”先经 embedding，再与数值拼接。

关键边界：

- TCN 是独立的操作历史分支，不处理已经融合完毕的全部战斗特征。
- 双方对象是当前观测的集合，不是 32 帧对象轨迹。
- 没有 GRU、LSTM、Transformer、Attention，也没有跨推理调用传递的循环隐藏状态。
- 有历史窗口，但没有宏动作输出、逐招取消图或网络内部的动作合法性 mask。
- Dueling 的两个分支是状态价值与动作优势，不是方向头与按钮头，也不是 PPO Actor/Critic。

实现入口：[网络组装](../soku_ai/models/q_network.py)、[模型工厂](../soku_ai/models/factory.py)。

## 3. 模型输入契约

### 3.1 张量名称、形状与类型

表 1 为经过 Dataset 批处理后进入模型的张量。单次实战推理在外层补 `B=1`。

| 输入键 | 默认形状 | 类型 | 含义 |
|---|---|---|---|
| `state_continuous` | `[B, 30]` | float32 | 当前人物及环境数值 |
| `state_categorical` | `[B, 8]` | int64 | 当前招式、角色、天气和场景类别 |
| `history_numerical` | `[B, 32, 10]` | float32 | 原始操作及少量战斗数值的历史 |
| `history_categorical` | `[B, 32, 4]` | int64 | 双方招式与阶段的历史 |
| `history_mask` | `[B, 32]` | bool | 历史有效位置 |
| `self_object_numerical` | `[B, 64, 18]` | float32 | 当前己方对象数值 |
| `self_object_categorical` | `[B, 64, 2]` | int64 | 当前己方对象招式类别 |
| `self_object_mask` | `[B, 64]` | bool | 己方对象有效槽位 |
| `opponent_object_numerical` | `[B, 64, 18]` | float32 | 当前对方对象数值 |
| `opponent_object_categorical` | `[B, 64, 2]` | int64 | 当前对方对象招式类别 |
| `opponent_object_mask` | `[B, 64]` | bool | 对方对象有效槽位 |

表 1：网络输入接口。训练开启 AMP 后部分内部运算可使用低精度，但不改变该字段契约。

### 3.2 当前状态：30 个数值字段

表 2 按 `STATE_CONTINUOUS_FEATURES` 的实际列顺序列出，索引从 0 开始。

| 索引 | 字段 | 含义 |
|---|---|---|
| 0 | `self_position_x` | 己方 x 坐标 |
| 1 | `self_position_y` | 己方 y 坐标 |
| 2 | `self_speed_x` | 己方水平速度 |
| 3 | `self_speed_y` | 己方垂直速度 |
| 4 | `self_direction` | 己方游戏朝向值 |
| 5 | `self_current_spirit` | 己方当前灵力 |
| 6 | `self_action_frame_count` | 己方当前招式经过帧数 |
| 7 | `self_hitstop` | 己方 hitstop 数值 |
| 8 | `self_combo_hits` | 己方记录的连击段数 |
| 9 | `self_combo_limit` | 己方 Combo Limit |
| 10 | `self_total_object_count` | 己方总对象数，未按输入槽位截断 |
| 11 | `opponent_position_x` | 对方 x 坐标 |
| 12 | `opponent_position_y` | 对方 y 坐标 |
| 13 | `opponent_speed_x` | 对方水平速度 |
| 14 | `opponent_speed_y` | 对方垂直速度 |
| 15 | `opponent_direction` | 对方游戏朝向值 |
| 16 | `opponent_current_spirit` | 对方当前灵力 |
| 17 | `opponent_action_frame_count` | 对方当前招式经过帧数 |
| 18 | `opponent_hitstop` | 对方 hitstop 数值 |
| 19 | `opponent_combo_hits` | 对方记录的连击段数 |
| 20 | `opponent_combo_limit` | 对方 Combo Limit |
| 21 | `opponent_total_object_count` | 对方总对象数，未按输入槽位截断 |
| 22 | `weather_counter` | 天气计数 |
| 23 | `relative_x` | 对方 x − 己方 x |
| 24 | `relative_y` | 对方 y − 己方 y |
| 25 | `relative_vx` | 对方水平速度 − 己方水平速度 |
| 26 | `relative_vy` | 对方垂直速度 − 己方垂直速度 |
| 27 | `distance_x` | `abs(relative_x)` |
| 28 | `distance_y` | `abs(relative_y)` |
| 29 | `euclidean_distance` | 双方平面欧氏距离 |

表 2：当前状态数值字段。

“萃香第一视角”首先指双方数据统一重命名为 `self/opponent`。当前 x/y、速度和相对位置仍保留屏幕坐标语义，并没有把整个 observation 都旋转或镜像到“己方永远朝右”。相对前后方向的转换发生在控制动作编码层。

### 3.3 当前状态：8 个类别字段

| 索引 | 字段 | Embedding 维数 |
|---|---|---:|
| 0 | `self_action` | 32 |
| 1 | `self_action_block_id` | 8 |
| 2 | `opponent_character_id` | 16 |
| 3 | `opponent_action` | 32 |
| 4 | `opponent_action_block_id` | 8 |
| 5 | `active_weather` | 8 |
| 6 | `displayed_weather` | 8 |
| 7 | `stage_id` | 8 |

表 3：当前状态类别字段。合计 `32+8+16+32+8+8+8+8=120D`，加上 30 个数值得到 `150D`。

这里的 `action` 是游戏内部招式/动作状态 ID，不是模型输出的 0～143 控制动作 ID。二者不能互换。

### 3.4 历史：每帧 10 个数值及 4 个类别

数值字段按存储顺序为：

```text
self_input_horizontal_raw
self_input_vertical_raw
self_input_a_raw
self_input_b_raw
self_input_c_raw
self_input_d_raw
self_action_frame_count
opponent_action_frame_count
relative_x
relative_y
```

类别字段按存储顺序为：

```text
self_action
self_action_block_id
opponent_action
opponent_action_block_id
```

两个招式 ID 使用各 32D embedding，两个阶段 ID 使用各 8D embedding，合计 `80D`；与 10 个数值拼成每帧 `90D`。

“原始输入”表示保留采集到的轴和 ABCD 回读数值，再进行数值归一化，没有先把它们全部压成二值，也没有先转成 Joint Action embedding。原始输入中包含的持续计数因此仍可被网络利用。该表没有独立的 `previous_action_duration` 字段。

注意：原始轴输入与输出动作的语义不同。历史水平轴保留原始屏幕轴数值；输出的水平类别则为 `NONE/FORWARD/BACKWARD`。

### 3.5 对象：每对象 18 个数值及 2 个类别

数值字段按顺序为：

```text
relative_position_x
relative_position_y
relative_speed_x
relative_speed_y
gravity_x
gravity_y
direction
action_frame_count
hitstop
hit_count
hit_box_count
hurt_box_count
frame_data_available
frame_flags
attack_flags
frame_damage
frame_spirit_damage
distance_to_suika
```

类别为 `action`、`action_block_id`，embedding 合计 `40D`，故单对象输入为 `18+40=58D`。

当前实现需特别区分以下事实：

- `gravity_x`、`gravity_y` 因原采集值不可信，被清理函数固定清零。结构仍保留 18 列，不等于 18 列都携带有效变化信息。
- `frame_flags`、`attack_flags` 仍作为原始数值列输入，没有拆成语义位 embedding。
- 输入有攻击框/受击框数量，但没有完整判定框坐标、旋转框顶点等精确几何。
- 默认每侧取相对己方萃香最近的最多 64 个已采集对象；己方、对方两组分开截断与汇总。
- 模型对象容量不等于游戏全部对象均已被底层采集；网络只能使用采集结果中可见的对象。

来源：[字段定义](../soku_ai/data/schemas.py)、[离线字段构造](../soku_ai/data/preprocess.py)、[实战字段构造](../soku_ai/live/observation.py)、[对象清理](../soku_ai/data/sanitization.py)。

### 3.6 当前没有进入网络的内容

这版 DQfD observation 不包含双方 HP、装备 Skill 类型/等级、手牌/卡槽/符卡能量、切卡/用卡输入，也没有 BC/CQL 的 `previous_joint_action_id` 和其 START token。

HP 会在网络外参与奖励与终局计算；“用于训练奖励”不等于“作为网络状态输入”。

## 4. 归一化、Embedding 与参数共享

### 4.1 数值归一化

当前实现分别拟合状态、历史、对象三组均值和标准差：

```text
normalized = (value - mean) / std
```

标准差小于 `1e-6` 时替换为 1。当前函数没有 BC/CQL 中的统一 `[-10, 10]` 裁剪步骤。对象先清理两个重力字段，再归一化；其他有效对象字段出现 NaN/Inf 会报错。

预处理入口使用训练集合拟合归一化；当前 DQfD 配置训练占比为 1.0，因此其“训练集归一化”可覆盖全部 81 份数据，不应套用 BC 的 8:2 设定。推理从 checkpoint 内恢复归一化，并校验字段顺序和禁用对象列。

来源：[归一化实现](../soku_ai/data/normalization.py)、[预处理入口](../scripts/preprocess_replays.py)。

### 4.2 类别词表与未知值

| 类别 | 正常词表容量 | Embedding 宽度 | 实际表行数 |
|---|---:|---:|---:|
| 游戏招式 `action` | 2048 | 32 | 2050 |
| 动作阶段 `block` | 512 | 8 | 514 |
| 角色 `character` | 64 | 16 | 66 |
| 天气 `weather` | 32 | 8 | 34 |
| 场景 `stage` | 64 | 8 | 66 |

表 4：默认 embedding 配置。每张表额外保留 unknown 和 padding 两行。

规则为：

1. 正常 ID 映射到对应行。
2. 越界 ID 映射到 `unknown_index=vocab_size`。
3. 原始 padding 值 `-2` 映射到 `padding_index=vocab_size+1`，优先于 unknown 处理。

主网络只创建一组 `GameEmbeddings`，供当前状态、历史和双方对象共同使用。双方对象还共享同一个对象 MLP；并不存在两套独立对象参数。

来源：[Embedding 实现](../soku_ai/models/embeddings.py)。

## 5. 当前状态编码器

`StateEncoder` 的默认结构为：

```text
[B,150]
  -> Linear(150,256)
  -> SiLU
  -> LayerNorm(256)
  -> Linear(256,256)
  -> SiLU
  -> [B,256]
```

LayerNorm 位于第一层 SiLU 之后；第二层 SiLU 后没有再放 LayerNorm。该分支没有 Dropout。

来源：[StateEncoder](../soku_ai/models/state_encoder.py)。

## 6. 历史 TCN 编码器

### 6.1 窗口与标签对齐

默认 `history_len=32`、`action_shift=1`。以原始采集行号 `t` 表示时：

```text
当前状态：采集行 t
历史窗口：最多使用采集行 t-31 ... t
监督标签：采集行 t+1 中记录的控制输入
下一状态：采集行 t+1
```

因此，历史中包含采集行 t 的已记录输入，但不包含待预测标签所在的 t+1 行输入。

Dataset 也支持 `action_shift=0`，此时历史排除当前行，最多使用 `t-32 ... t-1`。不过当前实战 `LiveObservationBuilder` 总是追加当前快照历史，没有相应的 shift=0 分支；本文实战说明仅按默认 shift=1 对齐。若使用 shift=0 模型，需另行核对实时对齐，不能只改 YAML 就宣称两端一致。

历史不跨 episode 起点；不足 32 帧时左侧补数值 0、类别 `-2`，并将对应 mask 置为 false。

### 6.2 输入投影和残差块

先进行：

```text
history_numerical [B,32,10]
  + history category embedding [B,32,80]
  -> concat [B,32,90]
  -> Linear(90,64)
  -> transpose [B,64,32]
  -> 乘 history_mask
```

接着通过四个残差块：

| 残差块 | 输入通道 | 输出通道 | 每块时序卷积数 | kernel | dilation | 残差支路 |
|---|---:|---:|---:|---:|---:|---|
| Block 1 | 64 | 64 | 2 | 3 | 1 | Identity |
| Block 2 | 64 | 64 | 2 | 3 | 2 | Identity |
| Block 3 | 64 | 128 | 2 | 3 | 4 | 1×1 Conv，64→128 |
| Block 4 | 128 | 128 | 2 | 3 | 8 | Identity |

表 5：DQfD 默认 TCN。四个残差块内总计八次时序卷积。

每块实际运算顺序为：

```text
x -> CausalConv1d -> GroupNorm(1,Cout) -> SiLU -> Dropout(0.05)
  -> CausalConv1d -> GroupNorm(1,Cout) -> SiLU -> Dropout(0.05)
  -> 加上 Identity 或 1×1 Conv 残差
  -> SiLU
  -> 块外再次乘 history_mask
```

卷积通过 padding 后裁掉右侧多余输出，实现卷积本身不读取相应位置之后的输入。时间长度保持 32，最后输出为 `[B,128,32]`。

### 6.3 感受野与“因果”的准确范围

仅按卷积路径计算，理论感受野为：

```text
R = 1 + 每块卷积数 × (kernel-1) × sum(dilation)
  = 1 + 2 × (3-1) × (1+2+4+8)
  = 61
```

这是卷积结构推导，不表示模型实际获取了 61 个真实游戏帧。输入窗口仍最多 32 帧，窗口外没有真实历史。

还需注意：输入到 `GroupNorm(1,C)` 的张量为 `[B,C,L]`，其统计在每个样本的通道和时间维内进行，而不是只统计同一帧的通道；推理模式也继续使用输入统计。结合本项目的张量布局，可以推知某个中间位置会受窗口内其他时间位置影响。[PyTorch GroupNorm 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)

所以，“卷积是因果卷积”不能扩展成“整个分支的每一个中间位置都严格逐帧因果”。最终网络只用截止决策时刻的历史窗口生成一次输出，不因此自动读到窗口之外的未来；但它也不能不加修改就当成 BC 那种逐位置因果预测器使用。

mask 在投影后和每个残差块后应用，没有传入 GroupNorm 做 masked statistics。因此 padding 不参与最终有效帧池化，但不能声称它对块内归一化完全没有影响。

### 6.4 时间池化和输出

对有效历史位置分别计算：

```text
masked mean over time -> [B,128]
masked max over time  -> [B,128]
concat                -> [B,256]
Linear(256,128) -> SiLU -> LayerNorm(128)
输出                  -> [B,128]
```

均值池化除数使用有效历史帧数，至少为 1；无效位置在 max pooling 前排除。整个窗口无有效帧时，两项池化值置零，但后续带可学习参数的投影不保证最终输出恒为零。

时间 mean/max 是对卷积后特征汇总，不是对原始输入直接求平均，也不是只取最后一个时间位置。网络可以先识别局部操作变化，再汇总到固定的 128D 历史特征。

来源：[TCN](../soku_ai/models/tcn.py)、[历史窗口构造](../soku_ai/data/transition_builder.py)。

## 7. 双方对象编码器

每个对象的默认编码为：

```text
18 数值 + action embedding 32D + block embedding 8D = 58D
  -> Linear(58,64)
  -> SiLU
  -> LayerNorm(64)
  -> Dropout(0.05)
  -> Linear(64,64)
  -> SiLU
  -> 每对象 64D
```

随后对每一侧的有效对象做 masked mean 64D 与 masked max 64D，拼成每侧 128D。默认 `object_output_dim=128=2×object_entity_dim`，输出投影是 Identity。

己方与对方分别调用同一个 `ObjectEncoder`。两侧对象不会混为一个池化集合，但共用 embedding、MLP 和投影参数。

不足 64 个对象时补零与类别 padding，并通过 mask 排除；某侧没有对象时，该侧默认输出为零。对象集合经过池化后，原始槽位顺序不作为独立序列传入融合层。

来源：[ObjectEncoder](../soku_ai/models/object_encoder.py)。

## 8. 特征融合与 Dueling Q Head

### 8.1 融合 MLP

严格按以下顺序拼接：

```text
当前状态 256D + 操作历史 128D + 己方对象 128D + 对方对象 128D = 640D
```

融合层为：

```text
Linear(640,512) -> SiLU -> LayerNorm(512)
  -> Linear(512,256) -> SiLU
  -> 最终战斗特征 [B,256]
```

这是四路特征首次统一融合的位置，没有 GRU 隐藏状态或另一个 Memory Fusion 分支。

### 8.2 价值和优势分支

```text
Value stream:     Linear(256,128) -> SiLU -> Linear(128,1)
Advantage stream: Linear(256,128) -> SiLU -> Linear(128,144)
```

合成为：

```text
Q(s,a) = V(s) + A(s,a) - mean_over_144_actions(A(s,*))
```

此处 `s` 表示完整 observation，包含历史。Value 分支为所有动作提供共同的状态价值；Advantage 分支描述不同动作相对于该状态的差别。

输出没有 softmax、sigmoid 或温度缩放。所有 Q 值都可能为负，动作仍按最大 Q 选择。Q 为负不表示禁止动作，也不意味着应选择最小 Q。

来源：[融合网络](../soku_ai/models/q_network.py)、[Dueling Head](../soku_ai/models/dueling_head.py)。

## 9. 144 维输出对应什么动作

### 9.1 控制组合

| 分量 | 编码 |
|---|---|
| 水平 `h` | 0=NONE，1=FORWARD，2=BACKWARD |
| 垂直 `v` | 0=NONE，1=UP，2=DOWN |
| `A` | 体术是否按下 |
| `B` | 轻弹幕是否按下 |
| `C` | 重弹幕是否按下 |
| `D` | Dash 是否按下 |

表 6：输出动作分量。总数 `3×3×2⁴=144`。

FORWARD 沿角色朝向，BACKWARD 与朝向相反。最终由控制层结合当前朝向转换成屏幕左/右键；不是在 Q 头中再做一次左右预测。

编码公式为：

```text
action_id = (h * 3 + v) * 16 + A * 8 + B * 4 + C * 2 + D
```

例如：

| ID | 含义 |
|---:|---|
| 0 | 水平空、垂直空、无按钮，即松键 |
| 8 | 原地按 A |
| 32 | 按下方向，无按钮 |
| 48 | 向前，无按钮 |
| 49 | 向前 + D |
| 56 | 向前 + A |

表 7：部分控制动作示例。`action_id=0` 不等于 BC/CQL 的 Joint432 中立 ID 192，二者映射不能混用。

### 9.2 输入、招式与轨迹的区别

模型输出的是一次控制状态，不是游戏保证执行的招式。例如“向前+A”是否产生对应攻击，还取决于地面/空中、当前动作、硬直和游戏自身规则。

连续输出相同按钮组合与“重复松开再点按”不是同一件事。多次推理形成的方向切换和按下/松开序列，才可能在游戏中表现为搓招、连段或立回。网络没有预先定义的 `236B`、`623C` 宏动作，也不输出切卡或用卡命令。

原始 REP 的动作标签将 ABCD 通过 `raw != 0` 转为布尔值；历史分支则保留原始数值再归一化。不要把“标签是二值按钮”误解成“历史也只剩二值”。

来源：[动作编解码](../soku_ai/data/action_space.py)。

## 10. 网络与轨迹训练的连接

### 10.1 一条训练样本

一条监督样本以一个有效转移为中心，主要包括：

```text
observation          当前状态 + 32 帧历史 + 当前双方对象
action               专家控制动作 ID
reward / done        当前转移奖励 / 终局标志
next_observation     下一状态对应的完整 observation
n_step_reward        最多 N 个连续转移的折扣奖励和
n_step_discount      gamma 的实际累计步数次方
n_step_done          累计过程中是否终局
n_step_observation   累计末端状态对应的完整 observation
is_demo              专家样本标志
training_weight      数据侧样本权重
```

因此 DQfD 有轨迹数据，也使用前后关系；但每个 observation 的前向仍输出一组 144 维 Q，不是一次预测整段轨迹的控制动作。历史窗口长度与 N 步回报长度是两个独立参数。

没有循环隐藏状态意味着不需要 GRU burn-in。历史帧也不需要各自产生独立监督损失，当前动作的损失可以沿 TCN 反向传播到历史特征相关参数。

### 10.2 Online 与 Target 网络

训练器构建两套同构完整网络：

- Online：由优化器更新，负责当前 Q 估计及下一动作选择。
- Target：初始化复制 Online，使用 eval 模式，在无梯度目标计算中为选中的下一动作评分。

Double DQN 的目标形式为：

```text
a_next = argmax_a Q_online(next_observation, a)
y_1 = reward + gamma * (1-done) * Q_target(next_observation, a_next)

R_k = r_t + gamma*r_(t+1) + ... + gamma^(k-1)*r_(t+k-1)
a_k = argmax_a Q_online(observation_(t+k), a)
y_k = R_k + gamma^k * (1-done_k) * Q_target(observation_(t+k), a_k)
```

`k` 为实际累计的有效步数，不跨 episode 边界。下一动作选择时实现会临时将 Online 切到 eval，再恢复原模式，避免 Dropout 随机性参与该选择；Target 参数不由 Online 的优化器直接更新。

基础配置使用 `gamma=0.99`、`n_step=3`；aggressive_v2 使用 `gamma=0.9995`、`n_step=30`。两者网络形状相同，但学习目标不同。默认 Target 为每 5000 个训练 step 硬更新；代码还支持配置软更新。

### 10.3 DQfD 损失边界

当前损失代码包含：单步 Huber TD、N 步 Huber TD、专家 large-margin、可配置空闲动作项和参数 L2；样本还可叠加 PER 重要性权重及数据侧训练权重。

基础 large-margin 形式为：

```text
L_demo = max_a [Q(s,a) + margin(a_expert,a)] - Q(s,a_expert)
margin(a_expert,a) = 0，若 a=a_expert；否则为配置 margin
```

这些损失与样本权重不是额外网络输入，也不是额外输出头。aggressive 配置的行为偏置不能由架构名推断，需核对具体 checkpoint 的配置和来源。详细奖励配置见 [奖励调参说明](reward_tuning.md)。

来源：[Dataset](../soku_ai/data/transition_builder.py)、[训练器](../soku_ai/training/demo_trainer.py)、[Double DQN](../soku_ai/rl/double_dqn.py)、[DQfD 损失](../soku_ai/rl/dqfd_loss.py)。

## 11. 实战推理的历史与网络调用

推理端加载 Online 权重并使用 `eval()`。通用 Agent 支持传入 epsilon 随机选动作，默认 epsilon=0；当前实战 Runtime 直接取 `argmax(Q)`。Dropout 在 eval 模式停用，GroupNorm/LayerNorm 仍执行。

实战历史由模型外的 deque 保存，而非网络内部记忆：

1. 读取共享内存快照，确定萃香所在侧。
2. 新游戏帧生成一行数值历史和类别历史。
3. 追加到最多 32 行的窗口，再归一化、构造 mask 和当前对象输入。
4. 网络输出 144 个 Q，选择动作 ID 并解码。
5. 在聚焦与安全条件允许时发送控制输入。

旧 DQfD 的缺帧处理是：同进程、同回合发生小间隔跳帧时，复制上一整行历史数值和类别，填入缺口；进程/回合切换、帧号异常或间隔超过窗口时清空。重复填充的行不是重新采集到的真实游戏状态。

因此，旧端显示历史数量 32，只能说明缓存有 32 个历史位置，不能保证这 32 个位置全部来自不同的真实采集帧；也不能保证每帧都及时执行了一次模型动作。

默认 shift=1 的采集对齐及 32 帧历史属于网络输入条件。要做模型间公平实战比较，必须同时核对观测连续性、推理延迟、实际输入回读，不能只比较 TCN 名称或窗口数字。

来源：[实战 ObservationBuilder](../soku_ai/live/observation.py)、[推理 Agent](../soku_ai/inference/agent.py)、[实战 Runtime](../soku_ai/live/runtime.py)。

## 12. Checkpoint 与兼容性

完整续训包包含 Online/Target 权重、优化器、配置、归一化、动作映射、训练步数、数据划分哈希及可选 AMP/PER 状态。轻量 `model_snapshot` 不包含完整续训状态；加载函数会拒绝将其作为带 Target 或优化器恢复的续训包使用。

不能把 BC/CQL 的 Joint432 checkpoint 直接作为本架构权重加载。至少存在以下语义和形状差异：

- 本模型为 144 控制动作，BC/CQL 为 432 类完整绝对控制状态。
- 本模型为独立原始操作历史 TCN，不是融合战斗特征后的 GRU/TCN。
- 状态字段、对象字段、对象容量和归一化契约不同。
- 本模型输出 Dueling Q，不是 BC logits，也不是当前 CQL 的单一 Joint Q head。

当前通用 checkpoint 加载函数检查包版本和权重匹配，并不等于全面自动审计所有语义元数据；归一化字段在对应恢复路径另行校验。修改输入顺序、动作含义、共享关系或归一化规则后，即使维度碰巧一样，也不能宣称旧权重等价兼容。

来源：[Checkpoint 实现](../soku_ai/training/checkpoint.py)。

## 13. 与当前 BC TCN32 的关键区别

本节仅帮助辨认实现，不构成性能优劣结论。

| 项目 | 本文 DQfD | 当前 BC TCN32 |
|---|---|---|
| TCN 输入 | 原始操作 + 少量战斗历史，投影前每帧 90D | 状态、对象、资源、上一动作已融合的每帧 256D |
| 时序分支位置 | 与当前状态、当前对象并列 | 在每帧战斗融合之后 |
| 时序结构 | 4 块、每块 2 次 kernel=3 卷积 | 5 块、每块 1 次 kernel=2 卷积 |
| dilation | 1/2/4/8 | 1/2/4/8/16 |
| 归一化 | 块内 GroupNorm | 每帧通道 LayerNorm |
| 历史汇总 | 时间 mean/max 后投影 | 使用对应时间位置的 TCN 特征 |
| 输出 | 144 维 Dueling Q | 432 维分类 logits |
| 方向标签 | 水平前/后相对朝向 | 屏幕绝对九宫格 |
| 对象容量 | 每侧最多 64 | 每侧最多 3 |
| 控制历史表示 | 轴与 ABCD 原始回读数值 | 每帧上一 Joint Action embedding + 方向持续时间 |

表 8：源码结构对比。BC 的 32 帧中每一帧都包含对应的上一帧动作信息，不是整个网络只看一个历史动作。

参照：[BC TCN](../../soku_bc/soku_bc/temporal.py)、[BC 网络](../../soku_bc/soku_bc/models.py)。

## 14. 限制与后续人工核对清单

### 14.1 不由网络结构保证的能力

- 不保证已掌握完整出招表、取消规则、连招或主动进攻策略。
- 不保证正确识别每一次伤害的具体来源；TD 回报传播不是精确因果归属。
- 不保证输入的原始 flag 数值具有良好的语义表达。
- 不保证 32 个缓存历史位置全部是真实采集帧。
- 不保证离线 Q 拟合改善一定对应实战能力提升。

### 14.2 待执行核对项目

以下仅为后续人工验收清单，本次未执行：

- [ ] 按默认配置核对各分支输出：256/128/128/128，融合 640→512→256，Q 输出 `[B,144]`。
- [ ] 对照公式核验完整动作编解码，特别是动作 0、向前/向后、ABCD 位顺序。
- [ ] 对照同一 REP 的离线/实战字段及 normalization，核对默认 shift=1 的时间关系。
- [ ] 验证短历史、空对象、未知类别、真实回合边界的输出有限性与 mask 行为。
- [ ] 如计划使用 shift=0 模型，先核对实战历史包含当前输入的问题。
- [ ] 区分实际采集历史、重复补帧历史和实际成功发键频率。
- [ ] 核对待比较 checkpoint 的内嵌配置、奖励/权重阶段和模型来源，不能仅按文件名判断。

### 14.3 源码索引与修订记录

| 模块 | 主文件 |
|---|---|
| 基础网络配置 | [dqfd_suika_v1.yaml](../configs/dqfd_suika_v1.yaml) |
| aggressive 网络与训练配置 | [dqfd_suika_aggressive_v2.yaml](../configs/dqfd_suika_aggressive_v2.yaml) |
| 输入 Schema | [schemas.py](../soku_ai/data/schemas.py) |
| 网络组装 | [q_network.py](../soku_ai/models/q_network.py) |
| 类别编码 | [embeddings.py](../soku_ai/models/embeddings.py) |
| 状态编码 | [state_encoder.py](../soku_ai/models/state_encoder.py) |
| 时序编码 | [tcn.py](../soku_ai/models/tcn.py) |
| 对象编码 | [object_encoder.py](../soku_ai/models/object_encoder.py) |
| Q 输出 | [dueling_head.py](../soku_ai/models/dueling_head.py) |
| Observation 与 N 步转移 | [transition_builder.py](../soku_ai/data/transition_builder.py) |
| 模型加载与预测 | [agent.py](../soku_ai/inference/agent.py) |

表 9：主要维护位置，所有项目内链接相对本文件解析。

| 版本 | 日期 | 修订内容 |
|---|---|---|
| V1.0 | 2026-09-10 | 首次按当前源码整理输入完整清单、各分支维度、TCN 池化与归一化边界、144 动作语义、训练接口及实战补帧行为；未实施运行验收 |
