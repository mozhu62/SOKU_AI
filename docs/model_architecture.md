# CQL Joint432 模型架构

## 版本与边界

- 网络：soku_cql_recurrent_joint432_v3。
- 动作 schema：soku_controller_joint432_v1。
- 观测 schema：soku_cql_joint432_observation_v1。

从随机参数开始，旧 CQL / PPO / DQN checkpoint 明确拒绝，不做部分迁移。旧文件保留。没有宏动作、Actor/Critic、GAE、Attention、人工威胁特征或专家 margin。本次 gamma、奖励定义和每侧最多 3 个对象的规则均未修改。

## 完整 Controller State

每个动作是一帧的全部按键，不是一招或一套连招：

| 组成 | 值域 | 含义 |
|---|---|---|
| direction | 1～9 | 屏幕绝对九宫格，不随朝向翻转 |
| combat_mask | 0～15 | bit0=A体术、bit1=D DASH、bit2=B轻弹幕、bit3=C重弹幕 |
| card_command | 0～2 | NONE、CHANGE_CARD、USE_CARD |

joint_action_id = (direction - 1) × 48 + combat_mask × 3 + card_command。
共432类，ID为0～431。Neutral是“5 + 无战斗按钮 + NONE”，ID192；ID0是左下且无按钮。

人类可读示例：5、6+A、6+D+A、2+B、5+USE_CARD、6+USE_CARD、4+CHANGE_CARD。展示顺序 D+A 不改变存储 bit 顺序 A/D/B/C。切卡与用卡同帧为1必须报 action schema 不兼容，不会合并、忽略或丢帧。

## 主干与输出（默认维度）

```text
人物主体 + Skill/Card + 上一帧实际输入 + 可用 max_spirit/hitstop
    Current Encoder: 878 → 256 → 256
                    每层 Linear → LayerNorm → SiLU
                                    ┐
己方对象≤3 → 对象MLP → mean64+max64=128 ├ concat 512
对方对象≤3 → 对象MLP → mean64+max64=128 ┘
    Fusion: 512 → 256 → 256（每层 Linear → LayerNorm → SiLU）
    Battle Feature: 256
       ├ 单层 GRU: input256 / hidden128
       └ 与 GRU 输出拼接: 256 + 128 = 384
    Memory Fusion: Linear384→256 → LayerNorm → SiLU
    Joint Q Head: Linear256→128 → SiLU → Linear128→432
    原始 Q 输出: [batch, sequence, 432]
```

末层没有 softmax、sigmoid 或温度。step_q() 返回 [batch,432] Q和 [batch,128] 记忆；act() 返回 argmax Joint ID和记忆，控制层统一decode成真实按键。

Current Encoder 输入维数增加，但隐藏层仍256。资源直接进入Current，不再额外加64D资源MLP或加宽Fusion。主干宽度由本版配置校验固定。

### Current Encoder 的 878D

| 输入组 | 维度 |
|---|---:|
| 人物主体：连续18 + tactical9 + 类别embedding88 | 115 |
| 双方资源：每侧361 | 722 |
| 上一帧 joint_action embedding | 32 |
| 上一帧方向组合持续时间 | 1 |
| 可选双方 max_spirit/hitstop 数值4 + mask4 | 8 |

人物主体包含双方 x/y、vx/vy、绝对direction、HP、current_spirit、action_frame_count、relative_x/y；action ID/block ID和active_weather使用embedding。已有9个tactical flags保留。详见[完整输入清单](cql_dataset_fields.md)。

原始CSV有双方max_spirit/hitstop时附加保存真实值和mask。旧v4 NPZ没有时明确无效，不把零伪装成游戏状态；训练集完全没有覆盖的字段，在验证和实战也不启用。

### Skill / Card

每侧四槽在网络中统一命名 skill_slot_1～4；236/623/214/22只用于适配原采集列。

每槽 variant embedding8 + learned level embedding4 + effective level embedding4 + mask1 = 17D；四槽共68D。真实值加1编码，token0未知，不将未知技能当默认技能或零级。

Card ID使用双方所有槽共享的16D embedding。现有每侧16个手牌采集槽按“当前选中卡在首槽、后续切卡顺序”concat，每槽embedding16 + normalized cost1 + mask1，共288D。没有mean pooling。Card ID0可以有效，不能表示空槽。

再加5个真实数值：card_gauge、card_count、hand_capacity、hand_count、hand_cards_used。每侧68+288+5=361D。当前选中卡ID/费用由首槽表达，不重复添加同样特征；原始selected_index/selected_id/selected_cost仍在NPZ保留并检查一致性。未抽取牌堆不输入模型。

所有连续特征（含卡牌数值、有效手牌费用、可选人物字段）仅用训练集拟合normalization；类别用embedding，mask和tactical flags不拟合。仅上一帧持续时间采用固定clip60/60缩放。

### Object Encoder：不变

每侧最多3个，仍按相对萃香距离及原始序号排序，不改变截断规则。

每对象8数值 + action embedding32 + action_block embedding8 = 48D。
共享MLP为48→64→64，每层Linear/LayerNorm/SiLU。
masked mean64与masked max64拼为每侧128D；padding不参与池化，空集合输出零。类别embedding保留当前separate/shared配置，默认双方separate。

### GRU与上一帧历史

保留单层GRU、sequence和burn-in逻辑，默认学习段32个转移、burn-in最多16帧。片段起点隐藏状态为0，burn-in不反传，不跨回合、断帧、终局拼接。

observation[t]读取action[t-1]，Embedding(433,32)。真实片段起点用独立START/PAD token432和duration0；随机从同一连续片段中部取序列时保留真实前一帧动作。padding embedding行为零，不与真实动作共用token。

action_duration只表示水平/垂直组合持续帧数，按钮变化不重置它，不是完整joint action或招式持续时长。只输入clip(duration[t-1],60)/60，绝不将action[t]或duration[t]放进当前observation。

默认action_shift=1：转移动作a_t来自下一采集快照中已消费的输入。因此实战快照t的实际回读对应a_(t-1)，而不是模型计划发送的动作。缺帧、暂停或新局重置历史。不允许混训不同action_shift；约定随normalization/checkpoint保存。

## 联合离散CQL

```text
q_data = gather(Q_online(s), dataset_joint_action_id)
a_next = argmax(Q_online(s_next), 432个动作)
y = reward + gamma × (1 - terminated) × Q_target(s_next, a_next)
TD = mean(Huber(q_data, stop_gradient(y)))
CQL = mean(T × logsumexp(Q_online(s)/T, 432个动作) - q_data)
loss = TD + cql_alpha × CQL
```

目标网络是一套完整网络，初始复制online、禁用梯度；有效优化后继续target_tau软更新。TD/CQL共用有效mask。不再有分项Q、六按钮loss或可加Q分解。T只用于CQL保守项，推理直接argmax。

## 诊断与使用

网页及日志提供数据/模型argmax Joint Action Top-N、Neutral占比、Q_data/Q_max均值标准差、Q_max−Q_data、CQL gap、TD MSE/MAE、Validation EV、完整动作expert agreement、模块梯度和参数变化。动作显示解码字符串，Q不是概率，一致率不是胜率。

在soku_cql目录从零训练：

```text
python scripts/train.py --config configs/cql_suika.yaml --headless
```

网页模式移除--headless。默认输出outputs/cql_suika_joint432_v3。已有合格资源v4 NPZ在加载时转换，不需要因为换Q头重采REP；若要补入旧NPZ未存的max_spirit/hitstop，可以用已有含对应列的CSV重生成NPZ，文件哈希变化后需指定新split_file。

[单元测试源码](../tests/)覆盖全432动作往返、卡命令互斥、历史泄漏/边界、模型形状、Double DQN/CQL、归一化隔离。按要求本次未运行测试、编译前端或启动对局。新网页须由使用者手动构建。
