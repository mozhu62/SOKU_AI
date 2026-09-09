# BC 网络与训练口径

## 1. 复用边界

直接复用 CQL 的 schema、动作编码、资源编码和主干设计，在独立 `soku_bc` 包中实现。代码没有导入 `soku_cql`、`soku_ppo` 或 `soku_ai`。schema 中保留 `soku_cql_*` 字符串是为了读取现有 NPZ 的数据协议，不代表训练仍是 CQL。

| 模块 | 实际结构 |
|---|---|
| Current State | 连续量/战术量/类别 embedding/资源/上帧动作拼接 → Linear(input,256) → LayerNorm → SiLU → Linear(256,256) → LayerNorm → SiLU |
| Object | 每侧最多 3 个；8 数值 + action/block embedding → 共享两层 64D MLP；masked mean 64 + max 64 = 每侧 128D |
| Fusion | 256 + 128 + 128 = 512 → 两层 256D MLP，各含 LayerNorm/SiLU |
| GRU | 单层 input=256、hidden=128，离线序列训练 |
| Memory Fusion | 当前特征 256 + GRU 128 = 384 → Linear(384,256) → LayerNorm → SiLU |
| BC Policy Head | Linear(256,128) → SiLU → Linear(128,432)，输出原始 logits |

保留现有对象截断和 embedding 共享方式，不扩对象数量，不加 Attention/Transformer/GNN。唯一分类头没有独立方向或按钮子头。

## 2. 输入与动作

- 基础输入仍为 18 连续状态、5 个类别字段、9 个战术标志，完整名字以 `soku_bc/schema.py` 和网页字段清单为准。
- 双方四个 skill_slot：variant、level、effective_level 的 embedding 和有效 mask；不将招式名称写死成网络节点。
- 双方有序手牌：16 个存储槽的 Card ID（共享 embedding）、费用和 mask，按槽位 concat，不 mean pooling。原始 card_count/card_gauge/hand_capacity/hand_count/hand_cards_used 五个数值由训练集归一化；选中卡沿用现有首槽语义，不造新字段。
- 可选的双方 max_spirit、hitstop 使用真实记录 mask；训练集未覆盖的列在验证/推理时也不当作已知值。
- 上一帧实际 `previous_joint_action_id` 使用 433×32 embedding；真实动作 0～431，START/PAD=432。方向组合持续时间取上一帧的 duration，clip 到 60 后除以 60，不是完整按键持续时间。
- 当前帧标签不输入当前 observation；NPZ 的 action_shift 用于核验既有状态/输入对齐约定，数据集先右移历史再切序列。终局、episode 边界、无效连续片段重置历史。

完整 Controller State：direction 1～9，combat_mask 四 bit 顺序 A/D/B/C，card_command=NONE/CHANGE_CARD/USE_CARD。

```text
joint_action_id = (direction - 1) * 48 + combat_mask * 3 + card_command
```

432 个离散类别表示同一帧完整按键，不是连招/招式宏。Neutral=192，即 5 + 无按钮 + NONE。同帧切卡与用卡冲突沿用现有清洗规则：清掉切卡、保留用卡，不删帧、不回写 NPZ。

## 3. BC 学习目标

一份样本是一个连续状态序列及每帧的专家完整按键标签。默认 B=32、学习长度 L=32、burn-in=16，最多 1024 个有效监督帧/更新。短片段补齐但不计损失，不跨局或断帧拼接。

1. burn-in 在 no_grad 下恢复 GRU 记忆；零长度历史严格从零记忆开始。
2. 学习片段输出 `[B,L,432]` 的 logits，只筛选有效帧。
3. 使用专家 Joint ID 的交叉熵：`loss = mean(-log softmax(logits)[expert_id])`。
4. AdamW 更新可训练模块；冻结参数 requires_grad=False 并清梯度，保留优化器分组和未冻结状态。

默认 label_smoothing=0，即标准 BC；可调到 0.2 以内做标签平滑。验证 NLL 永远使用未平滑真实标签，避免把平滑强度变化误当成准确度变化。

**不读取奖励，不计算 TD、gamma、N-step、CQL conservative loss、PER 或 DQfD margin；没有目标网络、Actor/Critic 双网或 GAE。** 验证仅 no_grad 前向，训练概率不是探索混合，也没有熵奖励。Neutral 不屏蔽、不特殊惩罚；按数据真实分布学习。

## 4. 效率与可复现性

复用 DQN/CQL 风格的有限容量 NPZ LRU 缓存；每批从少量 REP 批量索引；后台 CPU 预取、CUDA 固定页内存/非阻塞搬运、可选 AMP，序列通过融合 GRU 执行。只计算一个网络，没有未来 TD 状态尾段、目标网络前向或软更新。

采样按有效帧数加权、有放回。训练随机数由 seed/step 派生；暂停丢弃预取后继续不会错位。验证使用独立固定 seed 及批次计划，不消耗训练采样随机数；不是每次遍历整个验证集。训练帧数包含重复采样，不等于完整 epoch。

每记录间隔计算概率、直方图、模块梯度和权重差。吞吐率用最近记录窗口的实际取数/优化时间，不混入暂停或验证；保存、验证、暂停另计。是否比 CQL 更快仍需在实际硬件测量，本次没有跑基准。

## 5. 指标定义

| 指标 | 含义 |
|---|---|
| loss | 当前训练配置的交叉熵（可能包含标签平滑） |
| NLL | 未平滑专家负对数似然；越低越好，均匀 432 类约 6.07 |
| Joint Top-1 / Top-5 | 专家完整按键是否等于最大概率动作 / 位于概率最高的 5 个动作中 |
| direction/combat/card accuracy | 将单一 Joint 预测解码后的分项一致率，不是额外 loss |
| expert_probability | 真实专家动作获得概率的算术均值，不等于 exp(-平均 NLL) |
| confidence_mean | 每个状态最大分类概率的均值；可能自信但错误 |
| normalized_entropy | `H(p)/log(432)`，接近 1 为均匀，接近 0 为集中；不是训练奖励 |
| macro_recall | 对该次验证实际出现的动作，逐类召回率再平均；未出现动作显示未记录 |
| 多数动作基线 | 始终输出训练集最多动作，在这批验证上的命中率；不根据验证标签挑动作 |

所有分类误差按有效帧加权合并，频数直接求和。网页频率是多个状态的 argmax 统计，不能当作单帧概率。最佳模型按当前配置阶段最低验证 NLL 保存；胜负、伤害差需要之后在独立实战验证。

## 6. 模型格式与续训

`network_version=soku_bc_recurrent_joint432_v1`，`algorithm=bc`，输出语义 `categorical_logits`；仅保存一个 model state_dict、优化器、AMP scaler、配置、归一化、固定划分 hash、计数及 Torch RNG。临时文件写完后原子替换目标。

旧 CQL 的 Q 值不是概率 logits，明确拒绝加载；不做部分迁移。BC 的 `model.act(obs, memory)` 返回 argmax Joint ID 和新记忆，`step_logits` 便于推理端显示 softmax 概率；调用者按断帧/暂停/终局清空 memory，并使用 checkpoint 内归一化。`action_space.to_controller` 解码回方向和六个按钮。
