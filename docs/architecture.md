# BC Joint144 / TCN32 网络与训练说明

## 1. 本次实验边界

直接修改当前 BC，不创建 Git 分支，也不迁移旧权重。网络版本为 soku_bc_tcn32_joint144_v1。

删除双方卡牌全部输入、技能学习等级及生效等级；保留双方四个技能槽的必杀类型及有效 mask。输出去掉切卡和用卡，由 Joint432 改为 Joint144。当前状态编码由原来的 878→1024 改成 228→256。

不改变对象数量、对象截断规则、TCN 感受野、关键帧加权损失、优化器超参数、数据划分、采样规则、动作对齐及 checkpoint 选优规则。模型不含 GRU、Q/TD、奖励、Actor/Critic 或宏动作。

## 2. 当前状态输入：228D

这里的维度指类别经过 embedding 后拼接的状态向量，不是 228 个原始字段。对象分支独立计算，不算在 228D 内。

| 输入组 | 展开维度 |
|---|---:|
| 双方位置、速度、绝对朝向、当前灵力、动作帧、相对位置、双方 HP 等连续状态 | 18 |
| 双方防御/擦弹/受击/空中状态、对方弹幕攻击有效标志 | 9 |
| 双方 action embedding：2×32 | 64 |
| 双方 action_block embedding：2×8 | 16 |
| active_weather embedding | 8 |
| 双方四槽技能类型：2×4×(variant embedding 8 + mask 1) | 72 |
| 上一帧真实 Joint144 动作 embedding | 32 |
| 上一帧方向组合持续时间 | 1 |
| 双方 max_spirit、hitstop 数值及各自有效 mask | 8 |
| 合计 | **228** |

移除量：卡牌相关 586D，加上技能学习/生效等级 embedding 64D；878−586−64=228D。

技能槽在网络内部称 skill_slot_1～4，按槽位拼接；同一槽双方共享类型 embedding。原始 variant 0/1/2 编为 token 1/2/3，token 0 表示未知。每侧 skill_categorical 与 skill_mask 均为 [...,4]。

上一帧动作词表为 145×32：真实动作 0～143，START/PAD=144。observation[t] 只使用真实 action[t−1]。回合/连续片段起点不从上一段寻找历史；采样切片中途仍使用原始轨迹的真实前一帧。

duration 仍指方向水平/垂直组合的持续帧数，不是完整按键组合持续时间；输入 clip(duration[t−1],0,60)/60，边界为 0。

## 3. 四路编码与最终输出

~~~text
当前帧状态 228D ─ Linear(228,256) + LayerNorm + SiLU ─ 256D ┐
连续32帧状态    ─ 独立 TCN32 ───────────────────────── 256D ┤
己方最多3对象   ─ Object Encoder + masked mean/max ─── 128D ┤
对方最多3对象   ─ Object Encoder + masked mean/max ─── 128D ┘
                              concat 768D
                                   ↓
                   Linear(768,1024) + LayerNorm + SiLU
                                   ↓
                          Linear(1024,144)
                                   ↓
                       原始 logits [batch,length,144]
~~~

本次只缩小 Current Encoder，不把最终融合层也改成 256D；融合层隐藏宽度仍为 1024D。

### TCN32

输入 [B,T,228]，独立投影 Linear(228,256)+LayerNorm+SiLU。因果 stem 使用 kernel=2、dilation=1；四个残差块使用 dilation=1/2/4/8，每块两层 kernel=2 因果卷积，通道宽度 256，整体感受野恰好为当前帧加前 31 帧。

TCN 直接读取历史状态向量，不读取 Current Encoder 的输出，也不加入历史对象。删除卡牌和等级同时影响当前分支与历史分支，不存在从缓存继续输入这些量的路径。

### 对象分支

每侧最多 3 个对象，不改选择规则。每对象输入 8 个数值 + action embedding 32D + action_block embedding 8D，共 48D。对象 MLP 为 48→64→64，含 LayerNorm 和 SiLU；masked mean 64D 与 masked max 64D 拼成每侧 128D。无效对象按 mask 排除。

## 4. 单帧动作编码

Joint144 = 9 个绝对九宫格方向 × 16 个 ABCD 组合。固定战斗 bit 顺序为 A、D、B、C，即体术、Dash、轻弹幕、重弹幕。

~~~text
joint_action_id = (direction - 1) * 16 + combat_mask
direction = joint_action_id // 16 + 1
combat_mask = joint_action_id % 16
~~~

范围 0～143；5+无按钮是 ID 64。例如 6+D+A 是一次同时按下的控制状态，不是宏动作。模型不能发出切卡或使用符卡。

forward 返回原始 logits，无 softmax/sigmoid/temperature；训练 CE 接收 logits，推理 argmax 后解码为方向与四个战斗按钮。页面的 softmax 概率仅用于诊断，不代表胜率。

## 5. 数据兼容与训练

既有 v4 NPZ 保留六列原始按键与完整资源元数据，源文件及固定 split hash 不修改，也不需要重新录制。

ReplayStore 加载时：

1. 跳过卡牌及技能等级数组的读取、解压和归一化。
2. 验证技能类型和有效 mask，只将 ABCD 四列投影为 Joint144 标签，不删帧。
3. 若存在旧 Joint432 标签，只用于核对原始数据，不当作新的训练标签。
4. 在投影后构造上一帧动作，再采样序列，防止当前帧标签泄漏。
5. 重新统计 Joint144 的多数动作基线、复制基线和切换帧；卡牌独有变化成为 hold。

已有离线镜像 NPZ 仍使用其真实镜像后轴输入，因此当前动作和历史动作使用相同坐标约定；本次没有新增或变更镜像流程。

保留每批 32 段、31 帧前导 + 32 帧监督、训练集拟合 normalization、AMP、预取、学习率等现有设置。前导不产生独立 CE，但可接收后续监督通过 TCN 回传的梯度。

~~~text
w[t] = changepoint_weight（真实连续历史存在且 Joint144 标签发生改变）
       1（其他帧）
loss = sum(CE[t] × w[t] × valid[t]) / sum(w[t] × valid[t])
~~~

YAML 中关键帧权重仍为 4。验证 NLL、Top-1、Top-5 等按有效监督帧等权统计，best.pt 仍按当前阶段最低验证 NLL 选择。新增/保留的分组诊断不参加反向传播。

注意：忽略卡牌输入/输出不等于把原来的 REP 改成“双方没有用过卡”的对局。卡牌造成的技能替换、伤害或其他世界状态变化仍保留；本实验是不让模型观测和控制卡牌，并非重写游戏规则。

## 6. 推理、日志与兼容性

训练和推理共享动作编码、技能类型处理及 normalization schema。LiveFrames.v1 和 ctypes 布局保持不变，不需为这次网络实验重新构建 DLL。模型只消费同帧技能类型，不读取等级和卡牌。

推理继续要求真实连续 32 帧；缺帧、换局、模型切换按既有规则重置。保留 F10、失焦松键、看门狗、观测过期检查及自动续局。

诊断改为 144 类频率、Neutral=64、方向准确率和四按钮组合准确率，删除卡牌命令准确率。归一化熵除以 log(144)，均匀预测 NLL 约 4.97。旧 Joint432 的准确率、复制基线及 NLL 不能直接当成当前口径。

旧 checkpoint 明确拒绝训练和推理加载，不部分迁移。新版本默认输出 outputs/bc_suika_tcn32_joint144，原模型和日志保留不覆盖。所有保留参数重新随机初始化。

## 7. 启动与验收

在 soku_bc 目录从零开始：

~~~text
python scripts/train.py --config configs/bc_suika_tcn32.yaml --start
~~~

不要带旧模型的 --resume。新版本续训：

~~~text
python scripts/train.py --config configs/bc_suika_tcn32.yaml --resume outputs/bc_suika_tcn32_joint144/last.pt
~~~

网页源码有改动，用户需自行构建后重启服务：

~~~text
npm --prefix web run build
python scripts/play.py
~~~

网络重构阶段未代为编译、运行测试、训练或对局；验收源码包含 144 动作往返、旧数据投影、移除资源不影响输入、228D 拼接、流式/批量一致性及旧 checkpoint 拒绝。实际延迟改善尚需用户测量，不能仅凭缩小维度承诺低于一帧。

## 可选 PALR 训练正则

后续 PALR 任务保留上述网络与关键帧 CE，新增可选 `L_total = L_keyframe_bc + alpha * HSCIC(TCN_feature, previous_expert | current_expert)`。仅训练取出 `[B,L,256]` 的 TCN 输出；没有新增参数，默认推理接口不变。默认关闭，启用方式、数学来源和该次单元检查/短 benchmark 结果见 [PALR 专项说明](palr.md)。
