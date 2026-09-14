# BC 卡牌意图双头（spell_v2）

> 更新：完整卡表自动导出、直接训练格式采集和实战宏已接入源码，当前流程见 `card_training_pipeline.md`。本文后续涉及“手填卡表/宏未接入”的旧限制由新流程替代，实机验收仍待完成。

基于游戏直接接口：当前可用集合来自 canActivateCard，标签来自 usedCards 增量。
采集、转换、字段与命令见 [采集说明](card_capture_raw.md)。IQL 不动。

## 网络

保持现有 Combat 与 TCN 参数尺寸；卡牌关闭时不创建额外模块，旧单头 BC 仍兼容。
TCN32/256 由原配置决定；不增加 GRU 或历史卡牌输入，不破坏现有 TCN 缓存计算。

```text
当前状态 228D → Current Encoder → 256D ────────┐
历史状态 [32或256,228] → 原 TCN → 256D ────────┤
当前己方对象（最多3） → 128D ──────────────────┤
当前对方对象（最多3） → 128D ──────────────────┘
                      拼接768D → 原 Fusion → 1024D
                                                 │
当前可用集合 N维 → Linear(N,32)→SiLU→Linear(32,32)→SiLU
                                                 │
       1024D + Linear(concat(1024D,32D),1024) → 1024D
                           ├─ 原 Combat Linear → 144 logits
                           └─ Linear(1024,128)→SiLU→Linear(128,N+1)
```

N 是配置中的角色卡牌数，卡牌类0为 NONE，1..N 按卡表映射到真实 Card ID。
新增融合残差层零初始化，开始时不会随机扰动旧 Combat 特征。
当前集合同时影响两个输出头；不把全部手牌顺序、费用、天气附加字段塞入网络。

forward(return_aux=True) 返回 Combat logits 与 aux.spell_logits。
act() 对两个头分别 argmax，返回 combat_action、spell_action（类别）和 spell_card_id（真实ID，NONE=-1）。
**卡牌是意图输出，不以当前集合硬屏蔽 logits。** 不自行判断地空、取消或受击条件。
控制宏是否能立即执行是执行层问题，不改变标签。

## 损失与 Keyframe

Combat CE、Combat Keyframe、PALR 和已有序列采样保持原实现。
额外：total_loss = 原损失 + spell_training.loss_weight × SpellLoss。

- Passive NONE：无可用卡且没有使用事件，不训练 Spell；Combat 正常训练。
- Active NONE：有可用卡、有效事件观察中没有用卡，Spell 目标 NONE。
- Use：有效 usedCards 新增目标，训练对应 Card ID；即使输入集合为空也不丢弃真实事件。
- Unknown/同帧多事件：不当 NONE，屏蔽 Spell 监督。

SpellLoss = sum(weight × CE) / sum(weight)。默认 Active NONE=0.5、Use Keyframe=5.0，
两者独立可配置，不复用 Combat 的 changepoint_weight。无监督时返回可反传的零。
没有 masked softmax、额外 reward、Q/V 或即时合法性 loss。

诊断继续通过现有日志提供独立 Spell loss、监督数量、Use/NONE数量、混淆矩阵、
每卡 precision/recall、使用率与非NONE准确率；本轮不新增网页面板。

## checkpoint 与范围

新协议为 bc_spell_used_cards_v1，网络后缀 _spell_v2。
旧单头模型继续原样加载；旧实验 _spell_v1 因接口及损失语义改变明确拒绝，不静默迁移。
显式 init_combat_from 可迁移同尺寸单头参数并新建卡牌分支；仍要求原归一化一致，
不将“新数据重新拟合 normalization”伪装成等价参数迁移。默认 null 为随机初始化。

本轮未编译/运行测试或实机验收，未替换 DLL。
双头模型算法 act 已支持意图输出；旧实战控制循环的宏接线仍是独立待办。
