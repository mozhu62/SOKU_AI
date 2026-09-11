# Reward V2 调参说明

Reward 参数集中在 `configs/rewards/aggressive_v2.yaml`。训练结构、模型结构和 Reward 策略分离，调整奖励时不需要修改 Python 代码。

## 调整原则

1. 先看 `reports/aggressive_v2/dataset_report.md` 的 Reward 分量累计值和位置风险。
2. 一次只明显修改一到两个系数，并修改 `version`，便于区分实验。
3. 先保证实际伤害与回合胜负占主导，位置和移动只做小幅塑形。
4. 改完运行 `prepare_aggressive_v2.cmd`。脚本会比较 Reward 配置和 Reward 引擎版本的联合哈希，自动重建不匹配的数据。
5. 每 1000 Step 用固定快照实战比较，不要只看训练集 top-1。

## 伤害与胜负

| 参数 | 当前值 | 作用 | 调高风险 |
|---|---:|---|---|
| `damage.dealt_scale` | 0.0015 | 每点造成伤害的正奖励 | 可能只追求换血，不顾回合胜负 |
| `damage.taken_scale` | 0.0010 | 每点受到伤害的负奖励 | 可能过度保守 |
| `damage.self_corner_taken_multiplier` | 1.25 | 自己在墙角受伤时放大惩罚 | 可能为了出墙做危险移动 |
| `damage.opponent_corner_dealt_multiplier` | 1.15 | 对手在墙角受伤时放大奖励 | 可能过度执着压墙 |
| `damage.hit_confirm_bonus` | 0.05 | 每个实际造成伤害的 Transition 加分 | 可能偏爱多段低伤攻击 |
| `outcome.win_reward` | 10.0 | 回合胜利 | 太大时中间行为信号相对变弱 |
| `outcome.loss_penalty` | 10.0 | 回合失败 | 太大时败局 Q 值整体过低 |

`dealt_scale` 与 `taken_scale` 是完全独立的。要提高主动进攻倾向，优先小幅提高 `dealt_scale`；不要同时降低受伤惩罚，否则模型可能学成无脑换血。

## 位置与移动

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `positioning.corner_margin` | 120 | 离舞台边界多少距离算墙角 |
| `positioning.approach_min_distance` | 300 | 超过该距离才计算接近奖励 |
| `positioning.approach_reward_per_unit` | 0.0001 | 每真实接近一个位置单位的奖励 |
| `positioning.retreat_min_distance` | 350 | 超过该距离才惩罚继续后撤 |
| `positioning.retreat_penalty_per_unit` | 0.00005 | 每真实拉远一个位置单位的惩罚 |
| `positioning.corner_escape_reward_per_unit` | 0.0002 | 自己在墙角向场内移动的奖励 |
| `positioning.corner_hold_grace_frames` | 45 | 持续缩墙的宽限帧数 |
| `positioning.corner_hold_penalty_per_frame` | 0.001 | 超过宽限后每帧惩罚 |
| `positioning.opponent_corner_pressure_reward` | 0.0005 | 对手在墙角时有效前进或攻击的每帧奖励 |

接近、后撤和出墙都使用实际位置变化，不按按键本身计分。受击或 Hitstop 帧不做移动塑形，避免把击退误判成玩家主动移动。

当前数据没有可靠的“成功防御”事件字段，因此不直接奖励 BACKWARD。有效防守已经通过减少 `damage_taken`、避免墙角额外伤害和保留回合胜负得到回报。以后采集器若能提供防御成功、擦弹或灵力防御消耗事件，再增加独立的 `guard_success` 分量。

## 示范样本权重

`demonstration_weighting` 不直接改变环境 Reward，而是改变对应 Transition 对训练损失的贡献：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `win_round_weight` | 1.5 | 提高胜局动作的学习强度 |
| `loss_round_weight` | 0.65 | 降低败局动作的学习强度 |
| `winning_attack_multiplier` | 1.5 | 进一步提高胜局 A/B/C 攻击样本 |
| `losing_corner_backward_multiplier` | 0.5 | 降低败局墙角后退样本 |
| `minimum_weight` | 0.25 | 单样本最低权重 |
| `maximum_weight` | 3.0 | 单样本最高权重 |

这部分用于处理现有 Replay 中“萃香败局较多、后退较多”的示范偏差，但不会删除败局。败局仍提供受伤、错误站位和失败终局的信息。

## 推荐调参顺序

如果实战仍然缩墙，依次检查：

1. `predicted_backward_rate` 是否明显高于训练批次中的 `expert_backward_rate`。
2. 先把 `loss_round_weight` 从 0.65 降到 0.55，或把 `losing_corner_backward_multiplier` 从 0.5 降到 0.35。
3. 再把 `corner_escape_reward_per_unit` 小幅提高到 0.00025。
4. 最后才提高 `corner_hold_penalty_per_frame`，单次增幅不要超过 50%。

如果实战盲目地前冲，反向调整上述参数，并优先提高 `damage.taken_scale`，不要直接取消接近奖励。
