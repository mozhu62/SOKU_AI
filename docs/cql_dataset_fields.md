# CQL Replay 数据字段

## 角色连续状态（18 项）

1. `self_position_x`
2. `self_position_y`
3. `self_speed_x`
4. `self_speed_y`
5. `self_direction`
6. `self_current_spirit`
7. `self_action_frame_count`
8. `opponent_position_x`
9. `opponent_position_y`
10. `opponent_speed_x`
11. `opponent_speed_y`
12. `opponent_direction`
13. `opponent_current_spirit`
14. `opponent_action_frame_count`
15. `relative_x`
16. `relative_y`
17. `self_hp`
18. `opponent_hp`

## 可选人物状态（4 项 + mask）

新CSV存在时额外保存state_optional_continuous：self_max_spirit、self_hitstop、opponent_max_spirit、opponent_hitstop；state_optional_mask按相同顺序表示是否真实记录。旧v4 NPZ无这些数组则明确无效，不伪造游戏数值0。只有训练集记录过的字段才在验证和实战启用，归一化也仅拟合训练集有效值。

## 角色类别状态（5 项）

1. `self_action`
2. `self_action_block_id`
3. `opponent_action`
4. `opponent_action_block_id`
5. `active_weather`

## 战术状态（9 项）

1. `self_guarding`
2. `opponent_guarding`
3. `self_graze_active`
4. `opponent_graze_active`
5. `opponent_projectile_attack_active`
6. `self_hurt_state`
7. `self_airborne_flag`
8. `opponent_hurt_state`
9. `opponent_airborne_flag`

这些字段由游戏帧数据计算，不包含控制器内部的按键保持时间。

## 必杀技与卡槽资源

以下资源按帧保存，`self` 是萃香侧，`opponent` 是对手侧：

- `*_skill_valid_mask`：低 4 位依次表示 236、623、214、22 槽是否唯一识别。
- `*_skill_variants`：`[帧数, 4]`，每槽取 `0=默认、1=替换1、2=替换2、-1=未知`。
- `*_skill_levels`：`[帧数, 4]`，技能卡学习等级；未知为 `-1`。
- `*_skill_effective_levels`：`[帧数, 4]`，考虑当前游戏修正后的实际生效等级；未知为 `-1`。
- `*_card_state`：`[帧数, 8]`，依次是符卡能量、游戏卡牌计数、当前槽位索引、当前卡ID、当前卡费用、手牌容量、有效手牌数量、已使用手牌数。
- `*_hand_card_ids`、`*_hand_card_costs`：`[帧数, 16]`，从当前选中卡开始排列的卡槽内容及费用；系统卡、技能卡、符卡都按实际卡ID保存。
- `*_hand_mask`：对应卡槽是否有效；不能依靠填充值判断，因为卡ID 0本身可能有效。

萃香四槽对应关系：

| 输入槽 | 默认 | 替换1 | 替换2 |
|---|---|---|---|
| 236 | 妖鬼-密- | 元鬼玉 | 踏鞴 |
| 623 | 地霊-密- | 地霊-疎- | 火鬼 |
| 214 | 妖鬼-疎- | 厭霧 | 鬼神燐火術 |
| 22 | 萃鬼 | 疎鬼 | 攫鬼 |

原始 CSV 仍保存剩余卡组和原始卡组，便于核对采集。最终 NPZ 不把尚未抽到的卡组顺序放进可观测资源，防止模型获得画面上不可见的未来抽卡信息。

Joint432 网络将资源 embedding 和数值直接放入 Current Encoder；无额外资源 MLP。技能类型/学习和生效等级、有序手牌ID/费用及mask均输入；选中卡由首槽表示。card_gauge、card_count、hand_capacity、hand_count、hand_cards_used输入并使用训练集归一化。selected_index不重复输入。完整张量、结构、旧模型兼容与启动说明见 [资源模型](resource_model.md)。

## 对象状态

每个对象保存 8 项数值：

1. `relative_position_x`
2. `relative_position_y`
3. `relative_speed_x`
4. `relative_speed_y`
5. `direction`
6. `action_frame_count`
7. `hitstop`
8. `hit_count`

每个对象另保存 `action` 和 `action_block_id` 两项类别特征。双方对象分别按其与萃香的直线距离从近到远排序，每侧每帧最多写入 3 个，并通过 offsets 标记每个游戏帧对应的对象区间。少于 3 个时只保存实际存在的对象。

被裁掉的数量分别记录在分片 metadata 的 `self_projectiles_discarded` 和 `opponent_projectiles_discarded` 中。完整对象仍保留在原始 `.objects.csv`，不会进入最终模型数据。

## 动作监督标签

### 原始轴存储（不是独立 Q 头）

- `action_horizontal`：`int8`，取值 `-1/0/1`。
- `action_vertical`：`int8`，取值 `-1/0/1`。
- `action_duration`：`int32`，表示当前水平/垂直组合连续保持的帧数；episode 断开时重新从 1 计数。

### 原始按钮存储（不是独立 Q 头）

`action_buttons` 是 `[帧数, 6]` 的独立二值数组，六列依次是：

1. `melee`
2. `dash`
3. `light_projectile`
4. `heavy_projectile`
5. `change_card`
6. `use_spell_card`

六列全零代表无按钮；前四战斗按钮允许任意组合，切卡和使用符卡必须互斥。转换和加载扫描整份数据，同帧两卡命令为1明确报 action schema 不兼容。

训练标签仅为 joint_action_id=(direction-1)*48+combat_mask*3+card_command，范围0～431。combat_mask固定bit0～3为A/D/B/C，card_command为NONE=0、CHANGE_CARD=1、USE_CARD=2。Neutral为ID192。统一编码/解码见[action_space.py](../soku_cql/action_space.py)。

observation[t]加入previous_joint_action_id=action[t-1]，使用433×32 embedding；连续片段/回合起点及终局后使用独立START/PAD token432。previous_action_duration=clip(action_duration[t-1],60)/60，边界为0。待预测action[t]和duration[t]不作为当前输入。

## 转移与奖励

- `episode_id`：标记连续片段边界。
- `transition_valid`：当前状态、动作和下一状态是否属于同一连续片段。
- `rewards`：`damage_dealt * damage_dealt系数 - damage_taken * damage_taken系数`。
- `terminated`：下一有效帧任意一方 HP 归零。

伤害量只在转换过程中由相邻 HP 计算；双方 HP 已包含在状态中，因此不再额外保存逐帧伤害数组。

## 不写入最终分片的字段

`scene_id`、`battle_mode`、`battle_sub_mode`、`match_state`、`stage_id`、原始帧号和回合号都不进入最终 `.npz`。其中少数模式字段只在转换阶段用于过滤非战斗帧，过滤后立即丢弃。

转换器不会加载旧模型参数或旧训练配置。连续特征也不会在转换阶段套用外部统计量。
