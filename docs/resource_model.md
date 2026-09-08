# Joint432 的资源输入

当前网络为 soku_cql_recurrent_joint432_v3，详见[完整架构](model_architecture.md)。旧资源双头权重与新版不兼容，训练、实战均拒绝加载；旧文件保留。

## 真实来源与编码

资源版REP CSV → v4原始NPZ → Dataset统一输入转换 → Current Encoder。实战从主共享内存和同帧资源侧通道读取，检查PID、采集序号、战斗帧及回合一致，不用别的帧或默认技能补齐。

双方四技能槽读取variant、learned level、effective level及有效mask。网络统一使用skill_slot_1～4；原列236/623/214/22仅用于适配槽顺序。variant原值0/1/2，-1未知；level原值0～4，-1未知。真实值加1，token0表示未知。variant embedding8D，两个level各4D。

手牌读取Card ID、费用、槽顺序及有效mask；每侧16个手牌采集槽，不是16个弹幕。全部槽与双方共用16D Card ID embedding，按槽concat，不池化。选中卡ID/费用与首槽一致性在加载时检查；原始selected_index保留，不重复制造选中卡输入。

卡牌数值：card_gauge、card_count、hand_capacity、hand_count、hand_cards_used；另输入有效槽费用。只用训练集拟合归一化，不再沿用旧资源网络的/1000、/5固定缩放。没有未来抽卡或剩余牌堆输入。

每侧资源361D，双方722D直接进入Current Encoder；不增加独立资源MLP或Fusion分支。Current输出仍256，Fusion仍512→256→256。

## 数据复用

- 有资源数组的v4 NPZ直接在内存转换Joint Action标签，不改写文件，不要求重新播放REP。
- 缺少技能/手牌资源的旧数据拒绝加载，不凭空补造技能或卡ID。
- 原始CSV有双方max_spirit/hitstop时新转换器附加保存；旧NPZ未存则使用无效mask。训练集完全没记录的字段在验证/实战也不启用。
- 任意帧切卡和用卡同时为1，转换/加载明确报action schema不兼容，不保留其中一项蒙混过关。
- 旧normalization和checkpoint均不能迁移；默认从随机参数训练到outputs/cql_suika_joint432_v3。
- 已完成CSV含所需字段时无需再次采集。重生成NPZ改变哈希后，需新split_file，不能静默重划分。

## 可视化与核对

训练数据页显示Joint Action分布、技能槽覆盖及可选状态记录数。Current Encoder梯度包含资源embedding，不再显示独立resource_encoder。

实战动作页显示同次决策使用的技能、手牌、能量、上一帧实际Controller Action和432维Q。空槽或未知值不伪装成有效数据。

字段顺序见[数据清单](cql_dataset_fields.md)。本次未构建网页、运行测试或实战；新页面需手动构建后生效。
