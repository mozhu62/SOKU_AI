# BC 卡牌：游戏直接接口、采集和数据转换

> 更新：默认采集配置现已直接输出 BC 训练 NPZ，并自动导出游戏完整卡表。以下为原始档案模式历史说明；当前启动步骤见 `card_training_pipeline.md`，不再要求手填卡表或手动二次转换。

本轮只修改 BC、共享 DLL、共享 REP 采集/转换代码。IQL 不修改，旧数据不覆盖。
没有编译、运行测试、替换 DLL、启动游戏或训练。下文描述源码实现，不代表实机验收通过。

## 接口和 DLL

已删除此前的六个卡牌入口 Hook、跳板、父调用关系及交叉推断。

- 可用集合：遍历手牌，直接调用 `Player::canActivateCard(slot)`，收集对应 Card ID 并去重。
- 实际使用：读取 `Player::handInfo.usedCards`，比较同一逻辑步前后列表，输出新增 ID。
- 只检查读取是否有效、角色/回合是否变化、旧列表是否仍是新列表的完整前缀。
- 列表清空/改写、读失败、一次超过16条新增记录均标为事件无效，不改成没有使用。
- 不重新判断天气、受击、地空、取消条件。游戏接口的返回值就是本项目的可用定义。

沿用游戏逐逻辑帧回调，加速循环每个内部帧都独立读取，不只记录末帧。
无需知道玩家名字才能读取卡牌；左右两侧分别保存。

新二进制版本是 sokubin v3；卡牌扩展每帧240字节，含基础状态共1236字节（压缩前）。
旧 v1 仍可读；此前 Hook v2 明确拒绝，不能混淆字段。

`card_capture[F,2]` 每侧包含：

| 字段 | 定义 |
|---|---|
| before / after | 处理前/后可用集合，valid、count、最多16个 card_ids |
| event_valid | 本逻辑步 usedCards 增量是否可靠 |
| event_count | 本逻辑步新增使用记录数 |
| used_card_ids | 本逻辑步新增 Card ID，按游戏列表顺序 |

基础录制中原有的手牌/费用等字段继续保留，不代表它们都进入 BC 网络。
不再新增天气、手牌隐藏、用卡锁等卡牌推理字段。

正常游戏新增独立共享内存 `Local\SokuDataBridge.Cards.v1.<PID>`，272字节。
包含进程号、采集序号、游戏帧和相同 CardFrame；读写使用 sequence 防撕裂。
BC `live/card_state.py` 提供读取器与 N 维 observation 转换，必须匹配基础状态的采集序号。
State/Resources/LiveFrame 原协议不变。旧采集配置明确关闭卡牌读取；正常游戏默认发布新侧通道。

## 原始采集

REP → 二进制临时文件 → 压缩原始 NPZ，不经过 CSV。
配置 `soku_cql/configs/replay_cards_reimu.yaml` 默认灵梦、独立目录、单槽正常速度。
仍保留每个工作槽独占游戏目录、存储预算、CRC、完整录制凭据和断点续采。
原始 NPZ 验证成功后才清理登记的二进制临时文件。

原始档案 schema 为 `bc_card_capture_direct_v1`；所有原始行/按键保留，不生成奖励或划分。
源玩家元数据保存在 `metadata_json.capture.source_match_metadata`。
当前要求目标角色仅出现在一侧；同角色内战仍需单独处理视角，不自动猜测。

用户手动构建并关闭游戏后更新每份要使用的 DLL：

```bat
cd /d C:\FXTZ_AI\tools\soku-data-monitor
build_release.bat
```

复制 `package\game-module\SokuDataBridge\SokuDataBridge.dll` 到游戏的
`th123\modules\SokuDataBridge\SokuDataBridge.dll`。新导出标记为 SokuCardDirectV1；
采集脚本会在启动前拒绝旧 DLL。

```bat
cd /d C:\FXTZ_AI\soku_cql
python scripts\preprocess_replays.py --config configs\replay_cards_reimu.yaml --limit 1
cd /d C:\FXTZ_AI\soku_bc
python scripts\audit_raw_cards.py C:\FXTZ_AI\soku_cql\data\replay_cards_reimu_raw
```

先核对一局的 Card ID、新增时点、回合清空以及多费卡记录。这里的 Keyframe 是游戏
usedCards 更新时点，不擅自将它描述成更早的按键时点或宏请求时点。

## BC 训练格式

先在你的 BC 配置加入有序完整角色卡表，包含要学习的系统卡、技能卡和符卡：

```yaml
spell_system:
  enabled: true
  character_id: 0
  cards: []  # 必须填写真实的 {id: 整数, name: 名称}，空表明确拒绝
  init_combat_from: null
spell_training:
  active_none_weight: 0.5
  spell_use_weight: 5.0
  max_weight: 10.0
  loss_weight: 1.0
```

不能仅从少量 REP 中出现的卡声称得到了角色全部卡表；当前不伪造完整卡目录。
同一份有序清单同时决定输入位、输出类和 checkpoint，不能在续训时重排。

```bat
cd /d C:\FXTZ_AI\soku_bc
python scripts\prepare_spell_dataset.py --source C:\FXTZ_AI\soku_cql\data\replay_cards_reimu_raw --output data\replay_cards_reimu_training --config configs\你的BC配置.yaml
```

转换需要本机同级 soku_cql 源码，或通过 --cql-project 指定；训练本身不依赖 CQL 运行时。
它复用现有 Combat 转换，使用同一有效行过滤、episode 和三对象截断，不重新定义这些逻辑。
源档案不修改，不覆盖已存在输出。BC 不使用 reward，新 BC 分片的 reward 占位为0。

| 字段 | 形状 | 定义 |
|---|---|---|
| spell_available_mask | [F,N] bool | 当前处理后状态的可用集合 |
| spell_observation_valid | [F] bool | 当前集合读取有效 |
| spell_event_card_id | [F] int | 本帧新增使用ID，-1表示无单个事件，需配合有效位 |
| spell_event_valid | [F] bool | 单分类事件有效；列表重置及同帧多次事件不监督 |
| spell_use_keyframe | [F] bool | event_valid 且有一个使用ID |
| spell_game_frame | [F] int | 原始游戏帧，用于连续性校验 |

Dataset 使用 state[t] 的集合预测 event[t+1]，只跨真实连续、同 episode 的边。
当前待预测 Card ID 不进入 observation。当前发生的事件与最终监督段标签不是相同的行偏移。
同帧多次使用完整保留在原始档案，单分类头无法表达时仅屏蔽 Spell 监督，不删除 Combat。
未知可用集合不伪装成全零；卡表外ID明确报错，不能悄悄当 NONE。

转换脚本不创建固定划分、不拟合 normalization、不启动训练。
原 BC 训练入口目前仍保留旧的8:2划分逻辑；本轮没有实现此前讨论的按玩家9:1/最多30局规则，
因此不要直接让它为新高手数据自动生成划分。这是独立的后续数据划分事项。

## 剩余验收与实战边界

模型双头、Dataset/loss、采集与转换已接线，详见 [模型说明](spell_system_v1.md)。
卡牌共享内存读取器已提供，但旧 live 控制循环尚未接入搜卡/切卡/打卡宏，
所以当前实战入口仍明确拒绝卡牌双头 checkpoint，不会只执行 Combat 冒充支持。
没有声称完成宏实战或前端卡牌面板。
测试源码已更新，未执行。DLL 真实用卡覆盖与事件更新时点仍需实机验收。
