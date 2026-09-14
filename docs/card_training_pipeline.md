# 卡牌采集直接输出训练 NPZ

## 路径

`REP → 每帧二进制临时录制 → 自动转换 BC 训练 NPZ`，不经过 CSV，也不需要手动运行 prepare_spell_dataset。

`soku_cql/configs/replay_cards_reimu.yaml` 已改为 `output_kind: bc_card_training`，输出到独立的 `data/replay_cards_reimu_training`。旧 `replay_cards_reimu_raw` 文件不修改。`card_capture_raw` 模式仍可显式选择。

## 项目内置静态卡表

静态表位于 `soku_bc/catalogs/reimu.json`，包含通用卡 0～20、灵梦技能卡 100～111，以及符卡 200、201、204、206、207、208、209、210、214、219，共43张。输出为 NONE 加43张卡，共44类。铜钱ID12正常保留，不按费用过滤。

来源为 [TH105 继承条目](https://www.thpatch.net/wiki/Th105/Spell_cards/zh-hans) 与 [TH123 通用卡及新增/修改条目](https://www.thpatch.net/wiki/Th123/Spell_cards/zh-hans)。不纳入剧情符卡，不能把 TH105 的旧通用卡覆盖到 TH123 上。中文名称仅用于显示。

采集转换和训练均读取项目内的同一静态表，NPZ 和 checkpoint 保存展开的 cards。推理沿用 checkpoint 的类别顺序，不读取游戏资源、不联网、不依赖训练服务器路径。遇到表外 Card ID 明确报错；其他角色与修改卡牌的 MOD 需要另行配置静态表，不动态扩展网络。

已移除 DLL CardCatalog 导出模块、导出环境变量设置和 SokuCardCatalogV1 检查。不再生成或读取录制旁的 .cards.json，也不生成训练目录的 card_catalog.json。静态卡表内容加入转换签名，修改卡表后不能静默复用不兼容分片。

## 使用

已安装支持 Cards.v1 的 DLL（SokuCardDirectV1）即可采集，不需要额外的动态卡表导出版 DLL：

```bat
cd /d C:\FXTZ_AI\soku_cql
python scripts\preprocess_replays.py --config configs\replay_cards_reimu.yaml --limit 1
```

确认输出后去掉 limit 批量录制。至少有足够的独立 REP 能形成训练/验证两组，再启动：

```bat
cd /d C:\FXTZ_AI\soku_bc
python scripts\train.py --config configs\bc_reimu_cards.yaml
```

该配置从随机初始化训练，不假装旧 checkpoint 续训。玩家级划分由 BC 启动时自动创建，高手小号按 ID 合并，每人验证最多30份独立REP。归一化只在训练集拟合。旧原始 NPZ 现在可以直接用 prepare_spell_dataset 配合 bc_reimu_cards.yaml 转换到独立目录，不必重录或等待生成卡表，但不能覆盖同名已发布分片。

## 实战

原 BC 实战入口已接受同版本卡牌双头 checkpoint。保留批量 Streaming TCN，只在末帧融合处加入可用集合并取两个输出头。DLL Cards.v1 必须与基础状态同进程、同帧、同采集序号；未匹配时释放按键并等新帧，不填零。

模型输出目标 Card ID，宏检查手牌并用 A+B 切卡、B+C 用卡；每次按下后释放并重新读取。目标锁定、超时退出、暂停/失焦重置。读到实际事件可确认成功，目标消失只结束请求，不伪造成功。不重新判断费用、地空、取消条件。推理 JSON 包含 spell_catalog、spell_probabilities、spell_card_id、spell_card_name 和 card_macro。

本次交付源码，未编译、运行测试、启动游戏或训练；实战宏仍需实机验收。已存在的动态卡表文件不删除，但默认流程不再读取它们。
