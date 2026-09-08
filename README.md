# SOKU CQL 离线训练与 Replay 数据转换

## Git 同步与服务器训练

仓库只同步源码、配置模板、文档和依赖清单；`.gitignore` 已排除数据集、模型、日志、缓存、`node_modules` 和网页构建产物。本地已有数据不删除；服务器所需 NPZ 单独通过 SFTP/SCP 或数据盘传输，不进入 Git 云端。

训练网页默认无 token，通过 SSH 转发后直接打开 **http://localhost:8776/**。首次安装、Git 同步、单独放置数据、tmux 和端口冲突处理见 [服务器同步说明](docs/server_sync.md)。

## 使用模型实战验证（Windows）

Linux 训练完成后，将 CQL 的完整 `.pt` 文件复制回 Windows 游戏电脑。复用 PPO 浏览器工作台布局与操作，不再使用独立 Tk 窗口。更新前端源码后，在 `soku_cql` 目录由使用者手动构建并运行：

```text
npm --prefix web run build
python scripts/play.py
```

打开终端打印的实战链接，默认 `http://127.0.0.1:8777/play.html`；端口被占用时自动递增，不需要 token。顶部直接「选择本机模型 → 加载模型 → 继续」：Windows 原生文件窗口可选择桌面、其他盘的 `.pt`，不用搬进项目，不用改命令行或重启服务。也可从已登记下拉列表切换；当前已加载模型与待加载文件分开显示。先正常启动载入 `SokuDataBridge` 的游戏，进入 **1P 萃香 / 2P CPU 灵梦**。旧 DQN/PPO 模型不会被当作 CQL 加载；新文件校验失败不会卸载旧模型。第一次未安装网页依赖时先运行 `npm --prefix web ci`。

模式判断已经修正：SokuLib 主模式 `battleMode=2` 是人机对战，只有子模式 `battleSubMode=2` 是 REP；正常子模式 0/1 都放行。不再硬性要求 `mode=2, submode=0`，沿用 PPO 的有效战斗/角色判断。

- 固定权重、432 个完整 Controller State 上直接 argmax；无宏、不训练。切卡/用卡互斥。
- 默认 WASD 方向、J 体术、K DASH、I 轻弹幕、L 重弹幕；切卡 O、符卡 P 需与游戏设置一致。在「参数与记录」编辑，暂停后应用并重新加载。
- 默认评估 20 个完整小局，显示伤害差、胜负、Q 值、游戏按键回读及实际动作 ID。中途接管/暂停的片段单列，不计正式胜率。
- F10 或失焦立即松键；整场结束后自动连续按 Z 续局，可在配置中关闭。
- 报告保存到 `outputs/evaluations/<时间-编号>/`，不会修改或覆盖训练的 `last.pt`。

配置：[configs/live_eval.yaml](configs/live_eval.yaml)；完整使用、指标口径和人工验收清单：[实战验证说明](docs/live_evaluation.md)。

## 离线训练入口

训练直接读取 `data/replay_shards_resources_v4/*.npz`，不会启动游戏。网络包含状态编码、对象编码、双方技能/卡牌资源编码、融合 MLP 和 GRU，参数从零随机初始化。唯一 Joint Q 头输出 432 个完整逐帧动作 Q；Current Encoder 接收 Skill/Card 和上一帧实际 Controller Action。训练采用真正的联合离散 CQL。旧 CQL checkpoint 也明确拒绝加载，本版从随机初始化开始。当前默认模型目录为 `outputs/cql_suika_joint432_v3`，完整结构见 [模型架构](docs/model_architecture.md)，资源接入见 [资源模型](docs/resource_model.md)。

在 `soku_cql` 目录安装 Python 依赖与构建工作台（由使用者执行）：

```text
python -m pip install -e .
cd web
npm ci
npm run build
cd ..
python scripts/train.py --config configs/cql_suika.yaml
```

服务默认监听 `127.0.0.1:8776`，不使用 token。服务器训练时，在本地终端用 SSH 将本机 8776 转发到实际服务端口，浏览器打开 `http://localhost:8776/`。端口冲突时程序自动换端口并打印对应 SSH 命令；需要可信局域网直连可显式加 `--host 0.0.0.0`。完成数据准备后点击「开始 / 继续」。首次运行自动创建固定 8:2 划分及训练集归一化参数，不需要重新播放 REP。

无网页立即训练：

```text
python scripts/train.py --config configs/cql_suika.yaml --headless
```

继续 CQL 训练（默认使用 checkpoint 保存的配置）：

```text
python scripts/train.py --resume outputs/cql_suika_joint432_v3/last.pt
```

另开随机初始化实验：

```text
python scripts/train.py --config configs/cql_suika.yaml --output outputs/cql_experiment_02
```

`scripts/run.py` 是同一训练入口的别名。更多架构、CQL 公式、效率设置、配置来源和人工验收说明见 [离线训练说明](docs/offline_training.md)。旧采集入口保留如下；它与训练入口是两个独立流程。

本项目并行启动多个一次性游戏进程播放 REP，由 SokuDataBridge 采集主 CSV、对象 CSV 和完成标记，随后转换为 CQL 训练分片。

转换器是独立实现：不读取任何已有模型文件，不导入旧训练项目，也不使用预定义的组合指令表。后续 CQL 模型按随机参数初始化训练。

## 数据链路

```text
.rep
  -> Python采集器并行启动游戏播放
  -> SokuDataBridge.dll 采集
  -> 主 CSV / objects.csv / done
  -> scripts/preprocess_replays.py
  -> data/replay_shards_resources_v4/*.npz
```

## 使用

在 `soku_cql` 目录执行：

```text
python scripts/preprocess_replays.py --config configs/replay_dataset.yaml
```

这条命令会先处理 `collection.replay_dir` 中的 `.rep`，再转换采集结果。批处理器会跳过已经具有完整 CSV 三件套的 REP，因此中断后可以直接重新执行。

- `--overwrite-capture`：强制重新播放并采集已有 REP。
- `--overwrite`：强制重新生成已有 CQL 分片。
- `--skip-collection`：不启动游戏，只转换已经采集的 CSV。
- `--workers 数量`：临时指定同时运行的游戏实例数。
- `--limit 数量`：第二阶段只转换前若干份 CSV。

游戏、REP、采集目录、加速倍数和并行实例数都在 `configs/replay_dataset.yaml` 的 `collection` 分组配置。所有相对路径以本项目目录为基准。

## 动作标签

NPZ 保留以下原始控制字段；训练标签只有 joint_action_id：

- `action_horizontal`：`-1/0/1`，表示左/无/右。
- `action_vertical`：`-1/0/1`，保留 Replay 中垂直轴的原始正负方向。
- `action_duration`：该水平与垂直组合已经连续保持的游戏帧数。
- `action_buttons`：6个独立列，依次表示体术、DASH、轻弹幕、重弹幕、切卡、使用符卡。

前四列按 A/D/B/C 的固定 bit 顺序构成 combat_mask（0～15），后两列构成互斥 card_command（NONE/CHANGE_CARD/USE_CARD）。同帧切卡与用卡都为1必须报 action schema 不兼容。Dataset 将原始轴转换为 direction 1～9，再计算 joint_action_id=(direction-1)*48+combat_mask*3+card_command，共432类。完整中立动作是ID192。

`action_shift` 控制状态与按键标签的帧偏移，默认值1表示用下一采集帧中已执行的输入作为当前转移动作。observation[t] 只接收 action[t-1] 的 embedding 和 clip(action_duration[t-1],60)/60；片段起点使用 START token432。duration 仅指水平/垂直组合持续时间，不是完整动作时长。

## 分片内容

- 状态：`state_continuous`、`state_categorical`、`tactical_state`；CSV有真实字段时附加双方max_spirit/hitstop及有效mask，旧NPZ缺失则明确无效。
- Replay资源：双方236/623/214/22已安装技能类型、学习/生效等级，以及当前选中卡、完整手牌顺序、卡牌费用、卡槽数量和符卡能量。
- 弹幕对象：双方各自仅保留距离萃香最近的 3 个对象及其偏移索引；不足 3 个时按实际数量保存。
- 动作：水平、垂直、方向持续帧数和六按钮位图。
- 转移：`episode_id`、`transition_valid`、奖励和终止标记。

连续特征当前保存采集原值。应在 CQL 训练阶段仅根据固定训练数据拟合并保存独立归一化参数。完整字段见 [CQL 数据字段](docs/cql_dataset_fields.md)。

新版配置写入独立的 `captured_replays_resources_v4` 和 `replay_shards_resources_v4` 目录，不覆盖旧采集与旧分片。原始 `.objects.csv` 仍保留完整采集结果，方便追溯和重新生成数据；数量裁剪只发生在最终 CQL `.npz` 分片中。

`scene_id`、`battle_mode`、`stage_id`、原始帧号、回合号和双方逐帧伤害数组都不会写入最终分片。采集阶段需要的模式字段只用于筛选有效战斗帧，筛选完成后即丢弃。其余数组使用训练端可直接读取的常规类型，不再做位图和窄整数转换；文件仍使用压缩 NPZ。

本次交付源码和配置，未编译前端、未运行训练或测试。已有 NPZ 不会因新增训练模块而被改写。
