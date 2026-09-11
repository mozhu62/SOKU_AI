# 当前 NPZ 数据版 DQfD

## 1. 改造范围与备份

本版在 `soku_ai` 内改造 DQfD，使其直接训练当前 BC/CQL 采集流程生成的 `soku_cql_raw_axes_action_resources_v4` NPZ。没有改 BC/PPO/CQL 项目，没有修改源数据，没有引入宏动作、GRU、BC CrossEntropy 或 CQL conservative loss。

改造前代码、配置、文档和测试源码已备份到项目同级目录：

```text
../_rollback_backups/dqfd_before_resources_v4_20260911_211533/
```

备份共 93 个文件，已逐文件比对 SHA-256。数据、checkpoints、runs 等大文件未重复复制，也未删除或覆盖，仍在原位置。备份中的 `BACKUP.md` 记录范围。恢复时先保留实验产物，再将备份代码复制回去；备份不包含本次新增文件，若完全回退，需另行移出这些新增文件。

本次进行了源码静态核对、目录/清单只读检查与一个真实 NPZ 的元数据/数组头检查，并添加测试源码；没有运行测试、编译、训练、导出或对局。BC 配置指向的 `soku_bc/data` 不存在，但实际数据在 `soku_cql/data/replay_shards_resources_v4_mirror`。已确认 6000 个文件名与清单对应，训练 4800、验证 1200，3000 对原版/镜像没有跨训练/验证集合；抽查 `1005966.npz` 的 SHA-256 与清单一致，字段及 shape 符合资源版契约。这不是全数据读取或训练运行验收，没有新模型或性能实测结果。

## 2. 保留与删除的输入

### 2.1 当前状态

数值/标志输入共 **27 项**，按如下顺序进入网络：

```text
self_position_x
self_position_y
self_speed_x
self_speed_y
self_direction
self_current_spirit
self_action_frame_count
opponent_position_x
opponent_position_y
opponent_speed_x
opponent_speed_y
opponent_direction
opponent_current_spirit
opponent_action_frame_count
relative_x
relative_y
self_hp
opponent_hp
self_guarding
opponent_guarding
self_graze_active
opponent_graze_active
opponent_projectile_attack_active
self_hurt_state
self_airborne_flag
opponent_hurt_state
opponent_airborne_flag
```

类别输入共 **5 项**：

```text
self_action
self_action_block_id
opponent_action
opponent_action_block_id
active_weather
```

动作状态 ID 的 embedding 为 32D，动作块 ID 为 8D，天气为 8D。双方同类 ID 共享 embedding；它们是游戏角色动作状态，不是模型输出的控制按键编号。

所以当前状态编码器的实际输入为：

```text
27 + 2×32 + 2×8 + 8 = 115D
```

删除旧输入中的双方 combo_hits、combo_limit、total_object_count，weather_counter、opponent_character_id、displayed_weather、stage_id，以及 relative_vx/vy、distance_x/y、euclidean_distance。后五项部分可从现有几何量推导，但本版直接移除冗余输入，不补造列。

双方人物 hitstop 也从本版必需输入中删除：新采集有可选 hitstop 字段，但旧 v4 分片不保证有效，不能把无效值当真值。max_spirit 同样不接入这一最小适配版。对象自己的 hitstop 则是八项标准字段之一，继续保留。未使用 Skill、Skill level、手牌、卡槽和卡能量；这些不是原 DQfD 核心输入，本次不扩展它们。

### 2.2 32 帧历史

每个历史行为 **11 个数值 + 4 个类别**：

```text
previous_horizontal
previous_vertical
previous_a
previous_b
previous_c
previous_d
previous_axes_duration
self_action_frame_count
opponent_action_frame_count
relative_x
relative_y
```

类别为同一行的双方 action 和 action_block_id，embedding 合计 80D。因此每历史行是 **11 + 80 = 91D**，再送入旧 TCN。

重要区别：新数据只保留方向符号、按钮是否按下，以及“水平/垂直组合保持时长”。无法还原旧数据每个按钮自己的原始持续计数，所以删除那种计数语义，不把 0/1 冒充持续帧数。

观察行 t 的 `previous_*` 只读取 NPZ 已对齐标签 `action[t-1]`，持续时间也只读取 `duration[t-1]`。不把当前待预测标签 `action[t]` 放进当前输入。NPZ 已执行过 action_shift，本版仅校验配置与元数据一致，不再偏移一次。

历史仍是滑动窗口：连续数据足够时 32 行；回合起点、真实终局后和断帧后重新累计，不跨边界拼接。无历史位置数值为零、类别使用独立 padding 值 -2，并提供 mask。训练随机抽到轨迹中间时仍使用它真实的前序连续历史，不把每个抽样位置当成新回合。

### 2.3 对象

双方分别最多 **3 个**，严格沿用 NPZ 的最近三个对象及其顺序，不再筛选或重新截断。超过三个会报数据契约错误。

每对象八个数值：

```text
relative_position_x / relative_position_y
relative_speed_x / relative_speed_y
direction
action_frame_count
hitstop
hit_count
```

两个类别：action、action_block_id，分别 embedding 为 32D、8D。每对象输入 **8 + 32 + 8 = 48D**。

删除旧对象的 gravity_x/y、hit_box_count、hurt_box_count、frame_data_available、frame_flags、attack_flags、frame_damage、frame_spirit_damage、distance_to_suika。不填充伪造的攻击判定或标称伤害。

## 3. 网络结构

按新默认配置静态推导：

| 分支 | 计算结构 | 输出 |
|---|---|---|
| 当前状态 | 115 → Linear 256 → SiLU → LayerNorm → Linear 256 → SiLU | 256D |
| 历史 | 每行 91 → Linear 64；原四个残差 TCN 块，通道 64/64/128/128、dilation 1/2/4/8，每块两个 kernel=3 卷积；masked mean/max 后投影 | 128D |
| 己方对象 | 每对象 48 → 64 → 64，共享对象 MLP；masked mean 64 + max 64 | 128D |
| 对手对象 | 同上，共享参数 | 128D |
| 融合 | 640 → Linear 512 → SiLU → LayerNorm → Linear 256 → SiLU | 256D |
| Dueling Head | V 分支 256 → 128 → 1；A 分支 256 → 128 → 144 | 144 个 Q |

历史 TCN 的窗口长度为 32，不应把它和卷积堆叠的理论感受野混为一谈。沿用原 TCN 的 GroupNorm、Dropout、残差与池化，没有替换 BC 的 TCN。

输出：`Q(s,a) = V(s) + A(s,a) - mean(A(s,:))`，形状 `[batch,144]`。原始 Q 不经过 softmax。选动作依旧取 Q 最大项。

网络版本为 `tcn_entity_dueling_dqn_resources_v4_v1`，观察版本为 `dqfd_resources_v4_observation_v1`。旧配置仍使用旧维度和参数结构；新版不创建对手角色和场景 embedding。

## 4. 动作语义

输出保持原 DQfD：

```text
horizontal: NONE=0 / FORWARD=1 / BACKWARD=2
vertical:   NONE=0 / UP=1 / DOWN=2
A/B/C/D:    体术 / 小弹幕 / 重弹幕 / DASH

id = (horizontal×3 + vertical)×16 + 8A + 4B + 2C + D
```

共 144 个单帧完整按键组合，id=0 为无方向、无按钮。它不是 5A/236B 等宏招式，也不是 BC 的绝对九宫格 Joint144 编号；即使类别数量相同，不能直接交换 ID。

数据原按钮顺序是 melee、dash、light_projectile、heavy_projectile、change_card、use_spell_card；读取时明确重排为 A/B/C/D。物理左右按当前 self_direction 转成前进/后退，垂直按配置的轴符号转成上/下。

卡牌按钮不在原 DQfD 动作空间内，本版明确丢弃这两个按钮位，但不因为出现卡牌而整帧删除。比如只有用卡键的记录映射成方向 + 无 ABCD；同时有 A 和用卡则保留 A。日志记录含卡牌按钮的投影帧数。由于缺少卡牌动作，这些伤害不能成为“学会主动用卡”的依据，这是保留 144 动作空间的限制。

## 5. 数据与训练

### 5.1 数据隔离

直接读取 NPZ 必需字段，不加载 Skill/Card 和源 rewards 等无用数组，不生成新 NPZ。复用已有 BC/CQL 固定划分文件，校验清单 SHA-256、文件集合、逐文件 SHA-256、8:2 协议、训练/验证不交叉。不会自动重新划分，不把服务器绝对路径写进数据身份哈希。

只对训练分片拟合新版 state/history/object 的均值、标准差，再用于两套数据。DQfD 派生的 split_manifest 和 normalization 单独保存在 `data/dqfd_resources_v4`，不能覆盖源数据清单。两端迁移时 NPZ 文件内容、相对名称和划分需要一致；本版沿用原划分，不额外修正原划分的镜像分组策略。

每次进程启动会核对并扫描全部分片，过程显示进度；同一进程的训练/验证数据实例共用已准备的清单、索引与归一化。不是每个 step 再拟合一次。

`data.cache_gb` 是每个 Dataset 的解压后缓存上限，默认训练/验证各 4 GiB，不包含模型、PER 树、索引和运行库内存。超过预算按最近使用顺序淘汰。大数据随机 PER 采样若频繁解压，应提高预算；内存足够时设 `training.preload_shards: true` 并提供足够缓存。预加载预算不足会明确拒绝，不会声称完成全量预加载。

### 5.2 样本与边界

每条样本仍包括 observation、action、reward、next_observation、done，以及 n_step_reward、n_step_discount、n_step_observation、n_step_done。Observation 自带历史窗口与双方对象。

只纳入 source transition_valid 为真、当前双方 HP 大于零、下一行仍在同一 episode 的转移。文件最后一行不构造没有下一状态的样本。终局后的动画不作为连续战斗动作。

N 步遇终局或无效转移提前停止，折扣为 `gamma^实际步数`。真正终局禁止 bootstrap；断帧/截断不伪造输局，从最后一个已观测的有效下一状态 bootstrap，不穿越断点继续累加奖励。

### 5.3 奖励与优化

使用原 DQfD v1 规则重新从 HP 构造奖励，不读取当前 NPZ 中可能不同的 BC/CQL rewards：

```text
r = 0.001 × (对手掉血 - 己方掉血)
    + 10 × 本次真实转移击败对手
    - 10 × 本次真实转移己方被击败
```

终局结果只对真实 HP KO 判定；边界不可靠或无 KO 证据不伪造输赢。双方同时归零不额外赋胜负奖励。

训练保留：Double DQN 一步 TD、3 步 TD、Huber、专家间隔 0.8、PER alpha=0.6 / beta 从 0.4 到 1、示范优先级 bonus、AdamW、AMP、梯度裁剪及 target 每 5000 step 硬同步。默认 gamma=0.99，学习率 1e-4，batch=128。没有新增动作 0 屏蔽或额外 idle 惩罚；旧函数的这类可选项在新版配置中不启用。

验证仍报告专家 Top-1/Top-5、Q 分布、TD 与 margin 等，不把离线一致率宣称为实战胜率。因缺少对手角色 ID，原按对手角色的分组为空；天气分组使用 active_weather，并输出 `weather_field`，不冒充 displayed_weather。旧 best.pt 选择规则未改。

## 6. 启动与模型兼容

以下命令在 `soku_ai` 项目根目录、已安装原依赖的环境中执行。本次没有新增第三方依赖。

`configs/dqfd_suika_resources_v4.yaml` 的两个源路径已按本机设置；迁移服务器时再调整：

```yaml
data:
  source_dir: ../soku_cql/data/replay_shards_resources_v4_mirror
  source_split_file: ../soku_cql/data/replay_shards_resources_v4_mirror/train_val_split.json
```

默认值已指向本机实际数据，而非 BC 配置里不存在的目录。路径按项目根目录解析；数据也可保留在服务器其他磁盘，只需修改这两个位置。`action_shift` 要与采集时元数据一致，默认 1，已与抽查的真实分片核对。

随机初始化训练：

```powershell
python scripts/train_dqfd_demo.py --config configs/dqfd_suika_resources_v4.yaml
```

续训新版本：

```powershell
python scripts/train_dqfd_demo.py --config configs/dqfd_suika_resources_v4.yaml --resume checkpoints/dqfd_resources_v4/last.pt
```

独立离线验证：

```powershell
python scripts/evaluate_offline.py --config configs/dqfd_suika_resources_v4.yaml --checkpoint checkpoints/dqfd_resources_v4/last.pt --split validation
```

TensorBoard：

```powershell
tensorboard --logdir runs/dqfd_resources_v4 --port 6006
```

默认每 1000 step 验证、保存续训状态并保存轻量 snapshot。`last.pt` 在 `checkpoints/dqfd_resources_v4`，轻量版本在其 `snapshots` 子目录；Ctrl+C 沿用原中断保存逻辑。轻量 snapshot 不含 target/优化器/PER，不能当作完整续训文件。输出目录已有 last.pt 时，不允许不带参数从零覆盖，需显式续训或换输出目录。

输入与网络版本在加载权重前检查。旧 DQfD 权重拒绝载入新版，也没有静默部分迁移；新版需从随机初始化训练。以后新版自身的 resume 继续核对划分、归一化和预处理版本。

模型 shape 检查脚本和导出样例的张量尺寸已按新旧配置区分，但未实际运行。旧实战 `LiveObservationBuilder` 仍使用旧输入协议，本次未接入新版实时观测；它会明确拒绝新版，避免读错列后仍发送按键。旧模型的原实战入口保留。

## 7. 代码位置与后续人工验证

| 模块 | 文件 |
|---|---|
| 字段及版本 | `soku_ai/data/resources_schema.py` |
| NPZ 校验、动作投影、奖励、历史连续性 | `soku_ai/data/resources_reader.py` |
| 固定划分复用、训练集归一化 | `soku_ai/data/resources_setup.py` |
| 有界缓存、历史窗口、对象 padding、N 步样本 | `soku_ai/data/resources_dataset.py` |
| 新旧维度和 embedding 选择 | `soku_ai/models/factory.py`、`state_encoder.py`、`embeddings.py`、`q_network.py` |
| 训练/验证入口与权重检查 | `soku_ai/training/data_setup.py`、`demo_trainer.py`、`checkpoint.py`、`evaluator.py` |
| 新默认训练配置 | `configs/dqfd_suika_resources_v4.yaml` |
| 回归用例源码，未执行 | `tests/test_resources_v4.py` |

人工验收应覆盖：真实 NPZ 清单/字段读取；144 类双朝向动作映射；ABCD 重排；action_shift 0/1 无双重偏移；回合、断帧、终局与切片历史；HP 奖励和缩短 N 步的折扣；只有训练集拟合归一化；源文件未变化；旧模型拒载；训练中断后新模型可续训。没有运行这些用例，因此交付不能视为已通过运行验收。
