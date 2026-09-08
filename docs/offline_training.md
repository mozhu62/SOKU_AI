# CQL 离线训练说明

## 实现边界

本次复用 PPO 工作台的 React / TypeScript / shadcn 按钮与对话框 / Recharts / FastAPI 组织方式，以及角色编码器、集合对象编码器、融合层和单层 GRU 的结构。代码复制到 CQL 项目内部，不导入 `soku_ppo` 或 `soku_ai`，不加载任何 PPO/DQN 参数。旧项目和已有训练数据未修改。

离线算法参考 [CQL 论文](https://arxiv.org/abs/2006.04779) 与 [作者实现](https://github.com/aviralkumar2907/CQL)。本实现是432-way完整Controller State的联合离散CQL，使用Double DQN目标；不是把 PPO loss 重命名，也不是连续动作 SAC-CQL。

训练器负责离线训练和离线验证，不启动游戏或发送键盘，也不使用实时 rollout、PPO clipping、GAE 或在线熵奖励。训练完成后的模型由独立 Windows 实战入口 `scripts/play.py` 读取，实战入口只推理和统计，不修改模型。当前网络的完整结构见 [CQL 当前模型架构](model_architecture.md)。

## 默认网络

唯一Joint Q头输出432个逐帧完整Controller State，不使用方向/按钮分项Q。主体结构：

```text
Current Encoder（人物、资源、上一帧实际输入） → 256
双方各最近3个对象 → 每侧masked mean64+max64 → 128×2
concat512 → Fusion两层256
Battle256 → 单层GRU128
Battle256+GRU128 → Memory Fusion256
Joint Q Head: Linear256→128 → SiLU → Linear128→432
```

完整字段和默认878D Current输入展开见[模型架构](model_architecture.md)。原始v4资源NPZ可直接复用；新转换器额外保存原CSV实际存在的max_spirit/hitstop，旧NPZ缺失时用无效mask而非真实0。

action=(direction 1～9, combat_mask 0～15, card_command 0～2)，joint ID=(direction−1)×48+combat_mask×3+card_command。bit顺序A/D/B/C；卡命令NONE/CHANGE_CARD/USE_CARD互斥。原始输入同帧切卡与用卡重合时，自动清除切卡、保留用卡和其它按钮，不删除帧。Neutral为ID192。

卡键清洗在标签编码和上一帧历史生成前完成，新 NPZ 写入清洗后的按钮，旧 NPZ 只在加载内存中清洗。已有文件和固定划分哈希不变，无需重采/重转。启动日志逐份记录清洗数量；`dataset_summary.json` 按训练/验证分组记录 `card_overlap_cleaned_rows`（转换时已清洗 + 本次加载清洗）、`card_overlap_cleaned_on_load`（本次加载清洗）、`card_overlap_cleaned_files`（涉及分片），页面「数据与覆盖率」同时展示。统计范围为分片全部标签行，包含末帧，不等同于有效训练转移数；重复缓存加载不累加计数。规则 `prefer_use_card_v1` 不改变432类动作编码、网络、gamma或奖励。

当前observation仅加入previous_joint_action_id及上一帧方向组合持续时间clip60/60，不加入当前标签。START/PAD=432；Embedding(433,32)。技能slot使用variant8D、两个level各4D；Card ID双方所有槽共享16D并按槽concat。主干宽度不扩大，每侧对象仍最多3个。

网络soku_cql_recurrent_joint432_v3完全随机初始化；目标网络复制online。旧CQL/PPO/DQN checkpoint全部拒绝，包括实战，不做部分加载。默认独立目录outputs/cql_suika_joint432_v3。

## 离线 CQL 更新

每个样本只有一个joint_action_id标签。q_data直接从online的432维Q中gather，不拆方向和按钮loss。

更新使用：

```text
a* = argmax_a Q_online(s_next,a)
y  = r + gamma * (1 - terminated) * Q_target(s_next,a*)
TD = mean(Huber(Q_online(s,a_dataset), stop_gradient(y)))

CQL = mean(T * logsumexp_a(Q_online(s,a)/T) - Q_online(s,a_dataset))
loss = TD + cql_alpha * CQL
```

保守项直接对同次前向输出的全部432个完整动作计算logsumexp，不做可加分解。cql_temperature只用于保守项，不是策略温度；网络末层没有softmax，推理直接argmax。

在线网络负责下一动作选择，目标网络负责下一动作评分，目标值停止梯度。每次有效优化后按 `target_tau` 软更新目标网络。AMP 梯度溢出时跳过优化器和目标网络更新，并记录 `optimizer_skipped`。

奖励直接使用 NPZ 的 `rewards`；当前转换器的奖励由双方 HP 变化计算。训练 YAML 不会偷偷覆盖已存奖励，也不会添加在线奖励。`gamma` 是每个相邻游戏帧的折扣，延迟命中的回报通过多次 Bellman 更新往前传播。

## 样本、序列与有效性

默认一次批次含 32 条序列，每条最多 32 个学习转移，以及最多 16 帧 burn-in 与最后一个下一状态。完整序列的输入跨度最多 49 个游戏状态帧，32 个学习转移约为游戏时间 0.53 秒（60 FPS）。

随机选择一个有效转移作为学习起点，向前取同片段 burn-in 恢复记忆，向后取学习序列。burn-in 停止梯度。若临近片段起点，使用真实的前导长度与零初始记忆；补齐不能产生虚假历史。后端使用 packed GRU 处理不同 burn-in 长度。

片段结束、非连续转移、回合边界和终局都会切断序列。短片段右侧补齐，并用 loss mask 排除补齐位置；TD 和 CQL 使用相同有效 mask。终局下一状态可用于配对，但 bootstrap 为零。非终局片段末端仅在当前转移及下一状态有效时 bootstrap，不跨缺帧拼接。

默认每批先按有效转移数加权选择 4 个 REP，再从各 REP 中随机选多条序列，以减少解压/缓存切换。允许不同序列重叠；累计样本数不是不重复样本覆盖率。没有 PER，CQL 保守项按这个固定离线采样分布计算，未混入旧 DQfD 监督大间隔项。

## 训练/验证划分

完整运行预处理入口后，会按整份 REP 对应的资源版 NPZ 建立或校验 `data/train_val_split_resources_v4.json`；只有已有 NPZ 时也可执行 `python scripts/preprocess_replays.py --only-split`。训练首次启动发现清单不存在时仍会自动建立。默认 seed 42，约 80% REP 用于训练，剩余 20% 用于验证；按文件数而非帧数划分，整数取整。字节完全相同的重复 NPZ 归同组，重复组不会跨集合，重复组存在时最终文件比例可能略偏离 8:2。不同文件名但重新编码/重新压缩产生的重复录像不能仅靠文件哈希自动辨认。

划分清单持久化并校验内容哈希。新增、删除或改写 NPZ 后不自动重分：为新的数据版本指定新的 `data.split_file` 与模型输出目录。已有清单被篡改、两集合重叠或缺文件时启动失败，不静默跳过。

状态、对象、卡牌数值/有效费用、可选人物字段归一化只拟合训练REP；同一份参数用于训练和验证，标准差过小时设为 1。归一化值裁剪到 [-10, 10]，类别和战术二值不归一化。checkpoint 保存归一化、字段规范和划分哈希。

默认每 1000 step 验证 20 个固定随机采样批次，种子与训练采样分开，每次验证重用同样的样本计划。**这是固定验证集内的固定抽样验证，不是每次遍历所有验证帧。** 修改批量、序列设置或验证批次数后是新的比较阶段。

验证没有 backward、optimizer step、目标网络更新。显示 TD MSE/MAE、EV、Q/目标分布、完整Joint Action专家一致率、Neutral占比和Joint频次。验证 TD 目标随目标网络变化，因此 MSE 下降不证明实际对局变强；专家一致率也不是胜率。目标方差接近零时EV不计算（null），界面标明不适用，不填零。

## 效率与配置

效率实现参考现有 DQN 训练器的预读缓存、批量数据搬运、AMP 和目标网络做法：

- 启动扫描、校验并预读 NPZ；容量足够时全部常驻内存，不再每步读 CSV 或解压 NPZ。缓存不足时使用 LRU，工作台显示命中率。
- NumPy 批量索引角色和每侧三对象；不按每帧构建大量 Python 对象。
- 后台单线程提前准备两批，并在 GPU 模式下准备固定页内存；训练线程执行 non_blocking 设备搬运。
- 整段融合 GRU、AdamW、CUDA/ROCm 上 AMP；CPU 自动关闭 AMP。Bellman 目标和保守项用 float32。
- 重诊断按 `log_interval` 记录，网页摘要每秒最多两次、图表每秒一次；不逐步复制全模型给网页。

关键参数：

| 参数 | 默认 | 用途 |
|---|---:|---|
| data.cache_gb | 4 | NPZ 解压数组缓存上限，单位 GiB；不含批次、模型和临时数组 |
| batch_size | 32 | 一次更新的序列数，并非单帧样本数 |
| sequence_length | 32 | 每条序列最多参与损失的转移数 |
| burn_in | 16 | 前导状态帧数，不反向传播 |
| replays_per_batch | 4 | 一批混合几个随机 REP |
| prefetch_batches | 2 | 后台 CPU 批次预取深度 |
| amp | true | GPU 混合精度，CPU 不启用 |
| learning_rate | 0.0001 | AdamW 学习率 |
| gamma | 0.99 | 每帧 Bellman 折扣 |
| cql_alpha | 1 | 保守正则强度；0 退化为这里的 Double DQN TD 目标 |
| cql_temperature | 1 | CQL logsumexp 温度 |
| target_tau | 0.005 | 目标参数向在线参数靠近的比例 |
| max_grad_norm | 10 | 梯度范数裁剪 |
| log_interval | 20 | 指标与模块梯度/变化量记录间隔 |
| save_interval | 1000 | 保存 last.pt 和带步数历史版本 |
| validation_interval | 1000 | 自动验证间隔 |
| validation_batches | 20 | 每次固定验证批次数 |

根据页面的取数等待、缓存命中率和显存实际情况调整缓存/批量；这里没有执行性能测量，不能保证相较 DQN 的具体加速倍数。速度是最近记录步的样本吞吐，不包含暂停和验证；累积时间卡单独列出这些阶段。

## 工作台与控制

- 总览：训练与验证误差、速度、时间分解和具体诊断。
- 学习诊断：数据/模型argmax的Joint Action Top-N及全部432项；Neutral占比；Q_data/Q_max均值标准差、Q_max−Q_data、CQL gap、TD MSE/MAE、Validation EV与完整expert agreement；模块梯度/权重变化。
- 数据与覆盖：固定划分及输入字段，训练/验证动作分布，分页文件清单。
- 参数与模型：编辑值、运行值、来源、参数锁、模块冻结、模型保存位置。

仅允许暂停且当前更新/验证结束后应用热参数。页面提交阶段编号，过期配置会拒绝；参数锁需通过「保存参数锁」显式应用。正常更新、暂停和保存共用一个训练线程，浏览器不会直接操作模型。

模块冻结使用 requires_grad=False 并清空梯度，保留优化器状态；解冻后继续使用原来的 AdamW 状态。只冻结Joint Q头时共享主干仍可变化，因此输出不一定固定；资源embedding计入Current Encoder。全部冻结会拒绝。

修改参数后保存 `configs/stage_*.json` 和阶段记录；旧曲线按阶段区分，不将不同参数条件下的验证结果直接连在一起。普通应用不会修改输入维度、数据划分或动作语义。

API/网页默认监听 `127.0.0.1`，不使用 token。通过 SSH 转发后，本地浏览器直接打开 `http://localhost:8776/`，也支持本地端口与服务器端口不同的转发。Host 允许 localhost、回环、私有或链路本地 IP，浏览器 Origin 必须与 Host 完全一致。服务从 8776 开始直接绑定 socket，失败时自动尝试后续端口，并打印对应 SSH 转发指令；HTTP 与 WebSocket 使用同一个服务，无外部 CDN。确需可信局域网直接访问时，显式设置 `--host 0.0.0.0`；此服务没有登录认证，不要暴露到公网。代码同步、独立传输数据集及 SSH 操作见 [服务器同步说明](server_sync.md)。

网页断线不会停止离线训练，重连也不自动恢复已暂停任务。同一请求编号不会重复执行控制动作。两个训练实例不能同时使用同一个输出目录；可用 `--output` 指定独立目录运行。

## 保存与继续

默认模型目录为 `outputs/cql_suika_joint432_v3`：

```text
last.pt                       最新可续训状态
best.pt                       当前阶段最低验证 TD MSE 的模型
best_stage_N.pt               每个阶段保留的最佳验证版本
snapshots/step_*.pt            每 1000 step 或手动保存的历史版本
config.json                   最后应用的完整配置
configs/stage_*.json           修改历史快照
normalization.json            仅训练集拟合的归一化
dataset_summary.json          文件与动作覆盖统计
metrics.jsonl                 训练、验证和配置阶段记录
error.log                     如果发生运行异常，保存调用栈
```

「最佳」仅按离线验证 TD MSE 定义，不是实战最强保证；每个参数阶段单独比较。保存先写临时文件再原子替换，包含在线/目标网络、优化器、AMP scaler、随机状态、步数、配置及数据哈希。正常暂停不丢弃已经完成的更新；Ctrl+C、停止按钮都会等待当前更新并保存。异常训练不把未确认状态覆盖进旧 checkpoint，最多回到最近一次成功保存。

从头开始与续训指令：

```text
python scripts/train.py --config configs/cql_suika.yaml
python scripts/train.py --resume outputs/cql_suika_joint432_v3/last.pt
python scripts/train.py --resume outputs/cql_suika_joint432_v3/last.pt --config configs/cql_suika.yaml
python scripts/train.py --config configs/cql_suika.yaml --headless
python scripts/train.py --config configs/cql_suika.yaml --output outputs/cql_new --port 8786
```

续训不带 `--config` 时使用 checkpoint 中保存的配置（包括工作台调整）；显式给 `--config` 时以该 YAML 为准，可调项差异会产生新阶段。结构、seed、动作方向定义与划分不兼容会拒绝。无 `--resume` 时若输出目录已存在 last.pt 会拒绝覆盖，使用新的 `--output` 从随机参数开始。

新 CQL 模型只能由本项目 checkpoint 加载器加载；PPO / DQN checkpoint 会明确拒绝。

## 人工验收（本次未执行）

本次仅源码审阅，遵循要求未安装依赖、编译、运行测试、生成划分或启动训练/网页。

1. 用户构建前端并启动后，默认应打印本机地址与 SSH 转发指令；显式启用局域网监听时才打印局域网 IP 链接。8776 被占用时打印实际备用端口和匹配的转发指令，HTTP 与 WebSocket 均可通过同一个本地转发端口访问。
2. 同一 seed、同一数据重启使用相同划分；训练与验证文件集合不相交；归一化来自训练集。
3. 完整432种encode/decode逐一往返通过；前四按钮任意组合保留，切卡/用卡重合时仅清除切卡；不删帧、不改NPZ文件，重复加载计数不累加；上一帧输入没有标签泄漏。
4. 终局、缺帧前后不拼接序列；burn-in 与 padding 不参与损失，终局 bootstrap 为零。
5. 全部联合 Q 相等时，保守 gap 应为 T × log(432)；alpha=0 时总损失只含 TD。
6. 验证前后在线网络、目标网络和优化器状态不变；验证仅使用隔离的 validation 名单。
7. 冻结模块记录的参数变化为 0，未冻结模块可变化；解冻不重建优化器。
8. 保存/续训恢复 online、target、optimizer/scaler 与相同采样步计划；不读取 PPO 权重。
9. 同一控制请求重复发送不重复启动；网页关闭不停止线程；停止后 last.pt 可续训。
10. 在 1366×768 与 1024×768 下核心控制可见，表格局部滚动，缺失指标显示「未记录」。

新增测试源码位于tests/，本次未运行。用户后续可在soku_cql目录手动执行：

```text
python -m unittest discover -s tests -v
```
