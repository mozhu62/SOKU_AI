# PALR：历史动作泄漏正则

## 范围与运行方式

本次只在现有关键帧 BC 损失上附加 HSCIC。当前仓库已经是 **Joint144、当前状态 228→256D、TCN32 输出 256D**，不是附件旧描述中的 Joint432、Current 1024D。保留当前架构，参数仍为 **2,463,664**，不重新引入卡牌或旧动作空间。

默认配置和旧同架构 checkpoint 都关闭 PALR。没有更改 REP/NPZ、动作标签、action_shift、镜像、8:2 划分、normalization、序列采样、模型宽度或实战输入输出。未引入额外神经网络，也没有修改 best 的 Validation NLL 选优规则。

四套配置复制当前 `configs/bc_suika.yaml` 的实际设置：关键帧权重 **16**、缓存 **32 GiB**；不是将旧 `bc_suika_tcn32.yaml` 的权重 4 当作当前基线。数据目录和划分文件保持一致，输出目录分开。

| 配置 | Keyframe | PALR | Alpha | 抽样上限 | 输出目录 |
|---|---|---|---:|---:|---|
| configs/bc_palr_a.yaml | 开，权重 16 | 关 | 0.1（不生效） | 256 | outputs/bc_palr_a_joint144 |
| configs/bc_palr_b.yaml | 开，权重 16 | 开 | 0.01 | 256 | outputs/bc_palr_b_joint144 |
| configs/bc_palr_c.yaml | 开，权重 16 | 开 | 0.1 | 256 | outputs/bc_palr_c_joint144 |
| configs/bc_palr_d.yaml | 开，权重 16 | 开 | 1.0 | 256 | outputs/bc_palr_d_joint144 |

在 `soku_bc` 目录分别执行以下任一条；不要为了对照把四条同时跑在同一 GPU 上。网页出现后按开始，附加 `--headless` 则直接开始离线训练。

~~~text
python scripts/train.py --config configs/bc_palr_a.yaml
python scripts/train.py --config configs/bc_palr_b.yaml
python scripts/train.py --config configs/bc_palr_c.yaml
python scripts/train.py --config configs/bc_palr_d.yaml
~~~

这些命令是使用说明，交付时未执行正式训练。数据路径按项目根目录解析；如果服务器路径不同，四个配置要同步修改数据路径，继续使用同一划分。

续训 C 组并恢复其中的全部 PALR 设置：

~~~text
python scripts/train.py --resume outputs/bc_palr_c_joint144/last.pt
~~~

沿用原 CLI 语义：只传 `--resume` 从 checkpoint 取配置；同时显式传 `--config` 则采用该 YAML 配置，包括 PALR。需要保持续训条件时不要额外传不同配置；改变 Alpha 的实验请使用独立输出目录。PALR 参数本次由 YAML 配置，网页显示运行值，不新增在线改参协议。

## 配置

~~~yaml
palr:
  enabled: false
  alpha: 0.1
  sample_size: 256
  feature_kernel: rbf
  action_kernel: categorical
  regularization: 0.001
~~~

`alpha` 接受任意非负有限数，例如 0、0.001、0.01、0.1、1、10，不表示已选出最佳值。`regularization` 为每样本岭系数 λ，实际矩阵添加 NλI，数值保护要求至少 1e-6。`sample_size` 允许 2～4096，建议保持 256；核矩阵显存随 N²、稠密求解成本随 N³ 增长。

Keyframe 与 PALR 独立开关。PALR 开启且 Alpha=0 仍计算诊断并消耗正则抽样 RNG；需要严格复现无 PALR 路径应设 `enabled: false`。

## 实现核对（对应交付要求）

### 1. 修改文件与模块

| 模块 | 文件 | 作用 |
|---|---|---|
| 数学实现 | soku_bc/palr.py | categorical/RBF 核、HSCIC、合法帧子抽样 |
| 配置 | soku_bc/config.py | 默认关闭、参数校验；不改模型 defaults |
| 标签准备 | soku_bc/dataset.py | 内存中单独准备 previous_expert_action_id；不改磁盘 schema |
| 特征出口 | soku_bc/models.py | forward 的可选 return_aux；原默认输出不变 |
| 优化与验证 | soku_bc/learner.py | 附加 PALR、梯度诊断、固定验证 HSCIC 汇总 |
| 运行与持久化 | soku_bc/runtime.py、soku_bc/checkpoint.py | 指标日志、独立验证 RNG、旧配置默认值 |
| 工作台 | web/src/components/PalrDiagnostics.tsx、web/src/views/Diagnostics.tsx | 损失分解、运行设置、验证 HSCIC、有效帧数 |
| 对照配置 | configs/bc_palr_a.yaml～bc_palr_d.yaml | A/B/C/D 同条件实验 |
| 检查 | tests/test_palr.py | PALR 数学、梯度、边界、恢复和验证检查 |
| 既有测试适配 | tests/test_bc_dataset.py、tests/test_keyframe_loss.py | 新 batch 元数据、可选 aux；修正旧测试对主 YAML 权重仍为 4 的过时断言 |
| 短计时 | scripts/benchmark_palr.py | 内存合成批次，不读取真实数据或保存模型 |
| 文档与授权 | README.md、docs/palr.md、docs/architecture.md、docs/keyframe_weighting.md、docs/licenses/PALR-MIT.txt | 使用、口径与官方 MIT 声明 |

### 2. 特征位置及形状

`BCNetwork.forward(..., return_aux=True)` 返回 `(logits, aux)`；`aux["temporal_feature"]` 是 `self.tcn(state, history_mask)[:, burn_in:]`，形状 `[B,L,256]`，对应监督段。

它尚未和当前帧或对象特征进行 Fusion。筛选和抽样后为 `[N,256]`。没有 detach 该特征；`forward()` 默认返回 logits，`act()`、`step_logits()`、live、play 没有加入 PALR。默认输出与 aux 模式输出逐项相同。

### 3. 真实上一帧与有效性

`read_shard()` 在完整 REP 原始专家动作上调用既有 `previous_actions()`：episode 相同、前一转移有效、前一帧非终局，才填入真实前一标签，否则填 START=144。随后单独复制 observation 的动作历史，PALR 监督数组不引用 observation。

`ReplayStore.sample()` 按当前监督标签的原始行号，一并截取顶层 `previous_expert_action_id[B,L]`。这是 batch 元数据，不是新模型输入、不是新的 NPZ 字段。人工切片起点不会清掉在原轨迹中真实存在的前一帧。

`sampled_palr()` 复用 `build_changepoint_mask()` 的 eligible 规则，联合监督 mask 与真实动作范围。保持帧和切换帧均可参与；episode 起点、断流后第一帧、终局后的断点、padding 和 START/PAD 均排除。断帧依据既有 NPZ transition_valid/episode/terminated，不新造另一套帧边界。

### 4. 离散动作核

当前是 144 个真实类别；实现不依赖 ID 数值间距。对上一动作 Y 和当前动作 Z 都使用 `K[i,j] = float(action[i] == action[j])`。交换类别编号不会改变核，不使用动作 ID 的减法、RBF 或预测动作。

### 5. RBF 带宽

特征核 `Kx[i,j] = exp(-||phi_i-phi_j||² / bandwidth)`。带宽取正的非对角距离平方中位数并 detach；全部特征重合时使用安全回退。统一 detached 尺度缩放避免大数平方溢出，带宽下限 1e-12，指数比上限 80。对 phi 的梯度保留，带宽本身不反传。

### 6. HSCIC 数学来源

依据 [PALR 论文式 (7)](https://papers.nips.cc/paper_files/paper/2023/file/06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf) 和 [官方 core/hscic.py 的 estimate_hscic](https://github.com/KAIST-AILab/palr/blob/main/core/hscic.py)。已核对三个迹项和 Nλ 正则尺度，授权声明随代码保留。

令 X=TCN 特征，Y=真实上一动作，Z=真实当前动作；A=solve(Kz+NλI,Kz)，⊙ 为逐元素乘法。实现为：

~~~text
t1 = sum(A ⊙ ((Kx ⊙ Ky) @ A))
t2 = sum(A ⊙ (Kx @ A) ⊙ (Ky @ A))
t3 = sum(column_sum(A ⊙ (Kx @ A)) ⊙ column_sum(A ⊙ (Ky @ A)))
HSCIC = max(0, (t1 - 2*t2 + t3) / N)
~~~

这与官方估计式数学等价，不是普通 HSIC、相关系数或分类辅助任务。**核选择不是照抄原连续控制实验**：按本任务要求改用 categorical 动作核和稳健中位数特征带宽；估计式不变。浮点微小负值截为 0。

### 7～8. 求解与精度

不使用 `torch.inverse`，使用 `torch.linalg.solve`。kernel 和 solve 位于局部 `autocast(enabled=False)`，特征转换为 float32 后计算，identity/kernel 与其同设备同 dtype。梯度可经过 float32 转换回传外层 AMP 网络。CPU bfloat16 外层和 CUDA fp16 外层均有检查。

### 9. 子抽样与恢复

只从 PALR 合法位置随机无放回取最多 256 帧；不足上限全部保留，不足 2 帧返回可微零并记录跳过原因。BC 仍使用全部有效监督帧及原权重。

训练 `torch.randperm` 使用现有 CPU torch RNG，已由 Learner 的 seed 初始化并由 checkpoint 的 rng_cpu 保存恢复。不使用预取线程的 NumPy RNG，不改变 REP 采样计划。验证每批使用 seed/批号派生的独立 torch Generator，既固定比较样本，又不消耗训练 RNG。

### 10. 最终损失

~~~text
L_keyframe_bc = sum(valid * weights * per_frame_CE) / sum(valid * weights)
L_total = L_keyframe_bc + palr.alpha * HSCIC(phi, previous_expert | current_expert)
~~~

`classification_parts()` 原关键帧定义、label_smoothing、reduction 完全不变。PALR 关闭时直接返回原 CE 张量，不建核、不抽样、不额外加零。

### 11～12. 梯度及关闭等价性

检查确认：只对 PALR backward，至少一个 TCN 可训练参数获得非零有限梯度，Policy Head 不接收该项直接梯度。完整训练日志记录 TCN 的**总损失梯度**（裁剪前），不冒充 PALR 单独梯度。TCN 被人为冻结时，该范数可为零，不能用冻结实验要求它更新。

PALR 关闭时，新旧损失 `torch.equal`，全部参数的梯度逐项 `torch.equal`，训练 RNG 不变。辅助返回不改变 logits 或 state_dict 键。已验证旧同版本 checkpoint 缺少 palr 时加载为关闭；原先被拒绝的 Joint432 等架构仍不会因此变成兼容。

### 13. 日志与网页

原 metrics.jsonl 的 train 记录和控制台诊断间隔增加：

- `loss_total`、`loss_keyframe_bc`、`loss_palr`（别名 `palr_loss`）、`palr_weighted_loss`。
- `palr_enabled`、`palr_alpha`、`palr_sample_size`、`palr_config`。
- `palr_eligible_samples`、`palr_sampled_samples`、`palr_skipped`、`palr_skip_reason`。
- `temporal_feature_norm`：有效监督帧的特征平均 L2；关闭 PALR 时未额外采集，记 null。
- `tcn_gradient_norm`：TCN 总梯度范数；原 `module_gradients` 与 `module_changes` 保留。

验证增加 `validation_hscic`、`validation_palr_eligible_samples`、`validation_palr_sampled_samples`、`validation_palr_skipped_batches`。即使训练关闭 PALR 也测量 HSCIC。它是**各固定验证批次子样本估计的加权平均**，不是整份验证集一次性计算的大核 HSCIC。没有足够合法样本时为 null，不是假零。

网页“学习诊断”新增 PALR 卡片、损失分解曲线和验证 HSCIC 曲线；复用既有组件和配色，按 Sites 的既有工作台规范实现，没有建立第二套页面或部署。既有 Copycat/Keyframe 指标保留。旧日志缺字段显示未记录/断点。

### 14. 单元检查

2026-09-11，在本机 yolo 环境完成 PALR 和关键帧/数据/模型/checkpoint 的 **43 项定向检查，全部通过**。扩展检查共 72 项，71 项通过、1 项既有实战缓冲测试失败；未改该测试来掩盖问题。覆盖以下内容：

- categorical 核及类别重编号、官方三迹矩阵公式等价。
- episode/断帧/终局/padding/人工切片边界、独立监督元数据。
- 常数、重复和极端有限特征不产生 NaN/Inf，HSCIC 非负。
- 人工携带上一动作信息的特征比独立随机特征有更高 HSCIC；条件 HSCIC 不等于普通 HSIC。
- PALR 单独梯度、CPU/CUDA 外层 AMP、损失分解、关闭时损失/梯度/RNG 等价。
- 验证无梯度且不污染 RNG、空组不伪造零、runtime 继续按 NLL 选 best。
- checkpoint 恢复抽样与配置、旧无 PALR 配置兼容、四套消融条件一致。

扩展检查发现一项**本次未改动的既有失败**：`test_live_tcn_stream.test_ring_backfills_real_frames_and_detects_overwrite`。缓冲容量已经是 128，但测试仍期望只写入 36 个新帧便覆盖、丢弃 28 个；实际正确读出 36 个。测试文件和 live/frame_stream.py 的本次 git diff 都为空，按范围要求未修改实战代码或该测试。不能报告全仓检查全绿。

### 15. 短 benchmark

RTX 3050 Laptop GPU，PyTorch 2.13.0+cu132；B=32，监督 L=32，上下文 31，PALR N=256，关键帧权重 16，AMP 开启。合成批次预先放入设备，包含 forward/backward/AdamW，不包含数据 IO、诊断、保存。A/B/B/A，每段预热 3 步、计时 10 步，每种模式共 20 个计时步，不保存模型。

| 短测 | PALR 关闭中位数 | PALR 开启中位数 | 增量 |
|---|---:|---:|---:|
| 第一次 | 25.88 ms | 26.63 ms | +0.75 ms，约 +2.9% |
| 追加跳步计数检查 | 25.80 ms | 26.31 ms | +0.51 ms，约 +2.0% |

追加检查四段测量均无 AMP 跳过更新。短测有显卡时钟和系统负载波动，不能保证正式训练也只增加 2～3%，更不能直接推算服务器性能；真实数据还有 IO 和诊断开销。

可手动复现：

~~~text
python -m unittest tests.test_palr tests.test_keyframe_loss tests.test_bc_learning tests.test_bc_dataset tests.test_bc_checkpoint -q
python scripts/benchmark_palr.py --device cuda --steps 10
~~~

没有编译前端、启动实战或长时间训练。查看新网页需用户自行运行既有 `npm --prefix web run build`，否则旧 dist 不会出现新增组件。

## 如何解读效果

只有在相同核、λ、sample_size、固定验证计划下才比较 HSCIC。categorical 条件动作分布很稀疏时，一批中某些动作只有一个样本，可能使估计偏低；低 HSCIC 本身不是已经解决 Copycat 的证明。还要同时检查切换 Top-1、Hold Top-1、Model Copy Rate、总体 NLL 与真正实战表现。

此正则只约束 TCN 分支；当前状态分支仍按要求保留 previous-action 输入。因此不能声称完全封堵全部历史信息通路，也不保证实战单调变强。本次提供的是可隔离、可复现的 PALR 消融实验。
