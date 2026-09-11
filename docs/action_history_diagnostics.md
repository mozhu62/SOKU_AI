# 上一帧复制基线与动作切换诊断

这两个指标只增加统计、日志及界面，不参与 CrossEntropy 或 best 选优。当前项目已切换到 Joint144 / TCN32 网络，因此旧 checkpoint 的兼容性应以 architecture.md 为准；指标定义本身未改变。

## 定义及实现位置

`action_diagnostics.py` 负责计数和比率定义；`ReplayStore.__init__` 利用现有预读过程对每份分片统计，训练/验证分别合并并缓存到内存。运行时写入既有 `dataset_summary.json`，不向 checkpoint 增加必需字段，也不修改 NPZ。固定数据下不在每个训练 step 重新扫描基线。

全量统计的有效位置从现有 `segments` 还原，与正式 BC 监督标签相同；上一帧专家动作来自原有 `previous_actions()` 的结果，不来自模型输出、不自行向前搜索。仅 previous ID 在 0～143 的位置具有有效历史，START/PAD=144、episode 起点、终局/断帧后的片段起点均被排除。真实片段中途开始的随机采样序列，如果具有真实上一帧，则保留其历史。

设 N 为正式指标统计的全部有效帧，H 为其中具有真实上一帧的帧，K 为 H 中当前动作等于上一帧的数量，C 为 H 中动作发生变化的数量：

- 上一帧复制基线 = K/H。
- 动作切换帧占比 = C/N；分母按要求是全部有效帧，不是 H，因此一般不恰好等于 1-K/H。
- 切换 Top-1 = C 中模型 argmax 与当前专家动作一致的数量 / C。
- 切换 Top-5 = C 中专家动作进入模型前五候选的数量 / C。

H=0 时复制基线显示未记录；C=0 时切换准确率显示未记录，不是 0%。此时若 N>0，切换帧占比仍是实际的 0%。

`learner.classification_metrics()` 在 no_grad 中复用现有前向输出，不增加网络推理。`diagnostic_previous_actions()` 跳过 burn-in，再用同一批 CE mask 选择历史动作，排除 padding 和所有无效监督位置。若训练批已经同步镜像，则从增强后的标签与历史直接比较，不另外做镜像。批间合并准确率时先累加正确帧/切换帧数量再相除，不能平均各批百分比。

## 全量与抽样的比较口径

现有验证使用固定种子抽取序列，不是遍历全验证集，本次不改变这一行为。全量基线用于观察数据本身的按键保持倾向；本次验证还记录同批复制基线和具有真实历史位置的模型 Top-1，支持完全相同位置的比较。旧总体 Top-1 仍包含合法的起点帧，不因新增诊断而删除这些样本。

镜像与原始文件的相等/不等关系应保持不变；不执行新的随机镜像、不修改验证集文件清单。全量复制基线高不代表模型无效，但高总体 Top-1 也不能单独证明学会了操作切换或证明当前状态特征的因果贡献。

## 新日志字段

写入 `metrics.jsonl` 的训练/验证记录及实时状态：

| 字段 | 口径 |
|---|---|
| train_previous_action_baseline / val_previous_action_baseline | 固定训练/验证数据全量 K/H |
| train_action_change_accuracy / val_action_change_accuracy | 记录训练批 / 固定验证抽样的切换 Top-1 |
| train_action_change_top5_accuracy / val_action_change_top5_accuracy | 对应切换帧 Top-5 |
| train_action_change_fraction / val_action_change_fraction | 对应批次 C/N |
| train_dataset_action_change_fraction / val_dataset_action_change_fraction | 全量数据 C/N，不随模型训练变化 |
| train_batch_previous_action_baseline / val_batch_previous_action_baseline | 对应批次具有历史的 K/H，与模型在同批位置比较 |
| train_history_joint_accuracy / val_history_joint_accuracy | 对应批次 H 个位置的模型 Top-1 |
| train/val_previous_action_samples | H；不包括 START/PAD |
| train/val_previous_action_copy_correct | K |
| train/val_action_change_samples、action_change_correct、action_change_top5_correct | C 及正确分子，便于审计合并 |
| action_diagnostics_version | bc_action_history_diagnostics_v1 |

旧日志没有这些指标，不反推、不补零。同版本模型重启后开始产生新记录；按既有规则暂停后点“验证”可以记录当前模型结果。旧 Joint432 模型不能用于当前 Joint144 网络，必须从零训练。

## 前端

- 总览“验证动作一致率”图同时保留 Top-1、Top-5、多数动作基线，新增全量上一帧复制基线和切换 Top-1。
- 同区域新增对照表、中文定义、统计分母、本次/全量切换帧占比及同批同位置比较。
- 最近验证记录增加复制基线、切换 Top-1/Top-5、切换帧数和占比。
- 学习诊断增加训练/验证切换指标和训练切换趋势；数据页展示训练/验证全量分子、分母。

## 历史 Joint432 只读统计（2026-09-09，不适用于当前 Joint144）

以下数值保留作历史记录。当前已移除卡牌动作，复制基线和切换帧定义须由 Joint144 数据预读重新计算；本次未运行统计，不提供伪造的新数值。

数据：`soku_cql/data/replay_shards_resources_v4_mirror`；使用其中已有 `train_val_split.json`，没有生成新划分。

固定划分 SHA256：`25cd695ca24c01fccec902240b7cb11b667afc374c2a44442aa66600d5e19845`。统计时逐份校验 NPZ 内容 hash 与该清单一致。

| 项目 | 训练集 | 验证集 |
|---|---:|---:|
| 文件数（包括离线镜像） | 800 | 200 |
| 文件名标记镜像的数量 | 400 | 100 |
| 全部有效帧 N | 8,527,358 | 2,036,992 |
| 可比较历史帧 H | 8,525,416 | 2,036,510 |
| 保持相同操作 K | 7,582,714 | 1,809,398 |
| 动作切换帧 C | 942,702 | 227,112 |
| 上一帧复制基线 K/H | **88.94245161%** | **88.84798012%** |
| 切换帧占全部有效帧 C/N | 11.05503017% | **11.14938105%** |

本地没有 BC checkpoint 或训练日志，因此 **实际 val_action_change_accuracy 尚未取得**，不能从数据本身、旧 Top-1 汇总或 CQL 模型推算。基线统计没有运行神经网络，也没有编译、运行单元测试或训练。

本地 YAML 当前路径为 `data/replay_shards_resources_v4_mirror`，但用户提供的实际目录位于相邻 CQL 项目；本次不擅自改 YAML。只读统计可在 BC 目录运行：

```text
python -B scripts/inspect_action_baselines.py --directory ../soku_cql/data/replay_shards_resources_v4_mirror --split ../soku_cql/data/replay_shards_resources_v4_mirror/train_val_split.json
```

脚本只读取既有划分/NPZ并打印结果，不写文件、不加载模型。验收测试源码为 `tests/test_action_diagnostics.py`（本次未执行），覆盖边界/无效位置、burn-in、加权合并、无切换帧、诊断不改梯度以及同步镜像不改变切换定义。
