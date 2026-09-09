# BC：GRU 与连续 32 帧 TCN 对照

## 1. 范围与结论边界

本版提供两种时序结构和配套训练、网页比较、实战加载。实验是否提升模仿准确率、速度和实际对战表现，需要使用者运行后判断；本次按要求没有编译、执行测试、启动训练或游戏。

不改状态字段、Skill/Card、上一帧实际动作、Joint432、对象截断（每侧最多 3 个）、专家标签、CE、8:2 划分、normalization 或 best 的验证 NLL 选优规则。不引入奖励、TD、宏动作、PPO、Attention 或 Transformer；不修改旁边的旧工程。

## 2. 冻结与真正移除前向影响

| 模式 | GRU 权重 | GRU 前向 | 有效时序输入 |
|---|---|---|---|
| GRU 正常训练 | 更新 | 执行 | 256D 战斗特征＋128D 循环隐藏状态 |
| GRU 冻结实验 | 不更新，清空梯度 | 仍执行 | 同上，仍影响分类输出 |
| TCN32 | 强制冻结，不能在表单解冻 | 不执行 | 当前帧＋前 31 帧战斗特征 |

TCN 包中保留 GRU 参数以明确记录其旁路状态，不调用 GRU、不向 GRU 传梯度。冻结 GRU 本身不会固定整个策略：上游编码器和下游分类头仍可更新。若要验证“没有 GRU 的影响”，使用 TCN 模式，而不是只勾冻结。

## 3. 网络结构

公共部分保持：

```text
Current Encoder: input → 256 → 256，LayerNorm / SiLU
Object Encoder: 每侧最多 3 个对象，每对象 48 → 64 → 64
                masked mean 64 + masked max 64 = 每侧 128
Fusion: 256 + 128 + 128 = 512 → 256 → 256
时序分支: GRU 或 TCN，输出 128
Memory Fusion: 当前 Battle 256 + 时序 128 = 384 → 256
Policy Head: 256 → 128 → 432 原始 logits
```

TCN 分支：

1. Battle Feature 256 经 Linear、LayerNorm、SiLU 投影至 128。
2. 五个 128D 残差块，单层卷积 kernel=2，dilation 分别 1/2/4/8/16。
3. 每块为左侧 padding → Conv1d → 逐帧 LayerNorm/SiLU → 逐帧 Linear → 残差/SiLU。
4. 感受野 `1 + 1 + 2 + 4 + 8 + 16 = 32`，只覆盖当前及历史战斗特征，不覆盖未来。
5. 不做整段池化，不用跨时间归一化；每个输入时刻各输出 128D。

“32 帧”限定的是 TCN 的 Battle Feature 窗口。原 observation 仍含动作帧数和上一帧方向持续量等状态，不把这些合法状态删掉来制造严格的原始事件截断。

## 4. 序列、mask 与初始化

默认旧 GRU 配置仍为 `burn_in=16`。两份新对照配置统一使用：

```yaml
seed: 42
training:
  batch_size: 32
  sequence_length: 32
  burn_in: 31
  learning_rate: 0.0001
  label_smoothing: 0.0
```

每条读取最多 31 个前导帧和 32 个监督帧，CE 仍只作用于有效监督位置。短前导由 Dataset 右侧填充，TCN 分支将真实前导重新对齐到监督段左边，并逐层抹去无效填充，避免偏置制造虚假历史。GRU 保持原有 packed sequence 与长度处理。

- GRU：前导 no_grad，监督段进行截断时序反传。
- TCN：前导不产生独立 CE，但后续监督帧的梯度能流向实际使用的前导编码特征。
- 片段起点使用原 START/PAD 输入；不跨回合、终局、断帧拼接，不输入当前标签。
- 新实验全部随机初始化，不迁移旧 BC/CQL/PPO 权重。同 seed 时公共模块在创建 TCN 之前完成初始化，使公共层起始权重相同。
- 两组保留相同采样规则、监督位置和验证种子；但参数量、上下文梯度路径、实战记忆长度不同。这是时序结构对比，不是宣称参数量完全匹配的实验。

## 5. 命令行启动

命令在 `soku_bc` 目录执行。两份 YAML 默认只读相邻 CQL 工程中已有的镜像资源分片：

```yaml
data:
  directory: ../soku_cql/data/replay_shards_resources_v4_mirror
  split_file: ../soku_cql/data/replay_shards_resources_v4_mirror/train_val_split.json
```

服务器没有这个目录结构时，同时修改两份 YAML，指向已经传输的数据及相同划分；路径相对 `soku_bc` 项目根解析。不要因某组启动失败删除既有划分或模型。

前端依赖/构建由使用者执行（不要求重新安装 PyTorch）：

```text
npm --prefix web ci --include=dev
npm --prefix web run build
```

依次启动两组，避免资源争用干扰速度比较：

```text
python scripts/train.py --config configs/bc_suika_gru32_control.yaml
python scripts/train.py --config configs/bc_suika_tcn32.yaml
```

默认准备后暂停，网页端口 8796，占用时自动顺延。点击“开始 / 继续”训练；无网页可加 `--headless`，网页自动开始可加 `--start`。

独立模型目录分别为：

```text
outputs/temporal_experiments/gru32_seed42/
outputs/temporal_experiments/tcn32_seed42/
```

续训：

```text
python scripts/train.py --resume outputs/temporal_experiments/gru32_seed42/last.pt
python scripts/train.py --resume outputs/temporal_experiments/tcn32_seed42/last.pt
```

不传 YAML 时使用 checkpoint 内配置。`--temporal-mode tcn` 可作为新训练的覆盖参数，自动设前导 31；仍须另设新输出目录，不允许用 GRU checkpoint 跨架构续训。公平对照优先使用两份完整配置，避免将旧 16 帧前导与新 31 帧前导混比。

## 6. 网页实验与冻结

“参数与模型”中可以勾选冻结 GRU，暂停后应用。页面明确区分编辑冻结列表和实际旁路状态；“全部有效模块训练”不会重新启用 TCN 模式中被旁路的 GRU。

“时序实验”页：

1. 暂停，等待当前更新结束。若锁定参数，先显式解锁。
2. 选择 GRU 或 TCN32，填写唯一的 ASCII 实验名；GRU 可另勾冻结实验。
3. 确认保存当前并创建实验。新模型/优化器/计数独立，使用 31 帧前导，现有冻结列表重新选择。
4. 创建后保持暂停；点击开始才训练新组。
5. 对照表及最多四组曲线显示验证 NLL、Top-1、切换帧 Top-1、实际更新数、累计监督帧与吞吐。

已有目录不覆盖；创建失败不替换原模型。新组文件失败时保留供排查，换一个实验名重试。切换过程中原分支先保存 `last.pt`，原 best 和历史不清除；之后可以用原分支的 `last.pt` 继续训练。训练线程不会同时优化两组。

`comparison.json` 只保留本阶段最近 100 次验证摘要；完整记录仍在各组自己的 `metrics.jsonl`。网页目录发现只读取 `outputs/temporal_experiments` 下最多 50 份近期摘要，加上当前运行，不在刷新时扫描模型或完整日志。自行指定其他目录的旧实验不会自动列入历史比较。

比较条件记录 split hash、normalization hash、种子、公共网络配置、前导/监督长度、采样和优化参数、活动冻结模块及设备配置等。条件不同显示警告；无法自动证明真实硬件或后台负载一致，速度对照需由使用者控制。同训练步不是同 wall time；AMP 跳过更新时应结合实际更新次数比较。

日志增加 `temporal_mode`、`network_version`、`context_frames`、`prefix_frames` 和 `module_status`。GRU 旁路时 `module_changes.gru`、`module_gradients.gru` 应为 0；这些指标是诊断，不参与 CE 或 best 选优。

## 7. 实战验证

将两组各自的 checkpoint 复制到 Windows 游戏电脑，在 BC 网页实战选择并加载即可：

```text
python scripts/play.py
```

默认访问 `http://localhost:8797/`。模型版本自动决定 GRU 或 TCN，不需要更换 DLL 或重新采集 REP。

TCN 强制 `decision_interval_frames=1`：缓存最近最多 31 个已成功推理帧的 Battle Feature，追加当前帧后预测。暂停、换模型、终局、缺帧、失败发送或不连续推理都会清空。界面显示实际窗口帧数、重置次数和原因。Windows 异步读取仍可能漏帧；频繁重置表示短历史运行，不能把它当作稳定的满 32 帧实验。

两组都执行同一 Joint432 argmax、相同键盘控制，不添加宏动作或探索噪声；报告独立保存，包含网络版本和时序信息，不覆盖训练模型。比较使用相同对手、难度、键位、决策间隔与评估局数，结合伤害差、胜率、完整局数量、P95 推理耗时和缺帧情况判断。

## 8. 兼容与人工验收

- 原 GRU 版本不变；旧包缺少模式元数据时只在内存中补 `gru`，严格加载原权重与优化器。
- TCN 新版本 `soku_bc_tcn32_joint432_v1`，必须带 32 帧协议；GRU/TCN 跨模式续训明确报错。
- 动作/schema、资源字段和对象上限不变，无需重新生成 NPZ。
- 测试源见 `tests/test_temporal_experiment.py`、`tests/test_temporal_runtime.py`，本次未执行。

人工验收要点：

1. 两份配置使用相同数据/划分/归一化；公共层初始权重相同，数据标签和 CE 未改。
2. TCN 改未来帧不影响过去输出；改变超过 31 帧前的 Battle Feature 不影响当前输出。短历史、完整历史的批量与逐帧输出对齐。
3. TCN 更新不调用 GRU，GRU 梯度/权重变化为零；GRU 冻结仍执行前向，解冻恢复训练。
4. 旧 GRU 包可续训/推理；跨模式恢复拒绝；创建实验失败、重复请求或旧页面请求不覆盖/污染原模型。
5. 网页两组曲线不混合，参数差异提示正确；1024×768 和 1366×768 下控制可见，表格局部滚动。
6. 实战自动识别两种模型，TCN 每帧执行 Joint432；暂停/缺帧重新积累历史，未发送预测不提交记忆。
7. 用同条件实际对战确认效果，再决定采用哪种结构；不能仅凭同批训练损失下降判定 TCN 更好。
