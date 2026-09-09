# SOKU BC：离线行为克隆

独立复用 `soku_cql` 的人物/资源编码、每侧最多 3 个对象、Fusion、GRU 和 Memory Fusion。网络从随机初始化学习 REP 专家的完整单帧 Controller State；训练前端沿用现有 React 工作台，改为 BC 专用指标。

**不是在 CQL 上加一项模仿奖励。** 本项目只有一个策略分类网络与交叉熵损失；没有 Q 目标、TD、N 步回报、奖励、目标网络、PPO 或宏动作。原 CQL/PPO/DQN 工程、数据和模型均不修改。

## 启动

以下命令在 `soku_bc` 目录执行。Python 3.11+；前端沿用 Vite 7，使用满足锁文件要求的 Node.js（如 22.12+）。保留你已安装的 CUDA/ROCm 版 PyTorch，不要为了界面改装 CPU 版。

首次安装依赖、由使用者手动构建前端：

```text
python -m pip install -e .
npm --prefix web ci
npm --prefix web run build
```

打开工作台（数据准备后默认暂停，点击“开始 / 继续”）：

```text
python scripts/train.py --config configs/bc_suika.yaml
```

默认地址 **http://127.0.0.1:8796/**，无 token。占用时实际绑定后自动顺延并打印地址，不与 CQL 训练默认 8776 或实战 8777 共用端口。`scripts/run.py` 是同一训练入口的兼容别名。

服务器无界面立即训练：

```text
python scripts/train.py --config configs/bc_suika.yaml --headless
```

打开网页并自动训练，加 `--start`；结束按 Ctrl+C，等待保存完成。不要直接杀进程。

## 数据与保存

- 默认只读 `../soku_cql/data/replay_shards_resources_v4` 的现有资源版 NPZ，不重新采集或转换、不复制数据。
- 自己生成 `data/train_val_split_resources_v4.json`，按完整独立 REP 内容分组 8:2；相同 NPZ 副本不跨集合。连续量归一化只拟合训练集。
- 迁移服务器后，修改 YAML 的 `data.directory` 为实际相对目录；代码运行不依赖旁边的 CQL Python 包。
- 首次扫描、校验、拟合可能耗时，终端和网页显示进度；缓存和预取减少训练期间反复解压。NPZ 文件变化时明确拒绝沿用旧划分，需要另开划分和输出目录。
- 输出默认 `outputs/bc_suika_joint432_v1/`。启动准备完成即保存初始化/恢复后的 `last.pt`；默认每 1000 step 保存 `last.pt` 与带时间后缀的版本。
- `best.pt` 和 `best_stage_N.pt` 按当前阶段最低**验证 NLL**选取，不代表游戏胜率最强；改参会另开阶段，不混画旧曲线。
- `.gitignore` 排除数据集、模型、日志、依赖和构建产物；只同步源码、配置、文档和锁文件。

BC 续训（恢复权重、优化器、步数及随机状态）：

```text
python scripts/train.py --resume outputs/bc_suika_joint432_v1/last.pt
```

只指定 `--resume` 使用 checkpoint 内配置。要采用新的 YAML，同时传入 `--config configs/bc_suika.yaml`；架构、随机种子及动作轴定义不允许偷偷切换。新训练请指定新的 `--output outputs/bc_run_02`。**不接受旧 CQL/PPO/IQL checkpoint，也不做部分权重迁移。**

## SSH 浏览器访问

服务器启动网页模式后，在自己电脑的终端执行（替换用户和服务器地址）：

```text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8796:127.0.0.1:8796 USER@SERVER
```

然后打开 `http://localhost:8796/`。若服务器提示顺延到 8797，只将命令最后一个端口改为 8797。浏览器关闭不会停止训练。可信局域网可显式 `--host 0.0.0.0`；没有密码或 token，能访问该端口的人可控制训练，不要公网暴露。

## 工作台

- 总览：验证 NLL、Top-1/Top-5、多数动作基线、训练速度及时间分解。
- 学习诊断：专家概率、置信度、归一化熵、全部 432 类动作频率/召回率、模块梯度和权重变化。
- 数据与覆盖：整份 REP 划分、真实 Skill/Card 字段、可选状态覆盖、卡键重合清洗计数。
- 参数与模型：暂停后修改学习率、batch、标签平滑和保存/验证间隔；冻结编码器/GRU/分类头并保存参数锁。

模型的 `step_logits()` / `act()` 可供之后的 BC 推理接入，输入与现有资源版 CQL 相同，输出语义是分类 logits。**已有 CQL 实战入口不会直接加载 BC checkpoint；本次交付是独立 BC 网络、离线训练及训练前端，不改原实战端。**

动作历史诊断新增“上一帧复制基线”和“动作切换帧准确率”，与总体 Top-1/Top-5、多数动作基线并列显示。详见 [指标定义、实际数据统计与兼容性](docs/action_history_diagnostics.md)。这些指标只做统计，不影响损失或 best 选优，旧日志缺字段不补零。

详见 [网络与训练口径](docs/architecture.md) 和 [人工验收清单](docs/acceptance.md)。本次按要求仅交付源码并做静态核对，未编译、运行测试、启动训练或对局；测试源文件已提供，不代表已执行通过。
