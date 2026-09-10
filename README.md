# SOKU BC：离线行为克隆

该项目从随机初始化开始，使用固定 REP 数据监督学习专家每一帧的完整 Controller State。输出空间为 Joint432；损失仅为 CrossEntropy，不包含 Q、TD、奖励、目标网络、PPO 或宏动作。

## 网络结构

当前网络固定为宽 TCN32 架构：

~~~text
当前帧 878D 状态 ── Linear(878,1024) + LayerNorm + SiLU ── 1024D ┐
                                                                    │
最近 32 帧 878D 状态 ── 宽 TCN32（每块两层因果卷积）────── 256D ──┤
                                                                    ├─ concat 1536D
己方最近 3 个对象 ── 共享对象 MLP + mean/max ───────────── 128D ──┤
敌方最近 3 个对象 ── 共享对象 MLP + mean/max ───────────── 128D ──┘

1536D ── Linear(1536,1024) + LayerNorm + SiLU ── Linear(1024,432)
~~~

GRU 与旧 Memory Fusion 已删除。TCN 直接读取每帧 878D 状态，不再读取压缩后的 Battle Feature。四个残差块使用 dilation 1/2/4/8，每块含两层 kernel=2 因果卷积；输入 stem 使逐帧感受野严格覆盖当前帧及前 31 帧。

详细定义见 [网络与训练口径](docs/architecture.md)。

## 安装与启动

以下命令在 soku_bc 目录执行：

~~~text
python -m pip install -e .
npm --prefix web ci
npm --prefix web run build
python scripts/train.py --config configs/bc_suika.yaml
~~~

工作台默认打开 http://127.0.0.1:8796/，无 token；端口占用时自动顺延。无网页训练：

~~~text
python scripts/train.py --config configs/bc_suika.yaml --headless
~~~

恢复同版本模型：

~~~text
python scripts/train.py --resume outputs/bc_suika_wide_tcn32_v2/last.pt
~~~

旧 GRU、旧 256D TCN、CQL、PPO 和 DQfD checkpoint 均会被明确拒绝，不能静默部分加载。新架构必须去掉 --resume，使用新输出目录从随机初始化开始。

## 数据和模型

- 数据目录及固定 8:2 划分由 configs/bc_suika.yaml 指定。
- 数据集、模型、日志和前端构建产物均由 .gitignore 排除。
- last.pt 用于续训；best.pt 和 best_stage_N.pt 仍按当前阶段最低 Validation NLL 选择。
- 432 类动作、normalization、horizontal mirror augmentation、action shift 和 BC CrossEntropy 口径不变。
- 默认输出目录为 outputs/bc_suika_wide_tcn32_v2/。

## SSH 访问工作台

服务器监听回环地址时，在本地电脑执行：

~~~text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8796:127.0.0.1:8796 USER@SERVER
~~~

然后打开 http://localhost:8796/。若服务器实际端口顺延，修改命令右侧端口。可信局域网也可显式加 --host 0.0.0.0；服务没有密码或 token，不应暴露到公网。

## 实战推理

Windows 游戏电脑执行：

~~~text
npm --prefix web run build
python scripts/play.py
~~~

打开 http://localhost:8797/，选择新架构 .pt 并加载。实战端从 LiveFrames.v1 队列维护真实连续 32 帧；不足 32 帧、换局、断帧或协议不兼容时不发键。模型输出 432 个原始 logits，取 argmax 后解码为九宫格方向、四个战斗按钮及互斥卡牌命令。

详见 [实战推理说明](docs/live_inference.md) 和 [TCN 完整帧窗口](docs/tcn_live_frames.md)。

## 工作台

- 训练总览：NLL、Top-1/Top-5、基线、吞吐和耗时。
- 学习诊断：动作切换准确率、专家概率、熵、432 类频率、模块梯度与权重变化。
- 独立实验：只创建同一宽 TCN32 架构的随机初始化分支，用于比较随机种子、冻结项或超参数。
- 参数与模型：暂停后修改训练参数及冻结当前状态、对象、TCN、融合或分类头。

本次按项目约束仅修改源码和文档，没有编译前端、运行测试、启动训练或进入游戏。
