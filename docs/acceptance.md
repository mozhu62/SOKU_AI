# 宽 TCN32 人工验收清单

本次修改只重构 BC 网络及其训练、checkpoint、诊断和实战接线。Joint432、数据字段、数据划分、normalization、镜像增强、CrossEntropy 和 best 选优规则不变。

## 网络

1. CurrentStateEncoder 的拼接输入为 878D，只有一层 878→1024。
2. 模型实例不存在 gru 和 memory_fusion。
3. TCN 输入为 [batch,time,878]，输出 256D。
4. 四个 TCN 残差块各有两层因果卷积，整体感受野为 32 帧。
5. 对象分支仍为每侧最多 3 个对象、每侧 128D。
6. Fusion 第一层为 1536→1024，分类头为 1024→432。
7. 改变未来帧不能影响当前及以前输出。

## 训练

1. 默认 burn_in=31、sequence_length=32。
2. 31 帧前导不产生 CE，但可接收后续监督帧梯度。
3. 只对有效 mask 位置计算 Joint432 CrossEntropy。
4. 冻结模块列表只包含 current_encoder、object_encoder、tcn、fusion 和 policy_head。
5. 日志不再出现 GRU 或 Memory Fusion 梯度/参数变化。
6. 独立实验只创建相同宽 TCN32 的随机初始化分支。

## Checkpoint

1. 新模型版本为 soku_bc_wide_tcn32_joint432_v2。
2. 同版本 checkpoint 严格恢复模型、优化器、scaler、RNG、split hash 和 normalization。
3. 旧 GRU、旧 TCN、CQL/PPO/IQL/DQfD 均明确拒绝。
4. 不允许静默部分加载或覆盖现有模型目录。

## 实战

1. LiveFrames.v1 窗口达到真实连续 32 帧前不推理、不发键。
2. 窗口缓存的是 878D 状态；对象只取当前帧。
3. 缺帧、换局、帧回退或换进程时重新积累。
4. 完整 Joint432 解码后的方向和六个按钮均进入控制层。
5. 页面显示 32/32、首末帧号、完整 logits、最终动作、发送状态和回读。

## 静态验收源码

- tests/test_bc_learning.py：宽度、输出和冻结。
- tests/test_temporal_experiment.py：双卷积、32 帧感受野、流式/批量一致与 checkpoint 拒绝。
- tests/test_live_tcn_stream.py：真实帧队列、窗口与缓存。
- tests/test_temporal_runtime.py：独立实验和暂停连续性。

按项目要求，本次没有编译、运行测试、启动训练或进入游戏；以上均为后续人工验收项。
