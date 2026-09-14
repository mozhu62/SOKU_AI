# 批量 Graph 实战接入

CUDA/hybrid 且 streaming_tcn=true、tcn_cuda_graph=true 时，LiveAgent 加载 BatchGraphCandidate（沿用验证阶段的类名）。模型加载期间一次捕获所有支持长度，不在对局中首次捕获。

- 初始化：window_seed_eager，完整窗口建立真实缓存。
- 新增1帧：single_frame_graph。
- 新增2/3/4/8/18帧：batch_graph，一次批量 replay。
- 其他长度：batch_eager，批量普通前向；不丢弃、不补假帧，更新相同缓存后可重新进入Graph。

切换精度/模型/设备需重新加载。原权重、训练逻辑和输入历史长度不变。关闭 tcn_cuda_graph 可退回原普通增量实现。捕获失败明确停止加载，不静默降级。

此前已完成离线验证，本次只接线并静态检查，未启动游戏或编译前端。重启实战服务即可使用新的后端；参数页的运行后端显示实际路径。整体速度仍需实战对照，不等同于离线TCN耗时。
