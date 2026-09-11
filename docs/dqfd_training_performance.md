# DQfD resources-v4 训练吞吐优化

## 修改边界

只优化数据组批、搬运和诊断，不修改模型结构、单步/N 步 TD、gamma、reward、专家间隔、AMP、梯度裁剪、PER 参数、Target 更新、验证样本数及 best 规则。现有 resources-v4 checkpoint 可以按原入口续训，不需要重新录制、转换 NPZ 或拟合 normalization。

## 批量取数

`ResourceReplayDataset.get_batch()` 接收本步已经抽出的索引，按分片 gather 状态、32 帧历史和每侧最多三个对象。当前/下一步/N 步状态在同一分片一起构造，重复状态可以复用 CPU 构造结果；回到模型前仍保留每条样本，包括 PER 重复抽样。历史保留原始 episode_start、history_valid、padding；N 步逐 offset 累计，遇终局、断点按原规则停止。

训练保留原有稳定分片排序和对应 IS 权重，验证保留原验证索引顺序。没有对下一批提前采样：本步 TD 误差回 CPU、PER 更新后，才抽下一批。没有修改 RNG 抽样调用次数。旧格式 Dataset 不走新批量实现。

`PackedBatchTransfer` 将同 dtype 的张量合并到可复用的固定页 CPU 缓冲，分别搬运并恢复为原字典视图，减少许多小 H2D 调用。缓冲只保留各 dtype 的最大已用批量容量；复用前等待上次 H2D 完成，不覆盖仍在 DMA 读取的内存。GPU 缓冲不原地复用，避免破坏旧批次的 autograd 引用。CPU 模式不使用固定页缓冲。

## 配置与回退

`configs/dqfd_suika_resources_v4.yaml` 默认：

```yaml
training:
  vectorized_batches: true
  packed_transfer: true
```

两项均设为 false，可退回原逐条组批/逐张量搬运路径，诊断仍保留。旧 YAML 缺少这两个键时也默认启用；不是 checkpoint 权重字段，不改变加载形状。`num_workers` 仍不用于此 PER 主循环，调大它不会增加取数线程。

## 看哪些性能记录

日志每 `log_interval` 步新增性能行；TensorBoard 使用 `performance/` 分组：

- `active_steps_per_second`：该窗口取数、搬运、模型更新、TD 回读、PER 回写的主机活动时间吞吐，不含验证、保存和日志输出。
- `transitions_per_second`：active 吞吐乘 batch_size，指直接监督的 transition 数；32 帧历史和 bootstrap 计算不重复算成监督样本。
- 旧 `steps_per_second`：仍是日志间墙钟速度，包含期间的验证/保存等，便于与旧日志比较。
- `host_per_sample_ms`、`host_batch_cpu_ms`、`host_pack_cpu_ms`、`host_transfer_ms`、`host_online_forward_ms`、`host_bootstrap_ms`、`host_loss_ms`、`host_backward_optimizer_ms`、`host_td_readback_ms`、`host_per_update_ms`：窗口每步的主机阶段均值。
- `gpu_sample_*_ms`：仅 CUDA 日志步使用 Event 测量的搬运、Online、Bootstrap、Loss、反向/优化、TD 回读流上阶段耗时；不是全窗口平均，也不是独占 kernel 时间，可能包含流上的等待/空闲。CPU 不输出伪造的 GPU 时间。
- `validation_seconds`、`save_resume_seconds`、`save_snapshot_seconds`：各次验证、完整模型保存、轻量快照保存耗时。

CUDA 异步运行时，主机 forward 耗时常是提交时间，而 TD 回读会等待前面的 GPU 工作；不能把主机 `td_readback` 很长全部解释为复制数据慢。结合 CUDA Event 采样辨别。仅沿用日志步同步，不在每个阶段强制同步。活动吞吐在异步 Target 更新等边界仍不是严格端到端 GPU 时间，整体速度看 wall 指标。

批量/固定页搬运是可关闭的优化，不承诺一定达到 BC 的 50 step/s。若 host_batch_cpu 高，继续查缓存/组批；若 CUDA bootstrap 或 backward 高，瓶颈在网络工作量；若 active 高而 wall 低，查看验证/保存。

## 验证说明

提供 `tests/test_resource_batching.py`：逐条与批量全字段对照，覆盖重复索引、跨分片、历史首尾、缺帧/终局、无对象及三对象、N=1/3/40、验证元数据和 CPU/CUDA 搬运缓冲复用。本次按要求没有执行测试、编译、训练或基准测试，暂无实测提速倍数。不同搬运布局还可能影响 GPU 非确定性计算，因此不承诺逐 bit 复现训练权重。
