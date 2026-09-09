# 交付核对与人工验收

本次按要求不编译、不运行测试、不启动服务或训练。静态检查不能证明运行性能、浏览器布局和真实 NPZ 全部可用；以下是使用者后续验收步骤，不是已经通过的测试记录。

本次静态核对结果：动作/schema/资源/embedding 文件与复用来源内容一致；包内相对 Python 模块引用均有对应文件；前端锁文件只改项目名称，依赖与原工作台一致；已检查训练入口到数据采样、CE、保存恢复和网页指标字段的连接。默认 NPZ 目录在本机存在。运行结果、吞吐率、测试通过与浏览器显示效果未验证。

## 源码对应关系

| 要求 | 实现 |
|---|---|
| 独立复用 CQL 主干 | config.py、models.py、nn_modules.py、resource_encoder.py |
| 完整 432 类 BC | action_space.py、learner.classification_parts；唯一 policy_head |
| 只读现有 NPZ、上帧历史、不跨片段 | dataset.read_shard / ReplayStore.sample、action_space.previous_actions |
| 8:2 与训练集归一化 | dataset.split_replays、ReplayStore.__init__ |
| 缓存、预取、AMP、冻结 | dataset.get、prefetch.py、learner.py |
| 恢复/独立模型与阶段日志 | checkpoint.py、runtime.py |
| BC 训练前端及调参 | web/src/views/、web/src/App.tsx、web_service.py |
| 相对路径、端口冲突、SSH 无 token | config.resolve、web_service.bind_port / _private_host、configs/bc_suika.yaml |

## 已提供但未执行的测试源码

- `tests/test_bc_learning.py`：标准 CE、mask 梯度、聚合指标、网络形状、因果顺序、单步/序列一致性、冻结主干仅训头。
- `tests/test_bc_dataset.py`：全 432 动作往返、前一帧历史与断点、忽略奖励且不改 NPZ、8:2、训练集归一化和 BC 批次形状。
- `tests/test_bc_checkpoint.py`：保存/恢复权重、优化器和随机状态；拒绝 CQL；不接受不同固定划分。
- `tests/test_bc_controls.py`：控制幂等、快照隔离、配置限制、SSH 不同本地端口与来源限制。

需要执行时，在项目根目录手动运行：

```text
python -m unittest discover -s tests -t .
```

## 后续人工检查

1. 按 README 安装/构建后启动；确认端口默认 8796，页面先显示数据准备，完成后暂停。
2. 确认训练/验证 NPZ 数及 hash，归一化只来自训练集；原 CQL 数据、划分与输出文件不变化。
3. 从零启动后已有 step=0 的 last.pt；开始训练后 CE、NLL、Top-1/Top-5、概率及吞吐有实际数值，未记录项不显示假零。
4. 暂停后冻结主干、只训分类头；诊断中共享模块 L2=0、分类头 L2>0。解冻/改学习率保留优化器，新阶段分开显示。
5. 点击手动验证不增加 updates；固定模型重复验证结果应一致；验证显示样本量、未覆盖动作及多数动作基线。
6. 保存版本、Ctrl+C 等待退出；--resume 恢复 step/优化器；传 CQL checkpoint 明确拒绝，不覆盖已有 BC 模型。
7. 端口被占用时顺延并打印实际地址；SSH 转发后普通根路径可访问，不需要 token；浏览器断开不自动停止/恢复训练。
8. 在 1366×768 / 1024×768 下检查顶部按钮、核心卡片，表格局部滚动、全部动作下拉和参数锁；不从训练界面启动游戏。
