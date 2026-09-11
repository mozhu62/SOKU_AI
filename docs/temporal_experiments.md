# BC TCN32 独立实验

当前版本已经删除 GRU，不再提供 GRU/TCN 结构切换。实验室只创建相同TCN32 网络的独立随机初始化分支，用来比较随机种子、冻结模块或训练超参数。

## 固定结构

~~~text
228D → 256D 当前状态
228D × 32 → 256D TCN
对象 128D + 128D
concat 768D → 1024D → Joint144
~~~

TCN 的四个残差块各含两层因果卷积，感受野严格为 32 帧。训练固定 burn_in=31；这 31 帧是卷积上下文，不产生独立 CE。

## 创建实验

1. 暂停训练并等待当前更新完成。
2. 打开“独立实验”。
3. 输入唯一的 ASCII 名称并确认。
4. 工作台先保存原 last.pt，再创建随机初始化的新模型、优化器和配置快照。
5. 新分支保持暂停，由使用者手动开始。

新实验不会迁移当前模型权重，不覆盖原目录，不修改 NPZ，不增加原运行的 step。

## 比较规则

页面比较 Validation NLL、总体 Top-1、动作切换 Top-1、有效监督帧和吞吐。只有 split hash、normalization、模型配置、冻结项与训练超参数一致时，随机种子对照才可直接解释。

旧 comparison.json 使用 bc_temporal_comparison_v1，不会伪装成当前 bc_joint144_tcn32_comparison_v1 记录。

## 冻结实验

冻结模块包括 current_encoder、object_encoder、tcn、fusion 和 policy_head。冻结只停止参数更新，前向仍然执行；如果只训练分类头，应冻结前四项。

旧 GRU checkpoint 和旧 TCN checkpoint 不能由当前版本续训。需要保留旧实验时，应同时保留旧源码环境。
