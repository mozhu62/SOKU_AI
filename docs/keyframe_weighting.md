# Keyframe-Focused Weighted CE

此次只修改 BC 的损失归约、相关配置与诊断统计。模型、Joint432、采样、时序输入、镜像、动作对齐、归一化和 best checkpoint 选优规则均不变。

## 配置

在 YAML 顶层与 training 同级设置：

```yaml
keyframe_weighting:
  enabled: true
  changepoint_weight: 4.0
```

两份 bc_suika 配置均已启用权重 4。可测试 1、2、4、8、16；其他大于等于 1 的有限数值也有效。enabled=false 或 weight=1 时数学上等价于原来的等权平均 CE。未配置该段的旧配置/旧 checkpoint 默认关闭加权。

现有同网络版本 checkpoint 可续训。要对旧模型启用新损失，必须同时指定新 YAML，避免只使用 checkpoint 中的旧配置：

```text
python scripts/train.py --config configs/bc_suika.yaml --resume outputs/bc_suika_wide_tcn32_v2/last.pt
```

## 边界与损失

- keyframes.build_changepoint_mask(expert_actions, valid_mask, previous_expert_actions) 返回两个与标签同形状的 bool 张量。
- previous_expert_actions 取自 Dataset 已对齐的 previous_joint_action_id。原有 previous_actions 在完整轨迹上按 episode、有效转移、终局重置历史；起点使用 START=432。
- 不在 batch 内重新右移标签。监督段首帧若有真实连续上一帧，仍可作为 changepoint。
- 无真实历史的有效监督帧：changepoint_valid=False，但仍以权重 1 参与 CE。padding 和无效帧完全排除。
- learner.classification_parts 保留原有四个位置参数和 (loss, parts) 返回结构，只增加可选关键字参数传递真实历史与加权配置。
- 先对有效标签计算 reduction=none 的 CE，再散射为 [B,L]；weights、valid_mask、两个 changepoint mask 均保持 [B,L]。这样 padding 中的非法标签或 NaN 不会污染损失。
- 最终归约为 sum(CE * weights * valid_mask) / sum(weights * valid_mask)，不是 weighted CE 的普通 mean。

hold CE=1、change CE=2、weight=4 时，loss=(1+8)/(1+4)=1.8。

## 验证与日志

learner.classification_metrics 复用损失中生成的 mask；旧调用方式也通过同一个 build_changepoint_mask 构造判断。新增以下字段，训练诊断和验证记录均可产生；日志同时保留 train_/val_ 前缀别名：

| 字段 | 定义 |
|---|---|
| hold_top1 | 有效历史中保持帧的完整动作准确率 |
| changepoint_top1 | 有效历史中切换帧的完整动作准确率 |
| changepoint_top5 | 切换帧的 Top-5 准确率 |
| hold_nll | 保持帧的等权、未平滑 NLL |
| changepoint_nll | 切换帧的等权、未平滑 NLL |
| expert_changepoint_rate | 切换帧数 / 可比较历史帧数 |
| previous_action_baseline_accuracy | 保持帧数 / 可比较历史帧数；直接复制专家上一帧的基线 |
| model_copy_rate | 模型 argmax 等于专家上一帧动作的帧数 / 可比较历史帧数 |

原有 action_change_accuracy 等字段继续保留。原 action_change_fraction 的分母是全部有效监督帧，与 expert_changepoint_rate 的分母不同。

aggregate_metrics 先累加各组正确数、NLL 总和和分母，再计算分组指标，避免不同批次的切换帧数量影响统计口径。空组返回 None，日志不伪造 0%。原 Overall NLL/Top1/Top5、验证 loss 和 best 的 validation_nll_v1 规则保持等权；训练 loss 是加权 CE，二者可以不同。

训练记录另外包含 keyframe_weighting_enabled、changepoint_weight（实际生效权重）。参数仍随现有配置快照与 checkpoint 保存；此次未增加实时调参入口。

## 针对性验证

```text
python -m unittest tests.test_keyframe_loss tests.test_action_diagnostics -v
```

测试覆盖连续保持/切换、跨 episode、断帧、终局、padding、切片首帧的真实历史、burn-in 对齐、1.8 人工算例、无效 NaN 与零梯度、各组汇总及空分组、验证指标不受权重影响。weight=1 和 enabled=false 同时比较原 CE 的数值与梯度，包含 label_smoothing=0/0.1/0.2；按浮点容差核对，不宣称不同归约顺序逐 bit 相同。
