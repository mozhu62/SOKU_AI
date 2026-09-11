# TH123 Suika 数据集报告

- Replay：81
- 有效帧：882621
- Transition：882417
- Round：204
- Suika 左/右：38 / 43
- 对象截断率：0.560345%
- 平均/最大对象数：7.129 / 156
- 平均 Round 长度：4326.6 帧
- Reward 均值/标准差：0.000967 / 0.169645
- 训练权重均值/标准差：1.041807 / 0.506394
- 训练权重范围：0.325000 / 2.250000
- 无效帧：96499
- 有效战斗帧缺口：0
- 终局胜/负：88 / 116

## Reward 分量累计值

- approach：36.285199
- corner_escape：19.732546
- corner_hold：-27.310001
- damage_dealt：2389.495524
- damage_taken：-1711.541078
- far_retreat：-10.777457
- hit_confirm：472.450007
- opponent_corner_damage：123.852381
- opponent_corner_pressure：9.108000
- round_outcome：-280.000000
- self_corner_damage：-168.108758

## 位置风险统计

| 区域 | Transition | 造成伤害 | 受到伤害 | 每千帧受伤 | 受伤帧率 |
|---|---:|---:|---:|---:|---:|
| deep_corner | 179246 | 173167 | 566236 | 3158.99 | 2.665% |
| corner | 62917 | 127678 | 106199 | 1687.92 | 1.570% |
| side | 221055 | 520873 | 355153 | 1606.63 | 1.442% |
| center | 419199 | 771279 | 683953 | 1631.57 | 1.625% |

完整 Action、天气、对手分布和逐 Replay 统计见 `dataset_report.json`。
