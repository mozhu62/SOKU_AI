# 玩家级固定划分

在训练配置的 `data` 中加入以下设置。旧配置不指定 `split_mode` 时仍按旧 8:2 规则运行，不改写旧划分。

```yaml
data:
  split_mode: per_player
  split_character_id: 0
  player_validation_fraction: 0.1
  player_validation_max: 30
  player_aliases_file: ../tools/sokureplays_get/soku_dataset/reimu_expert_players.json
  split_file: data/train_val_split_cards_by_player.json
```

启动训练时自动调用现有 `split_replays` 入口生成或校验清单。使用新数据时另设输出目录，不续用旧划分与旧归一化缓存。

每位玩家的验证 REP 数为 `min(30, ceil(独立REP数 × 0.1), 独立REP数 - 1)`。只有一份 REP 的玩家全部进入训练。全体无法组成非空验证集时明确报错。小号按配置中的 user_ids 合并，不使用模糊昵称猜测。

使用源元数据中的目标角色确定 server/client 用户；同角色内战或缺失身份字段明确拒绝。相同 REP ID 的原片、镜像、复制文件同组，因此 30 的上限指独立 REP，不是镜像后的文件数。相同内容的冲突身份也拒绝。

规则、种子、玩家映射和文件哈希固定保存在清单中；变化时要求新清单，不静默重划分。卡牌格式转换会保留 source_match_metadata。

独立生成清单可调用 `scripts/split_player_replays.py --source 目录 --output 清单路径 --aliases 玩家表路径`。原始档案可以用于提前查看划分，但转换后文件哈希变化，训练必须针对转换后的分片生成新清单；原始清单不可直接复用。

本次未运行训练、转换或测试。
