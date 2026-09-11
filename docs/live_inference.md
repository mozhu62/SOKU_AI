# BC 实战推理说明

## 启动

在 Windows 游戏电脑的 soku_bc 目录执行：

~~~text
npm --prefix web run build
python scripts/play.py
~~~

打开 http://localhost:8797/，选择 soku_bc_tcn32_joint144_v1 checkpoint，加载后进入对战并点击继续。

## 模型加载

加载器要求 algorithm=bc、网络版本、Joint144 schema、observation manifest 和完整 state_dict 全部一致。旧 Joint432、旧 GRU/TCN、CQL、PPO、IQL 或 DQfD 权重都会被拒绝，不会部分迁移。

checkpoint 自带训练 normalization、状态字段清单和方向轴约定。实战 YAML 不选择时序模式，因为当前只有TCN32。

## 推理链

~~~text
DLL LiveFrames.v1 队列
  → 连续 32 个 State/Skill 配对帧（忽略卡牌和技能等级）
  → 每帧 228D 状态
  → 当前状态 256D + TCN 256D + 对象 128D + 128D
  → 768D → 1024D → 144 logits
  → argmax joint_action_id
  → direction + combat_mask
  → Windows 按键状态
~~~

模型输出是完整单帧 Controller State。方向和体术、Dash、轻弹幕、重弹幕均来自同一个 Joint144 ID，不存在只执行方向头的路径。

## 历史边界

- TCN 必须有当前帧和前 31 帧真实连续状态。
- previous_joint_action 来自游戏实际输入回读，不使用模型上一条计划动作。
- 上一帧 duration 仍只表示水平/垂直方向组合持续时间。
- episode、终局、断帧、帧号回退和进程变化会切断历史。
- 窗口不足时松键等待，不用假帧补齐。

## 控制安全

- F10、失焦、暂停、异常和停止时先松开全部键。
- 推理后再次校验战斗状态、最新帧与观测年龄；过期动作丢弃。
- 浏览器断开不会停止后台会话。
- 自动续局与模型推理解耦。
- 评估只读 checkpoint，不训练、不覆盖 last.pt 或 best.pt。

## 页面诊断

实战工作台显示：

- 当前 checkpoint、网络版本及TCN32 结构；
- 32 帧窗口的缓存、首末帧号、丢帧和重置原因；
- 完整 144 logits/概率排序及最终 Joint Action；
- 解码后的方向和全部按钮；
- 模型请求、实际发送与游戏输入回读；
- 每局伤害、动作频率、推理 P50/P95 和评估报告。

详细帧队列要求见 [TCN 完整帧窗口](tcn_live_frames.md)。
