# 宽 TCN32 实战：完整真实帧窗口

## 行为约定

- 输入必须来自同一游戏进程、同一小局且帧号连续的 32 个真实游戏帧。
- 新进入战斗、换局、真正缺帧或换模型后，窗口不足 32 帧时松键等待。
- 不复制旧帧、不用零 observation 凑满窗口，也不退化为单帧推理。
- 暂停和失焦只停止发键；游戏仍推进时继续采集真实观测。
- 推理期间经过的帧由 DLL 环形队列保留，下次批量补读。

32 帧约覆盖 0.53 秒游戏时间；完整历史不等于每秒一定完成 60 次推理，实际频率仍受模型计算和 Windows 控制开销限制。

## DLL 通道

实战端读取 Local\SokuDataBridge.LiveFrames.v1.<PID>：

- 环形队列容量 128。
- 每槽同时保存 StatePayload 与同次采样的 Skill/Card 资源。
- Python 校验协议版本、结构大小、进程、序号和提交一致性。
- 队列被覆盖或槽读取不一致时记录真实丢帧并重新积累窗口。

旧 State.v3 只保存最新快照，无法找回推理期间错过的游戏帧，因此不允许作为 TCN32 的降级来源。

## 网络输入

每个队列帧先由 ObservationBuilder 转成 observation，再由模型构造 878D 状态表示。窗口缓存：

~~~text
[32,878]
~~~

推理时：

- 历史 TCN 对完整窗口输出 256D。
- 当前帧 878D 经单层宽编码得到 1024D。
- 当前帧双方对象分别得到 128D。
- 四路拼成 1536D，经 1024D 融合后输出 432 logits。

历史对象不会缓存进 TCN，TCN 也不再读取旧 256D Battle Feature。

## 手动更新与启动

以下命令由使用者在 soku_bc 目录执行；本次交付没有代为执行：

~~~text
cmake -S ../tools/soku-data-monitor -B ../tools/soku-data-monitor/build-win32 -A Win32 -DSOKU_ENABLE_PPO_STREAM=ON
cmake --build ../tools/soku-data-monitor/build-win32 --config Release --target SokuDataBridge --parallel 1
npm --prefix web run build
python scripts/play.py
~~~

构建结果位于 ../tools/soku-data-monitor/build-win32/Release/SokuDataBridge.dll。退出游戏后再替换游戏实际加载的 DLL，并重启游戏。

## 诊断

页面 temporal_capture 分开显示：

- 当前缓存帧数与首末帧号；
- 最近一次推理实际使用帧数；
- 队列累计接收和覆盖丢失数量；
- 窗口重置次数及原因；
- padding_frames，固定为 0。

每次真实推理的 context_frames_used 必须为 32。模型切换后创建新 Agent 并清空特征缓存；previous_joint_action 仍来自游戏实际输入回读。

## 人工验收

1. 旧 DLL 明确提示缺少 LiveFrames.v1，不发键。
2. 31/32 帧时只等待；达到 32/32 后才允许推理。
3. 窗口滑动时只编码新增帧的 878D 状态。
4. 暂停后帧号继续推进，继续时不无故清空连续历史。
5. 真正缺帧、换局和进程切换时重新积累。
6. 批量窗口推理与逐帧状态缓存得到一致 logits。
7. 方向、ABCD、切卡和用卡都按 Joint432 解码执行。

对应静态验收源码为 tests/test_live_tcn_stream.py 和 tests/test_temporal_runtime.py；本次未执行。
