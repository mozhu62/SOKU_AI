# BC 实战推理端

## 启动与模型选择

仅在 Windows 游戏电脑运行。已有 BC Python/前端依赖即可；本次没有新增第三方依赖。全部命令在 `soku_bc` 目录执行：

```text
npm --prefix web run build
python scripts/play.py
```

若首次安装缺少依赖，先按 README 安装 Python 项目并执行 `npm --prefix web ci`。前端构建失败应先修复，不会降级为另一个 Tk 窗口。

服务默认只绑定 `127.0.0.1:8797`，网页直接打开 `http://localhost:8797/` 或 `http://127.0.0.1:8797/`。端口冲突自动顺延，以终端打印地址为准，不需要 token。8796 仍为离线训练默认端口。

模型选择步骤：

1. 网页顶部“选择本机模型”打开 Windows 原生文件选择窗口，选择从服务器复制回来的 BC `.pt`。
2. 点击“加载模型”，等待输入规格和权重校验。也可从 `outputs` 已登记模型列表选择。
3. 启动已经加载资源版 SokuDataBridge 的游戏，进入对战。默认接管 1P，不按角色 ID 拦截；REP 子模式不能接管。
4. 点击“继续”并切回游戏。无法自动聚焦时手动 Alt+Tab；点击浏览器会触发失焦松键，需要再次点击继续。

只选文件不会立刻切换模型；加载后默认保持暂停。换模型前先松键，候选模型格式/权重检查失败时不替换旧模型。加载成功会结束旧评估会话、清空记忆并新建报告。报告属于实战记录，不覆盖模型。

可选命令：

```text
python scripts/play.py --checkpoint outputs/bc_suika_joint432_v1/best.pt
python scripts/play.py --checkpoint outputs/bc_suika_joint432_v1/last.pt --device cpu --rounds 0 --port 8797
```

`--model` 是 `--checkpoint` 别名。`--start` 在模型和游戏连接后自动尝试开始一次；`--headless` 不启动网页并自动接管，只适合已确认设置的使用者。默认命令不自动接管。Ctrl+C 退出服务并松键；F10 仅暂停，网页继续保留。

## 模块与数据路径

| 模块 | 职责 |
|---|---|
| scripts/play.py、configs/live_eval.yaml | 独立 Windows 实战入口与相对路径配置 |
| live/agent.py | 严格加载 BC 包的 model 权重与规格；eval + inference_mode；不创建优化器 |
| live/shared_state.py、resource_state.py | 复用原战斗端的只读 v3 状态协议与同采集序号资源协议 |
| live/observation.py、resources.py | 对齐训练 observation、双方技能/手牌、checkpoint 归一化、每侧最近 3 个对象 |
| live/input_history.py | 读取游戏实际消费的按键，按 checkpoint action_shift 构建 previous action；断点用 START=432 |
| live/control.py、windows_api.py | 原扫描码 SendInput、按 PID 聚焦、独立看门狗、F10 与失焦松键 |
| live/runtime.py | 帧推进、GRU/TCN 历史、推理后观测复核、输入执行和回读统计 |
| live/workbench.py、repository.py、model_picker.py | 串行命令队列、网页换模型、登记文件访问；不允许网页任意路径读写 |
| live/statistics.py | 伤害差、完整小局/片段、输入频率和推理延迟报告 |
| live/web_service.py、web/src/live | 本机网页实战工作台与状态推送 |

BC 工程内保留一份原 CQL 控制实现并做 BC 输出适配，不需要旁边存在 CQL/PPO/DQN Python 包。原项目源码和 DLL 未改。实战不扫描 NPZ，不加载训练集划分，不重新拟合归一化；即使 checkpoint 配置里是 Linux 数据目录也不会访问该目录。

已有能运行资源版 CQL 实战的 DLL 可以复用。本次没有改变 DLL 协议。缺少资源通道、协议大小/版本不匹配或没有同帧资源时，界面明确显示原因，不把未知技能、卡牌补零后继续控制。

## 推理、记忆与输出

每个决策将实时 observation 输入 BC `step_logits(obs, memory)`，获得 `[1,432]` logits 和新的时序状态，选 `argmax(logits)`，再 decode 为九宫格方向、4 位战斗按钮、互斥的卡牌命令。所有按钮位都执行，不只是方向；方向使用屏幕绝对坐标，不因朝向自动镜像。

加载器按 checkpoint 的网络版本自动创建 GRU 或 TCN32，不需要在实战 YAML 中手动选择网络。TCN memory 保存最多 31 帧的 256D 战斗特征，与当前帧一起形成 32 帧窗口；GRU 不执行。TCN 拒绝 decision_interval_frames 不为 1 的配置，网页加载后锁定该输入；从稀疏决策 GRU 会话切换前，先把该值设为 1 再加载 TCN。

softmax 仅供显示，温度固定等效为 1；没有随机采样、额外探索、宏动作、合法招式过滤或无动作保护。相同完整按键连续选择会保持，而非每帧松开再按。模型是否学到出招仍需看实际游戏结果。

上一帧输入来自 DLL 实际回读，不是上一次计划发送的动作。方向持续量只表示水平/垂直方向组合持续时间，clip 到 60 后 /60；不是完整动作持续时间。`action_shift` 与 checkpoint 保持一致。GRU 初始为零，暂停/进程或回合变化/长缺帧/失效预测时重置；任何非连续帧处输入历史重置 START。不会跨断点补造控制历史。

推理后再检查前台、场景、小局、观测年龄和帧延迟。只有动作成功发送后提交新时序状态。无法执行的预测保留显示但标明“未发送/丢弃”，不计入已发送动作频率。TCN 额外检查观测帧与成功推理帧的连续性，任意缺口都重新积累窗口，不将稀疏决策压成连续游戏帧；不足 32 帧时使用真实短历史及网络因果零填充。

默认每游戏帧决策一次，但 Windows 轮询、渲染及推理可能漏帧；这是异步控制，不承诺严格锁步。超过延迟限额会丢弃旧动作并追新帧，不将旧局输入发送到新局。超时或异常先松键，后保存报告。

## 界面与按键

默认打开“战斗与动作”，顶部仍可切换模型。显示：

- 最近完整动作、所选概率、归一化熵、观测帧与实际出招 actionId。
- Top 20 / Top 50 / 全部 432 动作的原始 logits、softmax 概率与发送/回读次数；没有 Q 列。
- 六个按钮的“模型要求／当前发送状态／游戏回读”，与当前方向分开核对。游戏回读是较新的状态，不是同帧命中确认。
- 同帧 Skill/Card 输入、双方 HP、逐局伤害差、完整小局胜率、缺帧、过期预测、推理 P50/P95。
- 当前 GRU/TCN 结构；TCN 实际窗口帧数、时序重置次数及最近重置原因。

默认按键：W/A/S/D = 上/左/下/右；J = A 体术，K = D DASH，I = B 轻弹幕，L = C 重弹幕，O = 切卡，P = 使用符卡。参数页可修改映射，暂停后应用生效；十个游戏按键不可重复且不能占用 F10。长期配置编辑 `configs/live_eval.yaml`。

整场结束按 Z 连续点按续局；默认统计 20 个完整小局后暂停，`rounds: 0` 不限制。首次中途接管、暂停/失焦或不完整采样均保留为片段，不计完整局胜率。

报告写入 `outputs/evaluations/<时间-随机后缀>/session.json`、`rounds.jsonl`、`summary.json`。记录 BC 算法、模型 SHA256、网络版本、时序结构与窗口语义、确定性 logits 选择规则、键位及 CPU 难度。训练 checkpoint、CE、采样、数据划分与 best 选择完全不变。

## 人工验收（本次未执行）

- 重新构建前端后，两个根地址均可显示实战页，无 token；占用默认端口时使用新打印地址。
- 不在 Windows 运行时明确拒绝；没有训练数据仍可选择和加载兼容 BC 模型。
- BC 权重严格加载；误选 CQL/PPO 或损坏文件时报告错误，不以旧模型冒充成功。
- 模型选择为包含 A/B/C/D 的完整动作时，对应 J/I/L/K 显示并执行；概率共 432 项且与 logits argmax 一致。
- F10、失焦、异常、停止/换模型均先松键；继续不会延续上一段 GRU/控制历史。
- 人机主模式 2 不被当作 REP；切换角色不被拦截；REP 子模式 2 不发键。
- 整场结束可按 Z 续局；评估文件独立，训练模型文件不被覆盖。

本次只交付源码、静态核对与使用说明，未构建前端、运行测试、打开游戏或执行实战验收。
