# Git 代码同步与 SSH 浏览器访问

## 1. 哪些内容进入 Git

Git 仓库根目录是 `soku_cql`，不是外层 `FXTZ_AI`。

- 提交：Python/前端源码、测试源码、配置模板、文档、依赖清单、`web/package-lock.json`。
- 不提交：`data/`、`outputs/`、`checkpoints/`、`runs/`、日志、REP、CSV、NPZ、模型权重、压缩包。
- 不提交：`node_modules/`、`web/dist/`、Python 缓存/虚拟环境和安装产物。
- 本机配置另存 `configs/任意名称.local.yaml`，也不进入 Git。不要把账号密钥写进共享配置。

`.gitignore` 只影响 Git 收录，不会删除本机数据，也不会阻止训练读取它们。它不能自动清除已经跟踪或推送过的数据。本次检查时仓库尚未跟踪任何文件，无需取消跟踪。

第一次提交前，在本机 `soku_cql` 目录执行：

```text
git status --short
git add .
git diff --cached --stat
git commit -m "Add CQL joint432 training code"
```

确认暂存列表只有代码/配置/文档后再提交。创建云端代码仓库，按仓库提供的地址设置 remote 并 push；本文不代填地址，也不会自动提交或上传。后续本机修改后提交、推送，服务器通过 `git pull --ff-only` 更新代码。训练运行时不要切换源码版本，先停止并保存，再更新并重新启动。

`.gitattributes` 将文本统一为 LF，Windows 批处理保留 CRLF，减少 Windows/Linux 同步时的纯换行差异。

## 2. 数据集单独放到服务器，不经过 Git

Git clone 后不会有数据集。通过你已有的 SFTP/SCP/数据盘传输方式，把 NPZ 放到服务器项目的：

```text
soku_cql/data/replay_shards_resources_v4/
```

只训练已有 NPZ 时，不需要传 REP、CSV、游戏、DLL、`node_modules` 或旧 PPO/DQN 项目。

若服务器数据盘另有存放位置，可以复制配置模板为本机配置：

```bash
cp configs/cql_suika.yaml configs/cql_suika.local.yaml
```

只修改副本中的 `data.directory`，再用 `--config configs/cql_suika.local.yaml` 启动。相对路径均以 `soku_cql` 项目目录为基准；数据目录可以是项目内的符号链接，具体数据不会随 Git 上传。

首次训练由服务器创建固定 8:2 划分和归一化。如果搬迁已有训练并续训，除同一套 NPZ 外，还需单独复制匹配的划分 JSON 与新版完整 checkpoint，保持相对文件名及内容一致；这些文件也不经 Git。不能用重新划分代替原 checkpoint 对应的数据划分。

## 3. 服务器准备并启动前端服务

在 Linux 服务器通过 Git 获取代码，进入项目后执行：

```bash
python -m pip install -e .
npm --prefix web ci
npm --prefix web run build
python scripts/train.py --config configs/cql_suika.yaml --host 127.0.0.1 --port 8776
```

Python 要求 3.11 或更高，PyTorch 需要与你的服务器 GPU/驱动匹配；Node 版本应满足仓库锁定的 Vite 依赖要求。前端在服务器构建，Python 同时提供网页、API 和 WebSocket，不需要再开 Vite 开发端口。依赖/构建只需在首次或对应源码更新后重新处理。

不要加 `--headless`，该选项会关闭网页。默认等待你在网页点击“开始 / 继续”；希望直接训练可加 `--start`。

为了 SSH 断开后训练仍保持，可在服务器使用已有的 tmux：

```bash
tmux new -s cql
python scripts/train.py --config configs/cql_suika.yaml --host 127.0.0.1 --port 8776 --start
```

按 Ctrl+B，再按 D 脱离 tmux；之后用 `tmux attach -t cql` 回去。网页断线不会主动停止训练，但普通 SSH 终端退出仍可能结束其中的进程，因此长时间任务使用 tmux。

继续新版模型时：

```bash
python scripts/train.py --resume outputs/cql_suika_joint432_v3/last.pt --host 127.0.0.1 --port 8776
```

不带 `--config` 的续训沿用 checkpoint 配置；显式 `--host` 保证即使旧配置存了 `0.0.0.0`，本次也只监听回环。

## 4. 本地电脑建立 SSH 转发

保持服务器训练进程运行。在你的本地电脑另开一个终端执行，把 `用户名@服务器地址` 替换成实际 SSH 登录信息：

```text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8776:127.0.0.1:8776 用户名@服务器地址
```

本地浏览器只打开：

```text
http://localhost:8776/
```

也可使用 `http://127.0.0.1:8776/`。没有 token、登录页面、随机字符串或额外页面路径。SSH 仍使用服务器原有账号/密码或密钥认证，网页无需再次认证。转发终端需要保持运行；Ctrl+C 关闭的是隧道，不是 tmux 内的训练。

SSH 不是默认 22 端口时，在命令中增加 `-p 实际SSH端口`。网页端口与 SSH 登录端口是两回事。

### 端口冲突

- 服务器 8776 占用：程序保留自动检测/递增，终端会打印实际服务端口以及对应 SSH 命令。只调整 `-L` 的最后一个端口，本地地址仍可保持 `localhost:8776`。
- 例如服务实际位于服务器 8778：

```text
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8776:127.0.0.1:8778 用户名@服务器地址
```

- 本机 8776 占用：把 `-L` 中第一个端口改为 8876，浏览器打开 `http://localhost:8876/`。
- 后端 HTTP/WebSocket 均允许“回环来源 + 回环 Host”的这种不同端口转发，不需要额外配置 token 或代理。

## 5. 访问边界

默认仅监听服务器 `127.0.0.1`，不需要在服务器防火墙或云安全组开放 8776；只需正常 SSH 能连上。浏览器使用同一地址连接网页、API 和 WebSocket，Host/Origin 校验保留，不提供任意公网网站跨来源控制。

确实需要可信局域网直连时可以显式加 `--host 0.0.0.0`，用 `http://服务器局域网IP:8776/` 访问。服务没有鉴权，能访问它的人可以控制训练，所以不要把该端口暴露到公网；一般使用 SSH 转发即可。

本次交付忽略规则、源码调整和说明，未提交/push、未传输数据、未安装依赖、未构建前端或启动服务。
