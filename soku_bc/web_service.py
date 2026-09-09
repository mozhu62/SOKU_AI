from __future__ import annotations

import asyncio
import errno
import ipaddress
import socket
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EDITABLE, MODULES, ROOT

WEB_DIST = ROOT / "web" / "dist"


class Command(BaseModel):
    id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    name: str
    value: dict = Field(default_factory=dict)


def bind_port(host, first, attempts):
    """检测即绑定并保留 socket，避免先检测后启动之间端口被其他进程抢占。"""
    failures = []
    for port in range(first, min(65536, first + attempts)):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind((host, port))
            listener.listen(128)
            listener.setblocking(False)
            return listener, port, failures
        except OSError as error:
            listener.close()
            if error.errno not in (errno.EADDRINUSE, errno.EACCES, 10048, 10013):
                raise
            failures.append(port)
    raise OSError(f"本机端口 {first} 起的 {attempts} 个候选均无法绑定，请用 --port 指定其他起始端口")


def _loopback_client(client_host):
    try:
        return ipaddress.ip_address(client_host or "").is_loopback
    except ValueError:
        return False


def _private_host(value, port, *, client_host=None):
    try:
        parsed = urlsplit("//" + value)
        if (parsed.username is not None or parsed.password is not None or parsed.path or parsed.query
                or parsed.fragment or parsed.port is None or not 1 <= parsed.port <= 65535):
            return False
        name = parsed.hostname or ""
        if name == "localhost":
            local_name = True
        else:
            address = ipaddress.ip_address(name)
            if not (address.is_private or address.is_loopback or address.is_link_local):
                return False
            local_name = address.is_loopback
        # SSH -L 保留浏览器的 Host，客户端端口可以不同；仅允许回环连接使用回环名称转发。
        forwarded_local = local_name and _loopback_client(client_host)
        return parsed.port == port or forwarded_local
    except ValueError:
        return False


def _same_origin(origin, host):
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        return (parsed.scheme == "http" and parsed.netloc.lower() == host.lower()
                and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def _lan_addresses():
    addresses = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = ipaddress.ip_address(item[4][0])
            if address.is_private and not address.is_loopback:
                addresses.add(str(address))
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 9))
        address = ipaddress.ip_address(probe.getsockname()[0])
        probe.close()
        if address.is_private and not address.is_loopback:
            addresses.add(str(address))
    except OSError:
        pass
    return sorted(addresses)


def create_app(runtime, port):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        peer = request.client.host if request.client else None
        if (not _private_host(host, port, client_host=peer) or not _same_origin(request.headers.get("origin"), host)
                or request.headers.get("sec-fetch-site") == "cross-site"):
            return JSONResponse({"detail": "仅允许本机或私有局域网同来源访问"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'")
        return response

    @app.get("/api/status")
    def status():
        return {**runtime.snapshot(), "ui_mode": "bc_training"}

    @app.get("/api/parameters")
    def parameters():
        state = runtime.snapshot()
        return {"config": state["config"], "stage": state["stage"], "source": state["config_source"],
                "locks": state["locked_parameters"], "modules": MODULES,
                "fields": [{"key": key, "min": value[0], "max": value[1], "type": value[2], "label": value[3]}
                           for key, value in EDITABLE.items()]}

    @app.get("/api/history")
    def history(kind="train", offset: int = 0, limit: int = 200):
        if kind not in ("train", "validation", "configuration"):
            raise HTTPException(400, "未知记录类型")
        return runtime.history_page(kind, max(0, offset), max(1, min(limit, 1000)))

    @app.get("/api/dataset")
    def dataset(offset: int = 0, limit: int = 100):
        if not runtime.snapshot()["data"]:
            return {"rows": [], "total": 0}
        rows = list(runtime.store.info.values())
        offset = max(0, offset)
        return {"rows": rows[offset:offset + max(1, min(limit, 200))], "total": len(rows)}

    @app.get("/api/models")
    def models():
        rows = []
        for pattern in ("last.pt", "best.pt", "best_stage_*.pt", "snapshots/*.pt"):
            for path in runtime.output.glob(pattern):
                if path.is_file() and path.resolve().is_relative_to(runtime.output):
                    stat = path.stat()
                    rows.append({"name": path.relative_to(runtime.output).as_posix(),
                                 "size_mb": stat.st_size / 1024 ** 2, "modified": stat.st_mtime})
        return sorted(rows, key=lambda row: row["modified"], reverse=True)[:200]

    @app.post("/api/commands")
    def command(value: Command):
        try:
            return runtime.submit(value.id, value.name, value.value)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/commands/{identifier}")
    def request_status(identifier: str):
        result = runtime.request(identifier)
        if result is None:
            raise HTTPException(404, "请求不存在")
        return result

    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket):
        host = websocket.headers.get("host", "")
        peer = websocket.client.host if websocket.client else None
        protocols = [part.strip() for part in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        if (not _private_host(host, port, client_host=peer) or not _same_origin(websocket.headers.get("origin"), host)
                or websocket.headers.get("sec-fetch-site") == "cross-site"
                or protocols != ["soku-bc"]):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol="soku-bc")
        try:
            while True:
                await websocket.send_json({**runtime.snapshot(), "ui_mode": "bc_training"})
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="bc-workbench")
    return app


def run_workbench(runtime):
    import uvicorn
    if not (WEB_DIST / "index.html").is_file():
        raise RuntimeError("BC 前端尚未构建：在 web 目录执行 npm ci 和 npm run build；或使用 --headless")
    bind_host = runtime.config["web"]["host"]
    listener, port, occupied = bind_port(bind_host, runtime.config["web"]["port"], runtime.config["web"]["port_attempts"])
    try:
        if occupied:
            print(f"端口占用或不可用：{occupied}；已改用 {port}", flush=True)
        app = create_app(runtime, port)
        # 客户端地址必须来自实际连接，不能让转发头伪造回环来源。
        server = uvicorn.Server(uvicorn.Config(app, host=bind_host, port=port, access_log=False, proxy_headers=False))
        addresses = _lan_addresses() if bind_host == "0.0.0.0" else []
        print(f"BC 离线训练工作台：http://127.0.0.1:{port}/", flush=True)
        print(f"SSH 转发（在你的电脑执行）：ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8796:127.0.0.1:{port} 用户名@服务器地址", flush=True)
        print("转发后本地浏览器：http://localhost:8796/ （无 token、无额外路径）", flush=True)
        for address in addresses:
            print(f"局域网访问：http://{address}:{port}/", flush=True)
        if bind_host == "0.0.0.0" and not addresses:
            print(f"局域网访问：http://服务器局域网IP:{port}/", flush=True)
        print("网页中开始/暂停训练；Ctrl+C 停止并保存。此入口不启动游戏。", flush=True)
        runtime.start()
        try:
            try:
                server.run(sockets=[listener])
            except KeyboardInterrupt:
                # uvloop 在 Ctrl+C 时可能先抛 CancelledError，再转换为 KeyboardInterrupt；按正常停止处理。
                print("收到 Ctrl+C，正在停止训练线程；已有训练状态将按运行阶段保存。", flush=True)
        finally:
            runtime.stop_event.set()
            runtime.thread.join()
    finally:
        listener.close()
