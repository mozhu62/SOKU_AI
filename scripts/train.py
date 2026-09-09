from __future__ import annotations

import argparse
import logging
import time

import _bootstrap

from soku_bc.checkpoint import load
from soku_bc.config import load_config, resolve, validate
from soku_bc.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description="BC 离线训练；仅读取 NPZ，不启动游戏")
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--output", help="相对 soku_bc 的独立模型输出目录")
    parser.add_argument("--temporal-mode", choices=("gru", "tcn"),
                        help="新实验时序结构；tcn 自动设置 31 帧前导上下文，跨架构不能 --resume")
    parser.add_argument("--port", type=int, help="本机服务起始端口，冲突自动顺延")
    parser.add_argument("--host", choices=("0.0.0.0", "127.0.0.1"), help="网页监听地址")
    parser.add_argument("--headless", action="store_true", help="无网页，立即开始训练")
    parser.add_argument("--start", action="store_true", help="网页服务就绪后自动开始训练")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.resume and args.config is None:
        config = load(resolve(args.resume))["config"]
        source = "BC checkpoint 配置"
    else:
        config = load_config(args.config or "configs/bc_suika.yaml")
        source = args.config or "configs/bc_suika.yaml"
    if args.output:
        config["output"]["directory"] = args.output
    if args.temporal_mode:
        config["model"]["temporal_mode"] = args.temporal_mode
        if args.temporal_mode == "tcn":
            config["training"]["burn_in"] = 31
    if args.port:
        config["web"]["port"] = args.port
    if args.host:
        config["web"]["host"] = args.host
    validate(config)
    runtime = Runtime(config, args.resume, args.headless or args.start, source)
    if not args.headless:
        from soku_bc.web_service import run_workbench
        run_workbench(runtime)
    else:
        runtime.start()
        try:
            while runtime.thread.is_alive():
                state = runtime.snapshot()
                if state["state"] == "paused" and state["step"] >= config["training"]["total_steps"]:
                    runtime.stop_event.set()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("正在停止并保存，请等待完成。", flush=True)
        finally:
            runtime.stop_event.set()
            runtime.thread.join()
    if runtime.snapshot()["state"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
