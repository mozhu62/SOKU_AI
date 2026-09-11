from __future__ import annotations

import argparse
import ctypes
import tkinter as tk
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from soku_ai.live.gui import LiveAiApp, load_live_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TH123 萃香实时 AI 运行器")
    parser.add_argument(
        "--settings",
        default=str(PROJECT_ROOT / "configs" / "live_ai.json"),
        help="实时运行配置 JSON",
    )
    parser.add_argument("--checkpoint", help="覆盖配置中的模型文件")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        help="覆盖配置中的推理设备",
    )
    return parser.parse_args()


def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def main() -> None:
    args = parse_args()
    settings = load_live_settings(args.settings)
    enable_dpi_awareness()
    root = tk.Tk()
    LiveAiApp(
        root,
        project_root=Path(PROJECT_ROOT),
        settings=settings,
        checkpoint_override=args.checkpoint,
        device_override=args.device,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

