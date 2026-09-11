from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .runtime import LiveAiRuntime, RuntimeTelemetry
from .windows_control import KeyBindings, parse_virtual_key


def load_live_settings(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path).resolve()
    with settings_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"实时配置根节点必须是映射: {settings_path}")
    payload["_settings_path"] = str(settings_path)
    return payload


def find_latest_checkpoint(project_root: Path) -> Path:
    checkpoint_root = project_root / "checkpoints"
    snapshot_dirs = (
        checkpoint_root / "aggressive_v2" / "snapshots",
        checkpoint_root / "noop_finetune" / "snapshots",
        checkpoint_root / "snapshots",
    )
    candidates = [
        path
        for directory in snapshot_dirs
        if directory.is_dir()
        for path in directory.glob("step_*.pt")
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()
    last_candidates = (
        checkpoint_root / "aggressive_v2" / "last.pt",
        checkpoint_root / "noop_finetune" / "last.pt",
        checkpoint_root / "last.pt",
    )
    existing = [path for path in last_candidates if path.is_file()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime_ns).resolve()
    return last_candidates[-1].resolve()


class LiveAiApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        project_root: Path,
        settings: dict[str, Any],
        checkpoint_override: str | None = None,
        device_override: str | None = None,
    ) -> None:
        self.root = root
        self.project_root = project_root.resolve()
        self.settings = settings
        emergency_key = parse_virtual_key(settings.get("emergency_pause_key", "F10"))
        self.runtime = LiveAiRuntime(
            poll_interval_ms=int(settings.get("poll_interval_ms", 2)),
            focus_delay_ms=int(settings.get("focus_delay_ms", 300)),
            emergency_pause_key=emergency_key,
        )
        configured_checkpoint = checkpoint_override or str(settings.get("checkpoint", ""))
        if configured_checkpoint:
            checkpoint = Path(configured_checkpoint)
            if not checkpoint.is_absolute():
                checkpoint = self.project_root / checkpoint
            checkpoint = checkpoint.resolve()
        else:
            checkpoint = find_latest_checkpoint(self.project_root)

        self.checkpoint_var = tk.StringVar(value=str(checkpoint))
        self.device_var = tk.StringVar(
            value=device_override or str(settings.get("device", "auto"))
        )
        self.control_enabled_var = tk.BooleanVar(
            value=bool(settings.get("control_enabled", True))
        )
        default_bindings = KeyBindings.from_mapping(settings.get("key_bindings", {}))
        self.key_vars = {
            name: tk.StringVar(value=value)
            for name, value in default_bindings.as_names().items()
        }
        self.status_vars = {
            name: tk.StringVar(value="-")
            for name in (
                "state",
                "connection",
                "model",
                "device",
                "step",
                "pid",
                "frame",
                "round",
                "side",
                "action",
                "keys",
                "latency",
                "history",
            )
        }
        self._last_log_key: tuple[str, str] | None = None
        self._build_window()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._refresh)

    def _build_window(self) -> None:
        self.root.title("TH123 萃香实时 AI 运行器")
        self.root.geometry("940x730")
        self.root.minsize(820, 650)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        settings_frame = ttk.LabelFrame(container, text="模型与控制设置", padding=10)
        settings_frame.grid(row=0, column=0, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)
        ttk.Label(settings_frame, text="模型文件").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.checkpoint_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(settings_frame, text="浏览", command=self._browse_checkpoint).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Label(settings_frame, text="设备").grid(row=1, column=0, pady=(10, 0))
        ttk.Combobox(
            settings_frame,
            textvariable=self.device_var,
            values=("auto", "cuda", "cpu"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            settings_frame,
            text="启用键盘控制（关闭后仅做影子推理）",
            variable=self.control_enabled_var,
            command=self._on_control_mode_changed,
        ).grid(row=1, column=1, sticky="e", pady=(10, 0))

        key_frame = ttk.Frame(settings_frame)
        key_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        labels = (
            ("left", "左"),
            ("right", "右"),
            ("up", "上"),
            ("down", "下"),
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
            ("d", "D"),
        )
        for column, (name, label) in enumerate(labels):
            ttk.Label(key_frame, text=label).grid(row=0, column=column, padx=3)
            ttk.Entry(key_frame, textvariable=self.key_vars[name], width=8).grid(
                row=1, column=column, padx=3
            )

        controls = ttk.Frame(container, padding=(0, 10))
        controls.grid(row=1, column=0, sticky="ew")
        ttk.Button(controls, text="启动 / 继续", command=self._on_start).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="暂停", command=self._on_pause).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(controls, text="停止", command=self._on_stop).pack(side=tk.LEFT)
        ttk.Label(
            controls,
            text="F10：全局紧急暂停并释放全部按键",
            foreground="#9a3412",
        ).pack(side=tk.RIGHT)

        telemetry = ttk.LabelFrame(container, text="实时状态", padding=10)
        telemetry.grid(row=2, column=0, sticky="ew")
        telemetry.columnconfigure(1, weight=1)
        telemetry.columnconfigure(3, weight=1)
        fields = (
            ("运行状态", "state", "连接", "connection"),
            ("模型", "model", "设备 / Step", "device"),
            ("游戏 PID", "pid", "帧 / 回合 / 侧别", "frame"),
            ("模型动作", "action", "实际按键", "keys"),
            ("推理耗时 / FPS", "latency", "历史帧", "history"),
        )
        for row, (left_label, left_key, right_label, right_key) in enumerate(fields):
            ttk.Label(telemetry, text=left_label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ttk.Label(telemetry, textvariable=self.status_vars[left_key]).grid(
                row=row, column=1, sticky="w", pady=2
            )
            ttk.Label(telemetry, text=right_label).grid(
                row=row, column=2, sticky="w", padx=(18, 8), pady=2
            )
            value_key = right_key
            if right_key == "device":
                value_key = "step"
            ttk.Label(telemetry, textvariable=self.status_vars[value_key]).grid(
                row=row, column=3, sticky="w", pady=2
            )

        lower = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        lower.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        q_frame = ttk.LabelFrame(lower, text="Q 值 Top-5", padding=6)
        log_frame = ttk.LabelFrame(lower, text="运行日志", padding=6)
        lower.add(q_frame, weight=3)
        lower.add(log_frame, weight=2)

        self.q_table = ttk.Treeview(
            q_frame,
            columns=("rank", "action", "q", "description"),
            show="headings",
            height=9,
        )
        self.q_table.heading("rank", text="#")
        self.q_table.heading("action", text="动作 ID")
        self.q_table.heading("q", text="Q 值")
        self.q_table.heading("description", text="动作")
        self.q_table.column("rank", width=40, anchor=tk.CENTER, stretch=False)
        self.q_table.column("action", width=70, anchor=tk.CENTER, stretch=False)
        self.q_table.column("q", width=100, anchor=tk.E, stretch=False)
        self.q_table.column("description", width=250)
        self.q_table.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择模型快照",
            initialdir=str(self.project_root / "checkpoints"),
            filetypes=(("PyTorch 模型", "*.pt"), ("全部文件", "*.*")),
        )
        if selected:
            self.checkpoint_var.set(selected)

    def _read_bindings(self) -> KeyBindings:
        return KeyBindings.from_mapping(
            {name: variable.get() for name, variable in self.key_vars.items()}
        )

    def _on_start(self) -> None:
        try:
            if not self.runtime.is_running:
                self.runtime.start(
                    self.checkpoint_var.get(),
                    device=self.device_var.get(),
                    bindings=self._read_bindings(),
                    control_enabled=self.control_enabled_var.get(),
                )
            else:
                self.runtime.set_control_enabled(self.control_enabled_var.get())
            self.runtime.resume()
        except Exception as error:
            messagebox.showerror("无法启动实时 AI", str(error), parent=self.root)

    def _on_pause(self) -> None:
        self.runtime.pause()

    def _on_stop(self) -> None:
        self.runtime.stop()

    def _on_control_mode_changed(self) -> None:
        if self.runtime.is_running:
            self.runtime.set_control_enabled(self.control_enabled_var.get())

    def _append_log(self, telemetry: RuntimeTelemetry) -> None:
        log_key = (telemetry.state, telemetry.message)
        if log_key == self._last_log_key:
            return
        self._last_log_key = log_key
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{telemetry.message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _render(self, telemetry: RuntimeTelemetry) -> None:
        self.status_vars["state"].set(telemetry.message)
        self.status_vars["connection"].set(
            "游戏已连接" if telemetry.connected else "等待游戏 / DLL"
        )
        self.status_vars["model"].set(
            Path(telemetry.checkpoint_path).name if telemetry.checkpoint_path else "-"
        )
        self.status_vars["device"].set(telemetry.device or "-")
        self.status_vars["step"].set(
            f"{telemetry.device or '-'} / {telemetry.training_step:,}"
        )
        self.status_vars["pid"].set(
            str(telemetry.game_process_id) if telemetry.game_process_id else "-"
        )
        self.status_vars["frame"].set(
            f"{telemetry.battle_frame} / {telemetry.current_round} / {telemetry.self_side}"
            if telemetry.connected
            else "-"
        )
        self.status_vars["action"].set(
            f"{telemetry.action_id}: {telemetry.action_description}"
            if telemetry.action_id >= 0
            else "-"
        )
        self.status_vars["keys"].set(
            " + ".join(telemetry.pressed_keys) if telemetry.pressed_keys else "无"
        )
        self.status_vars["latency"].set(
            f"{telemetry.inference_ms:.2f} ms / {telemetry.inference_fps:.0f} FPS"
        )
        self.status_vars["history"].set(f"{telemetry.history_count} / 32")
        for item in self.q_table.get_children():
            self.q_table.delete(item)
        for item in telemetry.top_actions:
            self.q_table.insert(
                "",
                tk.END,
                values=(
                    item.rank,
                    item.action_id,
                    f"{item.q_value:.5f}",
                    item.description,
                ),
            )
        self._append_log(telemetry)

    def _refresh(self) -> None:
        telemetry = self.runtime.poll_telemetry()
        if telemetry is not None:
            self._render(telemetry)
        self.root.after(100, self._refresh)

    def _on_close(self) -> None:
        self.runtime.stop()
        self.root.destroy()
