from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from soku_ai.data.action_space import DecodedAction, decode_action
from soku_ai.inference.agent import SokuDQNAgent

from .observation import LiveObservationBuilder, LiveStateUnavailable
from .shared_state import SharedMemoryClient
from .windows_control import (
    EmergencyPauseKey,
    KeyBindings,
    KeyboardController,
    WindowsApi,
)


@dataclass(frozen=True)
class RankedAction:
    rank: int
    action_id: int
    q_value: float
    description: str


@dataclass(frozen=True)
class RuntimeTelemetry:
    state: str = "stopped"
    message: str = "尚未启动"
    connected: bool = False
    active: bool = False
    control_enabled: bool = True
    checkpoint_path: str = ""
    training_step: int = 0
    device: str = ""
    game_process_id: int = 0
    battle_frame: int = 0
    current_round: int = 0
    self_side: str = "-"
    action_id: int = -1
    action_description: str = "-"
    pressed_keys: tuple[str, ...] = ()
    top_actions: tuple[RankedAction, ...] = ()
    inference_ms: float = 0.0
    inference_fps: float = 0.0
    history_count: int = 0
    updated_at: float = 0.0


def describe_action(action: DecodedAction) -> str:
    horizontal = {
        "NONE": "水平空",
        "FORWARD": "前进",
        "BACKWARD": "后退",
    }[action.horizontal.name]
    vertical = {"NONE": "垂直空", "UP": "上", "DOWN": "下"}[
        action.vertical.name
    ]
    buttons = "".join(
        name
        for name, pressed in (
            ("A", action.a),
            ("B", action.b),
            ("C", action.c),
            ("D", action.d),
        )
        if pressed
    )
    return f"{horizontal} + {vertical} + {buttons or '无按钮'}"


class LiveAiRuntime:
    """在后台串联共享内存、模型推理和安全键盘控制。"""

    def __init__(
        self,
        *,
        poll_interval_ms: int = 2,
        focus_delay_ms: int = 300,
        emergency_pause_key: int = 0x79,
    ) -> None:
        self.poll_interval_seconds = max(1, int(poll_interval_ms)) / 1000.0
        self.focus_delay_seconds = max(0, int(focus_delay_ms)) / 1000.0
        self.emergency_pause_key = int(emergency_pause_key)
        self._stop_event = threading.Event()
        self._desired_active = threading.Event()
        self._thread: threading.Thread | None = None
        self._settings_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._telemetry_queue: queue.Queue[RuntimeTelemetry] = queue.Queue(maxsize=1)
        self._telemetry = RuntimeTelemetry()
        self._control_enabled = True
        self._focus_pending = False
        self._activation_not_before = 0.0
        self._latest_game_process_id = 0
        self._pause_message = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def latest_telemetry(self) -> RuntimeTelemetry:
        with self._telemetry_lock:
            return self._telemetry

    def poll_telemetry(self) -> RuntimeTelemetry | None:
        latest: RuntimeTelemetry | None = None
        while True:
            try:
                latest = self._telemetry_queue.get_nowait()
            except queue.Empty:
                return latest

    def _publish(self, **changes: Any) -> None:
        with self._telemetry_lock:
            self._telemetry = replace(
                self._telemetry,
                updated_at=time.time(),
                **changes,
            )
            value = self._telemetry
        try:
            self._telemetry_queue.put_nowait(value)
        except queue.Full:
            try:
                self._telemetry_queue.get_nowait()
            except queue.Empty:
                pass
            self._telemetry_queue.put_nowait(value)

    def start(
        self,
        checkpoint_path: str | Path,
        *,
        device: str,
        bindings: KeyBindings,
        control_enabled: bool,
    ) -> None:
        if self.is_running:
            raise RuntimeError("实时 AI 已经在运行")
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
        self._stop_event.clear()
        self._desired_active.clear()
        with self._settings_lock:
            self._control_enabled = bool(control_enabled)
            self._focus_pending = False
            self._activation_not_before = 0.0
            self._pause_message = ""
        self._publish(
            state="loading_model",
            message="正在加载模型",
            checkpoint_path=str(checkpoint),
            active=False,
            control_enabled=bool(control_enabled),
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(checkpoint, device, bindings),
            name="SokuLiveAiRuntime",
            daemon=True,
        )
        self._thread.start()

    def set_control_enabled(self, enabled: bool) -> None:
        with self._settings_lock:
            self._control_enabled = bool(enabled)
        self._publish(control_enabled=bool(enabled))

    def resume(self) -> bool:
        if not self.is_running:
            return False
        self._desired_active.set()
        focused = True
        with self._settings_lock:
            control_enabled = self._control_enabled
            self._focus_pending = control_enabled
            self._pause_message = ""
        if control_enabled and self._latest_game_process_id:
            api = WindowsApi()
            focused = api.focus_process_window(self._latest_game_process_id)
            with self._settings_lock:
                self._focus_pending = not focused
                if focused:
                    self._activation_not_before = (
                        time.monotonic() + self.focus_delay_seconds
                    )
        self._publish(
            state="resuming",
            message=(
                "正在切回游戏，稍后恢复控制"
                if control_enabled
                else "影子推理已继续"
            ),
            active=True,
        )
        return focused

    def pause(self, message: str = "已由界面暂停") -> None:
        self._desired_active.clear()
        with self._settings_lock:
            self._focus_pending = False
            self._pause_message = message
        self._publish(state="paused", message=message, active=False)

    def stop(self) -> None:
        self._desired_active.clear()
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if thread is None or not thread.is_alive():
            self._thread = None
            self._publish(
                state="stopped",
                message="已停止并释放全部按键",
                active=False,
                connected=False,
                pressed_keys=(),
            )

    def _safe_release(self, keyboard: KeyboardController) -> None:
        try:
            keyboard.release_all()
        except OSError as error:
            self._desired_active.clear()
            self._publish(
                state="input_error",
                message=f"释放按键失败: {error}",
                active=False,
                pressed_keys=(),
            )

    def _focus_if_pending(self, api: WindowsApi, process_id: int) -> bool:
        with self._settings_lock:
            pending = self._focus_pending
        if not pending:
            return True
        focused = api.focus_process_window(process_id)
        with self._settings_lock:
            self._focus_pending = not focused
            if focused:
                self._activation_not_before = (
                    time.monotonic() + self.focus_delay_seconds
                )
        return focused

    def _control_is_enabled(self) -> bool:
        with self._settings_lock:
            return self._control_enabled

    def _current_pause_message(self) -> str:
        with self._settings_lock:
            return self._pause_message

    def _build_ranked_actions(self, q_values: torch.Tensor) -> tuple[RankedAction, ...]:
        count = min(5, int(q_values.shape[1]))
        values, indices = torch.topk(q_values[0], count)
        value_list = values.detach().float().cpu().tolist()
        index_list = indices.detach().cpu().tolist()
        return tuple(
            RankedAction(
                rank=rank,
                action_id=int(action_id),
                q_value=float(q_value),
                description=describe_action(decode_action(int(action_id))),
            )
            for rank, (action_id, q_value) in enumerate(
                zip(index_list, value_list, strict=True), start=1
            )
        )

    def _run(
        self,
        checkpoint_path: Path,
        device: str,
        bindings: KeyBindings,
    ) -> None:
        api: WindowsApi | None = None
        keyboard: KeyboardController | None = None
        client: SharedMemoryClient | None = None
        try:
            agent = SokuDQNAgent.load_checkpoint(checkpoint_path, device=device)
            if agent.normalization is None:
                raise ValueError("checkpoint 不包含实时推理所需的归一化参数")
            builder = LiveObservationBuilder(agent.config, agent.normalization)
            api = WindowsApi()
            keyboard = KeyboardController(api, bindings)
            emergency = EmergencyPauseKey(api, self.emergency_pause_key)
            client = SharedMemoryClient()
            training_step = int(agent.checkpoint_metadata.get("training_step") or 0)
            self._publish(
                state="waiting_bridge",
                message="模型已加载，等待游戏与 SokuDataBridge.dll",
                training_step=training_step,
                device=str(agent.device),
            )

            last_serial = -1
            recent_inference_times: list[float] = []
            last_wait_message = ""
            while not self._stop_event.is_set():
                if emergency.poll_pressed_edge():
                    self.pause("F10 紧急暂停已触发")
                if not self._desired_active.is_set() and keyboard.pressed_names:
                    self._safe_release(keyboard)

                snapshot = client.read()
                if snapshot is None:
                    self._safe_release(keyboard)
                    builder.reset()
                    if last_wait_message != "bridge":
                        self._publish(
                            state="waiting_bridge",
                            message="等待游戏与 SokuDataBridge.dll",
                            connected=False,
                            pressed_keys=(),
                        )
                        last_wait_message = "bridge"
                    self._stop_event.wait(self.poll_interval_seconds)
                    continue

                payload = snapshot.payload
                process_id = int(payload.gameProcessId)
                self._latest_game_process_id = process_id
                serial = int(payload.sampleSerial)
                if serial == last_serial:
                    self._stop_event.wait(self.poll_interval_seconds)
                    continue
                last_serial = serial

                try:
                    live = builder.build(payload)
                except LiveStateUnavailable as error:
                    self._safe_release(keyboard)
                    message = str(error)
                    if message != last_wait_message:
                        self._publish(
                            state="waiting_battle",
                            message=message,
                            connected=bool(payload.initialized),
                            game_process_id=process_id,
                            active=self._desired_active.is_set(),
                            pressed_keys=(),
                        )
                        last_wait_message = message
                    self._stop_event.wait(self.poll_interval_seconds)
                    continue
                if live is None:
                    self._stop_event.wait(self.poll_interval_seconds)
                    continue
                last_wait_message = ""

                inference_started = time.perf_counter()
                q_values = agent.predict_q(live.observation)
                ranked = self._build_ranked_actions(q_values)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                active = self._desired_active.is_set()
                control_enabled = self._control_is_enabled()
                action_id = int(q_values.argmax(dim=1).item())
                action = decode_action(action_id)

                now = time.monotonic()
                recent_inference_times.append(now)
                cutoff = now - 1.0
                while recent_inference_times and recent_inference_times[0] < cutoff:
                    recent_inference_times.pop(0)
                inference_fps = float(len(recent_inference_times))

                state = "paused"
                message = self._current_pause_message() or "实时推理正常，键盘控制已暂停"
                if active and control_enabled:
                    if not self._focus_if_pending(api, process_id):
                        self._desired_active.clear()
                        self._safe_release(keyboard)
                        active = False
                        state = "paused"
                        message = "无法切换到游戏窗口，请再次点击继续"
                    elif api.foreground_process_id() != process_id:
                        self._desired_active.clear()
                        self._safe_release(keyboard)
                        active = False
                        state = "paused"
                        message = "游戏失去焦点，已自动暂停并释放按键"
                    elif time.monotonic() < self._activation_not_before:
                        self._safe_release(keyboard)
                        state = "resuming"
                        message = "游戏已聚焦，等待安全延迟"
                    else:
                        try:
                            keyboard.apply(action, live.facing_right)
                            state = "controlling"
                            message = "AI 正在控制萃香"
                        except OSError as error:
                            self._desired_active.clear()
                            self._safe_release(keyboard)
                            active = False
                            state = "input_error"
                            message = f"发送键盘输入失败: {error}"
                elif active:
                    self._safe_release(keyboard)
                    state = "shadow"
                    message = "影子推理中，不发送键盘输入"
                else:
                    self._safe_release(keyboard)

                self._publish(
                    state=state,
                    message=message,
                    connected=True,
                    active=active,
                    control_enabled=control_enabled,
                    game_process_id=live.game_process_id,
                    battle_frame=live.battle_frame,
                    current_round=live.current_round,
                    self_side=live.self_side,
                    action_id=action_id,
                    action_description=describe_action(action),
                    pressed_keys=keyboard.pressed_names,
                    top_actions=ranked,
                    inference_ms=inference_ms,
                    inference_fps=inference_fps,
                    history_count=live.history_count,
                )
                self._stop_event.wait(self.poll_interval_seconds)
        except Exception as error:
            self._desired_active.clear()
            self._publish(
                state="fatal_error",
                message=f"实时 AI 已停止: {type(error).__name__}: {error}",
                active=False,
                pressed_keys=(),
            )
        finally:
            if keyboard is not None:
                self._safe_release(keyboard)
            if client is not None:
                client.close()
            self._latest_game_process_id = 0
