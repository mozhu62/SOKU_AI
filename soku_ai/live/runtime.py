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
from soku_ai.data.resources_schema import enabled as resources_enabled

from .observation import LiveObservationBuilder, LiveStateUnavailable
from .shared_state import SharedMemoryClient
from .resources_observation import ResourceLiveObservationBuilder
from .frame_stream import LiveFrameClient
from .idle_guard import IdleGuard
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
    history_target: int = 32
    history_resets: int = 0
    dropped_frames: int = 0
    observation_schema: str = ""
    input_readback: str = ""
    raw_action_id: int = -1
    idle_guard_overridden: bool = False
    idle_guard_interventions: int = 0
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
        player_side: str = "left",
        cpu_threads: int = 1,
        max_observation_age_ms: int = 100,
        max_action_lag_frames: int = 4,
        idle_guard_enabled: bool = True,
        idle_guard_max_frames: int = 60,
    ) -> None:
        self.poll_interval_seconds = max(1, int(poll_interval_ms)) / 1000.0
        self.focus_delay_seconds = max(0, int(focus_delay_ms)) / 1000.0
        self.emergency_pause_key = int(emergency_pause_key)
        self.player_side = player_side
        self.cpu_threads = max(1, int(cpu_threads))
        self.max_observation_age_ms = max(1, int(max_observation_age_ms))
        self.max_action_lag_frames = max(0, int(max_action_lag_frames))
        self._reset_history = threading.Event()
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
        self._idle_guard = IdleGuard()
        self._reset_idle = threading.Event()
        self.set_idle_guard(idle_guard_enabled, idle_guard_max_frames)

    def set_idle_guard(self, enabled: bool, max_frames: int) -> None:
        if not 1 <= int(max_frames) <= 600:
            raise ValueError("无动作上限必须为 1～600 个游戏帧")
        with self._settings_lock:
            self._idle_guard_enabled = bool(enabled)
            self._idle_guard_max_frames = int(max_frames)
        self._reset_idle.set()

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
        self._reset_history.set()
        with self._settings_lock:
            self._control_enabled = bool(control_enabled)
            self._focus_pending = False
            self._activation_not_before = 0.0
            self._pause_message = ""
        self._idle_guard = IdleGuard()
        self._reset_idle.set()
        self._publish(
            state="loading_model",
            message="正在加载模型",
            checkpoint_path=str(checkpoint),
            active=False,
            control_enabled=bool(control_enabled),
            connected=False, action_id=-1, action_description="-", top_actions=(), pressed_keys=(),
            history_count=0, history_resets=0, dropped_frames=0, input_readback="", observation_schema="",
            inference_ms=0.0, inference_fps=0.0,
            raw_action_id=-1, idle_guard_overridden=False, idle_guard_interventions=0,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(checkpoint, device, bindings),
            name="SokuLiveAiRuntime",
            daemon=True,
        )
        self._thread.start()

    def set_control_enabled(self, enabled: bool) -> None:
        self._reset_idle.set()
        with self._settings_lock:
            self._control_enabled = bool(enabled)
        self._publish(control_enabled=bool(enabled))
        self._reset_history.set()

    def resume(self) -> bool:
        self._reset_idle.set()
        if not self.is_running:
            return False
        self._desired_active.set()
        self._reset_history.set()
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
        self._reset_idle.set()
        self._desired_active.clear()
        self._reset_history.set()
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
        self._idle_guard.reset()
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

    def _read_resource_observation(self, client, builder, keyboard):
        if self._reset_history.is_set():
            self._reset_history.clear()
            builder.reset("暂停/继续或控制模式切换")
            client.discard_pending()
        frames = client.read()
        if client.pid is not None:
            self._latest_game_process_id = int(client.pid)
        if client.reset_reason:
            builder.reset(client.reset_reason)
        if not frames:
            if not client.available or client.age_ms() > self.max_observation_age_ms:
                self._safe_release(keyboard)
                builder.reset("DLL 尚未就绪或停止发布新帧")
                self._publish(state="waiting_bridge", connected=client.available,
                    message=client.wait_reason if not client.available else "DLL 观测已过期，等待新帧",
                    pressed_keys=(), history_count=0, history_resets=builder.reset_count)
            return None
        self._latest_game_process_id = int(frames[-1].gameProcessId)
        live = None
        reason = "等待新的连续战斗帧"
        for index, payload in enumerate(frames):
            try:
                live = builder.build(payload, history_only=index < len(frames) - 1)
            except LiveStateUnavailable as error:
                live, reason = None, str(error)
        if live is None:
            self._safe_release(keyboard)
            self._publish(state="waiting_battle", message=reason, connected=True, pressed_keys=(),
                          history_count=len(builder.history_num), history_resets=builder.reset_count,
                          dropped_frames=client.dropped, game_process_id=self._latest_game_process_id)
            return None
        if client.age_ms(frames[-1]) > self.max_observation_age_ms:
            self._safe_release(keyboard)
            builder.reset("最新观测过期")
            self._publish(state="waiting_battle", message="观测过期，等待新帧", pressed_keys=(), history_count=0)
            return None
        if live.history_count < builder.history_len:
            self._safe_release(keyboard)
            self._publish(state="warming_history", connected=True,
                message=f"累计真实连续历史 {live.history_count}/{builder.history_len}，就绪后自动推理",
                game_process_id=live.game_process_id, battle_frame=live.battle_frame,
                current_round=live.current_round, self_side=live.self_side, pressed_keys=(),
                history_count=live.history_count, history_resets=builder.reset_count, dropped_frames=client.dropped)
            return None
        return live

    def _run(
        self,
        checkpoint_path: Path,
        device: str,
        bindings: KeyBindings,
    ) -> None:
        api: WindowsApi | None = None
        keyboard: KeyboardController | None = None
        client: SharedMemoryClient | LiveFrameClient | None = None
        try:
            torch.set_num_threads(self.cpu_threads)
            agent = SokuDQNAgent.load_checkpoint(checkpoint_path, device=device)
            if agent.normalization is None:
                raise ValueError("checkpoint 不包含实时推理所需的归一化参数")
            native = resources_enabled(agent.config)
            builder = (ResourceLiveObservationBuilder(agent.config, agent.normalization, self.player_side)
                       if native else LiveObservationBuilder(agent.config, agent.normalization))
            api = WindowsApi()
            keyboard = KeyboardController(api, bindings)
            emergency = EmergencyPauseKey(api, self.emergency_pause_key)
            client = LiveFrameClient() if native else SharedMemoryClient()
            training_step = int(agent.checkpoint_metadata.get("training_step") or 0)
            self._publish(
                state="waiting_bridge",
                message="模型已加载，等待游戏与 SokuDataBridge.dll",
                training_step=training_step,
                device=str(agent.device),
                observation_schema="resources_v4 DQfD · 144 Q" if native else "旧 DQfD v1 · 144 Q",
                history_target=builder.history_len,
                history_count=0, history_resets=0, dropped_frames=0,
            )

            recent_inference_times: list[float] = []
            while not self._stop_event.is_set():
                if self._reset_idle.is_set():
                    self._reset_idle.clear()
                    self._idle_guard.reset()
                if emergency.poll_pressed_edge():
                    self.pause("F10 紧急暂停已触发")
                if not self._desired_active.is_set() and keyboard.pressed_names:
                    self._safe_release(keyboard)

                if native:
                    live = self._read_resource_observation(client, builder, keyboard)
                    if live is None:
                        # 即使历史还在预热，也先完成用户请求的聚焦，避免失焦暂停的游戏无法积累帧。
                        if self._desired_active.is_set() and self._control_is_enabled() and self._latest_game_process_id:
                            self._focus_if_pending(api, self._latest_game_process_id)
                        self._stop_event.wait(self.poll_interval_seconds)
                        continue
                    process_id = live.game_process_id
                else:
                    live = self._read_legacy_observation(client, builder, keyboard)
                    if live is None:
                        self._stop_event.wait(self.poll_interval_seconds)
                        continue
                    process_id = live.game_process_id

                inference_started = time.perf_counter()
                # Q 只搬回 CPU 一次，排名、argmax 与界面共用，避免多次设备同步。
                q_values = agent.predict_q(live.observation).detach().float().cpu()
                if not torch.isfinite(q_values).all():
                    raise ValueError("模型输出含 NaN/Inf，已停止按键控制")
                ranked = self._build_ranked_actions(q_values)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                active = self._desired_active.is_set()
                control_enabled = self._control_is_enabled()
                action_id = int(q_values.argmax(dim=1).item())
                raw_action_id = action_id
                idle_overridden = False
                action = decode_action(action_id)
                action_facing_right = live.facing_right
                stale = False
                if native:
                    fresh = client.read_state()
                    try:
                        if fresh is None:
                            raise LiveStateUnavailable("等待新观测")
                        player, _, side = builder.validate(fresh.payload)
                        p = fresh.payload
                        stale = (int(p.gameProcessId) != live.game_process_id or int(p.currentRound) != live.current_round
                                 or side != live.self_side or not 0 <= int(p.battleFrame) - live.battle_frame <= self.max_action_lag_frames
                                 or client.age_ms(p) > self.max_observation_age_ms
                                 or client.age_ms() > self.max_observation_age_ms)
                        action_facing_right = bool(player.direction > 0) if builder.positive_right else bool(player.direction < 0)
                    except LiveStateUnavailable:
                        stale = True

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
                        self._reset_history.set()
                        self._safe_release(keyboard)
                        active = False
                        state = "paused"
                        message = "游戏失去焦点，已自动暂停并释放按键"
                    elif time.monotonic() < self._activation_not_before:
                        self._safe_release(keyboard)
                        state = "resuming"
                        message = "游戏已聚焦，等待安全延迟"
                    elif stale:
                        self._safe_release(keyboard)
                        state = "waiting_fresh"
                        message = "跳过已过期的推理结果，自动继续读取新帧"
                    else:
                        try:
                            if not self._desired_active.is_set() or self._stop_event.is_set():
                                self._safe_release(keyboard)
                                active, state, message = False, "paused", "已暂停"
                            else:
                                context = (live.game_process_id, live.current_round, live.self_side)
                                with self._settings_lock:
                                    idle_enabled, idle_limit = self._idle_guard_enabled, self._idle_guard_max_frames
                                control_frame = int(fresh.payload.battleFrame) if native else live.battle_frame
                                action_id, idle_overridden = self._idle_guard.select(
                                    q_values, context, control_frame, idle_enabled, idle_limit)
                                action = decode_action(action_id)
                                keyboard.apply(action, action_facing_right)
                                self._idle_guard.record(action_id, context, control_frame, idle_overridden)
                                state = "controlling"
                                message = "AI 正在控制游戏"
                                if idle_overridden:
                                    message = f"防持续站桩触发：原始动作 0 → 执行动作 {action_id}"
                        except OSError as error:
                            idle_overridden = False
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
                    state=state, message=message, connected=True, active=active,
                    control_enabled=control_enabled, game_process_id=live.game_process_id,
                    battle_frame=live.battle_frame, current_round=live.current_round, self_side=live.self_side,
                    action_id=action_id, action_description=describe_action(action), pressed_keys=keyboard.pressed_names,
                    top_actions=ranked, inference_ms=inference_ms, inference_fps=inference_fps,
                    history_count=live.history_count,
                    raw_action_id=raw_action_id, idle_guard_overridden=idle_overridden,
                    idle_guard_interventions=self._idle_guard.interventions,
                    history_resets=builder.reset_count if native else 0,
                    dropped_frames=client.dropped if native else 0,
                    input_readback=(" ".join(f"{name}={int(value)}" for name, value in
                        zip(("H", "V", "A", "B", "C", "D"), builder.previous_input)) if native else "旧协议"),
                )
                self._stop_event.wait(self.poll_interval_seconds)
        except Exception as error:
            self._desired_active.clear()
            self._publish(
                state="fatal_error", message=f"实时 AI 已停止: {type(error).__name__}: {error}",
                active=False, pressed_keys=(),
            )
        finally:
            if keyboard is not None:
                self._safe_release(keyboard)
            if client is not None:
                client.close()
            self._latest_game_process_id = 0

    def _read_legacy_observation(self, client, builder, keyboard):
        snapshot = client.read()
        if snapshot is None:
            self._safe_release(keyboard)
            builder.reset()
            self._publish(state="waiting_bridge", message="等待游戏与 SokuDataBridge.dll",
                          connected=False, pressed_keys=())
            return None
        payload = snapshot.payload
        self._latest_game_process_id = int(payload.gameProcessId)
        try:
            return builder.build(payload)
        except LiveStateUnavailable as error:
            self._safe_release(keyboard)
            self._publish(state="waiting_battle", message=str(error), connected=bool(payload.initialized),
                          game_process_id=int(payload.gameProcessId), active=self._desired_active.is_set(),
                          pressed_keys=())
            return None
