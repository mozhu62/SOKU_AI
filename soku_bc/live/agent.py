from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch

from ..checkpoint import load
from ..models import BCNetwork
from ..action_space import ACTION_COUNT, to_controller, decode, action_name
from .observation import ObservationBuilder
from .tcn_window import TCNObservationWindow
from .batch_graph_candidate import BatchGraphCandidate


class LiveAgent:
    def __init__(self, path: Path, config):
        if not path.is_file():
            raise FileNotFoundError(f"找不到实战模型：{path}；请从训练服务器复制完整的 .pt 文件")
        before = path.stat()
        torch.set_num_threads(config["cpu_threads"])
        requested = config["device"]
        self.hybrid = requested == "hybrid"
        if self.hybrid and not torch.cuda.is_available():
            raise ValueError("hybrid 需要可用 CUDA；请修复 CUDA 环境或显式选择 cpu")
        requested = "cpu" if self.hybrid else requested
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
        self.tcn_device = torch.device("cuda:0") if self.hybrid else self.device
        self.device_label = "hybrid (CPU + TCN CUDA:0)" if self.hybrid else str(self.device)
        package = load(path)
        self.spell_settings = package['config'].get('spell_system', {})
        self.model = BCNetwork(package["spec"]["model"], package["network_version"], spell_system=self.spell_settings)
        self.card_client = None
        self.card_pid = None
        self.card_observation = self.card_snapshot = None
        from .spell_macro import SpellMacro
        self.card_macro = SpellMacro()
        self.cards = self.spell_settings.get('cards', [])
        self.card_status = '等待卡牌快照' if self.model.spell_enabled else '当前模型没有卡牌头'
        if self.model.spec != package["spec"]:
            raise ValueError("模型输入/Joint Action schema 不兼容；只接受新版 144-way BC 模型")
        # BC 包的权重键是 model；不能把 CQL 的 online Q 网络或优化器当作策略加载。
        self.model.load_state_dict(package["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        # 仅移动 TCN；状态 embedding、对象编码器和融合头保留在 CPU。
        if self.hybrid:
            self.model.tcn.to(self.tcn_device)
        self.amp = config.get("amp", False) and self.tcn_device.type == "cuda"
        self.precision = "AMP FP16" if self.amp else "FP32"
        if self.hybrid:
            self.precision = "CPU FP32 / TCN " + self.precision
        self.temporal_mode = self.model.temporal_mode
        if config["environment"]["decision_interval_frames"] != 1:
            raise ValueError("TCN 实战 decision_interval_frames 必须为1，禁止稀疏决策冒充连续帧")
        # 回读方向使用模型训练时的轴约定；Joint Action 的方向已经是屏幕绝对九宫格。
        self.vertical_positive_is_down = bool(package["config"]["data"]["vertical_positive_is_down"])
        self.builder = ObservationBuilder(package["normalization"], config["environment"]["player_side"],
                                          self.vertical_positive_is_down)
        self.step = int(package["step"])
        self.action_selection = "deterministic_argmax_logits"
        self.path = str(path)
        with path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("加载过程中模型文件发生变化，请先复制为固定文件再开始评估")
        self.memory = None
        self.tcn_window = TCNObservationWindow(self.model.tcn.context_frames, config.get("streaming_tcn", True))
        if config.get('tcn_cuda_graph', True) and self.tcn_window.streaming and self.tcn_device.type == 'cuda':
            # 捕获失败明确中止加载，不能暗中退回慢路径而显示已经加速。
            self.tcn_window.graph = BatchGraphCandidate(self.model.tcn, self.tcn_device, self.amp)
        self.timing_calls = 0

    def reset(self):
        self.memory = None
        self.builder.reset()
        self.tcn_window.reset()
        self.card_macro.reset()
        self.card_observation = self.card_snapshot = None
        self.card_status = '等待卡牌快照' if self.model.spell_enabled else '当前模型没有卡牌头'

    def prepare_cards(self, payload):
        if not self.model.spell_enabled:
            return True
        from .card_state import CardMemoryClient, observation
        from .spell_macro import CardSnapshot
        if self.card_pid != int(payload.gameProcessId):
            if self.card_client is not None:
                self.card_client.close()
            self.card_pid = int(payload.gameProcessId)
            self.card_client = CardMemoryClient(self.card_pid)
        frame = self.card_client.read_matching(payload)
        if frame is None:
            self.card_macro.reset('missing_card_frame')
            self.card_snapshot = None
            self.card_status = 'Cards.v1 未匹配当前帧，暂停发键；检查 DLL 版本与连接'
            return False
        side = self.builder.side
        player = payload.left if side == 'left' else payload.right
        if int(player.characterId) != self.spell_settings['character_id']:
            raise ValueError('卡牌模型角色与控制侧角色不符，不能用另一角色的 Card ID 表执行')
        encoded = observation(frame, side, [card['id'] for card in self.cards])
        self.card_observation = torch.from_numpy(encoded['spell_available_mask'])[None].to(self.device)
        captured = frame.players[0 if side == 'left' else 1]
        count = int(player.handCount)
        if not 0 <= count <= 16:
            raise ValueError('实时手牌数量无效')
        self.card_snapshot = CardSnapshot(
            (self.card_pid, int(payload.currentRound)), int(payload.battleFrame),
            tuple(int(player.handCards[i].id) for i in range(count)),
            int(player.selectedCardData.id) if count else None,
            frozenset(captured.after.cardIds[:captured.after.count]), True,
            confirmed_event_serial=int(payload.sampleSerial),
            confirmed_card_id=int(captured.usedCardIds[0]) if captured.eventValid and captured.eventCount == 1 else None)
        self.card_status = '卡牌快照已匹配'
        return True

    def card_diagnostics(self):
        # 当前执行状态独立于最近预测；暂停后不能把历史宏指令当作仍在执行。
        snapshot = self.card_snapshot
        return {'enabled': self.model.spell_enabled, 'status': self.card_status,
                'catalog': self.cards, 'frame': snapshot.frame if snapshot else None,
                'hand_ids': list(snapshot.hand_ids) if snapshot else None,
                'available_ids': sorted(snapshot.available_ids) if snapshot else None,
                'selected_id': snapshot.selected_id if snapshot else None,
                'macro_state': self.card_macro.state, 'target_card_id': self.card_macro.target}

    def apply_prediction(self, control, prediction):
        if not self.model.spell_enabled:
            return control.apply_joint(prediction['joint_action_id'])
        from .spell_macro import resolve_action
        command = resolve_action(prediction['joint_action_id'], prediction['spell_card_id'],
                                 self.card_snapshot, self.card_macro)
        prediction['card_macro'] = command
        if command['kind'] == 'combat':
            return control.apply_joint(prediction['joint_action_id'])
        return control.apply_card_command(command['kind'])

    def observe_tcn_frame(self, payload, resources):
        key = (int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame))
        if key != self.tcn_window.key:
            self.tcn_window.append(key, self.builder.build(payload, resources))

    @torch.inference_mode()
    def predict(self, payload, resources=None):
        started = time.perf_counter()
        self.timing_calls += 1
        # 实战只保留总耗时；算子定位交给离线 TCN profiler，避免诊断扰动控制。
        timing = None
        key = (int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame))
        if self.tcn_window.key != key:
            raise ValueError("推理帧不对应 TCN 窗口末帧，禁止使用错位历史")
        with torch.autocast(device_type=self.tcn_device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.tcn_window.logits(self.model, self.device, timing, self.tcn_device,
                                           self.card_observation)
        spell_output = {}
        if self.model.spell_enabled:
            logits, card_logits = logits
            if not torch.isfinite(card_logits).all():
                raise ValueError('卡牌头输出非有限值，停止控制')
            values = card_logits[0].float().cpu()
            selected = int(values.argmax())
            spell_output = {'spell_class': selected, 'spell_logits': values.tolist(),
                'spell_card_id': None if selected == 0 else self.cards[selected - 1]['id'],
                'spell_card_name': 'NONE' if selected == 0 else self.cards[selected - 1]['name'],
                'spell_probabilities': values.softmax(-1).tolist(),
                'spell_catalog': self.cards}
        memory = None
        previous_id, previous_duration = self.tcn_window.previous_action, self.tcn_window.previous_duration
        if (logits.shape != (1, ACTION_COUNT) or not torch.isfinite(logits).all()
                or (memory is not None and not torch.isfinite(memory).all())):
            raise ValueError("BC 输出形状不是 [1,144] 或 logits/时序状态含 NaN/Inf，已停止控制")
        # 转 CPU 同步 CUDA，使耗时覆盖实际计算，而不是只测异步提交。
        joint_logits = logits[0].float().cpu()
        probabilities = joint_logits.softmax(-1)
        # 固定权重评估采用 argmax；softmax 仅用于显示，不采样、不加噪声或动作保护。
        action = int(joint_logits.argmax())
        entropy = -(probabilities * joint_logits.log_softmax(-1)).sum()
        if not torch.isfinite(probabilities).all() or not torch.isfinite(entropy):
            raise ValueError("BC 分类概率或熵含 NaN/Inf，已停止控制")
        direction, buttons = to_controller(action)
        _, combat = decode(action)
        timing_result = {}
        return {**spell_output, "joint_action_id": action, "action": action_name(action), "combat_mask": combat,
                "direction": direction, "buttons": tuple(int(x) for x in buttons),
                "joint_logits": joint_logits.tolist(), "joint_probabilities": probabilities.tolist(),
                "selected_probability": float(probabilities[action]), "entropy": float(entropy),
                "normalized_entropy": float(entropy) / math.log(ACTION_COUNT),
                "output_semantics": "categorical_logits", "action_selection": self.action_selection,
                "temporal_mode": self.temporal_mode,
                "context_frames_used": len(self.tcn_window),
                "tcn_computed_frames": self.tcn_window.processed_frames,
                "tcn_cache_rebuilt": self.tcn_window.rebuilt,
                "tcn_backend": self.tcn_window.graph.last_path if self.tcn_window.graph is not None else 'eager',
                "streaming_tcn": self.tcn_window.streaming, "inference_precision": self.precision,
                "context_first_frame": self.tcn_window.rows[0][0][2],
                "observation_frame": int(payload.battleFrame), "observation_round": int(payload.currentRound),
                "sample_serial": int(payload.sampleSerial),
                "previous_joint_action_id": previous_id,
                "previous_action_duration": previous_duration,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "timing": timing_result,
                "resource_inputs": self.builder.resource_summary}, memory
