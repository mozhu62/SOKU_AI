from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from ..checkpoint import load
from ..models import CQLNetwork
from ..action_space import to_controller, decode, action_name
from .observation import ObservationBuilder


class LiveAgent:
    def __init__(self, path: Path, config):
        if not path.is_file():
            raise FileNotFoundError(f"找不到实战模型：{path}；请从训练服务器复制完整的 .pt 文件")
        before = path.stat()
        torch.set_num_threads(config["cpu_threads"])
        requested = config["device"]
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
        package = load(path)
        self.model = CQLNetwork(package["spec"]["model"], package["network_version"])
        if self.model.spec != package["spec"]:
            raise ValueError("模型输入/Joint Action schema 不兼容；只接受新版 432-way CQL 模型")
        self.model.load_state_dict(package["online"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        # 回读方向使用模型训练时的轴约定；Joint Action 的方向已经是屏幕绝对九宫格。
        self.vertical_positive_is_down = bool(package["config"]["data"]["vertical_positive_is_down"])
        self.builder = ObservationBuilder(package["normalization"], config["environment"]["player_side"],
                                          self.vertical_positive_is_down)
        self.step = int(package["step"])
        self.path = str(path)
        with path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("加载过程中模型文件发生变化，请先复制为固定文件再开始评估")
        self.memory = None

    def reset(self):
        self.memory = None
        self.builder.reset()

    @torch.inference_mode()
    def predict(self, payload, resources=None):
        started = time.perf_counter()
        arrays = self.builder.build(payload, resources)
        obs = {name: torch.from_numpy(value).unsqueeze(0).to(self.device) for name, value in arrays.items()}
        q, memory = self.model.step_q(obs, self.memory)
        if not torch.isfinite(q).all():
            raise ValueError("模型 Q 值为 NaN/Inf，已停止控制")
        # 转 CPU 同步 CUDA，使耗时覆盖实际计算，而不是只测异步提交。
        joint_q = q[0].cpu()
        action = int(joint_q.argmax())
        direction, buttons = to_controller(action)
        _, combat, card = decode(action)
        return {"joint_action_id": action, "action": action_name(action), "combat_mask": combat, "card_command": card,
                "direction": direction, "buttons": tuple(int(x) for x in buttons), "joint_q": joint_q.tolist(),
                "previous_joint_action_id": int(arrays["previous_joint_action_id"]),
                "previous_action_duration": float(arrays["previous_action_duration"][0]),
                "inference_ms": (time.perf_counter() - started) * 1000,
                "resource_inputs": self.builder.resource_summary}, memory
