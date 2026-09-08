from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NETWORK_VERSION = "soku_cql_recurrent_joint432_v3"
MODEL_DEFAULTS = {
    "action_vocab_size": 2048, "block_vocab_size": 512, "weather_vocab_size": 32,
    "action_embedding_dim": 32, "block_embedding_dim": 8, "weather_embedding_dim": 8,
    "current_hidden_dim": 256, "object_hidden_dim": 64, "object_set_dim": 128,
    "fusion_dim": 256, "gru_hidden_dim": 128, "q_hidden_dim": 128,
    "object_embedding_mode": "separate",
    "skill_embedding_dim": 8, "skill_level_embedding_dim": 4, "previous_action_embedding_dim": 32,
    "card_vocab_size": 512, "card_embedding_dim": 16,
}
DEFAULTS = {
    "seed": 42,
    "data": {"directory": "data/replay_shards_resources_v4",
             "split_file": "data/train_val_split_resources_v4.json",
             "train_fraction": 0.8, "cache_gb": 4.0, "vertical_positive_is_down": True},
    "model": MODEL_DEFAULTS,
    "training": {"device": "auto", "total_steps": 100000, "batch_size": 32,
                 "sequence_length": 32, "burn_in": 16, "replays_per_batch": 4,
                 "learning_rate": 0.0001, "gamma": 0.99, "n_step": 5, "cql_alpha": 1.0,
                 "cql_temperature": 1.0, "expert_imitation_weight": 0.01,
                 "target_tau": 0.005, "max_grad_norm": 10.0,
                 "weight_decay": 0.0001, "log_interval": 20, "save_interval": 1000,
                 "validation_interval": 1000, "validation_batches": 20,
                 "cpu_threads": 4, "amp": True, "prefetch_batches": 2, "frozen_modules": []},
    "output": {"directory": "outputs/cql_suika_joint432_v3"},
    "web": {"host": "127.0.0.1", "port": 8776, "port_attempts": 30},
}
MODULES = ("current_encoder", "object_encoder", "fusion", "gru", "memory_fusion", "joint_head")
# 运行中可调项只在暂停后应用；改变输入、结构和数据划分必须另开训练。
EDITABLE = {
    "learning_rate": (1e-8, 0.01, "float", "学习率"),
    "gamma": (0.0, 1.0, "float", "折扣 γ（每游戏帧）"),
    "n_step": (1, 120, "int", "TD 回报步数 N（1 恢复单步）"),
    "cql_alpha": (0.0, 100.0, "float", "CQL 保守项系数"),
    "cql_temperature": (0.01, 10.0, "float", "CQL logsumexp 温度"),
    "expert_imitation_weight": (0.0, 1.0, "float", "高手动作微量模仿权重（0 关闭）"),
    "target_tau": (0.000001, 1.0, "float", "目标网络软更新系数"),
    "max_grad_norm": (0.01, 1000.0, "float", "梯度裁剪上限"),
    "weight_decay": (0.0, 0.1, "float", "权重衰减"),
    "batch_size": (1, 1024, "int", "每批序列数"),
    "total_steps": (1, 100000000, "int", "总更新步数"),
    "log_interval": (1, 10000, "int", "指标记录间隔"),
    "save_interval": (1, 1000000, "int", "模型保存间隔"),
    "validation_interval": (1, 1000000, "int", "验证间隔"),
    "validation_batches": (1, 1000, "int", "固定验证批次数"),
}


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def validate(config: dict) -> dict:
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    cfg = config["training"]
    for key, (lo, hi, kind, _) in EDITABLE.items():
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lo <= value <= hi
                or (kind == "int" and not isinstance(value, int))):
            raise ValueError(f"training.{key} 必须为 {lo} 到 {hi} 之间的 {kind}")
    for key, lo in (("sequence_length", 1), ("burn_in", 0), ("replays_per_batch", 1), ("cpu_threads", 1)):
        if type(cfg[key]) is not int or not lo <= cfg[key] <= 4096:
            raise ValueError(f"training.{key} 超出范围")
    if type(cfg["amp"]) is not bool or type(cfg["prefetch_batches"]) is not int or not 1 <= cfg["prefetch_batches"] <= 8:
        raise ValueError("amp 必须为布尔值，prefetch_batches 必须在 1 到 8 之间")
    frozen = cfg["frozen_modules"]
    if not isinstance(frozen, list) or any(x not in MODULES for x in frozen) or set(frozen) == set(MODULES):
        raise ValueError("冻结模块无效或全部模块均被冻结")
    if config["data"]["train_fraction"] != 0.8:
        raise ValueError("本版本固定按整份 REP 进行 8:2 划分")
    cache_gb = config["data"]["cache_gb"]
    if type(cache_gb) not in (int, float) or not math.isfinite(cache_gb) or not 0.1 <= cache_gb <= 1024:
        raise ValueError("data.cache_gb 必须在 0.1 到 1024 之间")
    if type(config["data"]["vertical_positive_is_down"]) is not bool:
        raise ValueError("vertical_positive_is_down 必须是布尔值")
    model = config["model"]
    for key, default in MODEL_DEFAULTS.items():
        if isinstance(default, int) and (type(model[key]) is not int or not 1 <= model[key] <= 65536):
            raise ValueError(f"model.{key} 必须为正整数")
    if model["object_set_dim"] != model["object_hidden_dim"] * 2:
        raise ValueError("object_set_dim 必须等于 object_hidden_dim 的两倍")
    if model["object_embedding_mode"] not in ("shared", "separate"):
        raise ValueError("object_embedding_mode 无效")
    for key in ("current_hidden_dim", "object_hidden_dim", "object_set_dim", "fusion_dim", "gru_hidden_dim", "q_hidden_dim"):
        if model[key] != MODEL_DEFAULTS[key]:
            raise ValueError(f"本版保持原主干宽度，model.{key} 必须为 {MODEL_DEFAULTS[key]}")
    if config["web"]["host"] not in ("0.0.0.0", "127.0.0.1"):
        raise ValueError("web.host 只允许 0.0.0.0 或 127.0.0.1")
    if (type(config["web"]["port"]) is not int or type(config["web"]["port_attempts"]) is not int
            or not 1024 <= config["web"]["port"] <= 65535 or not 1 <= config["web"]["port_attempts"] <= 100):
        raise ValueError("本机端口或尝试数量无效")
    return config


def load_config(path: str | Path) -> dict:
    supplied = yaml.safe_load(resolve(path).read_text(encoding="utf-8-sig")) or {}
    result = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key not in result:
            raise ValueError(f"未知配置项：{key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f"{key} 包含未知配置项")
            result[key].update(value)
        else:
            result[key] = value
    return validate(result)
