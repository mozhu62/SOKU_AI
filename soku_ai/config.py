from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import math

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML，并把所有数据路径解析为相对项目根目录的绝对路径。"""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件根节点必须是映射: {config_path}")

    result = deepcopy(config)
    for option in ("vectorized_batches", "packed_transfer"):
        if option in result.get("training", {}) and type(result["training"][option]) is not bool:
            raise ValueError(f"training.{option} 必须为布尔值")
    project_root = config_path.parent.parent
    result["_config_path"] = str(config_path)
    result["_project_root"] = str(project_root)

    reward_file = result.get("reward_file")
    if reward_file:
        reward_path = Path(str(reward_file))
        if not reward_path.is_absolute():
            reward_path = config_path.parent / reward_path
        reward_path = reward_path.resolve()
        with reward_path.open("r", encoding="utf-8") as file:
            reward_config = yaml.safe_load(file)
        if not isinstance(reward_config, dict):
            raise ValueError(f"Reward 配置文件根节点必须是映射: {reward_path}")
        result["reward"] = reward_config
        result["_reward_config_path"] = str(reward_path)

    path_keys = {
        "raw_dir",
        "processed_dir",
        "split_manifest",
        "normalization_file",
        "reports_dir",
        "source_dir",
        "source_split_file",
    }
    for key in path_keys:
        value = result.get("data", {}).get(key)
        if value is not None:
            candidate = Path(value)
            result["data"][key] = str(
                candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
            )

    for key in ("checkpoint_dir", "tensorboard_dir"):
        value = result.get("training", {}).get(key)
        if value is not None:
            candidate = Path(value)
            result["training"][key] = str(
                candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
            )
    if result.get("data", {}).get("format") == "resources_v4":
        from soku_ai.data.resources_schema import MODEL_TYPE, OBSERVATION_VERSION
        data, rl = result["data"], result["rl"]
        if result["model"]["model_type"] != MODEL_TYPE or result["model"]["num_actions"] != 144:
            raise ValueError("resources_v4 必须搭配资源版 DQfD 网络及 144 类原生 DQfD 动作")
        if data["preprocessing_version"] != OBSERVATION_VERSION or data["max_objects_per_side"] != 3:
            raise ValueError("resources_v4 输入版本不匹配，且每侧必须为最多三个对象")
        if type(data["action_shift"]) is not int or data["action_shift"] not in (0, 1):
            raise ValueError("action_shift 只能为 0/1，必须与 NPZ 元数据一致")
        for key in ("horizontal_positive_is_right", "vertical_positive_is_down", "direction_positive_faces_right"):
            if type(data[key]) is not bool:
                raise ValueError(f"{key} 必须为布尔值")
        if not 0.1 <= float(data.get("cache_gb", 4)) <= 1024:
            raise ValueError("cache_gb 必须为 0.1～1024；这是每个数据集实例的缓存预算")
        if (type(rl["n_step"]) is not int or not 1 <= rl["n_step"] <= 1024
                or not 0 < float(rl["gamma"]) <= 1 or not 1 <= int(data["history_len"]) <= 512):
            raise ValueError("n_step、gamma 或 history_len 无效")
        for key in ("damage_scale", "round_win_reward", "round_loss_reward"):
            if not math.isfinite(float(result["reward"][key])):
                raise ValueError(f"reward.{key} 必须有限")
        if float(result["reward"]["damage_scale"]) < 0:
            raise ValueError("damage_scale 不能为负")
    return result


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """保存可复现实验配置，内部派生字段不会写入文件。"""
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(serializable, file, allow_unicode=True, sort_keys=False)
