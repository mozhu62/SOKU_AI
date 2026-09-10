from __future__ import annotations

import hashlib
import json
import re

from .config import ROOT, active_modules


EXPERIMENT_ROOT = ROOT / "outputs" / "temporal_experiments"
COMPARISON_VERSION = "bc_wide_tcn32_comparison_v2"


def experiment_path(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("实验名使用 1～64 个英文字母、数字、下划线或连字符；不能填写路径")
    target = (EXPERIMENT_ROOT / name).resolve()
    if target.parent != EXPERIMENT_ROOT.resolve():
        raise ValueError("实验目录越界")
    return target


def comparison_conditions(config, split_hash, normalization):
    training = config["training"]
    keys = ("batch_size", "sequence_length", "burn_in", "replays_per_batch", "learning_rate",
            "weight_decay", "label_smoothing", "max_grad_norm", "validation_batches", "amp",
            "device", "cpu_threads", "prefetch_batches")
    conditions = {"split_hash": split_hash, "seed": config["seed"],
                  "training": {key: training[key] for key in keys},
                  "model": dict(config["model"]),
                  "frozen_active_modules": sorted(set(training["frozen_modules"]) & set(active_modules(config["model"]))),
                  "normalization_hash": hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest()}
    conditions["cache_gb"] = config["data"]["cache_gb"]
    return conditions, hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()


def comparison_catalog(current):
    # 网页只读小型实验摘要，不扫描 checkpoint，也不在刷新时遍历完整 metrics 日志。
    rows = []
    if EXPERIMENT_ROOT.is_dir():
        files = []
        for path in EXPERIMENT_ROOT.glob("*/comparison.json"):
            try:
                stat = path.stat()
                if path.resolve().is_relative_to(EXPERIMENT_ROOT.resolve()) and stat.st_size <= 512 * 1024:
                    files.append((stat.st_mtime, path))
            except OSError:
                continue
        for _, path in sorted(files, key=lambda item: item[0], reverse=True)[:50]:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(row, dict) and row.get("version") == COMPARISON_VERSION
                        and isinstance(row.get("output"), str) and isinstance(row.get("name"), str)
                        and row.get("temporal_mode") == "tcn" and isinstance(row.get("validation"), list)):
                    rows.append(row)
            except (OSError, ValueError):
                continue
    if current:
        rows = [row for row in rows if row["output"] != current["output"]]
        rows.insert(0, current)
    return rows
