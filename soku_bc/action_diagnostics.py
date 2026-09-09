from __future__ import annotations

import numpy as np

from .action_space import ACTION_COUNT


DIAGNOSTIC_VERSION = "bc_action_history_diagnostics_v1"
DATA_COUNT_KEYS = ("valid_samples", "previous_action_samples", "previous_action_copy_correct", "action_change_samples")
BATCH_COUNT_KEYS = ("previous_action_samples", "previous_action_copy_correct", "action_change_samples",
                    "action_change_correct", "action_change_top5_correct", "history_joint_correct")
BATCH_RATE_KEYS = ("previous_action_baseline", "action_change_accuracy", "action_change_top5_accuracy",
                   "action_change_fraction", "history_joint_accuracy")


def ratio(numerator, denominator):
    # 没有可比较的历史或切换帧时返回未记录，不能用 0% 冒充实际预测失败。
    return numerator / denominator if denominator else None


def dataset_action_counts(expert, previous, valid):
    """只读取已对齐的专家历史；不在原始行之间搜索上一帧，也不调用模型。"""
    expert, previous, valid = np.asarray(expert), np.asarray(previous), np.asarray(valid, dtype=bool)
    if expert.shape != previous.shape or expert.shape != valid.shape:
        raise ValueError("动作历史诊断的标签、上一帧动作和有效位置形状不一致")
    # 历史边界由原有 previous_actions 标成 432；排除所有非真实 Joint Action token。
    eligible = valid & (previous >= 0) & (previous < ACTION_COUNT)
    same = expert == previous
    return {"valid_samples": int(valid.sum()), "previous_action_samples": int(eligible.sum()),
            "previous_action_copy_correct": int((eligible & same).sum()),
            "action_change_samples": int((eligible & ~same).sum())}


def merge_dataset_counts(rows):
    counts = {key: sum(row[key] for row in rows) for key in DATA_COUNT_KEYS}
    return {**counts,
            "previous_action_baseline": ratio(counts["previous_action_copy_correct"], counts["previous_action_samples"]),
            "action_change_fraction": ratio(counts["action_change_samples"], counts["valid_samples"])}


def batch_history_rates(counts, valid_samples):
    return {"previous_action_baseline": ratio(counts["previous_action_copy_correct"], counts["previous_action_samples"]),
            "action_change_accuracy": ratio(counts["action_change_correct"], counts["action_change_samples"]),
            "action_change_top5_accuracy": ratio(counts["action_change_top5_correct"], counts["action_change_samples"]),
            "action_change_fraction": ratio(counts["action_change_samples"], valid_samples),
            "history_joint_accuracy": ratio(counts["history_joint_correct"], counts["previous_action_samples"])}


def prefixed_history_metrics(metrics, prefix):
    """全量基线另行缓存；这里标记当前训练批/固定验证批的统计口径。"""
    if "previous_action_samples" not in metrics:
        return {}
    return {**{f"{prefix}_{key}": metrics[key] for key in (*BATCH_COUNT_KEYS, *BATCH_RATE_KEYS)
               if key != "previous_action_baseline"},
            f"{prefix}_batch_previous_action_baseline": metrics["previous_action_baseline"]}
