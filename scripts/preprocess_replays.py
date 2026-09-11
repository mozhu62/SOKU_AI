from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from soku_ai.config import load_config
from soku_ai.data.normalization import fit_normalization, save_normalization
from soku_ai.data.preprocess import PreprocessResult, preprocess_replay, write_dataset_report
from soku_ai.data.replay_reader import discover_replay_pairs, filter_character_replays
from soku_ai.rewards import reward_config_hash
from soku_ai.data.split import (
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)


def _result_sidecar(shard: Path) -> Path:
    return Path(f"{shard}.report.json")


def _sidecar_matches(payload: dict, config: dict) -> bool:
    return (
        payload.get("reward_config_hash") == reward_config_hash(config["reward"])
        and payload.get("preprocessing_version")
        == str(config["data"]["preprocessing_version"])
    )


def _add_training_weight_stats(payload: dict, shard: Path) -> bool:
    if "training_weight_count" in payload:
        return False
    with np.load(shard, allow_pickle=False) as data:
        valid = data["transition_valid"]
        weights = data["training_weights"][valid].astype(np.float64, copy=False)
    payload.update(
        {
            "training_weight_sum": float(weights.sum()),
            "training_weight_sum_squares": float(np.square(weights).sum()),
            "training_weight_count": int(weights.size),
            "training_weight_min": float(weights.min()) if weights.size else 1.0,
            "training_weight_max": float(weights.max()) if weights.size else 1.0,
        }
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="把 Replay CSV 转换为逐场 NPZ 分片")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(arguments.config)
    data = config["data"]
    pairs = filter_character_replays(
        discover_replay_pairs(data["raw_dir"]), int(data["suika_character_id"])
    )
    manifest_path = Path(data["split_manifest"])
    if not manifest_path.exists():
        manifest = create_split_manifest(
            pairs,
            seed=int(config["training"]["seed"]),
            train_ratio=float(data["split_train_ratio"]),
            validation_ratio=float(data["split_validation_ratio"]),
            test_ratio=float(data["split_test_ratio"]),
        )
        save_split_manifest(manifest, manifest_path)
    manifest = load_split_manifest(manifest_path)
    allowed_replays = {
        replay_id
        for replay_ids in manifest["splits"].values()
        for replay_id in replay_ids
    }
    pairs = [pair for pair in pairs if pair.replay_id in allowed_replays]
    if arguments.limit is not None:
        pairs = pairs[: arguments.limit]

    processed_root = Path(data["processed_dir"])
    results: list[PreprocessResult] = []
    for index, pair in enumerate(pairs, start=1):
        shard = processed_root / f"{pair.replay_id}.npz"
        sidecar = _result_sidecar(shard)
        can_skip = shard.exists() and sidecar.exists() and not arguments.overwrite
        payload = (
            json.loads(sidecar.read_text(encoding="utf-8"))
            if can_skip
            else None
        )
        if payload is not None and _sidecar_matches(payload, config):
            sidecar_changed = _add_training_weight_stats(payload, shard)
            payload["action_histogram"] = tuple(payload["action_histogram"])
            result = PreprocessResult(**payload)
            if sidecar_changed:
                sidecar.write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(f"[{index}/{len(pairs)}] Skip {pair.replay_id}")
        else:
            if payload is not None:
                logging.info("Reward 或预处理版本已变化，重新生成 %s", pair.replay_id)
            result = preprocess_replay(
                pair,
                shard,
                data,
                config["reward"],
            )
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{index}/{len(pairs)}] {pair.replay_id}: "
                f"frames={result.frame_count} transitions={result.transition_count}"
            )
        results.append(result)

    write_dataset_report(results, data["reports_dir"])
    if arguments.limit is not None:
        print("使用了 --limit，仅生成 smoke 分片和局部报告，不拟合正式 normalization。")
        return

    result_by_id = {result.replay_id: result for result in results}
    train_ids = manifest["splits"]["train"]
    missing = [replay_id for replay_id in train_ids if replay_id not in result_by_id]
    if missing:
        raise RuntimeError(f"训练集存在未完成的分片: {missing[:5]}")
    train_paths = [result_by_id[replay_id].shard_path for replay_id in train_ids]
    normalization = fit_normalization(train_paths, train_ids)
    save_normalization(normalization, data["normalization_file"])
    total_objects = sum(result.object_count for result in results)
    total_truncated = sum(result.truncated_object_count for result in results)
    truncation_rate = total_truncated / total_objects if total_objects else 0.0
    if truncation_rate > 0.01:
        logging.warning("对象截断率 %.4f%%，建议评估增大 max_objects_per_side", truncation_rate * 100)
    print(f"Normalization: {data['normalization_file']}")


if __name__ == "__main__":
    main()
