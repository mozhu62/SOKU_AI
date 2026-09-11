from __future__ import annotations

from soku_ai.data.normalization import load_normalization
from soku_ai.data.split import load_split_manifest
from soku_ai.data.transition_builder import ProcessedReplayDataset
from soku_ai.data.resources_schema import enabled


def build_processed_dataset(config: dict, split_name: str) -> ProcessedReplayDataset:
    if enabled(config):
        from soku_ai.data.resources_setup import prepare_resources
        from soku_ai.data.resources_dataset import ResourceReplayDataset
        return ResourceReplayDataset(config, prepare_resources(config), split_name)
    data = config["data"]
    rl = config["rl"]
    manifest = load_split_manifest(data["split_manifest"])
    if split_name not in manifest["splits"]:
        raise KeyError(f"数据划分不存在: {split_name}")
    normalization = load_normalization(data["normalization_file"])
    return ProcessedReplayDataset(
        data["processed_dir"],
        manifest["splits"][split_name],
        normalization,
        history_len=data["history_len"],
        max_objects_per_side=data["max_objects_per_side"],
        action_shift=data["action_shift"],
        gamma=rl["gamma"],
        n_step=rl["n_step"],
        shard_cache_size=data["shard_cache_size"],
        include_evaluation_metadata=split_name != "train",
    )


def load_training_normalization(config):
    if enabled(config):
        from soku_ai.data.resources_setup import prepare_resources
        return prepare_resources(config)["normalization"]
    return load_normalization(config["data"]["normalization_file"])
