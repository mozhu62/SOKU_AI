from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from soku_ai.rl.double_dqn import double_dqn_target
from soku_ai.rl.dqfd_loss import compute_dqfd_loss

from .metrics import MetricAccumulator, object_count_bucket, q_metrics
from .utils import move_to_device


@torch.no_grad()
def evaluate_offline(
    model: torch.nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    seed: int,
    target_model: torch.nn.Module | None = None,
    rl_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    target_model = model if target_model is None else target_model
    target_model.eval()
    total_samples = min(len(dataset), int(batch_size) * int(max_batches))
    rng = np.random.default_rng(seed)
    indices = (
        np.arange(len(dataset), dtype=np.int64)
        if total_samples == len(dataset)
        else np.sort(rng.choice(len(dataset), size=total_samples, replace=False))
    )
    overall = MetricAccumulator()
    by_opponent: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_weather: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_objects: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        bulk = getattr(dataset, "config", {}).get("training", {}).get("vectorized_batches", True)
        batch = (dataset.get_batch(batch_indices) if bulk and hasattr(dataset, "get_batch") else
                 default_collate([dataset[int(index)] for index in batch_indices]))
        batch = move_to_device(batch, device)
        q_values = model(batch["observation"])
        actions = batch["action"]
        metrics = q_metrics(q_values, actions)
        if rl_config is not None:
            one_step_discounts = torch.full_like(
                batch["reward"], float(rl_config["gamma"])
            )
            td1_target = double_dqn_target(
                model,
                target_model,
                batch["next_observation"],
                batch["reward"],
                batch["done"],
                one_step_discounts,
            )
            n_target = double_dqn_target(
                model,
                target_model,
                batch["n_step_observation"],
                batch["n_step_reward"],
                batch["n_step_done"],
                batch["n_step_discount"],
            )
            loss = compute_dqfd_loss(
                q_values=q_values,
                actions=actions,
                td1_targets=td1_target,
                n_step_targets=n_target,
                is_demo=batch["is_demo"],
                importance_weights=torch.ones_like(batch["reward"]),
                online_network=model,
                config=rl_config,
                self_actions=batch["observation"]["state_categorical"][:, 0],
                sample_weights=batch["training_weight"],
            )
            metrics.update(
                {
                    "loss_total": float(loss.total.item()),
                    "loss_td1": float(loss.td1.item()),
                    "loss_nstep": float(loss.n_step.item()),
                    "loss_margin": float(loss.margin.item()),
                    "td_error_mean": float(loss.td_errors.abs().mean().item()),
                }
            )
        overall.add(metrics, len(batch_indices))

        correct = q_values.argmax(dim=1).eq(actions).float().cpu().numpy()
        top5 = q_values.topk(5, dim=1).indices.eq(actions.unsqueeze(1)).any(dim=1).float().cpu().numpy()
        opponents = batch["opponent_character_id"].cpu().numpy() if "opponent_character_id" in batch else None
        weather_key = "active_weather" if "active_weather" in batch else "displayed_weather"
        weathers = batch[weather_key].cpu().numpy()
        object_counts = batch["object_count"].cpu().numpy()
        for index in range(len(batch_indices)):
            item_metrics = {
                "expert_top1": float(correct[index]),
                "expert_top5": float(top5[index]),
            }
            if opponents is not None:
                by_opponent[str(int(opponents[index]))].add(item_metrics)
            by_weather[str(int(weathers[index]))].add(item_metrics)
            by_objects[object_count_bucket(int(object_counts[index]))].add(item_metrics)

    return {
        **overall.result(),
        "by_opponent_character": {key: value.result() for key, value in by_opponent.items()},
        "by_weather": {key: value.result() for key, value in by_weather.items()},
        "weather_field": "active_weather" if getattr(dataset, "evaluation_metadata", False) else "displayed_weather",
        "by_object_count": {key: value.result() for key, value in by_objects.items()},
        "samples": int(len(indices)),
    }
