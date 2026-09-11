from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.data.normalization import load_normalization
from soku_ai.data.split import load_split_manifest
from soku_ai.data.transition_builder import ProcessedReplayDataset
from soku_ai.models.factory import build_model
from soku_ai.rl.double_dqn import double_dqn_target
from soku_ai.rl.dqfd_loss import compute_dqfd_loss
from soku_ai.rl.target_update import hard_update
from soku_ai.training.checkpoint import build_checkpoint, save_checkpoint
from soku_ai.training.metrics import q_metrics
from soku_ai.training.utils import move_to_device, resolve_device, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="固定单场 Replay 和 batch，检查模型能否过拟合")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay-id")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config["training"]["device"] = arguments.device
    seed = int(config["training"]["seed"])
    set_random_seed(seed)
    device = resolve_device(arguments.device)
    manifest = load_split_manifest(config["data"]["split_manifest"])
    replay_id = arguments.replay_id or manifest["splits"]["train"][0]
    normalization = load_normalization(config["data"]["normalization_file"])
    dataset = ProcessedReplayDataset(
        config["data"]["processed_dir"],
        [replay_id],
        normalization,
        history_len=config["data"]["history_len"],
        max_objects_per_side=config["data"]["max_objects_per_side"],
        action_shift=config["data"]["action_shift"],
        gamma=config["rl"]["gamma"],
        n_step=config["rl"]["n_step"],
        shard_cache_size=1,
    )
    batch_size = min(int(config["rl"]["batch_size"]), len(dataset))
    if batch_size == 0:
        raise RuntimeError("指定 Replay 没有有效 Transition")
    rng = np.random.default_rng(seed)
    fixed_indices = rng.choice(len(dataset), size=batch_size, replace=False)
    batch = default_collate([dataset[int(index)] for index in fixed_indices])
    batch = move_to_device(batch, device)

    online = build_model(config).to(device)
    target = build_model(config).to(device)
    hard_update(target, online)
    target.eval()
    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=float(config["rl"]["learning_rate"]),
        weight_decay=float(config["rl"]["weight_decay"]),
    )
    writer = SummaryWriter(
        log_dir=Path(config["training"]["tensorboard_dir"]) / "overfit_single_replay"
    )
    ones = torch.ones(batch_size, device=device)
    gamma = torch.full_like(batch["reward"], float(config["rl"]["gamma"]))
    for step in range(1, arguments.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        q_values = online(batch["observation"])
        td1_target = double_dqn_target(
            online,
            target,
            batch["next_observation"],
            batch["reward"],
            batch["done"],
            gamma,
        )
        n_target = double_dqn_target(
            online,
            target,
            batch["n_step_observation"],
            batch["n_step_reward"],
            batch["n_step_done"],
            batch["n_step_discount"],
        )
        loss = compute_dqfd_loss(
            q_values=q_values,
            actions=batch["action"],
            td1_targets=td1_target,
            n_step_targets=n_target,
            is_demo=batch["is_demo"],
            importance_weights=ones,
            online_network=online,
            config=config["rl"],
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            online.parameters(), float(config["training"]["grad_clip"])
        )
        optimizer.step()
        if step % 25 == 0:
            metrics = q_metrics(q_values.detach(), batch["action"])
            writer.add_scalar("overfit/loss_total", loss.total.item(), step)
            writer.add_scalar("overfit/loss_td1", loss.td1.item(), step)
            writer.add_scalar("overfit/loss_margin", loss.margin.item(), step)
            writer.add_scalar("overfit/expert_top1", metrics["expert_top1"], step)
        if step % 250 == 0:
            hard_update(target, online)

    checkpoint = build_checkpoint(
        online_network=online,
        target_network=target,
        optimizer=optimizer,
        config=config,
        normalization=normalization,
        step=arguments.steps,
        epoch=0,
        seed=seed,
        best_metric=q_metrics(q_values.detach(), batch["action"])["expert_top1"],
    )
    destination = Path(config["training"]["checkpoint_dir"]) / "overfit_single.pt"
    save_checkpoint(checkpoint, destination)
    writer.close()
    print(f"Replay: {replay_id}")
    print(f"Checkpoint: {destination}")


if __name__ == "__main__":
    main()
