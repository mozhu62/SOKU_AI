from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter

from soku_ai.data.resources_schema import enabled as resource_data_enabled
from soku_ai.data.split import load_split_manifest
from soku_ai.models.factory import build_model
from soku_ai.replay_buffer.prioritized_buffer import PrioritizedIndexBuffer, linear_beta
from soku_ai.rl.double_dqn import concatenate_observations, double_dqn_target
from soku_ai.rl.dqfd_loss import compute_dqfd_loss
from soku_ai.rl.target_update import hard_update, update_target_if_needed

from .checkpoint import (
    build_checkpoint,
    build_model_snapshot,
    load_checkpoint,
    save_checkpoint,
)
from .data_setup import build_processed_dataset, load_training_normalization
from .evaluator import evaluate_offline
from .metrics import action_style_metrics, no_op_metrics, q_metrics
from .utils import move_to_device, resolve_device, set_random_seed


LOGGER = logging.getLogger(__name__)


class DemoTrainer:
    def __init__(
        self,
        config: dict,
        *,
        resume_path: str | None = None,
        init_from_path: str | None = None,
    ) -> None:
        if resume_path and init_from_path:
            raise ValueError("--resume 与 --init-from 不能同时使用")
        self.config = config
        training = config["training"]
        self.seed = int(training["seed"])
        set_random_seed(self.seed)
        self.device = resolve_device(training["device"])
        if "cpu_threads" in training:
            torch.set_num_threads(int(training["cpu_threads"]))
        if (resource_data_enabled(config) and not resume_path and not init_from_path
                and (Path(training["checkpoint_dir"]) / "last.pt").exists()):
            raise ValueError("输出目录已有 last.pt；续训请使用 --resume，从零训练请修改 checkpoint_dir")
        self.train_dataset = build_processed_dataset(config, "train")
        manifest = load_split_manifest(config["data"]["split_manifest"])
        validation_ids = manifest["splits"].get("validation", [])
        self.validation_dataset = (
            build_processed_dataset(config, "validation") if validation_ids else None
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError("训练集没有有效 Transition")
        if self.validation_dataset is None:
            LOGGER.info("验证集为空，已关闭离线验证；全部 Replay 仅用于训练")
        if bool(training.get("preload_shards", False)):
            LOGGER.info("正在把 %d 个训练分片预加载到内存", len(self.train_dataset.shard_paths))
            self.train_dataset.preload_shards()
            LOGGER.info("训练分片预加载完成")

        self.online = build_model(config).to(self.device)
        self.target = build_model(config).to(self.device)
        hard_update(self.target, self.online)
        self.target.eval()
        rl = config["rl"]
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=float(rl["learning_rate"]),
            weight_decay=float(rl["weight_decay"]),
        )
        self.amp_enabled = bool(training["amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.priorities = PrioritizedIndexBuffer(
            len(self.train_dataset),
            alpha=float(rl["prioritized_replay_alpha"]),
            priority_epsilon=float(rl["priority_epsilon"]),
            demo_priority_bonus=float(rl["demo_priority_bonus"]),
        )
        self.rng = np.random.default_rng(self.seed)
        self.step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.normalization = load_training_normalization(config)
        self.checkpoint_dir = Path(training["checkpoint_dir"])
        self.provenance: dict[str, Any] = {}
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True

        if init_from_path:
            checkpoint = load_checkpoint(
                init_from_path,
                online_network=self.online,
                map_location=self.device,
            )
            self._validate_source_checkpoint(checkpoint)
            hard_update(self.target, self.online)
            self.provenance = {
                "mode": "model_only_finetune",
                "parent_checkpoint": str(Path(init_from_path).resolve()),
                "parent_training_step": int(checkpoint.get("training_step", 0)),
            }
            LOGGER.info(
                "已从 Step %d 模型权重初始化微调；优化器、Target 与 PER 均已重置",
                self.provenance["parent_training_step"],
            )
        elif resume_path:
            checkpoint = load_checkpoint(
                resume_path,
                online_network=self.online,
                target_network=self.target,
                optimizer=self.optimizer,
                scaler=self.scaler,
                map_location=self.device,
                restore_rng=True,
            )
            self.step = int(checkpoint["training_step"])
            self.epoch = int(checkpoint.get("epoch", 0))
            self.best_metric = float(checkpoint.get("best_metric", float("-inf")))
            manifest = load_split_manifest(config["data"]["split_manifest"])
            if checkpoint["dataset_split_manifest_hash"] != manifest["sha256"]:
                raise ValueError("Resume checkpoint 的数据划分哈希与当前配置不一致")
            if checkpoint["preprocessing_version"] != config["data"]["preprocessing_version"]:
                raise ValueError("Resume checkpoint 的预处理版本与当前配置不一致")
            if checkpoint["normalization"] != self.normalization.to_dict():
                raise ValueError("Resume checkpoint 的 normalization 与当前训练集不一致")
            if "replay_buffer_state" in checkpoint:
                self.priorities.load_state_dict(checkpoint["replay_buffer_state"])
                sampler_rng_state = checkpoint["replay_buffer_state"].get("sampler_rng_state")
                if sampler_rng_state is not None:
                    self.rng.bit_generator.state = sampler_rng_state
            self.provenance = dict(checkpoint.get("provenance", {}))

        phase_name = str(
            training.get("phase_name")
            or ("noop_finetune" if init_from_path or self.provenance else "dqfd")
        )
        run_name = training.get("run_name") or (
            f"{phase_name}_{time.strftime('%Y%m%d-%H%M%S')}_from_{self.step:09d}"
        )
        run_dir = Path(training["tensorboard_dir"]) / str(run_name)
        self.writer = SummaryWriter(log_dir=run_dir, flush_secs=10)
        LOGGER.info("TensorBoard run: %s", run_dir)
        self.writer.add_scalar("run/dataset_transitions", len(self.train_dataset), self.step)
        self.writer.add_scalar("run/start_step", self.step, self.step)
        self.writer.add_scalar(
            "run/no_op_margin_weight",
            float(self.config["rl"].get("no_op_margin_weight", 1.0)),
            self.step,
        )
        if self.provenance:
            self.writer.add_scalar(
                "run/parent_training_step",
                int(self.provenance.get("parent_training_step", 0)),
                self.step,
            )
        self.writer.flush()

    def _validate_source_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        manifest = load_split_manifest(self.config["data"]["split_manifest"])
        if checkpoint["dataset_split_manifest_hash"] != manifest["sha256"]:
            raise ValueError("初始化模型的数据划分哈希与当前配置不一致")
        if checkpoint["preprocessing_version"] != self.config["data"]["preprocessing_version"]:
            raise ValueError("初始化模型的预处理版本与当前配置不一致")
        if checkpoint["normalization"] != self.normalization.to_dict():
            raise ValueError("初始化模型的 normalization 与当前训练集不一致")

    def _save_resume_checkpoint(self, name: str) -> None:
        replay_buffer_state = self.priorities.state_dict()
        replay_buffer_state["sampler_rng_state"] = self.rng.bit_generator.state
        checkpoint = build_checkpoint(
            online_network=self.online,
            target_network=self.target,
            optimizer=self.optimizer,
            config=self.config,
            normalization=self.normalization,
            step=self.step,
            epoch=self.epoch,
            seed=self.seed,
            best_metric=self.best_metric,
            scaler=self.scaler,
            replay_buffer_state=replay_buffer_state,
            provenance=self.provenance,
        )
        save_checkpoint(checkpoint, self.checkpoint_dir / name)

    def _save_model_snapshot(self) -> None:
        snapshot = build_model_snapshot(
            online_network=self.online,
            config=self.config,
            normalization=self.normalization,
            step=self.step,
            epoch=self.epoch,
            seed=self.seed,
            provenance=self.provenance,
        )
        name = f"step_{self.step:09d}.pt"
        save_checkpoint(snapshot, self.checkpoint_dir / "snapshots" / name)

    def _log_train(
        self,
        loss: Any,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        self_actions: torch.Tensor,
        training_weights: torch.Tensor,
        round_outcomes: torch.Tensor,
        beta: float,
        steps_per_second: float,
    ) -> None:
        detached_q = q_values.detach()
        metrics = {
            **q_metrics(detached_q, actions),
            **no_op_metrics(
                detached_q,
                actions,
                self_actions,
                int(self.config["rl"].get("no_op_action_id", 0)),
            ),
            **action_style_metrics(detached_q, actions),
        }
        values = {
            "loss_total": float(loss.total.item()),
            "loss_td1": float(loss.td1.item()),
            "loss_nstep": float(loss.n_step.item()),
            "loss_margin": float(loss.margin.item()),
            "loss_idle_no_op": float(loss.idle_no_op.item()),
            "td_error_mean": float(loss.td_errors.abs().mean().item()),
            "training_weight_mean": float(training_weights.float().mean().item()),
            "sample_win_rate": float(round_outcomes.gt(0).float().mean().item()),
            "sample_loss_rate": float(round_outcomes.lt(0).float().mean().item()),
            **metrics,
        }
        for key, value in values.items():
            self.writer.add_scalar(f"train/{key}", value, self.step)
        self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.step)
        self.writer.add_scalar("PER/beta", beta, self.step)
        self.writer.add_scalar("Replay_Buffer/size", len(self.train_dataset), self.step)
        self.writer.add_scalar("performance/steps_per_second", steps_per_second, self.step)
        self.writer.flush()
        LOGGER.info(
            "step=%d/%d speed=%.3f step/s loss=%.6f top1=%.4f "
            "pred0=%.4f idle_pred0=%.4f forward=%.4f backward=%.4f "
            "attack=%.4f q0_adv=%.4f q_abs_max=%.3f",
            self.step,
            int(self.config["training"]["num_steps"]),
            steps_per_second,
            values["loss_total"],
            values["expert_top1"],
            values["predicted_no_op_rate"],
            values.get("idle_predicted_no_op_rate", 0.0),
            values["predicted_forward_rate"],
            values["predicted_backward_rate"],
            values["predicted_attack_abc_rate"],
            values["q_no_op_advantage"],
            values["q_abs_max"],
        )
        training = self.config["training"]
        if (
            values["q_abs_mean"] > float(training["q_warning_mean_abs"])
            or values["q_abs_max"] > float(training["q_warning_max_abs"])
        ):
            LOGGER.warning(
                "Q 值超过告警阈值: mean_abs=%.3f max_abs=%.3f",
                values["q_abs_mean"],
                values["q_abs_max"],
            )

    def _evaluate(self) -> dict[str, Any]:
        if self.validation_dataset is None:
            raise RuntimeError("当前配置没有验证集，不能执行离线验证")
        training = self.config["training"]
        metrics = evaluate_offline(
            self.online,
            self.validation_dataset,
            device=self.device,
            batch_size=int(self.config["rl"]["batch_size"]),
            max_batches=int(training["validation_batches"]),
            seed=self.seed,
            target_model=self.target,
            rl_config=self.config["rl"],
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"validation/{key}", value, self.step)
        self.online.train()
        return metrics

    def _prepare_batch(self, sample: Any) -> tuple[dict[str, Any], torch.Tensor, np.ndarray]:
        # 同一分片的样本连续读取，降低随机访问造成的 CPU cache miss。
        shard_ids = self.train_dataset.index_shards[sample.indices]
        order = np.argsort(shard_ids, kind="stable")
        ordered_indices = sample.indices[order]
        batch = default_collate(
            [self.train_dataset[int(index)] for index in ordered_indices]
        )
        batch = move_to_device(batch, self.device)
        weights = torch.from_numpy(sample.weights[order]).to(self.device)
        return batch, weights, ordered_indices

    def train(self, num_steps: int | None = None) -> None:
        rl = self.config["rl"]
        training = self.config["training"]
        total_steps = int(training["num_steps"] if num_steps is None else num_steps)
        batch_size = int(rl["batch_size"])
        log_interval = max(1, int(training["log_interval"]))
        checkpoint_interval = max(1, int(training["checkpoint_interval"]))
        snapshot_interval = max(
            1, int(training.get("snapshot_interval", checkpoint_interval))
        )
        self.online.train()
        last_log_time = time.perf_counter()
        last_log_step = self.step
        try:
            while self.step < total_steps:
                beta = linear_beta(
                    self.step,
                    beta_start=float(rl["prioritized_replay_beta_start"]),
                    beta_end=float(rl["prioritized_replay_beta_end"]),
                    anneal_steps=int(rl["prioritized_replay_beta_anneal_steps"]),
                )
                sample = self.priorities.sample(batch_size, beta, self.rng)
                batch, weights, ordered_indices = self._prepare_batch(sample)
                discounts = torch.full_like(batch["reward"], float(rl["gamma"]))

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    q_values = self.online(batch["observation"])
                    bootstrap_observation = concatenate_observations(
                        (batch["next_observation"], batch["n_step_observation"])
                    )
                    bootstrap_targets = double_dqn_target(
                        self.online,
                        self.target,
                        bootstrap_observation,
                        torch.cat((batch["reward"], batch["n_step_reward"]), dim=0),
                        torch.cat((batch["done"], batch["n_step_done"]), dim=0),
                        torch.cat((discounts, batch["n_step_discount"]), dim=0),
                    )
                    td1_target, n_target = bootstrap_targets.split(batch_size)
                    loss = compute_dqfd_loss(
                        q_values=q_values,
                        actions=batch["action"],
                        td1_targets=td1_target,
                        n_step_targets=n_target,
                        is_demo=batch["is_demo"],
                        importance_weights=weights,
                        online_network=self.online,
                        config=rl,
                        self_actions=batch["observation"]["state_categorical"][:, 0],
                        sample_weights=batch["training_weight"],
                    )
                self.scaler.scale(loss.total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), float(training["grad_clip"])
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.priorities.update_priorities(
                    ordered_indices,
                    loss.td_errors.abs().float().cpu().numpy(),
                )

                self.step += 1
                self.epoch = self.step * batch_size // max(len(self.train_dataset), 1)
                update_target_if_needed(
                    self.target,
                    self.online,
                    step=self.step,
                    mode=rl["target_update_mode"],
                    interval=int(rl["target_update_interval"]),
                    tau=float(rl["target_soft_tau"]),
                )
                if self.step % log_interval == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    now = time.perf_counter()
                    steps_per_second = (self.step - last_log_step) / max(
                        now - last_log_time, 1e-9
                    )
                    self._log_train(
                        loss,
                        q_values,
                        batch["action"],
                        batch["observation"]["state_categorical"][:, 0],
                        batch["training_weight"],
                        batch["round_outcome"],
                        beta,
                        steps_per_second,
                    )
                    last_log_time = now
                    last_log_step = self.step
                eval_interval = int(training["eval_interval"])
                # 全量训练模式没有验证集，必须跳过周期评估和 best checkpoint 选择。
                if (
                    self.validation_dataset is not None
                    and eval_interval > 0
                    and self.step % eval_interval == 0
                ):
                    metrics = self._evaluate()
                    metric = float(metrics.get("expert_top1", float("-inf")))
                    if metric > self.best_metric:
                        self.best_metric = metric
                        self._save_resume_checkpoint("best.pt")
                if self.step % checkpoint_interval == 0:
                    self._save_resume_checkpoint("last.pt")
                if self.step % snapshot_interval == 0:
                    self._save_model_snapshot()
        except KeyboardInterrupt:
            LOGGER.warning("收到 Ctrl+C，正在保存 Step %d", self.step)
            self._save_resume_checkpoint("last.pt")
            self._save_model_snapshot()
            LOGGER.warning("中断保存完成，可使用 last.pt 继续训练")
            return
        else:
            self._save_resume_checkpoint("last.pt")
            self._save_model_snapshot()
        finally:
            self.writer.flush()
            self.writer.close()
