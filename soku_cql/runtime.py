from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
import traceback
from collections import OrderedDict, deque

import numpy as np
from filelock import FileLock

from . import checkpoint
from .action_space import ACTION_SCHEMA, action_catalog, frequency_rows
from .config import EDITABLE, resolve, validate
from .dataset import ReplayStore, split_replays
from .learner import Learner
from .n_step import target_spec
from .prefetch import BatchPrefetch
from .storage import atomic_json

LOGGER = logging.getLogger(__name__)


class Runtime:
    """网页只向命令队列写请求；模型、优化器、保存和参数应用均由同一工作线程执行。"""

    def __init__(self, config, resume=None, autostart=False, config_source="YAML"):
        self.config = copy.deepcopy(config)
        self.output = resolve(config["output"]["directory"])
        self.resume_path = resolve(resume) if resume else None
        self.autostart = autostart
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.commands = queue.Queue()
        self.requests = OrderedDict()
        self.history = deque(maxlen=10000)
        self.state = {"state": "initializing", "message": "准备离线训练数据", "step": 0, "updates": 0,
                      "samples": 0, "config": copy.deepcopy(config), "config_source": config_source,
                      "output": str(self.output), "stage": 0, "latest_train": None, "latest_validation": None,
                      "best": None, "error": None, "data": None, "locked_parameters": []}
        self.thread = threading.Thread(target=self._run, name="cql-learner", daemon=False)
        self.learner = None
        self.prefetch = None
        self.step = self.updates = self.samples = self.stage = 0
        self.best = None
        self.paused = True
        self.timings = {"data_wait": 0.0, "optimization": 0.0, "validation": 0.0, "save": 0.0, "paused": 0.0}
        self.last_progress = 0.0

    def start(self):
        self.thread.start()

    def _progress(self, message):
        self.publish(message=message)
        if time.monotonic() - self.last_progress >= 1.0:
            LOGGER.info(message)
            self.last_progress = time.monotonic()

    def publish(self, **values):
        with self.lock:
            self.state.update(copy.deepcopy(values))

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def history_page(self, kind, offset, limit):
        with self.lock:
            rows = [copy.deepcopy(row) for row in self.history if row["kind"] == kind]
        end = max(0, len(rows) - offset)
        return {"rows": rows[max(0, end - limit):end], "total": len(rows), "retention": 10000}

    def submit(self, identifier, name, value):
        fingerprint = json.dumps([name, value], sort_keys=True)
        with self.lock:
            if identifier in self.requests:
                old = self.requests[identifier]
                if old["fingerprint"] != fingerprint:
                    raise ValueError("同一请求编号不能用于不同命令")
                return copy.deepcopy(old)
            if name not in ("resume", "pause", "save", "stop", "validate", "configure", "locks"):
                raise ValueError("未知控制命令")
            if self.state["state"] in ("error", "stopped"):
                raise ValueError("训练线程已退出，请重新启动程序")
            row = {"id": identifier, "name": name, "fingerprint": fingerprint, "status": "queued", "message": "等待更新边界执行"}
            self.requests[identifier] = row
            while len(self.requests) > 256:
                first = next(iter(self.requests))
                if self.requests[first]["status"] == "queued":
                    break
                self.requests.pop(first)
        if name == "stop":
            self.stop_event.set()
        self.commands.put((identifier, name, copy.deepcopy(value)))
        return copy.deepcopy(row)

    def request(self, identifier):
        with self.lock:
            return copy.deepcopy(self.requests.get(identifier))

    def _finish_request(self, identifier, status, message):
        with self.lock:
            row = self.requests[identifier]
            row.update(status=status, message=message)
            self.state["last_request"] = copy.deepcopy(row)

    def _record(self, kind, values):
        row = {**values, **target_spec(self.config["training"]), "action_schema": ACTION_SCHEMA,
               "kind": kind, "step": self.step, "stage": self.stage, "time": time.time()}
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with self.lock:
            self.history.append(row)

    def _save(self, name="last.pt", snapshot=False):
        started = time.perf_counter()
        checkpoint.save(self.output / name, self.learner, self.config, self.split, self.store.normalization,
                        self.step, self.updates, self.samples, self.best, self.stage)
        if snapshot:
            checkpoint.save(self.output / "snapshots" / f"step_{self.step:09d}_{time.time_ns()}.pt",
                            self.learner, self.config, self.split, self.store.normalization,
                            self.step, self.updates, self.samples, self.best, self.stage)
        self.timings["save"] += time.perf_counter() - started
        self.publish(last_saved={"path": str(self.output / name), "step": self.step, "time": time.time()}, timings=self.timings)

    def _close_prefetch(self):
        if self.prefetch:
            self.prefetch.close()
            self.prefetch = None

    def _apply(self, values):
        if not self.paused:
            raise ValueError("请先暂停，待当前更新结束后再应用参数")
        if set(values) - {"training", "expected_stage"}:
            raise ValueError("配置请求包含未知字段")
        if values.get("expected_stage") != self.stage:
            raise ValueError("页面配置已过期，请刷新后重新编辑")
        changes = values.get("training", {})
        if not isinstance(changes, dict) or set(changes) - (set(EDITABLE) | {"frozen_modules"}):
            raise ValueError("只能修改表单列出的运行参数")
        locks = self.snapshot()["locked_parameters"]
        if set(changes) & set(locks):
            raise ValueError("修改包含锁定参数，请显式解锁")
        config = copy.deepcopy(self.config)
        config["training"].update(changes)
        validate(config)
        self._close_prefetch()
        old = self.config
        self.config = config
        self.learner.apply_settings(config)
        self.stage += 1
        self.best = None
        atomic_json(self.output / "configs" / f"stage_{self.stage}_{time.time_ns()}.json", config)
        atomic_json(self.output / "config.json", config)
        self._record("configuration", {"previous": old["training"], "current": config["training"]})
        self.publish(config=config, config_source="工作台应用值（已保存配置快照）", stage=self.stage, best=None,
                     latest_train=None, latest_validation=None)
        self._save()

    def _handle_commands(self):
        while True:
            try:
                identifier, name, value = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if name == "resume":
                    if self.step >= self.config["training"]["total_steps"]:
                        raise ValueError("已达到总步数，请先提高 total_steps")
                    self.paused = False
                    self.publish(message="正在从固定离线数据进行 CQL 更新")
                elif name == "pause":
                    self.paused = True
                    self._close_prefetch()
                elif name == "save":
                    self._save(snapshot=True)
                elif name == "stop":
                    self.stop_event.set()
                    continue
                elif name == "validate":
                    if not self.paused:
                        raise ValueError("手动验证需先暂停；训练中会按间隔自动验证")
                    self._validate()
                elif name == "configure":
                    self._apply(value)
                elif name == "locks":
                    if not self.paused:
                        raise ValueError("请暂停后修改参数锁")
                    keys = value.get("keys", [])
                    if not isinstance(keys, list) or set(keys) - (set(EDITABLE) | {"frozen_modules"}):
                        raise ValueError("参数锁列表无效")
                    atomic_json(self.output / "parameter_locks.json", keys)
                    self.publish(locked_parameters=keys)
                self._finish_request(identifier, "completed", "已执行" if name != "stop" else "正在停止并保存")
            except ValueError as error:
                self._finish_request(identifier, "failed", str(error))
            self.publish(state="paused" if self.paused else "training")

    def _validate(self):
        self._close_prefetch()
        started = time.perf_counter()
        self.publish(state="validating", message="固定验证样本上评估，不更新权重")
        cfg = self.config["training"]
        rows = []
        for index in range(cfg["validation_batches"]):
            if self.stop_event.is_set():
                break
            # 每次验证重用相同 seed 和批次形状；结果不会消耗训练采样随机数。
            rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], index, 1]))
            rows.append(self.learner.validate_batch(self.store.sample(rng, cfg, "validation")))
            self.publish(message=f"验证批次 {index + 1}/{cfg['validation_batches']}")
        if not rows or len(rows) != cfg["validation_batches"]:
            return
        count = sum(row["samples"] for row in rows)
        result = {}
        for key in rows[0]:
            if key in ("samples", "ev", "joint_data_top", "joint_pred_top"):
                continue
            if isinstance(rows[0][key], list):
                result[key] = sum(np.asarray(row[key], np.int64) for row in rows).tolist()
            else:
                result[key] = sum(row[key] * row["samples"] for row in rows) / count
        # 跨批用二阶矩合并 EV；不能简单平均各批 EV。
        variance = sum((row["target_variance"] + row["target_mean"] ** 2) * row["samples"] for row in rows) / count - result["target_mean"] ** 2
        error_mean = result["q_data_mean"] - result["target_mean"]
        error_variance = max(0, result["td_mse"] - error_mean ** 2)
        for prefix in ("q_data", "q_max"):
            variance_q = max(0, sum((row[f"{prefix}_std"] ** 2 + row[f"{prefix}_mean"] ** 2) * row["samples"]
                                    for row in rows) / count - result[f"{prefix}_mean"] ** 2)
            result[f"{prefix}_std"] = variance_q ** 0.5
        result.update(target_variance=variance, target_std=max(0, variance) ** 0.5,
                      error_variance=error_variance,
                      q_abs_max=max(row["q_abs_max"] for row in rows))
        result["joint_data_top"] = frequency_rows(result["joint_data"])
        result["joint_pred_top"] = frequency_rows(result["joint_pred"])
        result.update(samples=count, ev=1 - error_variance / variance if variance > 1e-8 else None,
                      batches=len(rows), seconds=time.perf_counter() - started,
                      selection_metric="validation_td_mse_v1", step=self.step, stage=self.stage,
                      **target_spec(cfg))
        self.timings["validation"] += result["seconds"]
        self._record("validation", result)
        self.publish(latest_validation=result, timings=self.timings)
        # 仅代表当前实验阶段的离线 Bellman 拟合；不宣称它是实战最强模型。
        if self.best is None or result["td_mse"] < self.best["value"]:
            self.best = {"metric": "validation_td_mse_v1", "value": result["td_mse"], "step": self.step,
                         "stage": self.stage, **target_spec(cfg)}
            self._save("best.pt")
            self._save(f"best_stage_{self.stage}.pt")
            self.publish(best=self.best)

    def _run(self):
        output_lock = None
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            output_lock = FileLock(str(self.output / ".training.lock"))
            output_lock.acquire(timeout=0)
            if self.resume_path is None and (self.output / "last.pt").exists():
                raise ValueError("输出目录已有模型；续训请指定 --resume，随机新训练请使用 --output 新目录")
            package = checkpoint.load(self.resume_path) if self.resume_path else None
            if package and target_spec(package["config"]["training"]) != target_spec(self.config["training"]):
                LOGGER.info("TD 目标切换：N=%d → %d，gamma=%s → %s；保留权重和优化器，验证另起阶段",
                            package["config"]["training"]["n_step"], self.config["training"]["n_step"],
                            package["config"]["training"]["gamma"], self.config["training"]["gamma"])
            split_path = resolve(self.config["data"]["split_file"])
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(split_path) + ".lock", timeout=0):
                self.split = split_replays(self.config, self._progress, self.stop_event.is_set)
            self.store = ReplayStore(self.config, self.split, self._progress,
                                     self.stop_event.is_set, package["normalization"] if package else None)
            if self.stop_event.is_set():
                raise InterruptedError("数据准备已取消")
            self.learner = Learner(self.config)
            LOGGER.info("TD 目标：N=%d，gamma=%s；终局停止 bootstrap，非终局片段末端缩短回报后 bootstrap",
                        self.config["training"]["n_step"], self.config["training"]["gamma"])
            if package:
                for key in ("model", "seed"):
                    if package["config"][key] != self.config[key]:
                        raise ValueError(f"续训不能修改 {key}")
                if package["config"]["data"]["vertical_positive_is_down"] != self.config["data"]["vertical_positive_is_down"]:
                    raise ValueError("续训不能翻转动作方向定义")
                checkpoint.restore(package, self.learner, self.split)
                self.step, self.updates, self.samples = package["step"], package["updates"], package["samples"]
                self.best, self.stage = package["best"], package["stage"]
                if package["config"]["training"] != self.config["training"]:
                    self.stage += 1
                    self.best = None
            if (self.output / "metrics.jsonl").exists():
                # 仅启动时读取一次历史，网页刷新不反复扫描整份日志。
                with (self.output / "metrics.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            self.history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            # 从旧版本分支续训时使用新的阶段编号，避免不同权重的同一步数混成一条曲线。
            history_stage = max((row.get("stage", 0) for row in self.history), default=-1)
            if any(row.get("step", 0) > self.step for row in self.history):
                self.stage, self.best = max(self.stage, history_stage + 1), None
            elif package and package["config"]["training"] != self.config["training"]:
                self.stage = max(self.stage, history_stage + 1)
            locks_path = self.output / "parameter_locks.json"
            locks = json.loads(locks_path.read_text(encoding="utf-8")) if locks_path.exists() else []
            atomic_json(self.output / "normalization.json", self.store.normalization)
            atomic_json(self.output / "config.json", self.config)
            atomic_json(self.output / "configs" / f"stage_{self.stage}_{time.time_ns()}.json", self.config)
            summary = self.store.summary()
            atomic_json(self.output / "dataset_summary.json", {**summary, "action_schema": ACTION_SCHEMA,
                        "action_catalog": action_catalog(), "files": list(self.store.info.values())})
            self.publish(data=summary, model_spec=self.learner.online.spec, action_catalog=action_catalog(),
                         parameter_counts={name: sum(p.numel() for p in module.parameters())
                                           for name, module in self.learner.online.module_groups().items()},
                         device=str(self.learner.device), amp=self.learner.amp, locked_parameters=locks,
                         step=self.step, updates=self.updates, samples=self.samples, stage=self.stage, best=self.best)
            self._save()
            self.paused = not self.autostart
            self.publish(state="paused" if self.paused else "training", message="离线数据就绪；验证集仅用于评估")
            while not self.stop_event.is_set():
                self._handle_commands()
                if self.stop_event.is_set():
                    break
                if self.step >= self.config["training"]["total_steps"]:
                    if not self.paused:
                        self._save(snapshot=True)
                    self.paused = True
                    self._close_prefetch()
                    self.publish(state="paused", message="已达到总步数；可提高 total_steps 后继续")
                if self.paused:
                    self.publish(state="paused")
                    before = time.perf_counter()
                    self.stop_event.wait(0.2)
                    self.timings["paused"] += time.perf_counter() - before
                    continue
                cfg = self.config["training"]
                if self.prefetch is None:
                    self.prefetch = BatchPrefetch(self.store, self.config, self.step, self.learner.device.type == "cuda")
                self.publish(state="training", message="正在从固定离线数据进行 CQL 更新")
                start = time.perf_counter()
                raw = self.prefetch.get()
                wait = time.perf_counter() - start
                self.timings["data_wait"] += wait
                diagnostic = self.step == 0 or (self.step + 1) % cfg["log_interval"] == 0
                result = self.learner.train_batch(raw, diagnostic)
                self.step += 1
                self.updates += int(not result["optimizer_skipped"])
                self.samples += result["samples"]
                self.timings["optimization"] += result["optimization_seconds"]
                active = wait + result["optimization_seconds"]
                if diagnostic:
                    result.update(data_wait_seconds=wait, steps_per_second=1 / max(active, 1e-9),
                                  samples_per_second=result["samples"] / max(active, 1e-9),
                                  learning_rate=cfg["learning_rate"], cql_alpha=cfg["cql_alpha"],
                                  cache_gb=self.store.bytes / 1024 ** 3,
                                  cache_hit_rate=self.store.hits / max(1, self.store.hits + self.store.misses),
                                  step=self.step, stage=self.stage, **target_spec(cfg))
                    self._record("train", result)
                    self.publish(latest_train=result, timings=self.timings)
                    LOGGER.info("step=%d/%d N=%d effective_N=%.2f loss=%.5f TD=%.5f CQL=%.5f speed=%.2f step/s",
                                self.step, cfg["total_steps"], cfg["n_step"], result["n_step_mean"],
                                result["loss"], result["td_mse"], result["cql_gap"], result["steps_per_second"])
                self.publish(step=self.step, updates=self.updates, samples=self.samples)
                if self.step % cfg["validation_interval"] == 0:
                    self._validate()
                if self.step % cfg["save_interval"] == 0:
                    self._save(snapshot=True)
            self._close_prefetch()
            self._save()
            self.publish(state="stopped", message="已停止并保存 last.pt")
        except InterruptedError:
            self.publish(state="stopped", message="已取消数据准备")
        except Exception as error:
            LOGGER.exception("CQL 运行失败")
            self.publish(state="error", error=str(error), message="训练已停止；已保存的模型仍保留")
            try:
                (self.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
        finally:
            self._close_prefetch()
            with self.lock:
                for row in self.requests.values():
                    if row["status"] == "queued":
                        if row["name"] == "stop" and self.state["state"] == "stopped":
                            row.update(status="completed", message=self.state["message"])
                        else:
                            row.update(status="failed", message="工作线程已退出，命令未执行")
            if output_lock is not None and output_lock.is_locked:
                output_lock.release()
