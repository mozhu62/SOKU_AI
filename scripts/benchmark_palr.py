"""短合成批次计时：不读取 REP、不保存模型、不启动正式训练或游戏。"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time

import _bootstrap
import torch

from soku_bc.config import load_config
from soku_bc.learner import Learner, tensor_batch
from tests.fixtures import tensor_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 2 <= args.steps <= 50 or not 1 <= args.batch_size <= 32:
        parser.error("短 benchmark 只允许 2～50 步、1～32 条序列")
    config = load_config("configs/bc_palr_c.yaml")
    config["training"].update(device=args.device, batch_size=args.batch_size)
    rng = torch.Generator().manual_seed(8821)
    b, length, burn = args.batch_size, 32, 31
    obs = tensor_observation(b, length + burn)
    obs["state_continuous"] = torch.randn(b, length + burn, 18, generator=rng)
    actions = torch.randint(0, 144, (b, length + burn), generator=rng)
    obs["previous_joint_action_id"][:, 1:] = actions[:, :-1]
    batch = {"observation": obs, "burn_lengths": torch.full((b,), burn, dtype=torch.long),
             "joint_action_id": actions[:, burn:], "previous_expert_action_id": actions[:, burn-1:-1],
             "mask": torch.ones(b, length, dtype=torch.bool)}
    samples = {False: [], True: []}
    runs = []
    # 交错 A/B/B/A 顺序，减少预热和显卡时钟漂移对结论的影响；每轮同种子重新初始化。
    for enabled in (False, True, True, False):
        cfg = copy.deepcopy(config)
        cfg["palr"]["enabled"] = enabled
        learner = Learner(cfg)
        resident_batch = tensor_batch(batch, learner.device)
        for _ in range(3):
            learner.train_batch(resident_batch)
        elapsed = []
        skipped = 0
        for _ in range(args.steps):
            if learner.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = learner.train_batch(resident_batch)
            skipped += int(result["optimizer_skipped"])
            elapsed.append((time.perf_counter() - started) * 1000)
        runs.append({"enabled": enabled, "median_ms": statistics.median(elapsed), "optimizer_skipped": skipped})
        samples[enabled].extend(elapsed)
        del resident_batch, learner
    off, on = (statistics.median(samples[mode]) for mode in (False, True))
    print(json.dumps({"torch": torch.__version__, "device": args.device,
                      "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                      "batch_size": b, "supervised_length": length, "context": burn,
                      "palr_sample_size": cfg["palr"]["sample_size"], "keyframe_weight": cfg["keyframe_weighting"]["changepoint_weight"],
                      "amp": learner_amp(config, args.device), "measured_steps_per_mode": len(samples[False]),
                      "disabled_median_ms": off, "enabled_median_ms": on, "extra_ms": on-off,
                      "extra_percent": (on/off-1)*100, "runs": runs,
                      "scope": "resident synthetic batch; forward/backward/AdamW; no dataset IO, diagnostics or checkpoint"},
                     ensure_ascii=False, indent=2))


def learner_amp(config, device):
    return bool(config["training"]["amp"] and device == "cuda")


if __name__ == "__main__":
    main()
