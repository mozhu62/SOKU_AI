from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.data.replay_reader import discover_replay_pairs
from soku_ai.data.validator import save_validation_report, validate_replay_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="校验主 CSV、对象 CSV、帧连续性与对象计数")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    pairs = discover_replay_pairs(config["data"]["raw_dir"])
    if arguments.limit is not None:
        pairs = pairs[: arguments.limit]
    results = []
    for index, pair in enumerate(pairs, start=1):
        result = validate_replay_pair(
            pair,
            valid_match_states=config["data"]["valid_match_states"],
            valid_battle_sub_modes=config["data"]["valid_battle_sub_modes"],
        )
        results.append(result)
        print(f"[{index}/{len(pairs)}] {pair.replay_id}: {'OK' if result.valid else 'FAILED'}")
    report_path = Path(config["data"]["reports_dir"]) / "raw_validation_report.json"
    save_validation_report(results, report_path)
    print(f"报告: {report_path}")
    if not all(result.valid for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
