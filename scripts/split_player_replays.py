"""只生成玩家级固定划分清单，不转换数据、不训练、不修改 NPZ。"""
import argparse
import json
from pathlib import Path

import _bootstrap
from soku_bc.dataset import split_replays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--aliases', required=True)
    parser.add_argument('--character', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = {'seed': args.seed, 'data': {
        'directory': str(Path(args.source).resolve()),
        'split_file': str(Path(args.output).resolve()),
        'split_mode': 'per_player', 'split_character_id': args.character,
        'player_validation_fraction': 0.1, 'player_validation_max': 30,
        'player_aliases_file': str(Path(args.aliases).resolve()),
    }}
    result = split_replays(config, print)
    print(json.dumps(result['players'], ensure_ascii=False, indent=2))
    print(f"训练文件 {len(result['train'])}，验证文件 {len(result['validation'])}")


if __name__ == '__main__':
    main()
