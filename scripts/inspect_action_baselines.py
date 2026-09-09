"""只读统计既有固定划分的专家动作；不建模型、不生成划分、不修改 NPZ。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

import _bootstrap

from soku_bc.action_diagnostics import DIAGNOSTIC_VERSION, dataset_action_counts, merge_dataset_counts
from soku_bc.config import load_config, resolve
from soku_bc.dataset import digest, read_shard


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bc_suika.yaml')
    parser.add_argument('--directory', help='仅本次读取的数据目录，不改 YAML')
    parser.add_argument('--split', help='已有划分 JSON，必须与该次 BC 训练一致')
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve(args.directory or config['data']['directory'])
    path = resolve(args.split or config['data']['split_file'])
    split = json.loads(path.read_text(encoding='utf-8'))
    if split.get('sha256') != digest({key: value for key, value in split.items() if key != 'sha256'}):
        raise ValueError('固定划分校验失败；不会自动重建')
    groups = [set(split[group]) for group in ('train', 'validation')]
    if not all(groups) or groups[0] & groups[1] or groups[0] | groups[1] != set(split['files']):
        raise ValueError('固定划分存在重复、遗漏或空集合')
    if {split['files'][name] for name in groups[0]} & {split['files'][name] for name in groups[1]}:
        raise ValueError('训练与验证包含相同 NPZ 内容')
    result = {'action_diagnostics_version': DIAGNOSTIC_VERSION, 'split_hash': split['sha256'],
              'directory': str(root), 'split_file': str(path)}
    for group in ('train', 'validation'):
        counts = []
        mirror_files = 0
        for index, name in enumerate(split[group]):
            file = (root / name).resolve()
            if not file.is_relative_to(root):
                raise ValueError('划分路径越出数据目录')
            with file.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != split['files'][name]:
                    raise ValueError(f'分片内容与固定划分不一致：{name}')
            shard = read_shard(file, config['data']['vertical_positive_is_down'])
            valid = np.zeros(len(shard['joint_action_id']), bool)
            for start, end in shard['segments']:
                valid[start:end] = True
            counts.append(dataset_action_counts(shard['joint_action_id'], shard['previous_joint_action_id'], valid))
            mirror_files += int(file.stem.endswith('__mirror'))
            if index % 50 == 0:
                print(f'{group}: {index + 1}/{len(split[group])}', file=sys.stderr, flush=True)
        result[group] = {**merge_dataset_counts(counts), 'files': len(counts), 'mirror_named_files': mirror_files}
    result.update(train_previous_action_baseline=result['train']['previous_action_baseline'],
                  val_previous_action_baseline=result['validation']['previous_action_baseline'],
                  val_dataset_action_change_fraction=result['validation']['action_change_fraction'],
                  val_action_change_accuracy=None,
                  model_result_note='此脚本不运行模型；动作切换准确率请读取更新后 BC 验证日志')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
