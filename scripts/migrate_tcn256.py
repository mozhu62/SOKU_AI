"""显式迁移 BC TCN32 到 TCN256，不覆盖源模型，不恢复旧优化器。"""
import argparse
import copy

import _bootstrap
from soku_bc import checkpoint
from soku_bc.config import load_config, resolve
from soku_bc.learner import Learner
from soku_bc.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/bc_suika.yaml')
    args = parser.parse_args()
    target = resolve(args.output)
    if target.exists():
        raise FileExistsError('迁移目标已存在，禁止覆盖')
    source = checkpoint.load(resolve(args.source))
    if source['config']['model']['temporal_mode'] != 'tcn':
        raise ValueError('只接受当前 Joint144 BC TCN32 来源')
    config = load_config(args.config)
    if config['model']['temporal_mode'] != 'tcn256':
        raise ValueError('目标配置必须为 tcn256，burn_in=255')
    if config['data']['vertical_positive_is_down'] != source['config']['data']['vertical_positive_is_down']:
        raise ValueError('迁移不能改变方向定义')
    requested_device = config['training']['device']
    cpu_config = copy.deepcopy(config)
    cpu_config['training']['device'] = 'cpu'
    learner = Learner(cpu_config)
    state = learner.model.state_dict()
    old = source['model']
    expected = {key for key in state if any(key.startswith(f'tcn.blocks.{i}.') for i in (4, 5, 6))}
    if set(old)-set(state) or set(state)-set(old) != expected:
        raise ValueError('迁移只允许新增三个TCN块，不能缺少任何旧层')
    for key, value in old.items():
        if value.shape != state[key].shape:
            raise ValueError(f'旧权重形状不匹配：{key}')
        state[key] = value
    learner.model.load_state_dict(state, strict=True)
    config['training']['device'] = requested_device
    checkpoint.save(target, learner, config, {'sha256': source['split_hash']}, source['normalization'],
                    0, 0, 0, None, 0)
    atomic_json(target.with_suffix('.migration.json'), dict(source=str(resolve(args.source)),
                source_step=source['step'], migration='bc_tcn32_to_tcn256', optimizer_reset=True))
    print(f'已迁移到 {target}；旧层复用，新增块随机初始化，优化器/计数/排名重置。需重新训练评估。')


if __name__ == '__main__':
    main()
