"""按目标角色玩家划分完整 REP；镜像和原片必须留在同一集合。"""
import json
import math
from collections import defaultdict

import numpy as np

from .config import resolve


def split_settings(config):
    data = config['data']
    mode = data.get('split_mode', 'legacy')
    if mode not in ('legacy', 'per_player', 'fixed'):
        raise ValueError('data.split_mode 只能为 legacy、per_player 或 fixed')
    if mode == 'fixed':
        # 固定清单模式不读取划分时的外部玩家映射。
        return {'mode': 'fixed'}
    if mode == 'legacy':
        return None
    fraction = data.get('player_validation_fraction', 0.1)
    cap = data.get('player_validation_max', 30)
    character = data.get('split_character_id', 0)
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError('player_validation_fraction 必须在 0 与 1 之间')
    if type(cap) is not int or cap < 1 or type(character) is not int or character < 0:
        raise ValueError('玩家划分上限或角色 ID 无效')
    aliases = {}
    if data.get('player_aliases_file'):
        alias_path = resolve(data['player_aliases_file'])
        if not alias_path.is_file():
            # 不允许悄悄忽略映射，否则同一玩家的小号会被当成不同玩家重新划分。
            raise FileNotFoundError(
                f'玩家映射文件不存在：{alias_path}。请同步该 JSON 并修改 '
                'data.player_aliases_file；已有划分的训练请使用 data.split_mode=fixed。'
                '相对路径以 BC 项目根目录为基准。')
        catalog = json.loads(alias_path.read_text(encoding='utf-8-sig'))
        for name, info in catalog.items():
            for uid in info['user_ids']:
                key = str(uid)
                if key in aliases and aliases[key] != name:
                    raise ValueError(f'玩家小号重复归属：{uid}')
                aliases[key] = name
    return {'mode': mode, 'validation_fraction': fraction, 'validation_max': cap,
            'character_id': character, 'aliases': aliases}


def player_partition(root, entries, settings, seed, cancelled):
    players = defaultdict(dict)
    ownership = {}
    for name, sha in entries.items():
        if cancelled():
            raise InterruptedError('玩家划分已取消')
        with np.load(root / name, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata_json'].item()))
        capture = meta.get('capture', {})
        match = meta.get('source_match_metadata') or capture.get('source_match_metadata')
        if not match:
            raise ValueError(f'{name} 缺少玩家元数据，不能猜测玩家归属')
        sides = [side for side in ('server', 'client')
                 if match.get(side + 'Character') == settings['character_id']]
        if len(sides) != 1:
            raise ValueError(f'{name} 目标角色视角不唯一')
        uid = match.get(sides[0] + 'UserId')
        replay = match.get('id')
        if uid is None or replay is None:
            raise ValueError(f'{name} 缺少玩家 ID 或 REP ID')
        player = settings['aliases'].get(str(uid), 'user:' + str(uid))
        key = str(replay)
        # 同 REP 的原片、镜像和复制文件按元数据 ID 归组，而不是按文件名抽样。
        for identity in ('rep:' + key, 'sha:' + sha):
            owner = (player, key)
            if identity in ownership and ownership[identity] != owner:
                raise ValueError(f'{name} REP/内容哈希与玩家元数据冲突')
            ownership[identity] = owner
        players[player].setdefault(key, []).append(name)
    train, validation, summary = [], [], {}
    rng = np.random.default_rng(seed)
    for player in sorted(players):
        groups = players[player]
        keys = sorted(groups)
        rng.shuffle(keys)
        # 向上取整；只有一局的玩家全部用于训练，始终为每人保留训练数据。
        count = min(settings['validation_max'], len(keys) - 1,
                    math.ceil(len(keys) * settings['validation_fraction']))
        validation.extend(name for key in keys[:count] for name in groups[key])
        train.extend(name for key in keys[count:] for name in groups[key])
        summary[player] = {'replays': len(keys), 'train_replays': len(keys) - count,
                           'validation_replays': count}
    if not train or not validation:
        raise ValueError('玩家独立 REP 数不足，无法形成非空训练/验证集')
    return {'train': sorted(train), 'validation': sorted(validation), 'players': summary}
