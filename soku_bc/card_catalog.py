"""完整游戏卡表的固定编码；checkpoint 保存展开后的表，不依赖运行机路径。"""
import hashlib
import json
from pathlib import Path


def load_catalog(path, character_id):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('schema') != 'soku_static_card_catalog_v1' or value.get('character_id') != character_id:
        raise ValueError('游戏卡表协议或角色不匹配')
    rows = sorted(value['cards'], key=lambda row: row['id'])
    ids = [row['id'] for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 65535 for i in ids):
        raise ValueError('游戏卡表为空或 Card ID 重复/非法')
    cards = []
    for row in rows:
        name = row.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('静态卡表缺少名称')
        cards.append({'id': row['id'], 'name': name})
    return cards


def static_catalog_path(character_id):
    if character_id != 0:
        raise ValueError('当前静态卡表仅配置灵梦，请为其他角色明确提供卡表')
    return Path(__file__).with_name('catalogs') / 'reimu.json'


def static_cards(character_id):
    return load_catalog(static_catalog_path(character_id), character_id)


def catalog_hash(cards):
    return hashlib.sha256(json.dumps(cards, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
