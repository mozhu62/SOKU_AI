from __future__ import annotations

from collections import deque

import numpy as np
import torch

from soku_ai.data.resources_schema import (
    MODEL_TYPE, OBSERVATION_VERSION, STATE_FIELDS, TACTICAL_FIELDS, HISTORY_FIELDS, OBJECT_FIELDS,
)
from soku_ai.data.transition_builder import CATEGORICAL_PADDING_VALUE
from .observation import LiveObservation, LiveStateUnavailable


class ResourceLiveObservationBuilder:
    """与 resources_reader 对齐的实战输入；按真实采集帧累计历史。"""

    def __init__(self, config, normalization, player_side="left"):
        data = config["data"]
        if config["model"]["model_type"] != MODEL_TYPE or config["model"]["num_actions"] != 144:
            raise ValueError("实战模型必须为 resources_v4 DQfD 144 类网络")
        if data["preprocessing_version"] != OBSERVATION_VERSION or data["max_objects_per_side"] != 3:
            raise ValueError("实战观察版本或对象数量与训练不一致")
        self.history_len = int(data["history_len"])
        if not 1 <= self.history_len < 128:
            raise ValueError("真实帧队列支持 1～127 帧历史，当前 checkpoint 窗口不兼容")
        self.shift = data["action_shift"]
        if self.shift not in (0, 1) or player_side not in ("left", "right", "auto"):
            raise ValueError("action_shift 或 player_side 无效")
        self.side = player_side
        self.positive_right = bool(data["direction_positive_faces_right"])
        self.norm = {}
        if normalization.get("version") != OBSERVATION_VERSION:
            raise ValueError("checkpoint 归一化不是 resources_v4 版本")
        for group, fields in (("state", STATE_FIELDS + TACTICAL_FIELDS), ("history", HISTORY_FIELDS), ("object", OBJECT_FIELDS)):
            if normalization.get(f"{group}_features") != list(fields):
                raise ValueError(f"checkpoint {group} 字段顺序与实时输入不一致")
            mean = np.asarray(normalization[f"{group}_mean"], np.float32)
            std = np.asarray(normalization[f"{group}_std"], np.float32)
            if (mean.shape != (len(fields),) or std.shape != mean.shape or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or (std <= 0).any()):
                raise ValueError(f"checkpoint {group} 归一化参数无效")
            self.norm[group] = mean, std
        self.history_num = deque(maxlen=self.history_len)
        self.history_cat = deque(maxlen=self.history_len)
        self.reset_count = 0
        self.last_reset_reason = "首次连接"
        self.reset()

    def reset(self, reason="暂停、场景变化或观测失效"):
        if getattr(self, "key", None) is not None:
            self.reset_count += 1
        self.last_reset_reason = reason
        self.key = None
        self.previous_input = None
        self.axes_duration = 0
        self.history_num.clear()
        self.history_cat.clear()

    def normalize(self, values, group):
        values = np.asarray(values, np.float32)
        if not np.isfinite(values).all():
            raise LiveStateUnavailable(f"实时 {group} 输入有 NaN/Inf")
        mean, std = self.norm[group]
        result = (values - mean) / std
        if not np.isfinite(result).all():
            raise LiveStateUnavailable(f"实时 {group} 归一化溢出")
        return result

    def players(self, payload):
        side = self.side
        if side == "auto":
            left, right = int(payload.left.characterId) == 9, int(payload.right.characterId) == 9
            if left == right:
                raise LiveStateUnavailable("自动选侧无法确定萃香，请在实战配置 player_side 指定 left 或 right")
            side = "left" if left else "right"
        return (payload.left, payload.right, side) if side == "left" else (payload.right, payload.left, side)

    def validate(self, payload):
        if not payload.initialized or not payload.inBattle or int(payload.matchState) != 2:
            raise LiveStateUnavailable("等待实际战斗帧")
        if int(payload.battleSubMode) == 2:
            raise LiveStateUnavailable("当前是 REP 回放，不发送实战按键；人机主模式 2 可以正常运行")
        player, opponent, side = self.players(payload)
        if player.hp <= 0 or opponent.hp <= 0:
            raise LiveStateUnavailable("本小局已结束，等待下一小局")
        if int(player.direction) not in (-1, 1):
            raise LiveStateUnavailable("角色朝向无效")
        return player, opponent, side

    @staticmethod
    def categories(payload, player, opponent):
        return np.asarray([player.action, player.actionBlockId, opponent.action, opponent.actionBlockId,
                           payload.activeWeather], np.int64)

    def objects(self, owner, player):
        count = int(owner.objectCount)
        if not 0 <= count <= len(owner.objects):
            raise LiveStateUnavailable("DLL 对象数量超出协议容量")
        entities = list(owner.objects[:count])
        dx = np.asarray([obj.positionX for obj in entities], np.float32) - np.float32(player.positionX)
        dy = np.asarray([obj.positionY for obj in entities], np.float32) - np.float32(player.positionY)
        if not np.isfinite(dx).all() or not np.isfinite(dy).all():
            raise LiveStateUnavailable("DLL 对象坐标无效")
        # 与 REP 转换相同：float32 距离，并列时按源对象序号排序，只使用最近三个。
        order = np.lexsort((np.arange(count), np.hypot(dx, dy)))[:3]
        numerical = np.zeros((3, 8), np.float32)
        categorical = np.full((3, 2), CATEGORICAL_PADDING_VALUE, np.int64)
        mask = np.arange(3) < len(order)
        active = False
        for slot, index in enumerate(order):
            obj = entities[index]
            numerical[slot] = self.normalize([dx[index], dy[index],
                np.float32(obj.speedX) - np.float32(player.speedX), np.float32(obj.speedY) - np.float32(player.speedY),
                obj.direction, obj.actionFrameCount, obj.hitstop, obj.hitCount], "object")
            categorical[slot] = [obj.action, obj.actionBlockId]
            active |= bool(obj.frameDataAvailable) and any(box.valid and not rotation.valid
                for box, rotation in zip(obj.hitBoxes, obj.hitRotationBoxes))
        return numerical, categorical, mask, active

    def build(self, payload, *, history_only=False):
        try:
            return self._build(payload, history_only=history_only)
        except LiveStateUnavailable:
            self.reset()
            raise

    def _build(self, payload, *, history_only):
        player, opponent, side = self.validate(payload)
        key = (int(payload.gameProcessId), int(payload.currentRound), int(payload.battleFrame), side)
        if key == self.key:
            return None
        continuous = (self.key is not None and key[:2] == self.key[:2] and key[3] == self.key[3]
                      and key[2] == self.key[2] + 1)
        if self.key is not None and not continuous:
            self.reset("真实游戏帧不连续或回合/控制侧变化")
        sign = lambda value: (int(value) > 0) - (int(value) < 0)
        current_input = np.asarray([sign(player.inputHorizontal), sign(player.inputVertical),
                                   player.inputA > 0, player.inputB > 0, player.inputC > 0, player.inputD > 0], np.float32)
        old_duration = self.axes_duration
        self.axes_duration = (old_duration + 1 if continuous and
                              np.array_equal(current_input[:2], self.previous_input[:2]) else 1)
        relative_x = np.float32(opponent.positionX) - np.float32(player.positionX)
        relative_y = np.float32(opponent.positionY) - np.float32(player.positionY)
        categories = self.categories(payload, player, opponent)
        if continuous:
            # shift=1 的标签来自下一采集帧，因此当前回读输入就是训练的 action[t-1]。
            # shift=0 则必须使用上一采集帧回读，二者不能再共同向前错移一帧。
            control = current_input if self.shift == 1 else self.previous_input
            duration = self.axes_duration if self.shift == 1 else old_duration
            history = np.r_[control, duration, player.actionFrameCount, opponent.actionFrameCount, relative_x, relative_y].astype(np.float32)
            self.history_num.append(self.normalize(history, "history"))
            self.history_cat.append(categories[:4].copy())
        self.key, self.previous_input = key, current_input
        if history_only:
            return None
        observation = {}
        threat = False
        for name, entity in (("self", player), ("opponent", opponent)):
            num, cat, mask, active = self.objects(entity, player)
            observation.update({f"{name}_object_numerical": num, f"{name}_object_categorical": cat,
                                f"{name}_object_mask": mask})
            if name == "opponent":
                threat = active
        state = []
        for entity in (player, opponent):
            state.extend([entity.positionX, entity.positionY, entity.speedX, entity.speedY,
                          entity.direction, entity.currentSpirit, entity.actionFrameCount])
        state.extend([relative_x, relative_y, max(0, player.hp), max(0, opponent.hp)])
        flag = lambda entity, bit: bool(entity.frameDataAvailable and int(entity.frameFlags) & (1 << bit))
        state.extend([flag(player, 11), flag(opponent, 11), flag(player, 10), flag(opponent, 10), threat,
                      50 <= player.action < 150, flag(player, 2), 50 <= opponent.action < 150, flag(opponent, 2)])
        count = len(self.history_num)
        history_num = np.zeros((self.history_len, 11), np.float32)
        history_cat = np.full((self.history_len, 4), CATEGORICAL_PADDING_VALUE, np.int64)
        history_mask = np.zeros(self.history_len, bool)
        if count:
            history_num[-count:] = np.stack(self.history_num)
            history_cat[-count:] = np.stack(self.history_cat)
            history_mask[-count:] = True
        observation.update(state_continuous=self.normalize(state, "state"), state_categorical=categories,
                           history_numerical=history_num, history_categorical=history_cat, history_mask=history_mask)
        return LiveObservation({name: torch.from_numpy(value) for name, value in observation.items()},
            key[0], int(payload.sampleSerial), key[2], key[1], side, int(player.direction),
            bool(player.direction > 0) if self.positive_right else bool(player.direction < 0), int(player.hitstop), count)
