from __future__ import annotations

# 枚举来自 tools/SokuLib-dev/src/BattleMode.hpp；主模式与子模式不是同一个编号空间。
MODE_NAMES = {0: "剧情", 1: "街机", 2: "人机对战", 3: "本地双人", 4: "联机客户端",
              5: "联机服务器", 6: "观战", 7: "时间挑战", 8: "练习"}
SUBMODE_NAMES = {0: "正常游戏 PLAYING1", 1: "正常游戏 PLAYING2", 2: "REP 回放"}


def mode_details(payload):
    mode, submode = int(payload.battleMode), int(payload.battleSubMode)
    return {"battle_mode": mode, "battle_mode_name": MODE_NAMES.get(mode, f"未知主模式 {mode}"),
            "battle_submode": submode, "battle_submode_name": SUBMODE_NAMES.get(submode, f"未知子模式 {submode}")}


def role_error(payload, config):
    if not payload.inBattle:
        return None
    # 沿用 PPO 对真实战斗/角色的校验，不再用 mainMode==2 且 subMode==0 拦截正常人机。
    if int(payload.battleSubMode) == 2:
        return "检测到 REP 回放（battleSubMode=2），不向回放发送控制按键；主模式 2 本身是人机对战"
    player, opponent = (payload.left, payload.right) if config["player_side"] == "left" else (payload.right, payload.left)
    if player.characterId != config["self_character"] or opponent.characterId != config["opponent_character"]:
        return f"角色/控制侧不符：己方={player.characterId}，对方={opponent.characterId}"
    return None
