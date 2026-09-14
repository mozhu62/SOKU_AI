from __future__ import annotations

import time

from .statistics import frame_key


def ingest_frames(runtime, frames):
    for snapshot in frames:
        payload = snapshot.payload
        active = runtime._observe(payload)
        if active:
            runtime.agent.observe_tcn_frame(payload, snapshot.resources)
        else:
            runtime._reset_memory("离开有效战斗或小局结束")


def run_tcn_loop(runtime, client, restart):
    """先收全真实观测，再对最新窗口决策；暂停发键不停止采集、不抹去观测。"""
    env = runtime.config["environment"]
    last_packet_at = time.monotonic()
    last_game_key, last_game_advance_at = None, time.monotonic()
    while not runtime.closed.wait(env["poll_seconds"]):
        cycle_started = time.perf_counter()
        runtime.control.touch()
        runtime._handle_commands()
        ctrl = runtime.control.snapshot()
        if ctrl["revision"] != runtime.control_revision:
            runtime.agent.card_macro.reset('control_changed')
            runtime.control_revision = ctrl["revision"]
            runtime.stats.cut(ctrl["reason"])
            runtime.pending = None
            restart.reset(runtime.control)
        read_started = time.perf_counter()
        frames, dropped = client.read()
        read_ms = (time.perf_counter() - read_started) * 1000
        if client.reset_reason:
            runtime._reset_memory(client.reset_reason)
        if dropped:
            runtime.frame_queue_dropped += dropped
            runtime._reset_memory(f"帧队列覆盖或槽读取失败：{dropped} 个采集槽")
            runtime.stats.mark_partial("真实帧队列缺失，重新积累 TCN 窗口")
        runtime.frame_queue_connected = client.address is not None
        if not frames:
            runtime.message = client.wait_reason
            if not client.available:
                runtime.control.release()
            if time.monotonic() - last_packet_at > env["stale_timeout_seconds"]:
                runtime.control.release()
                runtime._reset_memory("真实帧队列停止推进或连接断开")
                if ctrl["active"]:
                    runtime.control.pause(runtime.message)
                runtime.stats.cut("帧队列连接中断")
                client.close()
                runtime.frame_queue_connected = False
            runtime._publish()
            continue
        last_packet_at = time.monotonic()
        runtime.frame_queue_received += len(frames)
        ingest_started = time.perf_counter()
        ingest_frames(runtime, frames)
        ingest_ms = (time.perf_counter() - ingest_started) * 1000
        snapshot = frames[-1]
        p = snapshot.payload
        active = runtime._active_battle(p)
        key = frame_key(p)
        if key != last_game_key or not active:
            last_game_key, last_game_advance_at = key, time.monotonic()
        elif time.monotonic() - last_game_advance_at > env["stale_timeout_seconds"]:
            # DLL 可以继续发布同一游戏帧；不能仅凭新采集序号持续持键。
            runtime.control.release()
            runtime.message = "游戏帧停止推进，已松键；保留逻辑帧历史并等待继续"
            runtime._publish("等待游戏帧")
            continue
        age = runtime.control.publisher_age_ms(p)
        if age > env["max_snapshot_age_ms"]:
            # 没有新的可用动作不等于过去的真实观测无效；保留缓存，继续追读队列。
            runtime.control.release()
            runtime.message = f"最新队列帧已过期 {age} ms，保留真实历史并等待新帧"
            if age > env["stale_timeout_seconds"] * 1000:
                runtime._reset_memory("队列最新帧长时间过期")
                runtime.control.pause("DLL 未及时发布新帧，请检查游戏后继续")
                client.close()
                runtime.frame_queue_connected = False
            runtime._publish()
            continue
        if not runtime.control.ready():
            runtime.agent.card_macro.reset('control_not_ready')
            runtime.message = runtime.control.reason
            runtime._publish()
            continue
        if not active:
            if runtime.match_finished:
                runtime.message = restart.tick(runtime.control)
            else:
                runtime.control.release()
                runtime.message = "等待有效战斗；窗口不跨小局"
            runtime._publish()
            continue
        restart.reset(runtime.control)
        count = len(runtime.agent.tcn_window)
        required = runtime.agent.tcn_window.size
        if count < required:
            runtime.control.release()
            runtime.tcn_warmup_waits += 1
            runtime.message = f"正在积累真实连续历史 {count}/{required}；满窗后开始决策，不重复补帧"
            runtime._publish("积累历史")
            continue
        if runtime.last_inferred_key == key:
            runtime._publish()
            continue
        if not runtime.agent.prepare_cards(p):
            runtime.control.release()
            runtime.message = '等待与当前状态同序号的 Cards.v1 数据；不使用旧卡牌状态'
            runtime._publish()
            continue
        prediction, _ = runtime.agent.predict(p, snapshot.resources)
        prediction['loop_timing'] = {'read_ms': read_ms, 'ingest_ms': ingest_ms,
                                     'received_frames': len(frames), 'age_before_ms': age}
        prediction.update(execution_status="pending", execution_reason="等待发键前安全检查")
        runtime.last_prediction = prediction
        fresh = client.read_state()
        valid = fresh is not None and fresh.payload.initialized
        if valid:
            latest = fresh.payload
            lag = int(latest.battleFrame) - int(p.battleFrame)
            prediction['loop_timing']['lag_after_frames'] = lag
            valid = (frame_key(latest)[:2] == key[:2] and runtime._active_battle(latest)
                     and 0 <= lag <= env["max_inference_lag_frames"]
                     and runtime.control.publisher_age_ms(latest) <= env["max_snapshot_age_ms"])
        if runtime.closed.is_set():
            prediction.update(execution_status="not_sent", execution_reason="会话已停止")
            runtime.stats.inference_attempt(prediction)
            break
        if not valid:
            runtime.agent.card_macro.reset('stale_prediction')
            prediction.update(execution_status="discarded", execution_reason="推理后观测已变化；历史保留，追读队列后重算")
            runtime.stale_predictions += 1
            runtime.control.release()
            runtime.message = "旧动作未发送；保留已采集历史，下一次补收积压帧后重算"
            finish_attempt(runtime, prediction, cycle_started)
            continue
        if not runtime.agent.prepare_cards(latest):
            runtime.control.release()
            prediction.update(execution_status='discarded', execution_reason='发键前卡牌快照未匹配')
            finish_attempt(runtime, prediction, cycle_started)
            continue
        if runtime.agent.apply_prediction(runtime.control, prediction):
            prediction.update(execution_status="sent", execution_reason=f"已发送；使用 {required} 个真实连续帧")
            runtime.last_inferred_key = key
            runtime.stats.decision(prediction)
            if runtime.pending:
                runtime.unconfirmed += 1
            # 宏执行的是组合键，不能将未执行的 Combat 预测标为按键确认目标。
            runtime.pending = None if prediction.get('card_macro', {}).get('kind', 'combat') != 'combat' else {
                "key": key, "serial": int(latest.sampleSerial),
                "action": (prediction["direction"], prediction["buttons"])}
            runtime.message = f"TCN{required} 实战中：真实连续窗口；采集独立于发键，不训练模型"
        else:
            runtime.control.release()
            prediction.update(execution_status="not_sent", execution_reason="暂停或失焦，已保留真实观测窗口")
        finish_attempt(runtime, prediction, cycle_started)


def finish_attempt(runtime, prediction, cycle_started):
    publish_started = time.perf_counter()
    runtime._publish()
    prediction['loop_timing']['publish_ms'] = (time.perf_counter() - publish_started) * 1000
    prediction['loop_timing']['cycle_ms'] = (time.perf_counter() - cycle_started) * 1000
    runtime.stats.inference_attempt(prediction)
