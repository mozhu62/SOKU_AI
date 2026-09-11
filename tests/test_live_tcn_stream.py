import ctypes as ct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from soku_bc.config import MODEL_DEFAULTS
from soku_bc.models import BCNetwork
from soku_bc.live.agent import LiveAgent
from soku_bc.live.frame_stream import (FrameHeader, FrameSlot, LiveFrameClient, CAPACITY, MAPPING_BYTES,
                                      MAGIC, VERSION, validate_header, captured_frame)
from soku_bc.live.shared_state import StatePayload, SharedMemoryProtocolError
from soku_bc.live.resource_state import Resources, matches
from soku_bc.live.tcn_window import TCNObservationWindow
from soku_bc.live.tcn_runtime import ingest_frames, run_tcn_loop
from soku_bc.live.runtime import LiveRuntime
from soku_bc.live.input_history import ControllerHistory
from tests.fixtures import tensor_observation


def header():
    return FrameHeader(MAGIC, VERSION, MAPPING_BYTES, ct.sizeof(StatePayload), ct.sizeof(Resources),
                       CAPACITY, ct.sizeof(FrameSlot), 7, 0, 0, 0, 1)


def fake_client():
    # 只操作测试自己分配的内存，不打开游戏映射、不发送按键。
    storage = ct.create_string_buffer(MAPPING_BYTES)
    root = ct.addressof(storage)
    current = header()
    ct.memmove(root, ct.addressof(current), ct.sizeof(current))
    client = LiveFrameClient.__new__(LiveFrameClient)
    client.address, client.pid, client.cursor, client.generation = root, 7, 0, None
    client.reset_reason, client.wait_reason = None, ''
    client.read_state = lambda: SimpleNamespace(payload=SimpleNamespace(gameProcessId=7, initialized=1))
    client._connect = lambda pid: True
    return storage, client


def write_frames(client, start, stop):
    for serial in range(start, stop + 1):
        slot = FrameSlot.from_address(client.address + ct.sizeof(FrameHeader) + ((serial - 1) % CAPACITY) * ct.sizeof(FrameSlot))
        slot.writeSequence = 0
        slot.serial = serial
        slot.payload.gameProcessId = 7
        slot.payload.initialized = slot.payload.inBattle = 1
        slot.payload.battleMode, slot.payload.battleSubMode, slot.payload.matchState = 2, 0, 2
        slot.payload.left.hp = slot.payload.right.hp = 10000
        slot.payload.sampleSerial = serial + 500
        slot.payload.battleFrame, slot.payload.currentRound = serial, 1
        slot.payload.publisherTickMs = serial * 17
        slot.resources.left.skills[0].variant = serial % 3
    FrameHeader.from_address(client.address).latestSerial = stop


class LiveTCNStreamTests(unittest.TestCase):
    """真实队列、窗口满额和暂停隔离验收源码；不随交付自动执行。"""

    def test_protocol_and_same_slot_resources(self):
        validate_header(header(), 7)
        bad = header()
        bad.slotSize += 8
        with self.assertRaises(SharedMemoryProtocolError):
            validate_header(bad, 7)
        slot = FrameSlot()
        slot.payload.gameProcessId, slot.payload.sampleSerial = 7, 100
        slot.payload.battleFrame, slot.payload.currentRound = 80, 2
        slot.resources.left.skills[0].variant = 2
        frame = captured_frame(slot)
        self.assertTrue(matches(frame.resources, frame.payload))
        self.assertEqual(frame.resources.resources.left.skills[0].variant, 2)

    def test_ring_backfills_real_frames_and_detects_overwrite(self):
        storage, client = fake_client()
        write_frames(client, 1, 40)
        frames, dropped = client.read()
        self.assertEqual([x.payload.battleFrame for x in frames], list(range(1, 41)))
        self.assertEqual(dropped, 0)
        write_frames(client, 41, 44)
        frames, dropped = client.read()
        self.assertEqual([x.payload.battleFrame for x in frames], [41, 42, 43, 44])
        for frame in frames:
            self.assertTrue(matches(frame.resources, frame.payload))
            self.assertEqual(frame.resources.resources.left.skills[0].variant, frame.payload.battleFrame % 3)
        write_frames(client, 45, 80)
        frames, dropped = client.read()
        self.assertEqual(len(frames), CAPACITY)
        self.assertEqual(dropped, 28)
        self.assertEqual(frames[0].payload.battleFrame, 73)
        self.assertEqual(frames[-1].payload.battleFrame, 80)

    def test_ring_odd_slot_is_not_a_valid_frame(self):
        storage, client = fake_client()
        write_frames(client, 1, 4)
        slot = FrameSlot.from_address(client.address + ct.sizeof(FrameHeader) + ct.sizeof(FrameSlot))
        slot.writeSequence = 1
        frames, dropped = client.read()
        self.assertEqual([x.payload.battleFrame for x in frames], [1, 3, 4])
        self.assertEqual(dropped, 1)
        FrameHeader.from_address(client.address).generation = 2
        slot.writeSequence = 2
        frames, _ = client.read()
        self.assertEqual(client.reset_reason, 'DLL 帧队列重新初始化')
        self.assertEqual(len(frames), 4)

    def test_window_requires_real_32_and_reuses_only_new_features(self):
        torch.manual_seed(37)
        model = BCNetwork({**MODEL_DEFAULTS, 'temporal_mode': 'tcn'}).eval()
        obs = tensor_observation(1, 40)
        obs['state_continuous'].normal_()
        window = TCNObservationWindow()
        row = lambda i: {name: value[0, i].numpy() for name, value in obs.items()}
        for i in range(31):
            window.append((7, 1, i), row(i))
        with self.assertRaisesRegex(ValueError, '不足'):
            window.logits(model, 'cpu')
        window.append((7, 1, 31), row(31))
        with torch.inference_mode():
            expected = model(obs)
            with patch.object(model, 'state_features', wraps=model.state_features) as encode:
                actual = window.logits(model, 'cpu')
                self.assertEqual(encode.call_args.args[0]['state_continuous'].shape[0], 32)
                self.assertTrue(torch.allclose(actual, expected[:, 31], atol=1e-5))
                # 即使中间没有推理/发键，真实帧依然进入窗口，不会清空。
                for i in range(32, 36):
                    window.append((7, 1, i), row(i))
                actual = window.logits(model, 'cpu')
                self.assertEqual(encode.call_args.args[0]['state_continuous'].shape[0], 4)
                self.assertTrue(torch.allclose(actual, expected[:, 35], atol=1e-5))
                self.assertEqual(window.rows[0][0][2], 4)
                self.assertEqual(len(window.features), 32)
                window.logits(model, 'cpu')
                self.assertEqual(encode.call_count, 2)
        window.append((7, 1, 35), row(35))
        self.assertEqual(len(window), 32)
        with self.assertRaisesRegex(ValueError, '不连续'):
            window.append((7, 1, 37), row(37))
        with self.assertRaisesRegex(ValueError, '不连续'):
            window.append((7, 2, 36), row(36))

    def test_paused_capture_and_true_gap_reset(self):
        runtime = LiveRuntime({'environment': {'active_match_states': [2], 'max_memory_gap_frames': 6}})
        agent = LiveAgent.__new__(LiveAgent)
        agent.temporal_mode, agent.memory, agent.tcn_window = 'tcn', None, TCNObservationWindow()
        history = ControllerHistory(0)
        def observe(p):
            history.observe((int(p.gameProcessId), int(p.currentRound), int(p.battleFrame)), 6, [0] * 4)
        def build(p, resources):
            observe(p)
            return history.observation()
        agent.builder = SimpleNamespace(reset=history.reset, build=build, observe_inputs=observe)
        runtime.agent = agent
        runtime.control = SimpleNamespace(ready=lambda: False, snapshot=lambda: {'active': False}, pause=Mock())
        runtime.stats = Mock(target=0, completed=0)
        storage, client = fake_client()
        write_frames(client, 1, 40)
        frames, _ = client.read()
        ingest_frames(runtime, frames)
        self.assertEqual(len(agent.tcn_window), 32)
        self.assertEqual(agent.tcn_window.key, (7, 1, 40))
        self.assertIsNone(runtime.last_inferred_key)
        self.assertEqual(runtime.temporal_resets, 0)
        write_frames(client, 42, 42)
        snapshot = client._slot(42)
        ingest_frames(runtime, [snapshot])
        self.assertEqual(runtime.temporal_resets, 1)
        self.assertEqual(len(agent.tcn_window), 1)
        self.assertEqual(agent.tcn_window.previous_action, 144)

    def test_missing_queue_never_falls_back_to_short_prediction(self):
        runtime = LiveRuntime({'environment': {'poll_seconds': .001, 'stale_timeout_seconds': 1}})
        runtime.closed = Mock(wait=Mock(side_effect=[False, True]))
        runtime.control = Mock()
        runtime.control.snapshot.return_value = {'revision': 0, 'active': False}
        runtime.agent = Mock()
        runtime._publish = Mock()
        client = SimpleNamespace(read=lambda: ([], 0), reset_reason=None, address=None, available=False,
                                 wait_reason='需要更新 DLL')
        run_tcn_loop(runtime, client, Mock())
        runtime.agent.predict.assert_not_called()
        runtime.control.apply_joint.assert_not_called()
        self.assertEqual(runtime.message, '需要更新 DLL')

    def test_loop_gates_short_history_and_keeps_full_history_on_discard(self):
        for count, lag, send_allowed in ((31, 0, True), (32, 0, True), (32, 10, True), (32, 0, False)):
            with self.subTest(count=count, lag=lag, send_allowed=send_allowed):
                runtime = LiveRuntime({'environment': {'poll_seconds': .001, 'stale_timeout_seconds': 1,
                                      'max_snapshot_age_ms': 250, 'max_inference_lag_frames': 6}})
                runtime.closed = Mock(wait=Mock(side_effect=[False, True]), is_set=Mock(return_value=False))
                runtime.control = Mock()
                runtime.control.snapshot.return_value = {'revision': 0, 'active': True}
                runtime.control.ready.return_value = True
                runtime.control.publisher_age_ms.return_value = 0
                runtime.control.apply_joint.return_value = send_allowed
                prediction = {'joint_action_id': 64, 'direction': 5, 'buttons': [0] * 4,
                              'context_frames_used': 32, 'inference_ms': 1.0}
                runtime.agent = SimpleNamespace(tcn_window=list(range(count)),
                                                predict=Mock(return_value=(prediction, None)))
                runtime.stats, runtime._publish, runtime._reset_memory = Mock(), Mock(), Mock()
                runtime._active_battle = lambda payload: True
                payload = SimpleNamespace(gameProcessId=7, currentRound=1, battleFrame=40, initialized=1)
                latest = SimpleNamespace(gameProcessId=7, currentRound=1, battleFrame=40 + lag,
                                         initialized=1, sampleSerial=600)
                snapshot = SimpleNamespace(payload=payload, resources=None)
                client = SimpleNamespace(read=lambda: ([snapshot], 0), reset_reason=None, address=1,
                                         available=True, read_state=lambda: SimpleNamespace(payload=latest))
                with patch('soku_bc.live.tcn_runtime.ingest_frames'):
                    run_tcn_loop(runtime, client, Mock())
                runtime._reset_memory.assert_not_called()
                self.assertEqual(len(runtime.agent.tcn_window), count)
                if count < 32:
                    runtime.agent.predict.assert_not_called()
                    runtime.control.apply_joint.assert_not_called()
                    self.assertEqual(runtime.tcn_warmup_waits, 1)
                elif lag > 6:
                    runtime.control.apply_joint.assert_not_called()
                    self.assertEqual(runtime.stale_predictions, 1)
                    self.assertEqual(prediction['execution_status'], 'discarded')
                elif send_allowed:
                    runtime.control.apply_joint.assert_called_once_with(64)
                    self.assertEqual(runtime.last_inferred_key, (7, 1, 40))
                else:
                    self.assertEqual(prediction['execution_status'], 'not_sent')
                    self.assertIsNone(runtime.last_inferred_key)


if __name__ == '__main__':
    unittest.main()
