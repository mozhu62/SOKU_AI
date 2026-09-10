import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from soku_bc.config import DEFAULTS
from soku_bc.learner import Learner
from soku_bc.runtime import Runtime
from soku_bc.live.runtime import LiveRuntime
from tests.fixtures import normalization


@contextmanager
def isolated_runtime():
    # 只在使用者手动执行测试时创建临时输出；不启动训练线程、服务或键盘控制。
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder).resolve()
        config = copy.deepcopy(DEFAULTS)
        config['training'].update(device='cpu', amp=False, burn_in=31)
        config['output']['directory'] = str(root / 'original')
        runtime = Runtime(config)
        runtime.output.mkdir()
        runtime.learner = Learner(config)
        runtime.split = {'sha256': 'fixture'}
        runtime.store = SimpleNamespace(normalization=normalization(), summary=lambda: {})
        runtime.step, runtime.updates, runtime.samples = 7, 7, 100
        runtime.publish(state='paused')
        target = root / 'branches' / 'candidate'
        values = {'name': 'candidate', 'confirm': True,
                  'expected_stage': 0, 'expected_output': str(runtime.output)}
        try:
            with patch('soku_bc.runtime.experiment_path', return_value=target), \
                    patch('soku_bc.runtime.resolve', side_effect=lambda path: root if str(path) == '.' else Path(path)):
                yield runtime, values, target
        finally:
            for lock in runtime.experiment_locks:
                lock.release()


class TemporalRuntimeTests(unittest.TestCase):
    """实验隔离与实战连续性验收源码，本次交付不执行。"""

    def test_random_branch_preserves_original_and_pauses(self):
        with isolated_runtime() as (runtime, values, target):
            original, old_output = runtime.learner, runtime.output
            weights = {key: value.clone() for key, value in original.model.state_dict().items()}
            runtime._new_experiment(values)
            self.assertIsNot(runtime.learner, original)
            self.assertTrue(runtime.paused)
            self.assertEqual((runtime.step, runtime.updates, runtime.samples), (0, 0, 0))
            self.assertEqual(runtime.output, target)
            self.assertEqual(runtime.learner.model.temporal_mode, 'tcn')
            self.assertTrue((old_output / 'last.pt').is_file())
            self.assertTrue((target / 'last.pt').is_file())
            self.assertTrue((target / 'comparison.json').is_file())
            for key, value in original.model.state_dict().items():
                self.assertTrue(torch.equal(weights[key], value))
            # 原页面的 stage 同为 0，也不能把参数应用到另一个实验目录。
            with self.assertRaisesRegex(ValueError, '旧页面'):
                runtime._apply({'training': {'learning_rate': 0.0002}, 'expected_stage': 0,
                                'expected_output': str(old_output)})

    def test_failed_branch_keeps_model_counts_and_rng(self):
        with isolated_runtime() as (runtime, values, target):
            original, config, output = runtime.learner, copy.deepcopy(runtime.config), runtime.output
            rng = torch.get_rng_state().clone()
            def fail(_):
                torch.manual_seed(999)
                raise RuntimeError('模拟新模型初始化失败')
            with patch('soku_bc.runtime.Learner', side_effect=fail):
                with self.assertRaisesRegex(ValueError, '原模型仍保留'):
                    runtime._new_experiment(values)
            self.assertIs(runtime.learner, original)
            self.assertEqual(runtime.config, config)
            self.assertEqual(runtime.output, output)
            self.assertEqual((runtime.step, runtime.updates, runtime.samples), (7, 7, 100))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertFalse((target / 'last.pt').exists())
            self.assertTrue(runtime.paused)

    def test_experiment_requires_pause_and_cannot_overwrite(self):
        with isolated_runtime() as (runtime, values, target):
            runtime.paused = False
            with self.assertRaisesRegex(ValueError, '暂停'):
                runtime._new_experiment(values)
            self.assertFalse(target.exists())
            runtime.paused = True
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, '目录已存在'):
                runtime._new_experiment(values)

    def test_experiment_command_is_idempotent(self):
        runtime = Runtime(copy.deepcopy(DEFAULTS))
        value = {'name': 'candidate', 'confirm': True}
        runtime.submit('experiment_request_01', 'experiment', value)
        runtime.submit('experiment_request_01', 'experiment', value)
        self.assertEqual(runtime.commands.qsize(), 1)
        with self.assertRaises(ValueError):
            runtime.submit('experiment_request_01', 'experiment', {**value, 'name': 'other'})

    def test_live_resume_keeps_tcn_observations(self):
        runtime = LiveRuntime({})
        runtime.agent = SimpleNamespace(temporal_mode='tcn', memory=None, tcn_window=[1, 2, 3])
        runtime.agent.reset = Mock(side_effect=lambda: setattr(runtime.agent, 'memory', None))
        runtime.control = SimpleNamespace(resume=Mock(), reason='已请求继续')
        runtime.stats = SimpleNamespace(target=0, completed=0)
        runtime.latest = SimpleNamespace(gameProcessId=1)
        request = runtime.command('resume')
        runtime._handle_commands()
        self.assertTrue(request.done())
        runtime.agent.reset.assert_not_called()
        self.assertEqual(runtime.agent.tcn_window, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
