import copy
import unittest

from soku_bc.config import DEFAULTS, validate
from soku_bc.runtime import Runtime
from soku_bc.web_service import _private_host, _same_origin


class BCControlTests(unittest.TestCase):
    def test_request_deduplication_and_snapshot_isolation(self):
        runtime = Runtime(copy.deepcopy(DEFAULTS))
        first = runtime.submit('request_123', 'pause', {})
        second = runtime.submit('request_123', 'pause', {})
        self.assertEqual(first, second)
        self.assertEqual(runtime.commands.qsize(), 1)
        with self.assertRaises(ValueError):
            runtime.submit('request_123', 'resume', {})
        snapshot = runtime.snapshot()
        snapshot['config']['training']['learning_rate'] = 1.0
        self.assertEqual(runtime.config['training']['learning_rate'], DEFAULTS['training']['learning_rate'])

    def test_no_online_reward_parameters_and_freeze_validation(self):
        config = copy.deepcopy(DEFAULTS)
        self.assertTrue({'gamma', 'n_step', 'cql_alpha', 'target_tau'}.isdisjoint(config['training']))
        config['training']['frozen_modules'] = ['joint_head']
        with self.assertRaises(ValueError):
            validate(config)
        config['training']['frozen_modules'] = []
        config['training']['label_smoothing'] = 0.5
        with self.assertRaises(ValueError):
            validate(config)

    def test_ssh_forwarded_port_and_cross_origin_protection(self):
        self.assertTrue(_private_host('localhost:8796', 8797, client_host='127.0.0.1'))
        self.assertFalse(_private_host('localhost:8796', 8797, client_host='192.168.1.2'))
        self.assertTrue(_same_origin('http://localhost:8796', 'localhost:8796'))
        self.assertFalse(_same_origin('https://example.com', 'localhost:8796'))


if __name__ == '__main__':
    unittest.main()
