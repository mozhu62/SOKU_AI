from __future__ import annotations

from soku_ai.env.mock_env import MockSokuEnv
from soku_ai.models.factory import build_model
from soku_ai.replay_buffer.prioritized_buffer import PrioritizedReplayBuffer
from soku_ai.training.online_trainer import OnlineDQNTrainer


def test_mock_environment_collection(model_config) -> None:
    env = MockSokuEnv(history_len=32, max_objects=32, horizon=2)
    model = build_model(model_config)
    buffer = PrioritizedReplayBuffer(8)
    trainer = OnlineDQNTrainer(
        env,
        model,
        buffer,
        num_actions=144,
        seed=1,
    )
    trainer.collect_step(epsilon=0.0)
    trainer.collect_step(epsilon=1.0)
    assert buffer.size == 2
    assert trainer.state.environment_steps == 2
    trainer.close()

