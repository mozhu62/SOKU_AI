import torch
import pytest

from soku_bc.config import MODEL_DEFAULTS, context_frames_for
from soku_bc.models import network_spec
from soku_bc.temporal import TemporalConvEncoder
from soku_bc.live.tcn_window import TCNObservationWindow


def test_bc256_spec():
    cfg = {**MODEL_DEFAULTS, 'temporal_mode': 'tcn256'}
    spec = network_spec(cfg)
    assert context_frames_for(cfg) == 256
    assert spec['temporal']['dilations'] == [1, 2, 4, 8, 16, 32, 64]
    assert 1+1+2*sum(spec['temporal']['dilations']) == 256
    assert network_spec(MODEL_DEFAULTS)['temporal']['context_frames'] == 32


def test_bc256_causal_window():
    torch.manual_seed(3)
    tcn = TemporalConvEncoder(4, 8, 8, context_frames=256).eval()
    sequence = torch.randn(1, 287, 4)
    with torch.no_grad():
        full = tcn(sequence)
        torch.testing.assert_close(full[:, -1], tcn(sequence[:, -256:])[:, -1], atol=1e-5, rtol=1e-5)
        changed = sequence.clone()
        changed[:, 260:] += 2
        torch.testing.assert_close(full[:, :260], tcn(changed)[:, :260])


def test_live_requires_full_window():
    window = TCNObservationWindow(256)
    assert window.rows.maxlen == 256
    with pytest.raises(ValueError, match='256'):
        window.logits(None, 'cpu')
