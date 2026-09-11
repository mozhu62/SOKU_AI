from __future__ import annotations

import pytest

from soku_ai.rl.nstep import calculate_n_step_return


def test_n_step_return() -> None:
    value, discount, done, steps = calculate_n_step_return(
        [1.0, 2.0, 3.0, 4.0],
        [False, False, False, False],
        gamma=0.5,
        n_step=3,
    )
    assert value == pytest.approx(2.75)
    assert discount == pytest.approx(0.125)
    assert not done
    assert steps == 3


def test_n_step_stops_at_terminal() -> None:
    value, discount, done, steps = calculate_n_step_return(
        [1.0, 2.0, 100.0],
        [False, True, False],
        gamma=0.5,
        n_step=3,
    )
    assert value == pytest.approx(2.0)
    assert discount == pytest.approx(0.25)
    assert done
    assert steps == 2

