import numpy as np

from soku_bc.weather import compress_weather


def test_weather_compression_matches_iql_mapping():
    raw = np.arange(22, dtype=np.int64)
    expected = np.zeros(22, dtype=np.int64)
    expected[[5, 8, 10, 12, 13, 16, 17, 18]] = np.arange(1, 9)
    np.testing.assert_array_equal(compress_weather(raw), expected)


def test_weather_compression_supports_live_scalar():
    assert compress_weather(5) == 1
    assert compress_weather(21) == 0
