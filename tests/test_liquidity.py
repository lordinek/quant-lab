"""Likidite supurmesi testleri.

Merkez ayrim: fitille asip geri donmek SUPURME, kapanisla asmak KIRILMA.
Bu ikisini karistiran bir model donus ararken devam sinyaline girer.
"""
import pandas as pd
import pytest

from signals.liquidity import (Sweep, _is_upper_level, build_levels,
                               equal_levels, find_sweeps, to_frame)


def make_bars(rows: list[tuple[float, float, float, float]],
              start: str = "2026-09-04 12:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def const_level(bars: pd.DataFrame, name: str, value: float) -> pd.DataFrame:
    return pd.DataFrame({name: [value] * len(bars)}, index=bars.index)


# --- supurme cekirdegi ----------------------------------------------------

def test_wick_above_level_with_close_below_is_a_sweep():
    bars = make_bars([
        (99, 100, 98, 99),
        (99, 105, 98, 99),     # fitil 105 (seviye 100'u asti), kapanis 99 -> supurme
        (99, 100, 98, 99),
    ])
    sweeps = find_sweeps(bars, const_level(bars, "pdh", 100.0))
    assert len(sweeps) == 1
    s = sweeps[0]
    assert (s.side, s.level_name) == ("high", "pdh")
    assert s.penetration == 5.0


def test_close_above_level_is_not_a_sweep():
    """Kapanis seviyenin ustunde = kirilma, donus beklentisi yok."""
    bars = make_bars([
        (99, 100, 98, 99),
        (99, 105, 98, 104),    # kapanis 104 > 100
    ])
    assert find_sweeps(bars, const_level(bars, "pdh", 100.0)) == []


def test_low_side_sweep():
    bars = make_bars([
        (101, 102, 100, 101),
        (101, 102, 95, 101),   # fitil 95 (seviye 100'un altina), kapanis 101
    ])
    sweeps = find_sweeps(bars, const_level(bars, "pdl", 100.0))
    assert len(sweeps) == 1
    assert sweeps[0].side == "low"
    assert sweeps[0].penetration == 5.0


def test_close_exactly_on_level_counts_as_sweep():
    """Seviyeye tam kapanmak otesinde kalmak degildir - supurme sayilir."""
    bars = make_bars([(99, 105, 98, 100)])
    assert len(find_sweeps(bars, const_level(bars, "pdh", 100.0))) == 1


def test_min_penetration_filters_noise():
    bars = make_bars([(99, 100.01, 98, 99)])
    levels = const_level(bars, "pdh", 100.0)
    assert len(find_sweeps(bars, levels)) == 1
    assert find_sweeps(bars, levels, min_penetration=0.5) == []


def test_nan_levels_are_skipped():
    bars = make_bars([(99, 105, 98, 99), (99, 105, 98, 99)])
    levels = pd.DataFrame({"pdh": [float("nan"), 100.0]}, index=bars.index)
    sweeps = find_sweeps(bars, levels)
    assert len(sweeps) == 1
    assert sweeps[0].index == 1


def test_sweeps_are_sorted_by_bar():
    bars = make_bars([
        (99, 105, 98, 99),      # ust supurme
        (99, 100, 90, 99),      # alt supurme
    ])
    levels = pd.DataFrame({"pdh": [100.0] * 2, "pdl": [95.0] * 2}, index=bars.index)
    sweeps = find_sweeps(bars, levels)
    assert [s.index for s in sweeps] == [0, 1]


# --- seviye adi -> taraf --------------------------------------------------

def test_level_side_inferred_from_name():
    assert _is_upper_level("pdh") is True
    assert _is_upper_level("pdl") is False
    assert _is_upper_level("asia_high") is True
    assert _is_upper_level("equal_lows") is False
    with pytest.raises(ValueError):
        _is_upper_level("bilinmeyen")


# --- esit seviyeler -------------------------------------------------------

def test_equal_highs_need_multiple_touches():
    # Ayni zirveye iki kez dokunuluyor.
    rows = [(10, 12, 9, 10)] + [(10, 11, 9, 10)] * 5 + [(10, 12, 9, 10)]
    rows += [(10, 11, 9, 10)] * 5
    bars = make_bars(rows)
    eq = equal_levels(bars, lookback=8, min_touches=2)
    assert eq["equal_highs"].notna().any()


def test_equal_levels_never_look_ahead():
    """Bir barin esit-seviye degeri yalnizca GECMIS pencereden gelmeli."""
    rows = [(10, 11, 9, 10)] * 20
    bars = make_bars(rows)
    eq_short = equal_levels(bars, lookback=10)

    longer = make_bars(rows + [(10, 99, 9, 10)] * 5)
    eq_long = equal_levels(longer, lookback=10)

    # Ilk 20 barin degerleri, sonradan eklenen ucuk barlardan etkilenmemeli.
    pd.testing.assert_series_equal(
        eq_short["equal_highs"], eq_long["equal_highs"].iloc[:20], check_freq=False)


# --- seviye insasi --------------------------------------------------------

def test_build_levels_has_previous_day_and_asia():
    idx = pd.date_range("2026-07-13 00:00", periods=72, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
                        index=idx)
    levels = build_levels(bars)
    assert {"pdh", "pdl"}.issubset(levels.columns)
    assert {"asia_high", "asia_low"}.issubset(levels.columns)


def test_to_frame_is_empty_safe():
    assert to_frame([]).empty
