"""Piyasa yapisi testleri.

Odak: teyit gecikmesi ve BOS/CHoCH ayrimi. Bir swing sagindaki barlar
olusmadan BILINEMEZ; bu gecikmeyi atlayan kod backtest'te harika gorunup
canlida cuker.
"""
import pandas as pd

from signals.structure import (find_breaks, find_swings, to_frame, trend_series)


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-09-04 12:00", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def ramp(values: list[float]) -> pd.DataFrame:
    """Kapanis serisinden basit barlar (high/low kapanisin biraz disinda)."""
    return make_bars([(v, v + 0.5, v - 0.5, v) for v in values])


# --- swing tespiti --------------------------------------------------------

def test_swing_high_is_confirmed_late():
    # Tepe 5. barda; left=right=2 ile ancak 7. barda BILINIR.
    bars = ramp([10, 11, 12, 13, 14, 20, 14, 13, 12, 11])
    swings = find_swings(bars, left=2, right=2)
    highs = [s for s in swings if s.kind == "high"]
    assert len(highs) == 1
    assert highs[0].index == 5
    assert highs[0].confirmed_index == 7      # gecikme = right


def test_swing_low_detected():
    bars = ramp([20, 19, 18, 17, 10, 17, 18, 19, 20])
    lows = [s for s in find_swings(bars) if s.kind == "low"]
    assert len(lows) == 1
    assert lows[0].index == 4


def test_flat_region_produces_no_swings():
    """Yatay bolgede her bar 'tepe' sayilirsa yapi anlamsizca parcalanir."""
    bars = ramp([10] * 12)
    assert find_swings(bars) == []


def test_too_short_input_is_safe():
    assert find_swings(ramp([1, 2, 3])) == []


# --- BOS / CHoCH ----------------------------------------------------------

def test_first_break_upward_is_bos_then_reversal_is_choch():
    # Once tepe kurulup yukari kirilir (BOS bull), sonra dip asagi kirilir (CHoCH).
    bars = ramp([
        10, 11, 12, 11, 10,      # 2. barda swing high = 12.5
        11, 12, 13,              # 13 > 12.5 -> yukari kirilma
        12, 11, 10, 9, 8,        # asagi donus
        7, 6, 5,                 # onceki dip kirilir
    ])
    breaks = find_breaks(bars, find_swings(bars))
    assert len(breaks) >= 2

    first, second = breaks[0], breaks[1]
    assert (first.kind, first.direction) == ("bos", "bull")
    assert (second.kind, second.direction) == ("choch", "bear")


def test_break_uses_close_not_wick():
    """Fitille seviyeyi delip geri donmek yapiyi kirmaz - o likidite supurmesi."""
    bars = make_bars([
        (10, 10.5, 9.5, 10),
        (11, 11.5, 10.5, 11),
        (12, 12.5, 11.5, 12),     # swing high = 12.5
        (11, 11.5, 10.5, 11),
        (10, 10.5, 9.5, 10),
        # Fitil 13'e cikiyor (12.5'i asiyor) ama KAPANIS 11 - kirilma yok.
        (11, 13.0, 10.5, 11),
    ])
    breaks = find_breaks(bars, find_swings(bars))
    assert all(b.direction != "bull" for b in breaks)


def test_same_swing_is_not_broken_twice():
    """Ayni seviyenin ustunde kalan her bar yeni bir kirilma saymamali."""
    bars = ramp([10, 11, 12, 11, 10, 11, 12, 13, 13, 13, 13, 13])
    breaks = find_breaks(bars, find_swings(bars))
    bull = [b for b in breaks if b.direction == "bull"]
    assert len(bull) == 1


def test_trend_series_applies_from_break_bar_onward():
    bars = ramp([10, 11, 12, 11, 10, 11, 12, 13, 12, 11])
    breaks = find_breaks(bars, find_swings(bars))
    trend = trend_series(bars, breaks)
    b = breaks[0]
    # Kirilmadan onceki bar henuz trendi bilmiyor.
    assert trend.iloc[b.index - 1] is None
    assert trend.iloc[b.index] == b.direction


def test_no_lookahead_appending_bars_keeps_past_breaks():
    base = [10, 11, 12, 11, 10, 11, 12, 13, 12, 11]
    short = find_breaks(ramp(base), find_swings(ramp(base)))
    longer_bars = ramp(base + [50, 60, 70, 80])
    longer = find_breaks(longer_bars, find_swings(longer_bars))

    assert len(longer) >= len(short)
    for a, b in zip(short, longer):
        assert (a.index, a.kind, a.direction) == (b.index, b.kind, b.direction)


def test_to_frame_is_empty_safe():
    assert to_frame([]).empty
