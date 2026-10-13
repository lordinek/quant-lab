"""FVG / IFVG testleri.

Barlar elle kuruluyor: her testin bekledigi orunte gozle dogrulanabilsin diye.
Rastgele veriyle test etmek burada ise yaramaz - bir bosluk ya vardir ya yoktur,
onemli olan sinir durumlari.
"""
import pandas as pd
import pytest

from signals.fvg import (Gap, find_gaps, inverse_gaps, signal_direction,
                         to_frame, track_lifecycle)


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """(open, high, low, close) listesinden 1 dakikalik bar tablosu."""
    idx = pd.date_range("2026-09-04 12:00", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# --- tespit ---------------------------------------------------------------

def test_bullish_gap_detected():
    # bar0.high=10 < bar2.low=12  -> 10..12 arasi bullish bosluk
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 13),
    ])
    gaps = find_gaps(bars)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.direction == "bull"
    assert (g.bottom, g.top) == (10.0, 12.0)
    # Bosluk UCUNCU bar kapaninca bilinir - gelecege bakmamanin temeli.
    assert g.created_at == bars.index[2]


def test_bearish_gap_detected():
    # bar0.low=20 > bar2.high=18  -> 18..20 arasi bearish bosluk
    bars = make_bars([
        (21, 22, 20, 20),
        (20, 20, 17, 17),
        (17, 18, 16, 17),
    ])
    gaps = find_gaps(bars)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.direction == "bear"
    assert (g.bottom, g.top) == (18.0, 20.0)


def test_touching_bars_are_not_a_gap():
    # bar0.high == bar2.low: fitiller degiyor, arada islem gormemis bolge YOK.
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 10, 13),
    ])
    assert find_gaps(bars) == []


def test_min_size_filter_rejects_noise():
    bars = make_bars([
        (100, 100.01, 99, 100.01),
        (100, 103, 100, 103),
        (103, 104, 100.02, 103),
    ])
    assert len(find_gaps(bars)) == 1                      # bosluk var (0.01)
    assert find_gaps(bars, min_size_pct=0.5) == []        # ama gurultu kadar kucuk


# --- yasam dongusu --------------------------------------------------------

def test_gap_inverts_when_close_crosses_fully():
    # 10..12 bullish bosluk, sonra fiyat 9'a KAPANIYOR -> ters cevrildi.
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 13),
        (13, 13, 9, 9),
    ])
    gaps = track_lifecycle(bars, find_gaps(bars))
    g = gaps[0]
    assert g.inverted_at == bars.index[3]
    assert g.mitigated_at == bars.index[3]
    # Bullish bosluk ters cevrilince islem yonu SHORT olur.
    assert signal_direction(g) == "short"
    assert len(inverse_gaps(gaps)) == 1


def test_wick_through_gap_is_not_inversion():
    # Fitil 8'e iniyor ama KAPANIS 12.5 - bu supurme, rol degisimi degil.
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 13),
        (13, 13, 8, 12.5),
    ])
    gaps = track_lifecycle(bars, find_gaps(bars))
    g = gaps[0]
    assert g.inverted_at is None
    assert g.mitigated_at == bars.index[3]   # dokundu, ama cevirmedi
    assert inverse_gaps(gaps) == []


def test_gap_expires_without_inversion():
    rows = [(9, 10, 9, 10), (10, 13, 10, 13), (13, 14, 12, 13)]
    rows += [(13, 13.5, 12.5, 13)] * 10      # bosluga hic donmeyen sakin barlar
    bars = make_bars(rows)
    gaps = track_lifecycle(bars, find_gaps(bars), max_age=5)
    g = gaps[0]
    assert g.inverted_at is None
    assert g.expired_at is not None


def test_gap_cannot_invert_on_its_own_creation_bar():
    """Boslugu olusturan barin kendisi onu ters ceviremez.

    Ucuncu bar hem boslugu yaratip hem ayni anda gecersiz kilarsa, model
    bir olayi iki kez sayar ve sinyal ayni mumun icinde dogup olur.
    """
    # bar2 bullish boslugu olusturuyor ve kapanisi boslugun altinda (9).
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 9),
    ])
    gaps = track_lifecycle(bars, find_gaps(bars))
    assert gaps[0].inverted_at is None


def test_no_lookahead_future_bars_cannot_change_past_stamps():
    """Veriye YENI barlar eklemek, gecmiste damgalanmis olaylari degistirmemeli.

    Gelecege bakma hatasi sessizdir: backtest'i harika gosterir, canlida
    hicbir sey tutturmaz. BTC projesindeki test_no_lookahead'in muadili.
    """
    base = [
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 13),
        (13, 13, 9, 9),
    ]
    short = track_lifecycle(make_bars(base), find_gaps(make_bars(base)))
    long_bars = make_bars(base + [(9, 50, 9, 50), (50, 60, 49, 59)])
    long_ = track_lifecycle(long_bars, find_gaps(long_bars))

    assert short[0].inverted_at == long_[0].inverted_at
    assert short[0].created_at == long_[0].created_at


def test_multiple_gaps_tracked_independently():
    bars = make_bars([
        (9, 10, 9, 10),
        (10, 13, 10, 13),
        (13, 14, 12, 13),     # bosluk A: 10..12 bull
        (14, 20, 13, 20),     # low=13 = bar1.high -> degiyor, burada bosluk YOK
        (20, 21, 16, 20),     # bosluk B: 14..16 bull
        (20, 20, 11, 11),     # kapanis 11: B'yi ters cevirir (11<14), A'yi cevirmez (11>10)
    ])
    gaps = track_lifecycle(bars, find_gaps(bars))
    assert len(gaps) == 2
    a, b = gaps[0], gaps[1]
    assert (a.bottom, a.top) == (10.0, 12.0)
    assert (b.bottom, b.top) == (14.0, 16.0)
    assert b.inverted_at == bars.index[5]
    assert a.inverted_at is None


def test_to_frame_is_empty_safe():
    assert to_frame([]).empty


def test_signal_direction_requires_inversion():
    g = Gap(index=2, direction="bull", top=12, bottom=10,
            created_at=pd.Timestamp("2026-09-04 12:02", tz="UTC"))
    with pytest.raises(ValueError):
        signal_direction(g)
