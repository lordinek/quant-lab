"""
Kurulum zinciri testleri: supurme -> CHoCH -> IFVG -> giris.

Bu dosya projenin en kritik kod yolunu koruyor. Dort sinyal katmani ayri ayri
test edilmisti ama BIRLESIMLERI edilmemisti - ve gercek gelecege bakma hatasi
tam da orada cikti (esikler tum verinin ortalamasindan hesaplaniyordu).

Buradaki senaryolar elle kuruluyor: her adimin neden tetiklendigi (veya neden
tetiklenmedigi) gozle izlenebilsin diye.
"""
import pandas as pd
import pytest

from signals.fvg import Gap
from signals.liquidity import Sweep
from signals.structure import StructureBreak
from strategy.setup import (DEFAULT_CONFIG, MAX_GAP_CANDIDATES, Setup,
                            _allocate_gaps, _first_choch, _first_retest,
                            _inverted_gaps, _stop_from_sweep, find_setups,
                            to_frame)


def bars_from(rows, start="2026-09-04 12:00"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def sweep_at(index=10, side="high", level=100.0, pen=5.0):
    return Sweep(index=index, timestamp=pd.Timestamp("2026-09-04 12:00", tz="UTC"),
                 side=side, level=level, level_name="pdh", penetration=pen)


def brk(index, kind, direction, level=100.0):
    return StructureBreak(index=index, timestamp=pd.Timestamp("2026-09-04 12:00", tz="UTC"),
                          kind=kind, direction=direction, level=level, swing_index=index - 3)


def gap_at(inverted_index, direction, top=105.0, bottom=100.0):
    g = Gap(index=inverted_index - 5, direction=direction, top=top, bottom=bottom,
            created_at=pd.Timestamp("2026-09-04 12:00", tz="UTC"))
    g.inverted_index = inverted_index
    g.inverted_at = pd.Timestamp("2026-09-04 12:30", tz="UTC")
    return g


# --- zincir sirasi: her adim bir oncekinden SONRA olmali -------------------

def test_choch_must_come_after_sweep():
    breaks = [brk(5, "choch", "bear"), brk(20, "choch", "bear")]
    # Supurme 10. barda: 5. bardaki CHoCH gecmiste kaldi, secilmemeli.
    found = _first_choch(breaks, after=10, within=50, direction="short")
    assert found.index == 20


def test_choch_must_match_direction():
    """Ust likidite supuruldu -> asagi bekleniyor -> bull CHoCH kabul edilmez."""
    breaks = [brk(20, "choch", "bull")]
    assert _first_choch(breaks, after=10, within=50, direction="short") is None


def test_bos_is_not_accepted_as_choch():
    """BOS trend DEVAMI, CHoCH karakter DEGISIMI. Model donus ariyor."""
    breaks = [brk(20, "bos", "bear")]
    assert _first_choch(breaks, after=10, within=50, direction="short") is None


def test_choch_outside_window_is_rejected():
    breaks = [brk(200, "choch", "bear")]
    assert _first_choch(breaks, after=10, within=50, direction="short") is None


def test_inverted_gap_direction_is_opposite_of_trade():
    """Ters cevrilmis BULLISH bosluk artik dirençtir -> SHORT verir."""
    gaps = [gap_at(30, "bull")]
    assert _inverted_gaps(gaps, after=20, within=50, direction="short")
    assert not _inverted_gaps(gaps, after=20, within=50, direction="long")


def test_gap_candidates_are_capped():
    """Yedek bosluk sinirsiz degil: bes bosluk da baskasina gittiyse o bolge
    doymustur, altincisi ayni hikayenin parcasi degil."""
    gaps = [gap_at(30 + i, "bull") for i in range(MAX_GAP_CANDIDATES + 4)]
    found = _inverted_gaps(gaps, after=20, within=100, direction="short")
    assert len(found) == MAX_GAP_CANDIDATES


# --- stop yonu: en pahali sessiz hata -------------------------------------

def test_every_setup_has_stop_on_the_losing_side():
    """Stop, girisin ZARAR tarafinda olmali - short'ta yukari, long'ta asagi.

    Bu kural yoktu ve kurulumlarin %32,5'i stop'u KAR yonunde tasiyordu:
    stop ve hedef ayni tarafta kaliyor, ters hareket eden pozisyonu yalnizca
    zaman asimi kapatiyor, yani risk olculemez hale geliyordu. Ustelik
    backtest'te o "stop"lar kar uretip kazanma oranini sisiriyordu.

    Senaryo elle kurulmuyor: gercek veri uzerinde TUM kurulumlar taraniyor,
    cunku hata tek bir katmanda degil zincirin birlesiminde doguyor.
    """
    from data.histdata import download_range, resample

    minutes = download_range("NAS100", "2026-01", "2026-06", verbose=False)
    bars = resample(minutes, "5min")
    setups = find_setups(bars)

    assert setups, "veri yuklenemedi - test anlamsiz"
    ters = [s for s in setups
            if (s.direction == "short" and s.stop <= s.entry_price)
            or (s.direction == "long" and s.stop >= s.entry_price)]

    assert not ters, (
        f"{len(ters)}/{len(setups)} kurulumda stop kar yonunde; "
        f"ornek: {ters[0].direction} entry {ters[0].entry_price:.1f} "
        f"stop {ters[0].stop:.1f}")


def test_risk_and_reward_point_opposite_ways():
    """Stop ve hedef ZIT taraflarda olmali.

    Ayni tarafta olduklarinda pozisyon tek yonlu bir bahis degil, cikisi
    olmayan bir bekleyise donusuyor.
    """
    from data.histdata import download_range, resample

    minutes = download_range("NAS100", "2026-01", "2026-06", verbose=False)
    bars = resample(minutes, "5min")

    for s in find_setups(bars):
        assert (s.stop - s.entry_price) * (s.target - s.entry_price) < 0, (
            f"{s.direction} entry {s.entry_price:.1f} stop {s.stop:.1f} "
            f"target {s.target:.1f} - ikisi ayni tarafta")


# --- bosluk tahsisi: bir bosluk = bir islem -------------------------------

def setup_at(sweep_index, entry_index, gap, direction="short"):
    return Setup(entry_index=entry_index,
                 entry_time=pd.Timestamp("2026-09-04 12:00", tz="UTC"),
                 direction=direction, entry_price=100.0, stop=105.0,
                 target=92.5, sweep=sweep_at(index=sweep_index),
                 choch=brk(sweep_index + 1, "choch", "bear"), gap=gap)


def test_gap_goes_to_the_first_entry_not_the_first_sweep():
    """Ayni bosluga iki aday uzaniyorsa onu ilk GIREN alir, ilk SUPUREN degil.

    Supurmeye gore vermek gelecege bakmaktir: gec girenin boslugu tuketme ani,
    erken girenin karar aninin ILERISINDEDIR. Gerekce: _allocate_gaps.
    """
    g = gap_at(30, "bull")
    once_supurdu_gec_girdi = setup_at(sweep_index=5, entry_index=60, gap=g)
    gec_supurdu_once_girdi = setup_at(sweep_index=25, entry_index=40, gap=g)

    chosen = _allocate_gaps([once_supurdu_gec_girdi, gec_supurdu_once_girdi])

    assert len(chosen) == 1
    assert chosen[0].entry_index == 40


def test_allocation_does_not_depend_on_candidate_order():
    """Ayni adaylar hangi sirayla gelirse gelsin sonuc ayni olmali.

    Pencere kaydikca adaylarin bulunma sirasi degisir; sonuc degisirse canli
    bot backtest'ten ayrisir - bu hatanin ta kendisiydi.
    """
    g = gap_at(30, "bull")
    a = setup_at(sweep_index=5, entry_index=60, gap=g)
    b = setup_at(sweep_index=25, entry_index=40, gap=g)

    assert ([s.entry_index for s in _allocate_gaps([a, b])]
            == [s.entry_index for s in _allocate_gaps([b, a])])


def test_different_gaps_both_survive():
    """Ayri supurmeler ayri bosluklara giriyorsa ikisi de yasar."""
    a = setup_at(sweep_index=5, entry_index=40, gap=gap_at(30, "bull"))
    b = setup_at(sweep_index=25, entry_index=60, gap=gap_at(50, "bull"))

    assert len(_allocate_gaps([a, b])) == 2


def test_one_sweep_yields_at_most_one_trade():
    """Yedek bosluklar ayni hikayenin alternatifi, ek pozisyon degil.

    Bir supurme birden fazla bosluga aday olabilir (MAX_GAP_CANDIDATES), ama
    zincir tektir: ondan yalnizca bir islem cikar.
    """
    a = setup_at(sweep_index=5, entry_index=40, gap=gap_at(30, "bull"))
    b = setup_at(sweep_index=5, entry_index=60, gap=gap_at(50, "bull"))

    chosen = _allocate_gaps([a, b])

    assert len(chosen) == 1
    assert chosen[0].entry_index == 40      # once GIREN alternatif kazanir


# --- giris ----------------------------------------------------------------

def test_retest_enters_at_near_edge_not_mid():
    """Giris boslugun YAKIN kenarinda dolmali; ortayi almak her islemi
    sistematik olarak birkac puan iyi gosterir."""
    # Kontrol barlari boslugun ALTINDA kalmali (high < 100), yoksa retest
    # daha ilk barlarda tetiklenir.
    bars = bars_from([(95, 96, 94, 95)] * 5 + [(95, 106, 94, 100)])
    g = gap_at(2, "bull", top=105.0, bottom=100.0)
    hit = _first_retest(bars, g, after=2, within=10, direction="short")
    assert hit is not None
    index, price = hit
    assert price == g.bottom      # short icin alt kenar
    assert index == 5


def test_retest_returns_none_when_price_never_returns():
    bars = bars_from([(50, 51, 49, 50)] * 10)
    g = gap_at(2, "bull", top=105.0, bottom=100.0)
    assert _first_retest(bars, g, after=2, within=5, direction="short") is None


def test_retest_respects_window():
    bars = bars_from([(50, 51, 49, 50)] * 8 + [(100, 106, 99, 100)])
    g = gap_at(0, "bull", top=105.0, bottom=100.0)
    assert _first_retest(bars, g, after=0, within=3, direction="short") is None
    assert _first_retest(bars, g, after=0, within=20, direction="short") is not None


# --- stop -----------------------------------------------------------------

def test_stop_sits_beyond_sweep_extreme():
    s = sweep_at(level=100.0, pen=5.0, side="high")
    stop = _stop_from_sweep(s, "short", buffer_pct=0.0)
    assert stop == 105.0                      # supurmenin zirvesi

    with_buffer = _stop_from_sweep(s, "short", buffer_pct=1.0)
    assert with_buffer > stop                 # tampon disariya dogru


def test_stop_does_not_depend_on_dataset_average():
    """Stop yalnizca supurmenin kendi seviyesinden turemeli.

    Onceki hal veri setinin ortalama fiyatini kullaniyordu - yani gelecegi.
    Butunsel denetim bunu yakalamisti; burada regresyon olarak kilitliyoruz:
    fonksiyonun imzasinda artik boyle bir parametre YOK.
    """
    import inspect
    params = inspect.signature(_stop_from_sweep).parameters
    assert "avg_price" not in params
    assert set(params) == {"sweep", "direction", "buffer_pct"}


def test_long_stop_is_below_entry_side():
    s = sweep_at(level=100.0, pen=5.0, side="low")
    stop = _stop_from_sweep(s, "long", buffer_pct=0.0)
    assert stop == 95.0


# --- uctan uca ------------------------------------------------------------

def _scripted_short_scenario() -> pd.DataFrame:
    """Elle kurulmus tam bir short senaryosu.

    Once yukselen yapi kurulur, bir tepe supurulur, yapi asagi kirilir,
    bir bullish bosluk ters cevrilir, sonra fiyat bosluga geri doner.
    """
    rows = []
    # Yukselen yapi (swing high/low olussun diye dalgali).
    for base in (100, 102, 101, 104, 103, 106, 105, 108):
        rows.append((base, base + 1.5, base - 1.5, base))
    # Bullish bosluk yaratan hizli yukselis (bar[i-2].high < bar[i].low).
    rows.append((108, 112, 107, 112))
    rows.append((112, 118, 111, 117))
    rows.append((117, 120, 114, 119))
    # Tepe supurmesi: fitil yukari, kapanis geri.
    rows.append((119, 130, 118, 119))
    # Asagi kirilma - bosluk ters cevrilir.
    for px in (115, 110, 105, 100, 95):
        rows.append((px + 2, px + 3, px - 3, px))
    # Bosluga geri donus (retest).
    for px in (100, 108, 112, 114):
        rows.append((px - 2, px + 2, px - 3, px))
    rows += [(112, 113, 108, 110)] * 20
    return bars_from(rows)


def test_end_to_end_pipeline_runs_and_is_consistent():
    """Uctan uca: kurulum bulunursa alanlari kendi icinde tutarli olmali."""
    bars = _scripted_short_scenario()
    setups = find_setups(bars, {"min_risk_pct": 0.001, "min_gap_pct": 0.0,
                                "sweep_min_pen_pct": 0.0,
                                "killzones": ["london", "newyork", "asia",
                                              "newyork_pm"]})
    for s in setups:
        assert s.direction in ("long", "short")
        assert s.risk > 0
        # Zincir sirasi korunmus olmali.
        assert s.sweep.index < s.choch.index
        assert s.choch.index <= s.gap.inverted_index
        assert s.gap.inverted_index < s.entry_index
        # Stop ve hedef giris etrafinda dogru taraflarda.
        if s.direction == "short":
            assert s.stop > s.entry_price and s.target < s.entry_price
        else:
            assert s.stop < s.entry_price and s.target > s.entry_price


def test_min_risk_filter_removes_impossible_setups():
    """Stop cok yakinsa kurulum GERCEKTE alinamaz - filtre olmadan bir islem
    -200R gostermisti (2 puan maliyet / 0.01 puan risk)."""
    bars = _scripted_short_scenario()
    loose = find_setups(bars, {"min_risk_pct": 0.0, "min_gap_pct": 0.0,
                               "sweep_min_pen_pct": 0.0,
                               "killzones": ["london", "newyork", "asia",
                                             "newyork_pm"]})
    strict = find_setups(bars, {"min_risk_pct": 1000.0, "min_gap_pct": 0.0,
                                "sweep_min_pen_pct": 0.0,
                                "killzones": ["london", "newyork", "asia",
                                              "newyork_pm"]})
    assert len(strict) <= len(loose)
    assert strict == []


def test_killzone_filter_blocks_setups_outside_window():
    bars = _scripted_short_scenario()
    none_zone = find_setups(bars, {"killzones": [], "min_risk_pct": 0.001,
                                   "min_gap_pct": 0.0, "sweep_min_pen_pct": 0.0})
    assert none_zone == []


def test_empty_input_is_safe():
    empty = pd.DataFrame(columns=["open", "high", "low", "close"],
                         index=pd.DatetimeIndex([], tz="UTC"))
    assert find_setups(empty) == []
    assert to_frame([]).empty


def test_default_config_has_required_keys():
    """Config anahtarlarini kilitliyoruz: sessizce silinen bir anahtar
    varsayilan davranisi degistirir ve hicbir yerde hata vermez."""
    required = {"killzones", "swing_left", "swing_right", "min_gap_pct",
                "gap_max_age_hours", "sweep_min_pen_pct", "sweep_to_choch_hours",
                "choch_to_ifvg_hours", "ifvg_to_entry_hours", "stop_buffer_pct",
                "rr_target", "min_risk_pct", "exclude_entry_hours"}
    assert required.issubset(DEFAULT_CONFIG)

    # Zaman pencereleri SAAT cinsinden olmali, bar cinsinden degil: bar sayisi
    # zaman dilimine gore anlam degistirir ve bu sessizce baska bir strateji
    # uretir. Isim degisikligi kazara geri alinmasin diye kilitleniyor.
    for key in DEFAULT_CONFIG:
        assert not key.endswith(("_bars", "_max_age")), \
            f"{key}: zaman parametreleri saat cinsinden tanimlanmali"
