"""Kaydirmali pencerenin ZAMAN cinsinden tanimli olmasini kilitler.

Bu dosyanin tek derdi bir birim hatasi. `windows.py` pencereyi bir zamanlar
`BARS_PER_DAY_5MIN = 288` sabitiyle hesapliyordu; 15 dakikalik veride ayni
"90 gunluk pencere" sessizce 270 gune donusuyor, getiriler sisiyor ve pencere
sayisi dusuyordu. Zaman dilimleri karsilastirilirken bu tamamen olcum
artefakti uretir.

Ayni hatanin bir baska turu projede daha once yasandi: pencereler bar
cinsindeyken "15 dakikalik daha iyi" olculup karar verilmis, sonra geri
alinmisti. Bu yuzden kural artik testle sabit: bir pencerenin kapsadigi
TAKVIM SURESI zaman diliminden bagimsiz olmali.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.windows import DEFAULT_WINDOW_DAYS, _bars_per_day, run_windows


def _bars(minutes: int, days: int = 400) -> pd.DataFrame:
    n = days * 24 * 60 // minutes
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{minutes}min", tz="UTC")
    price = np.linspace(100.0, 200.0, n)
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price}, index=idx)


def test_bars_per_day_zaman_diliminden_turetiliyor():
    assert _bars_per_day(_bars(5)) == 288
    assert _bars_per_day(_bars(15)) == 96
    assert _bars_per_day(_bars(60)) == 24


def test_pencere_ayni_takvim_suresini_kapsiyor():
    """Asil kural: 5dk ve 15dk veride pencere ayni GUN sayisini kapsamali."""
    spans = {}
    for minutes in (5, 15, 60):
        bars = _bars(minutes)
        per_day = _bars_per_day(bars)
        window_bars = int(round(DEFAULT_WINDOW_DAYS * per_day))
        spans[minutes] = (bars.index[window_bars - 1] - bars.index[0]).days

    # Hepsi ~90 gun olmali; bar araligindan kaynakli bir barlik sapma normal.
    for minutes, gun in spans.items():
        assert abs(gun - DEFAULT_WINDOW_DAYS) <= 1, f"{minutes}dk -> {gun} gun"


@dataclass
class _Setup:
    entry_index: int
    entry_price: float = 100.0
    direction: str = "long"
    risk: float = 1.0


@dataclass
class _Trade:
    setup: _Setup
    exit_index: int
    r_multiple: float = 0.5
    outcome: str = "target"


def _trades_every(bars: pd.DataFrame, step: int) -> list[_Trade]:
    """Takvimde duzenli araliklarla dagilmis sahte islemler.

    `step` BAR degil GUN cinsinden dusunulsun diye her zaman diliminde bar
    sayisina cevriliyor - testin kendisi olcmek istedigi hataya dusmesin.
    """
    per_day = _bars_per_day(bars)
    stride = max(1, int(round(step * per_day)))
    return [_Trade(_Setup(entry_index=i), exit_index=min(i + 1, len(bars) - 1))
            for i in range(0, len(bars) - 1, stride)]


def test_pencere_sayisi_zaman_diliminden_bagimsiz():
    """Ayni takvim araligi, farkli cozunurluk -> ayni sayida pencere.

    Eski sabit kodla 15 dakikalik veri UC KAT az pencere uretiyordu (28 -> 8);
    testin yakaladigi sey tam olarak budur. Islemler gercekten veriliyor,
    yoksa bos listeyle her zaman dilimi sifir pencere dondurur ve test yanlis
    sebeple gecer.
    """
    counts = {}
    for minutes in (5, 15, 60):
        bars = _bars(minutes)
        counts[minutes] = len(run_windows(bars, [], _trades_every(bars, step=3)))

    assert min(counts.values()) > 0, f"hic pencere uretilmedi: {counts}"
    assert len(set(counts.values())) == 1, f"pencere sayilari uyusmuyor: {counts}"
