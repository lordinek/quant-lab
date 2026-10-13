"""
Piyasa yapisi: swing noktalari, BOS ve CHoCH.

SMC'de "trend" bir indikatorle degil, fiyatin kendi tepe ve dipleriyle
tanimlanir. Yukselen trend = her tepe ve dip bir oncekinden yuksek. Bu yapinin
kirilmasi iki farkli sey anlatir:

    BOS   (Break of Structure)  - trend YONUNDE kirilma: devam sinyali.
    CHoCH (Change of Character) - trend TERSINE kirilma: ilk donus uyarisi.

CHoCH bu projede kritik: IFVG kurulumlari genelde CHoCH'tan sonra aranir,
cunku yapinin karakter degistirdigi yerde ters cevrilmis bosluklar gercek
bir emir kumesine denk gelir.

GECIKME UYARISI - bu dosyanin en onemli kismi:
Bir swing noktasi ancak SAGINDA yeterli bar olustuktan sonra dogrulanabilir.
`left=2, right=2` ile calisirken i. bardaki tepe ancak i+2 anda BILINIR.
Bu yuzden her swing'in iki damgasi var: olustugu bar (`index`) ve teyit
edildigi bar (`confirmed_index`). Backtest ve canli kod DAIMA
`confirmed_index` kullanmali. Bunu karistirmak klasik gelecege bakma
hatasidir: sonuclari muhtesem gosterir, canlida hicbir sey tutturmaz.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Swing:
    """Bir salinim noktasi.

    index          : tepe/dibin olustugu bar
    confirmed_index: bu noktanin BILINEBILIR oldugu bar (index + right)
    """
    index: int
    confirmed_index: int
    kind: str            # 'high' | 'low'
    price: float
    timestamp: pd.Timestamp


@dataclass
class StructureBreak:
    """Yapinin kirildigi an."""
    index: int
    timestamp: pd.Timestamp
    kind: str            # 'bos' | 'choch'
    direction: str       # 'bull' | 'bear' - kirilmanin yonu
    level: float         # kirilan swing seviyesi
    swing_index: int     # kirilan swing'in bari


def find_swings(bars: pd.DataFrame, left: int = 2, right: int = 2) -> list[Swing]:
    """Fraktal swing tepe ve dipleri.

    Bir bar, solundaki `left` ve sagindaki `right` barin hepsinden yuksek
    (veya dusuk) ise swing sayilir. Buyuk `right` degeri daha guvenilir ama
    daha GEC teyit edilen noktalar uretir - dogrudan bir gecikme takasi.
    """
    n = len(bars)
    if n < left + right + 1:
        return []

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    stamps = bars.index

    swings: list[Swing] = []
    for i in range(left, n - right):
        window_h = high[i - left: i + right + 1]
        window_l = low[i - left: i + right + 1]

        # Esitlik durumunda swing saymiyoruz: yatay bolgelerde her bar
        # "tepe" olur ve yapi anlamsiz sekilde parcalanir.
        if high[i] == window_h.max() and (window_h == high[i]).sum() == 1:
            swings.append(Swing(i, i + right, "high", float(high[i]), stamps[i]))
        elif low[i] == window_l.min() and (window_l == low[i]).sum() == 1:
            swings.append(Swing(i, i + right, "low", float(low[i]), stamps[i]))

    return swings


def _last_confirmed(swings: list[Swing], kind: str, upto: int) -> Swing | None:
    """`upto` barinda BILINEN en son swing (teyit gecikmesine saygi duyar).

    Tek tek arama; find_breaks bunu KULLANMAZ (orada iki isaretcili O(n)
    gecis var). Burada tekil sorgular ve testler icin duruyor.
    """
    best = None
    for s in swings:
        if s.kind == kind and s.confirmed_index <= upto:
            best = s
        elif s.confirmed_index > upto:
            break
    return best


def find_breaks(bars: pd.DataFrame, swings: list[Swing]) -> list[StructureBreak]:
    """Yapi kirilmalarini (BOS / CHoCH) sirayla bulur.

    Kirilma KAPANISLA olculuyor, fitille degil: fitilin bir seviyeyi delip
    geri donmesi likidite supurmesidir - yapinin kirilmasi degil. Ayni ayrim
    fvg.py'de de var ve ikisi ayni felsefeden geliyor.

    Trend durumu kirilmalarla guncelleniyor: ilk kirilma yonu trendi kurar,
    sonraki kirilmalar ayni yondeyse BOS, tersse CHoCH olur ve trendi cevirir.
    """
    if not swings:
        return []

    close = bars["close"].to_numpy(dtype=float)
    stamps = bars.index

    breaks: list[StructureBreak] = []
    trend: str | None = None
    # Kirilan bir seviyeyi tekrar tekrar saymamak icin en son kirilan
    # swing'leri hatirliyoruz.
    used_high: int | None = None
    used_low: int | None = None

    # Iki isaretcili gecis: her bar icin swing listesini bastan taramak
    # O(bar x swing) demekti - 400 bin barlik veride bu dakikalarca suruyor.
    # Swing'ler zaten bar sirasinda, teyit damgalari da oyle; isaretcileri
    # ileri kaydirmak yeterli.
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    hi_ptr = lo_ptr = 0
    sh: Swing | None = None
    sl: Swing | None = None

    for i in range(len(bars)):
        while hi_ptr < len(highs) and highs[hi_ptr].confirmed_index <= i:
            sh = highs[hi_ptr]
            hi_ptr += 1
        while lo_ptr < len(lows) and lows[lo_ptr].confirmed_index <= i:
            sl = lows[lo_ptr]
            lo_ptr += 1

        if sh is not None and sh.index != used_high and close[i] > sh.price:
            kind = "choch" if trend == "bear" else "bos"
            breaks.append(StructureBreak(i, stamps[i], kind, "bull", sh.price, sh.index))
            trend, used_high = "bull", sh.index

        elif sl is not None and sl.index != used_low and close[i] < sl.price:
            kind = "choch" if trend == "bull" else "bos"
            breaks.append(StructureBreak(i, stamps[i], kind, "bear", sl.price, sl.index))
            trend, used_low = "bear", sl.index

    return breaks


def trend_series(bars: pd.DataFrame, breaks: list[StructureBreak]) -> pd.Series:
    """Her bar icin o an gecerli yapi yonu ('bull'/'bear'/None).

    Kirilmanin YASANDIGI bardan itibaren gecerli - o bardan once trend
    henuz degismemistir.
    """
    # pd.Series(None, ...) NaN uretir, None degil - "trend henuz belli degil"
    # durumunu NaN ile temsil etmek karsilastirmalari sessizce bozar.
    trend = pd.Series([None] * len(bars), index=bars.index, dtype=object)
    for b in breaks:
        trend.iloc[b.index:] = b.direction
    return trend


def to_frame(breaks: list[StructureBreak]) -> pd.DataFrame:
    if not breaks:
        return pd.DataFrame(columns=["timestamp", "kind", "direction", "level"])
    return pd.DataFrame([{
        "timestamp": b.timestamp, "kind": b.kind,
        "direction": b.direction, "level": b.level,
    } for b in breaks])
