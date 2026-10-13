"""
Fair Value Gap (FVG) ve Inverse FVG (IFVG) tespiti.

FVG: uc barlik bir orunte. Ortadaki bar o kadar hizli hareket eder ki birinci
ve ucuncu barin fitilleri hic ortusmez - arada fiyatin "atladigi", islem
gormedigi bir bosluk kalir.

    Bullish FVG:  bar[i-2].high < bar[i].low    -> bosluk yukari harekette
    Bearish FVG:  bar[i-2].low  > bar[i].high   -> bosluk asagi harekette

Klasik SMC beklentisi: fiyat bu bosluga geri doner ve oradan ilk yonune devam
eder (bullish gap destek, bearish gap direnc).

IFVG (bu projenin asil ilgi alani): fiyat boslugu tutmayip TAMAMEN gecerse,
o bolge rolunu tersine cevirir - basarisiz destek dirence, basarisiz direnc
destege doner. ICT'nin gozlemine gore bir seviyenin tutmamasi, tuttugu
durumdan daha bilgilendiricidir: orada duran emirler yenilmis demektir.

Zaman damgalari konusunda tek kural: bir FVG ancak ucuncu bar KAPANDIGINDA
bilinebilir. `created_at` bu yuzden bar i'nin damgasidir, i-2'nin degil.
Bunu kaydirmak klasik gelecege bakma hatasidir ve sonuclari muhtesem
gosterip canlida hicbir sey tutturmaz.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Bir FVG sonsuza kadar yasamaz: cok eski bosluklar pratikte anlamini yitirir
# ve aktif liste sisirse tarama yavaslar.
DEFAULT_MAX_AGE = 300


@dataclass
class Gap:
    """Tek bir FVG'nin yasam dongusu.

    top/bottom fiyat sinirlari; yon 'bull' veya 'bear'. Zaman alanlari None
    ise o olay hic yasanmadi.
    """
    index: int
    direction: str
    top: float
    bottom: float
    created_at: pd.Timestamp
    mitigated_at: pd.Timestamp | None = None   # fiyat bosluga ilk dokundu
    inverted_at: pd.Timestamp | None = None    # bosluk tamamen gecildi -> IFVG
    inverted_index: int | None = None
    expired_at: pd.Timestamp | None = None     # yaslandi, hic ters cevrilmedi

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


def find_gaps(bars: pd.DataFrame, min_size_pct: float = 0.0) -> list[Gap]:
    """Uc barlik FVG oruntulerini bulur (vektorize tarama).

    `min_size_pct`: boslugu fiyatin yuzdesi olarak minimum buyukluk. Sifir
    birakilirsa bir tick'lik gurultu bosluklari da sayilir; day trade'de
    bunlarin cogu spread'in altinda kalir, yani islem edilemez.
    """
    if len(bars) < 3:
        return []

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    stamps = bars.index

    # i ekseni: 2..n-1. Karsilastirma i-2 ile i arasinda.
    bull = high[:-2] < low[2:]
    bear = low[:-2] > high[2:]

    gaps: list[Gap] = []
    for offset in np.nonzero(bull | bear)[0]:
        i = int(offset) + 2
        if bull[offset]:
            bottom, top, direction = high[i - 2], low[i], "bull"
        else:
            bottom, top, direction = high[i], low[i - 2], "bear"

        if min_size_pct > 0.0:
            if (top - bottom) / close[i] * 100.0 < min_size_pct:
                continue

        gaps.append(Gap(index=i, direction=direction, top=float(top),
                        bottom=float(bottom), created_at=stamps[i]))

    return gaps


def track_lifecycle(bars: pd.DataFrame, gaps: list[Gap],
                    max_age: int = DEFAULT_MAX_AGE) -> list[Gap]:
    """Her boslugun dokunulma / ters cevrilme / yaslanma anlarini damgalar.

    Tek gecisli: her bar'da yalnizca o an AKTIF olan bosluklar kontrol edilir.
    Her bosluk icin tum gelecegi ayri ayri taramak O(n*m) olurdu - 220 bin
    barlik bir veri setinde bu dakikalar surer.

    Ters cevrilme KAPANIS fiyatiyla olculuyor, fitille degil: bir fitilin
    boslugu delip geri donmesi supurme davranisidir, rol degisimi degil.
    Bu ayrim modelin tamaminin dayandigi noktadir.
    """
    if not gaps:
        return gaps

    by_creation: dict[int, list[Gap]] = {}
    for g in gaps:
        by_creation.setdefault(g.index, []).append(g)

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    stamps = bars.index

    active: list[Gap] = []

    for i in range(len(bars)):
        still: list[Gap] = []
        for g in active:
            if i - g.index > max_age:
                g.expired_at = stamps[i]
                continue

            # Dokunma: bar araligi bosluk bolgesiyle kesisti mi.
            if g.mitigated_at is None and low[i] <= g.top and high[i] >= g.bottom:
                g.mitigated_at = stamps[i]

            # Ters cevrilme: kapanis boslugun tamamen ters tarafinda.
            if g.direction == "bull" and close[i] < g.bottom:
                g.inverted_at, g.inverted_index = stamps[i], i
                continue
            if g.direction == "bear" and close[i] > g.top:
                g.inverted_at, g.inverted_index = stamps[i], i
                continue

            still.append(g)

        active = still
        # Bu barda olusan bosluklar ancak SONRAKI barlarda test edilebilir:
        # ucuncu barin kendisi boslugu olusturan bar, onu kendi kapanisiyla
        # ters cevirmesi mantiksiz olurdu.
        active.extend(by_creation.get(i, []))

    return gaps


def to_frame(gaps: list[Gap]) -> pd.DataFrame:
    """Bosluk listesini incelenebilir bir tabloya cevirir."""
    if not gaps:
        return pd.DataFrame(columns=["direction", "top", "bottom", "size", "created_at",
                                     "mitigated_at", "inverted_at", "expired_at"])
    return pd.DataFrame([{
        "direction": g.direction, "top": g.top, "bottom": g.bottom, "size": g.size,
        "created_at": g.created_at, "mitigated_at": g.mitigated_at,
        "inverted_at": g.inverted_at, "expired_at": g.expired_at,
    } for g in gaps])


def inverse_gaps(gaps: list[Gap]) -> list[Gap]:
    """Yalnizca ters cevrilmis bosluklar - IFVG modelinin ham malzemesi.

    Ters cevrilmis bir bullish bosluk artik DIRENC bolgesidir (ve tersi):
    yon alani hala orijinal olusumu anlatir, ticaret yonu onun tersidir.
    Bu karisikligi onlemek icin `signal_direction()` kullanilmali.
    """
    return [g for g in gaps if g.inverted_at is not None]


def signal_direction(gap: Gap) -> str:
    """Ters cevrilmis boslugun islem yonu (olusum yonunun tersi)."""
    if gap.inverted_at is None:
        raise ValueError("bosluk ters cevrilmemis - IFVG sinyali yok.")
    return "short" if gap.direction == "bull" else "long"
