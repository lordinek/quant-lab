"""
Likidite havuzlari ve supurme (liquidity sweep / stop hunt) tespiti.

SMC'nin temel iddiasi: fiyat rastgele dolasmaz, EMIRLERIN BIRIKTIGI yerlere
gider. Stop emirleri belirgin tepe ve diplerin hemen otesinde kumelenir -
onceki gunun zirvesi, Asya seansinin dibi, esit tepeler. Buyuk katilimcinin
pozisyon alabilmesi icin karsi tarafta emir gerekir; o emirler oralarda durur.

Supurme (sweep): fiyat boyle bir seviyeyi FITILLE asip KAPANISLA geri doner.
Emirler toplanmis, ama yon devam etmemistir. Bu, IFVG kurulumunun en guclu
tetikleyicisi: supurme + yapinin ters kirilmasi + ters cevrilmis bosluk.

Bu dosyadaki her sey su ayrimin uzerine kurulu:
    fitil seviyeyi asti + kapanis geri dondu  -> SUPURME (donus beklentisi)
    kapanis seviyenin otesinde                -> KIRILMA (devam beklentisi)
Ayni ayrim fvg.py ve structure.py'de de var; ucu birden ayni felsefeden gelir.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.sessions import map_daily_to_bars, previous_day_levels, session_range


@dataclass
class Sweep:
    """Bir likidite seviyesinin supuruldugu an."""
    index: int
    timestamp: pd.Timestamp
    side: str            # 'high' = ustteki likidite alindi -> SHORT egilimi
    level: float
    level_name: str      # 'pdh' | 'pdl' | 'asia_high' | 'asia_low' | 'equal_highs' ...
    penetration: float   # seviyenin ne kadar otesine gidildi (fiyat birimi)


def find_sweeps(bars: pd.DataFrame, levels: pd.DataFrame,
                min_penetration: float = 0.0) -> list[Sweep]:
    """Verilen seviyelerin supurulme anlari.

    `levels`: bars ile ayni indekste, her kolonu bir seviye serisi
    (or. pdh, pdl). NaN olan barlar atlanir.

    Supurme tanimi taraf'a gore:
        ust seviye: high > seviye VE close <= seviye
        alt seviye: low  < seviye VE close >= seviye

    `min_penetration` gurultu filtresi: seviyeyi bir tick asmak supurme
    sayilmamali, spread'in altinda kalir.
    """
    sweeps: list[Sweep] = []
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    stamps = bars.index

    # Vektorize: bar bar Python dongusu 400 bin barlik veride her seviye
    # kolonu icin bastan donmek demekti. Maske bir kerede hesaplanip
    # yalnizca eslesen barlar geziliyor.
    for name in levels.columns:
        series = levels[name].to_numpy(dtype=float)
        valid = ~np.isnan(series)

        if _is_upper_level(name):
            pen = np.where(valid, high - series, np.nan)
            hit = valid & (high > series) & (close <= series) & (pen >= min_penetration)
            side = "high"
        else:
            pen = np.where(valid, series - low, np.nan)
            hit = valid & (low < series) & (close >= series) & (pen >= min_penetration)
            side = "low"

        for i in np.nonzero(hit)[0]:
            sweeps.append(Sweep(int(i), stamps[i], side, float(series[i]),
                                name, float(pen[i])))

    sweeps.sort(key=lambda s: s.index)
    return sweeps


def _is_upper_level(name: str) -> bool:
    """Seviye ismi ust likiditeyi mi gosteriyor.

    Isim tabanli, cunku seviyeler farkli kaynaklardan geliyor (gunluk, seans,
    esit tepeler) ve hepsinin tek ortak imzasi ismi.
    """
    lowered = name.lower()
    if lowered.endswith("high") or lowered == "pdh" or "highs" in lowered:
        return True
    if lowered.endswith("low") or lowered == "pdl" or "lows" in lowered:
        return False
    raise ValueError(f"seviye adindan taraf anlasilmiyor: {name} "
                     f"(ad 'high'/'low' ile bitmeli veya pdh/pdl olmali)")


def build_levels(bars: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Standart likidite seviyelerini tek tabloda toplar.

    Iceri girenler:
      pdh / pdl             - onceki islem gununun zirve ve dibi
      asia_high / asia_low  - Asya seansi araligi (o gunun kendi seansi)

    Asya araligi AYNI gunun seansidir ve Londra/NY killzone'larindan ONCE
    biter - yani o seviyeler kullanildiginda çoktan bilinmektedir, gelecege
    bakma yok. Ama Asya seansi SIRASINDA bu seviyeler henuz tamamlanmamistir;
    bu yuzden Asya saatlerindeki barlarda kullanilmamalidir.
    """
    daily = previous_day_levels(bars, tz=tz)
    levels = map_daily_to_bars(bars, daily, tz=tz)

    asia = session_range(bars, "asia")
    if not asia.empty:
        asia = asia.rename(columns={"high": "asia_high", "low": "asia_low"})
        # Asya araligi Tokyo gunune gore hesaplandi; barlara ayni gun uzerinden
        # eslesiyor.
        mapped = map_daily_to_bars(bars, asia, tz="Asia/Tokyo")
        levels = levels.join(mapped)

    return levels


def equal_levels(bars: pd.DataFrame, tolerance_pct: float = 0.02,
                 lookback: int = 50, min_touches: int = 2) -> pd.DataFrame:
    """Esit tepe/dip bolgeleri (equal highs / equal lows).

    Ayni seviyeye birden fazla dokunus, oranin altinda stop birikimi oldugunun
    en gorunur isaretidir - "cift tepe" klasik teknik analizde donus sinyali
    sayilir, SMC'de ise supurulecek hedeftir.

    Kaydirmali pencere kullaniliyor: her bar icin YALNIZCA gecmis `lookback`
    bar taraniyor, boylece gelecege bakma imkansiz.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(bars)

    eq_high = pd.Series(float("nan"), index=bars.index)
    eq_low = pd.Series(float("nan"), index=bars.index)

    for i in range(lookback, n):
        window_h = high[i - lookback:i]
        window_l = low[i - lookback:i]

        peak = window_h.max()
        tol = peak * tolerance_pct / 100.0
        if (window_h >= peak - tol).sum() >= min_touches:
            eq_high.iloc[i] = peak

        trough = window_l.min()
        tol = trough * tolerance_pct / 100.0
        if (window_l <= trough + tol).sum() >= min_touches:
            eq_low.iloc[i] = trough

    return pd.DataFrame({"equal_highs": eq_high, "equal_lows": eq_low})


def to_frame(sweeps: list[Sweep]) -> pd.DataFrame:
    if not sweeps:
        return pd.DataFrame(columns=["timestamp", "side", "level", "level_name",
                                     "penetration"])
    return pd.DataFrame([{
        "timestamp": s.timestamp, "side": s.side, "level": s.level,
        "level_name": s.level_name, "penetration": s.penetration,
    } for s in sweeps])
