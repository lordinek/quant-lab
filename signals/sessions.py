"""
Seans ve killzone filtreleri.

SMC/ICT modellerinin tamami seans yapisina dayanir: Asya aralikta birikir,
Londra o araligin bir tarafini suprur, New York gercek hareketi yapar. Bu
yuzden "saat kac" bu stratejide bir filtre degil, modelin cekirdegi.

KRITIK: killzone'lar UTC'de SABIT DEGIL. New York borsasi 09:30 YEREL saatte
acilir; yaz saatinde bu 13:30 UTC, kis saatinde 14:30 UTC. UTC'ye sabit saat
yazmak yilin yarisinda killzone'u bir saat kaydirir - ve bu hata hicbir yerde
hata olarak gorunmez, sadece sonuclari sessizce bozar. Bu yuzden seanslar
YEREL saat diliminde tanimlanip pandas'in tz donusumuyle uygulaniyor.
"""
import pandas as pd

# Seanslar (yerel saat, baslangic dahil - bitis haric).
# Kaynak saat dilimi DST'yi kendi hallediyor.
SESSIONS = {
    # Asya: likiditenin biriktigi dar aralik. Tokyo saati.
    "asia": ("Asia/Tokyo", 9, 15),
    # Londra killzone: gunun ilk supurme hareketi genelde burada.
    "london": ("Europe/London", 7, 10),
    # New York killzone: ABD acilisi etrafindaki en yogun pencere.
    "newyork": ("America/New_York", 8, 11),
    # NY ogleden sonra: ters donuslerin sik gorundugu ikinci pencere.
    "newyork_pm": ("America/New_York", 13, 16),
}


def _local_hours(index: pd.DatetimeIndex, tz: str) -> pd.Series:
    """Bar zaman damgalarinin ilgili sehirdeki yerel saati (DST dahil)."""
    if index.tz is None:
        index = index.tz_localize("UTC")
    return pd.Series(index.tz_convert(tz).hour, index=index)


def in_session(index: pd.DatetimeIndex, session: str) -> pd.Series:
    """Her bar icin: bu bar `session` penceresinin icinde mi (bool seri)."""
    if session not in SESSIONS:
        raise ValueError(f"bilinmeyen seans: {session} (secenekler: {list(SESSIONS)})")
    tz, start, end = SESSIONS[session]
    hours = _local_hours(index, tz)
    return (hours >= start) & (hours < end)


def in_killzone(index: pd.DatetimeIndex, zones: list[str] | None = None) -> pd.Series:
    """Verilen killzone'lardan HERHANGI birinin icinde mi.

    Varsayilan Londra + New York: ICT modellerinin islem yaptigi iki pencere.
    """
    zones = zones or ["london", "newyork"]
    mask = pd.Series(False, index=index)
    for z in zones:
        mask |= in_session(index, z)
    return mask


def session_range(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    """Her islem gunu icin bir seansin high/low araligi.

    Bu aralik SMC'de likidite havuzudur: Asya araliginin ustu ve altinda
    stop emirleri birikir, sonraki seanslar oralari supurur.

    Gun ayrimi seansin KENDI saat diliminde yapiliyor - Asya seansi UTC
    gununu ortadan boler, UTC gunune gore gruplamak iki ayri gunun
    yarilarini birlestirir.
    """
    tz, _, _ = SESSIONS[session]
    mask = in_session(bars.index, session)
    inside = bars.loc[mask]
    if inside.empty:
        return pd.DataFrame(columns=["high", "low"])

    local_day = inside.index.tz_convert(tz).date
    grouped = inside.groupby(local_day).agg(high=("high", "max"), low=("low", "min"))
    grouped.index = pd.to_datetime(grouped.index)
    return grouped


def previous_day_levels(bars: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Her gun icin bir ONCEKI islem gununun high/low'u (PDH/PDL).

    SMC'nin en cok kullanilan likidite seviyeleri. `shift(1)` sart:
    o gunun kendi high/low'unu kullanmak gelecege bakmaktir.
    """
    local_day = bars.index.tz_convert(tz).date
    daily = bars.groupby(local_day).agg(high=("high", "max"), low=("low", "min"))
    daily.index = pd.to_datetime(daily.index)
    prev = daily.shift(1)
    return prev.rename(columns={"high": "pdh", "low": "pdl"}).dropna()


def map_daily_to_bars(bars: pd.DataFrame, daily: pd.DataFrame,
                      tz: str = "America/New_York") -> pd.DataFrame:
    """Gunluk seviyeleri bar seviyesine yayar (her bar o gunun seviyelerini tasir)."""
    local_day = pd.to_datetime(bars.index.tz_convert(tz).date)
    out = daily.reindex(local_day)
    out.index = bars.index
    return out
