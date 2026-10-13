"""Seans / killzone testleri.

Asil derdi: yaz saati. UTC'ye sabit saat yazan bir uygulama bu testlerin
yarisinda dusuyor - ve gercek veride sessizce yanlis pencerede islem aciyor.
"""
import pandas as pd

from signals.sessions import (in_killzone, in_session, map_daily_to_bars,
                              previous_day_levels, session_range)


def hourly(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC")


def test_newyork_killzone_shifts_with_dst():
    """NY killzone yerel 08:00-11:00. Yazin 12:00 UTC, kisin 13:00 UTC'de baslar.

    Ayni UTC saatinin yilin iki farkli gununde farkli cevap vermesi
    GEREKIYOR - bu testin butun amaci o.
    """
    yaz = pd.DatetimeIndex(["2026-07-15 12:00"], tz="UTC")    # yerel 08:00 EDT
    kis = pd.DatetimeIndex(["2026-01-15 12:00"], tz="UTC")    # yerel 07:00 EST

    assert bool(in_session(yaz, "newyork").iloc[0]) is True
    assert bool(in_session(kis, "newyork").iloc[0]) is False

    kis_ic = pd.DatetimeIndex(["2026-01-15 13:00"], tz="UTC")  # yerel 08:00 EST
    assert bool(in_session(kis_ic, "newyork").iloc[0]) is True


def test_london_session_window():
    idx = hourly("2026-07-15 05:00", 8)          # UTC 05..12, yazin Londra = UTC+1
    mask = in_session(idx, "london")             # yerel 07:00-10:00 -> UTC 06:00-09:00
    inside = [str(t.hour) for t, m in zip(idx, mask) if m]
    assert inside == ["6", "7", "8"]


def test_killzone_is_union_of_zones():
    idx = hourly("2026-07-15 00:00", 24)
    both = in_killzone(idx, ["london", "newyork"])
    lon = in_session(idx, "london")
    ny = in_session(idx, "newyork")
    assert (both == (lon | ny)).all()
    assert both.sum() == 6      # 3 saat Londra + 3 saat NY, ortusmuyor


def test_naive_index_is_treated_as_utc():
    naive = pd.date_range("2026-07-15 12:00", periods=1, freq="1h")
    assert bool(in_session(naive, "newyork").iloc[0]) is True


def test_session_range_groups_by_local_day():
    """Asya seansi UTC gununu boler; gruplama UTC'ye gore yapilirsa iki ayri
    gunun yarilari birlesir ve aralik oldugundan genis cikar."""
    idx = pd.date_range("2026-07-14 00:00", periods=48, freq="1h", tz="UTC")
    bars = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }, index=idx)
    rng = session_range(bars, "asia")
    assert len(rng) == 2                    # iki ayri Tokyo gunu
    assert (rng["high"] == 101.0).all()


def test_previous_day_levels_do_not_leak_today():
    idx = pd.date_range("2026-07-13 13:00", periods=72, freq="1h", tz="UTC")
    bars = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }, index=idx)
    # Ikinci gunun high'ini belirgin sekilde yukselt.
    day2 = (bars.index.tz_convert("America/New_York").date
            == pd.Timestamp("2026-07-14").date())
    bars.loc[day2, "high"] = 150.0

    levels = previous_day_levels(bars)
    mapped = map_daily_to_bars(bars, levels)

    # 14 Temmuz barlarinin PDH'i 150 OLMAMALI - o o gunun kendi high'i.
    on_day2 = mapped.loc[day2, "pdh"].dropna()
    assert (on_day2 != 150.0).all()

    # 15 Temmuz barlari ise 150'yi onceki gun seviyesi olarak gormeli.
    day3 = (bars.index.tz_convert("America/New_York").date
            == pd.Timestamp("2026-07-15").date())
    assert (mapped.loc[day3, "pdh"].dropna() == 150.0).all()
