"""
Haber penceresi filtresi: makro veri anlarinda islem yapmamak.

NEDEN VAR. `backtest/slippage_tick.py` 15 islem gununde 9.588 stop
tetiklenmesi olctu. Kayma normal zamanda yok sayilabilir (medyan 0,29 puan,
%90'i 1,27) ama EN KOTU 10 olcum uc ayri gune ait ve ucu de ayni saniye
araliginda:

    2026-09-03 12:30:04 UTC  -> 76,8 puan kayma, tick akisinda 3,7 sn bosluk
    2026-09-04 12:30:10 UTC  -> 80,8 puan kayma, 9,0 sn bosluk
    2026-09-11 12:30:10 UTC  -> 88,5 puan kayma, 10,4 sn bosluk

12:30 UTC = 08:30 ET: ABD makro veri saati (tarim disi istihdam, TUFE, UFE,
haftalik issizlik basvurulari). Piyasa birkac saniye DONUYOR, sonra 80-130
puan atliyor. Stop emri o bosluga denk gelirse istedigin yerden degil,
bosluktan sonraki ilk fiyattan dolar.

88 puan = modelin stop mesafesinin (19,7 puan medyan) 4,5 kati, yani tek
islemde 4,5 R. Kendi hesabinda can yakar; PROP hesabinda oldurur: %1 riskte
$2.250, Topstep'in gunluk limiti $1.000, Apex'in yastigi $2.500.

SAAT DILIMI TUZAGI. Pencereler YEREL saatte (New York) tanimli, UTC'de DEGIL.
08:30 ET yaz saatinde 12:30 UTC, kis saatinde 13:30 UTC'dir. UTC'ye sabit
saat yazmak filtreyi yilin yarisinda bir saat kaydirir ve bu hata hicbir
yerde hata olarak gorunmez - `sessions.py`'deki killzone dersinin aynisi.

TAKVIM YOK, SAAT VAR. Dogrusu gercek bir haber takvimi (Forex Factory vb.)
baglamak olurdu: o zaman yalnizca veri ACIKLANAN gunler kapatilir. Elimizde
2021-2026'yi kapsayan bir takvim yok, bu yuzden pencere HER GUN kapali
tutuluyor. Bu daha temkinli ama daha pahali - maliyeti olculdu, bkz.
`backtest/news_filter_test.py`.
"""
import pandas as pd

# (yerel saat, dakika, once_dk, sonra_dk) - saat dilimi NEWS_TZ.
#
# 08:30 : ABD makro veri blogu. Olculen felaketlerin tamami burada.
# 10:00 : ISM, tuketici guveni, JOLTS. Olcumde saat 14 UTC'de 22,3 puanlik
#         en kotu deger bu pencereye denk geliyor - ilkinden kucuk ama var.
# 14:00 : FOMC karar saati. Yalnizca FOMC gunlerinde anlamli; takvimimiz
#         olmadigi icin varsayilan olarak KAPALI (asagida yorum satiri).
#         Her gun kapatmak 14:00 ET sonrasi NY seansini biciyor.
NEWS_TZ = "America/New_York"
NEWS_WINDOWS = [
    (8, 30, 5, 15),
    (10, 0, 5, 10),
    # (14, 0, 5, 30),   # FOMC - takvim baglaninca acilmali
]


def in_news_window(index: pd.DatetimeIndex,
                   windows: list | None = None) -> pd.Series:
    """Her zaman damgasi icin: haber penceresinin icinde mi (bool seri).

    Pencere [saat:dakika - once_dk, saat:dakika + sonra_dk) araligidir.
    Baslangic dahil, bitis haric - `sessions.in_session` ile ayni kural.
    """
    windows = NEWS_WINDOWS if windows is None else windows
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert(NEWS_TZ)
    # Gun icindeki dakika: DST donusumu tz_convert tarafindan zaten yapildi.
    minute_of_day = pd.Series(local.hour * 60 + local.minute, index=index)

    mask = pd.Series(False, index=index)
    for hh, mm, before, after in windows:
        center = hh * 60 + mm
        mask |= (minute_of_day >= center - before) & (minute_of_day < center + after)
    return mask


def next_news_time(ts: pd.Timestamp, windows: list | None = None) -> pd.Timestamp:
    """`ts`'den sonraki ilk haber penceresinin BASLANGICI.

    Acik pozisyonu haberden once kapatmak icin gerekiyor: "kac dakika sonra
    pencere aciliyor" sorusunun cevabi bu. Pencere zaten aciksa `ts` doner.
    """
    windows = NEWS_WINDOWS if windows is None else windows
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(NEWS_TZ)

    best = None
    for day_offset in (0, 1):
        day = (local + pd.Timedelta(days=day_offset)).normalize()
        for hh, mm, before, _after in windows:
            start = day + pd.Timedelta(hours=hh, minutes=mm - before)
            if start >= local and (best is None or start < best):
                best = start
    return best.tz_convert(ts.tz) if best is not None else None
