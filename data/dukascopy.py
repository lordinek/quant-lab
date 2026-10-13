"""
Dukascopy tick verisinden 1 dakikalik bar uretir. Ucretsiz, anahtar gerekmez,
NAS100 ve forex majorlerinde 2000'li yillara kadar gecmis var.

Neden yfinance degil: yfinance 5 dakikalik veride sadece 60 gun veriyor. SMC/
IFVG bir day trade modeli, 5-15 dakikalik barda calisir - 60 gunle kaydirmali
pencere testi yapilamaz, yani stratejinin farkli rejimlerde ayakta kaldigi
KANITLANAMAZ. Altin ve BTC projelerinde asil degeri o kanit uretmisti.

bi5 formati: LZMA sikistirilmis, her tick 20 byte big-endian:
    uint32 ms_offset (saat basindan), uint32 ask, uint32 bid,
    float32 ask_vol, float32 bid_vol
Fiyatlar tamsayi: gercek fiyat = deger / 10**decimals.

URL'de AY 0-INDEKSLI (Ocak = 00). Klasik tuzak, iki kez yakalanmayalim.

RATE LIMIT birinci dereceden kisit: hizli ardisik istek atinca Dukascopy once
503 doner, israr edilirse baglantiyi komple keser (WinError 10054). Olculdu,
tahmin degil. Bu yuzden istekler arasi adaptif bekleme + ustel geri cekilme
var ve her gun diske yaziliyor - indirme yarida kesilirse kaldigi gunden
devam eder, bastan baslamaz.
"""
import io
import lzma
import os
import struct
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
TICK_SIZE = 20
USER_AGENT = "Mozilla/5.0"

# Istekler arasi bekleme saniyesi. Basari serisinde tabana dogru iner,
# 503 gorunce tavana dogru cikar.
PAUSE_MIN = 0.4
PAUSE_MAX = 20.0
PAUSE_START = 0.6
MAX_RETRY = 5

# Ban yendiginde ISRAR ETMEK BANI UZATIYOR (olculdu: 28 hizli istekten sonra
# EURUSD dahil her sembol kapandi, art arda deneme acmadi). Dogru davranis
# geri cekilip uzun bir mola vermek.
COOLDOWN = 600
MAX_COOLDOWN_ROUNDS = 6

# Dukascopy fiyatlari tamsayi tutar; olcek enstrumana gore degisir.
# Yanlis olcek sessizce 100x yanlis fiyat uretir - sanity_check() ile dogrula.
DECIMALS = {
    "EURUSD": 5,
    "GBPUSD": 5,
    "AUDUSD": 5,
    "USDCHF": 5,
    "USDCAD": 5,
    "USDJPY": 3,
    "EURJPY": 3,
    "GBPJPY": 3,
    # Index CFD'leri 3 haneli. Once 5 varsayilmisti ve fiyatlar 291.5 cikti
    # (gercegi ~29.150) - tam da uyarilan hata: yanlis olcek HICBIR hata
    # vermez, sadece her fiyati 100 kat kaydirir. sanity_check() bu yuzden var.
    "USATECHIDXUSD": 3,   # NAS100 - dogrulandi 2026-09-08
    "USA500IDXUSD": 3,    # S&P 500
    "USA30IDXUSD": 3,     # Dow 30
    "DEUIDXEUR": 3,       # DAX
}


class RateLimited(Exception):
    """Dukascopy kapiyi kapatti. Cagiran uzun bir mola vermeli."""


def _hour_url(symbol: str, when: datetime) -> str:
    return (f"{BASE_URL}/{symbol}/{when.year}/{when.month - 1:02d}/"
            f"{when.day:02d}/{when.hour:02d}h_ticks.bi5")


def _fetch_hour(symbol: str, when: datetime, pause: float) -> tuple[bytes, float]:
    """Bir saatlik ham bi5 icerigi ve bir sonraki istek icin onerilen bekleme.

    Bos icerik (b"") gecerli bir cevaptir: piyasa o saat kapaliydi.
    """
    url = _hour_url(symbol, when)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(MAX_RETRY):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            # Basari: baskiyi yavasca azalt.
            return raw, max(PAUSE_MIN, pause * 0.9)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Dosya yok = o saat islem olmamis. Normal.
                return b"", max(PAUSE_MIN, pause * 0.9)
            if exc.code != 503:
                raise
        except Exception:
            pass   # timeout / baglanti kesilmesi - rate limit'in diger yuzu

        pause = min(PAUSE_MAX, max(pause, 1.0) * 2)
        time.sleep(pause * (attempt + 1))

    raise RateLimited(f"{symbol} {when:%Y-%m-%d %H}h: {MAX_RETRY} denemede alinamadi.")


def _parse_ticks(raw: bytes, when: datetime, decimals: float) -> pd.DataFrame:
    """Ham bi5'i tick tablosuna cevirir. Bos girdi bos tablo doner."""
    if not raw:
        return pd.DataFrame(columns=["ts", "bid", "ask", "volume"])

    data = lzma.decompress(raw)
    n = len(data) // TICK_SIZE
    if n == 0:
        return pd.DataFrame(columns=["ts", "bid", "ask", "volume"])

    scale = 10.0 ** decimals
    rows = []
    for i in range(n):
        ms, ask, bid, ask_v, bid_v = struct.unpack_from(">IIIff", data, i * TICK_SIZE)
        rows.append((when + timedelta(milliseconds=ms), bid / scale, ask / scale,
                     ask_v + bid_v))

    return pd.DataFrame(rows, columns=["ts", "bid", "ask", "volume"])


def _ticks_to_minutes(ticks: pd.DataFrame) -> pd.DataFrame:
    """Tick'leri 1 dakikalik OHLCV + ortalama spread'e indirger.

    Spread'i saklamak bilincli: BTC projesinde islem maliyetinin stratejiyi
    oldurebildigini gorduk. Day trade'de maliyet daha da belirleyici - sonradan
    tahmin etmek yerine gercek olculen spread'i tasiyoruz.

    Fiyat olarak mid (bid+ask)/2 kullaniliyor; maliyet ayri kolonda duruyor,
    boylece simulator giris/cikis tarafini kendi modelleyebilir.
    """
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"])

    df = ticks.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spr"] = df["ask"] - df["bid"]
    df = df.set_index("ts")

    out = df["mid"].resample("1min").ohlc()
    out["volume"] = df["volume"].resample("1min").sum()
    out["spread"] = df["spr"].resample("1min").mean()
    return out.dropna(subset=["open"])


def _hour_cache_path(symbol: str, when: datetime) -> str:
    """Cache birimi SAAT, gun degil.

    Gun birimi denendi ve yanlis cikti: gunun 15. saatinde ban yenince o gune
    ait indirilmis 15 saatin tamami cope gidiyordu, cunku dosya ancak gun
    tamamlaninca yaziliyordu. Saat biriminde ilerleme hicbir zaman kaybolmaz.

    Klasor yil/ay diye bolunuyor: 3 yillik veri ~26 bin dosya, hepsini tek
    klasore koymak Windows'ta dizin islemlerini yavaslatir.
    """
    return os.path.join(CACHE_DIR, symbol, f"{when:%Y}", f"{when:%m}",
                        f"{when:%d}-{when:%H}.csv")


def _load_hour_cache(path: str) -> pd.DataFrame:
    """Cache'lenmis saatlik barlar. Bos dosya = o saat piyasa kapaliydi."""
    if os.path.getsize(path) <= 1:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"])
    return pd.read_csv(path, index_col=0, parse_dates=True)


def download_hour(symbol: str, when: datetime, pause: float = PAUSE_START,
                  ) -> tuple[pd.DataFrame, float]:
    """Bir saatin 1 dakikalik barlari, diske cache'lenir.

    Bos saatler (hafta sonu, seans disi) de bos dosya olarak yazilir - yoksa
    her calistirmada bosuna tekrar sorgulanirlar.

    Ban yenirse mola verilip AYNI saat tekrar denenir. Saati atlamak veride
    sessiz bir delik birakir; o delik backtest'te hicbir yerde hata olarak
    gorunmez, sadece sonuclari bozar.
    """
    path = _hour_cache_path(symbol, when)
    if os.path.exists(path):
        return _load_hour_cache(path), pause

    ticks, pause = _hour_ticks(symbol, when, pause)
    bars = _ticks_to_minutes(ticks)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    bars.to_csv(path)
    time.sleep(pause)
    return bars, pause


def _hour_ticks(symbol: str, when: datetime, pause: float) -> tuple[pd.DataFrame, float]:
    """Bir saatin tick tablosu (indirme + ban molasi mantigi burada tek yerde).

    Onceden bu blok `download_hour` icindeydi; tick tarafi eklenince ayni
    mantigin ikinci kopyasi cikacakti. Ban davranisi iki yerde ayri ayri
    bakim istemesin diye buraya alindi.
    """
    decimals = DECIMALS.get(symbol)
    if decimals is None:
        raise ValueError(f"{symbol} icin fiyat olcegi tanimli degil - DECIMALS'a ekle.")

    for round_no in range(MAX_COOLDOWN_ROUNDS):
        try:
            raw, pause = _fetch_hour(symbol, when, pause)
            break
        except RateLimited:
            if round_no == MAX_COOLDOWN_ROUNDS - 1:
                raise
            print(f"  ! ban: {when:%Y-%m-%d %H}h - {COOLDOWN // 60} dk mola "
                  f"({round_no + 1}/{MAX_COOLDOWN_ROUNDS})", flush=True)
            time.sleep(COOLDOWN)
            pause = PAUSE_START

    return _parse_ticks(raw, when, decimals), pause


TICK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_ticks")


def download_hour_ticks(symbol: str, when: datetime,
                        pause: float = PAUSE_START) -> tuple[pd.DataFrame, float]:
    """Bir saatin HAM TICK'leri: ts, bid, ask, volume.

    NEDEN AYRI BIR YOL: `download_hour` tick'i indirip hemen 1 dakikalik bara
    indirgiyor ve tick'i atiyor. Spread olcumu icin bu yetiyordu (ortalama
    spread kolonu bara tasiniyor), ama KAYMA olculemiyor: kayma stop'un
    tetiklendigi tick ile emrin doldugu tick arasindaki farktir - saniye-alti
    bir olay. Dakikalik bar o pencereyi hic gostermez.

    Cache ayri klasorde ve pickle olarak tutuluyor: tick dosyalari CSV'den
    cok daha buyuk, ve bu veri yalnizca kayma olcumu icin gerekiyor - normal
    backtest yolunu agirlastirmamali.
    """
    path = os.path.join(TICK_CACHE_DIR, symbol, f"{when:%Y}", f"{when:%m}",
                        f"{when:%d}-{when:%H}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path), pause

    ticks, pause = _hour_ticks(symbol, when, pause)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ticks.to_pickle(path)
    time.sleep(pause)
    return ticks, pause


def download_day(symbol: str, day: date, pause: float = PAUSE_START) -> tuple[pd.DataFrame, float]:
    """Bir gunun 1 dakikalik barlari (24 saatlik cache biriminden birlestirilir)."""
    frames = []
    for hour in range(24):
        when = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        bars, pause = download_hour(symbol, when, pause)
        if not bars.empty:
            frames.append(bars)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"]), pause
    return pd.concat(frames).sort_index(), pause


def _day_is_cached(symbol: str, day: date) -> bool:
    return all(os.path.exists(_hour_cache_path(
        symbol, datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc)))
        for h in range(24))


def download_range(symbol: str, start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """[start, end] arasi 1 dakikalik barlar, gun gun indirilip cache'lenir.

    Bugun DAHIL EDILMEZ: gun henuz kapanmadigi icin yarim veri cache'lenir ve
    bir daha guncellenmez - sessiz bir veri bosluguna donusur.
    """
    first = pd.Timestamp(start).date()
    last = min(pd.Timestamp(end).date(), datetime.now(timezone.utc).date() - timedelta(days=1))

    pause = PAUSE_START
    frames = []
    day = first
    total = (last - first).days + 1
    done = 0

    while day <= last:
        cached = _day_is_cached(symbol, day)
        # Ban yonetimi saat seviyesinde (download_hour icinde) - orada
        # ilerleme kaybolmadan mola verilebiliyor.
        bars, pause = download_day(symbol, day, pause)

        if not bars.empty:
            frames.append(bars)
        done += 1

        if verbose and (not cached or done % 50 == 0):
            mark = "cache" if cached else f"indi ({len(bars):4} bar, bekleme {pause:.1f}s)"
            print(f"[{done:5}/{total}] {symbol} {day:%Y-%m-%d}  {mark}", flush=True)

        day += timedelta(days=1)

    if not frames:
        raise RuntimeError(f"{symbol} icin {start}..{end} araliginda hic veri yok.")
    return pd.concat(frames).sort_index()


def resample(minutes: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1 dakikalik barlari daha buyuk bir zaman dilimine toplar ('5min', '15min', '1h').

    Bosluklar atilir: forex/index'te hafta sonu ve seans disi saatler gercekten
    yoktur, onlari sifir hacimli bar olarak uydurmak backtest'i yaniltir.
    """
    out = minutes.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "spread": "mean",
    })
    return out.dropna(subset=["open"])


def sanity_check(symbol: str, day: str = "2026-09-04") -> None:
    """Fiyat olceginin dogrulugunu insan gozuyle dogrulamak icin.

    Yanlis DECIMALS degeri hicbir hata vermez, sadece tum fiyatlari 100 kat
    kaydirir - ve o hata backtest sonuclarinda cok gec fark edilir.
    """
    bars, _ = download_day(symbol, pd.Timestamp(day).date())
    if bars.empty:
        print(f"{symbol}: {day} gununde veri yok (piyasa kapali olabilir).")
        return
    print(f"{symbol}  {day}  {len(bars)} dakikalik bar")
    print(f"  fiyat araligi : {bars['low'].min():.5f} - {bars['high'].max():.5f}")
    print(f"  ortalama spread: {bars['spread'].mean():.5f} "
          f"({bars['spread'].mean() / bars['close'].mean() * 100:.4f}% fiyatin)")
    print(f"  ilk bar: {bars.index[0]}  son bar: {bars.index[-1]}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("kullanim: python -m data.dukascopy check <SEMBOL> [gun]")
        print("          python -m data.dukascopy get <SEMBOL> <baslangic> <bitis>")
        sys.exit(0)

    if args[0] == "check":
        sanity_check(args[1], args[2] if len(args) > 2 else "2026-09-04")
    elif args[0] == "get":
        df = download_range(args[1], args[2], args[3])
        print(f"\n{args[1]}: {len(df)} dakikalik bar | {df.index[0]} -> {df.index[-1]}")
