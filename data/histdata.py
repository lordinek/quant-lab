"""
HistData.com'dan 1 dakikalik bar verisi. Ucretsiz, kayit gerekmez.

Neden Dukascopy yerine: Dukascopy tick verisini SAATLIK dosyalar halinde
veriyor - 3 yillik NAS100 icin ~26.000 istek demek. Rate limit'e takildik ve
ban inatci cikti (saatte 2 dosya). HistData ayni veriyi AYLIK ZIP olarak
veriyor: ayni donem icin 36 istek. Yaklasik 700 kat az.

Tavizi: HistData M1 barlari SPREAD ICERMEZ (tek taraf fiyat, hacim de
gercek degil - tick sayaci). Islem maliyeti bu yuzden sabit ve konservatif
varsayilmali. BTC projesinde maliyetin stratejiyi oldurebildigini gorduk;
day trade'de etkisi daha da buyuk, o yuzden bu bir "sonra bakariz" konusu
degil - dukascopy.py gercek spread'i olcmek icin duruyor.

ZAMAN DILIMI TUZAGI: HistData damgalari EST'dir ve YAZ SAATI UYGULANMAZ
(sabit UTC-5). Bunu "US/Eastern" sanip pandas'a cevirtmek yilin yarisinda
veriyi bir saat kaydirir - ve killzone'lar tam da o saatlerde tanimli
oldugu icin hata dogrudan sinyallere siçrar. Sabit offset kullaniyoruz.
"""
import io
import os
import re
import time
import zipfile

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_histdata")
PAGE_URL = ("https://www.histdata.com/download-free-forex-historical-data/"
            "?/ascii/1-minute-bar-quotes/{pair}/{year}/{month}")
YEAR_URL = ("https://www.histdata.com/download-free-forex-historical-data/"
            "?/ascii/1-minute-bar-quotes/{pair}/{year}")
POST_URL = "https://www.histdata.com/get.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
REQUEST_PAUSE = 1.5

# HistData damgalari sabit UTC-5 (yaz saati YOK).
EST_OFFSET_HOURS = 5

PAIRS = {
    "NAS100": "nsxusd",     # Nasdaq 100
    "SP500": "spxusd",      # S&P 500
    "DOW": "wsxusd",        # Dow 30
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "XAUUSD": "xauusd",     # altin
}


def _cache_path(pair: str, year: int, month: int | None) -> str:
    if month is None:
        return os.path.join(CACHE_DIR, pair, f"{year}-full.csv")
    return os.path.join(CACHE_DIR, pair, f"{year}-{month:02d}.csv")


def _fetch(pair: str, year: int, month: int | None) -> bytes:
    """Bir ayin (veya `month=None` ise tum yilin) ZIP icerigi.

    HistData GECMIS YILLARI TEK DOSYA olarak veriyor, yalnizca son ~20 ayi
    aylik tutuyor. Aylik istek eski bir yil icin yapildiginda sayfada `tk`
    jetonu hic bulunmuyor - ilk indirmede 48 ay bu yuzden atlandi.

    Indirme iki adimli: once sayfadan gizli `tk` jetonu okunur, sonra o
    jetonla POST atilir. Referer basligi ZORUNLU - onsuz sunucu bos doner.
    """
    label = f"{year}" if month is None else f"{year}-{month:02d}"
    page = (YEAR_URL.format(pair=pair, year=year) if month is None
            else PAGE_URL.format(pair=pair, year=year, month=month))
    session = requests.Session()
    session.headers.update(HEADERS)

    resp = session.get(page, timeout=60)
    resp.raise_for_status()

    match = re.search(r'id="tk"\s+value="([^"]+)"', resp.text)
    if not match:
        raise RuntimeError(f"{pair} {label}: sayfada tk jetonu yok "
                           f"(veri mevcut olmayabilir).")

    payload = {
        "tk": match.group(1),
        "date": str(year),
        "datemonth": str(year) if month is None else f"{year}{month:02d}",
        "platform": "ASCII",
        "timeframe": "M1",
        "fxpair": pair.upper(),
    }
    post = session.post(POST_URL, data=payload,
                        headers={"Referer": page}, timeout=300)
    post.raise_for_status()

    if not post.content or len(post.content) < 500:
        raise RuntimeError(f"{pair} {label}: bos/kucuk cevap "
                           f"({len(post.content)} byte).")
    return post.content


def _parse_zip(blob: bytes) -> pd.DataFrame:
    """ZIP icindeki CSV'yi bar tablosuna cevirir.

    Format: YYYYMMDD HHMMSS;open;high;low;close;volume  (ayrac ';')
    Hacim gercek islem hacmi degil tick sayacidir - saklıyoruz ama
    strateji kararlarinda kullanmiyoruz.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("ZIP icinde CSV yok.")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, sep=";", header=None,
                             names=["ts", "open", "high", "low", "close", "volume"])

    ts = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    # Sabit UTC-5 -> UTC. tz_localize("US/Eastern") KULLANMA: HistData yaz
    # saati uygulamaz, o donusum yilin yarisinda bir saat kaydirir.
    df.index = (ts + pd.Timedelta(hours=EST_OFFSET_HOURS)).dt.tz_localize("UTC")
    return df.drop(columns=["ts"]).sort_index()


def download_month(pair: str, year: int, month: int | None = None,
                   force: bool = False) -> pd.DataFrame:
    """Bir ayin (month=None ise tum yilin) 1 dakikalik barlari, cache'lenir."""
    key = PAIRS.get(pair, pair).lower()
    path = _cache_path(key, year, month)

    if os.path.exists(path) and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    bars = _parse_zip(_fetch(key, year, month))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    bars.to_csv(path)
    time.sleep(REQUEST_PAUSE)
    return bars


def download_year(pair: str, year: int, force: bool = False) -> pd.DataFrame:
    """Tum yilin barlari, tek dosyada."""
    return download_month(pair, year, None, force=force)


def download_range(pair: str, start: str, end: str, verbose: bool = True) -> pd.DataFrame:
    """[start, end] arasi butun veri.

    Yil yil ilerliyor ve her yil icin ONCE tek dosyayi deniyor; HistData
    gecmis yillari boyle veriyor ve bu 12 istegi 1'e indiriyor. Yillik dosya
    yoksa (genelde ICINDE BULUNULAN yil) aylara duşuluyor.

    Eksikler sessizce gecilmiyor: veride delik birakmak, backtest'te hicbir
    yerde hata olarak gorunmeyen bir bozulmadir.
    """
    first = pd.Timestamp(start).replace(day=1)
    last = pd.Timestamp(end).replace(day=1)

    frames, missing = [], []
    for year in range(first.year, last.year + 1):
        try:
            bars = download_year(pair, year)
            frames.append(bars)
            if verbose:
                print(f"  {pair} {year} (yillik): {len(bars):7} bar", flush=True)
            continue
        except Exception:
            pass   # yillik yok - aylara dusuluyor

        for month in range(1, 13):
            cursor = pd.Timestamp(year=year, month=month, day=1)
            if cursor < first or cursor > last:
                continue
            try:
                bars = download_month(pair, year, month)
                frames.append(bars)
                if verbose:
                    print(f"  {pair} {cursor:%Y-%m}: {len(bars):7} bar", flush=True)
            except Exception as exc:
                missing.append((f"{cursor:%Y-%m}", str(exc)[:80]))
                if verbose:
                    print(f"  {pair} {cursor:%Y-%m}: ATLANDI - {str(exc)[:60]}", flush=True)

    if not frames:
        raise RuntimeError(f"{pair} icin {start}..{end} araliginda hic veri alinamadi.")

    if missing and verbose:
        print(f"\n  UYARI: {len(missing)} donem eksik - veride delik var:")
        for m, why in missing:
            print(f"    {m}: {why}")

    combined = pd.concat(frames).sort_index()
    # Yillik + aylik dosyalar ayni gunleri icerebilir; tekrarlari at.
    return combined[~combined.index.duplicated(keep="first")]


def resample(minutes: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1 dakikalik barlari daha buyuk zaman dilimine toplar ('5min', '15min', '1h').

    Bosluklar atilir: forex/index'te hafta sonu ve seans disi saatler gercekten
    yoktur, onlari sifir hacimli bar diye uydurmak backtest'i yaniltir.
    """
    out = minutes.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return out.dropna(subset=["open"])


def sanity_check(pair: str, year: int, month: int) -> None:
    """Fiyat araligi ve seans saatleri makul mu - insan gozuyle dogrulama.

    Ozellikle zaman dilimi: NAS100 icin barlarin New York borsa saatlerinde
    yogunlasmasi gerekir. Yogunlugun kaydigi goruluyorsa offset yanlistir.
    """
    bars = download_month(pair, year, month)
    ny_hour = bars.index.tz_convert("America/New_York").hour
    by_hour = pd.Series(1, index=ny_hour).groupby(level=0).sum()

    print(f"{pair} {year}-{month:02d}: {len(bars)} dakikalik bar")
    print(f"  tarih araligi : {bars.index[0]} -> {bars.index[-1]}")
    print(f"  fiyat araligi : {bars['low'].min():.2f} - {bars['high'].max():.2f}")
    print(f"  en yogun NY saatleri: "
          f"{', '.join(f'{h:02d}:00 ({n})' for h, n in by_hour.nlargest(5).items())}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("kullanim: python -m data.histdata check <CIFT> <yil> <ay>")
        print("          python -m data.histdata get <CIFT> <YYYY-MM> <YYYY-MM>")
        sys.exit(0)

    if args[0] == "check":
        sanity_check(args[1], int(args[2]), int(args[3]))
    elif args[0] == "get":
        df = download_range(args[1], args[2], args[3])
        print(f"\n{args[1]}: {len(df)} bar | {df.index[0]} -> {df.index[-1]}")
