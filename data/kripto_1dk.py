"""
Kripto 1 dakikalik veri - Binance spot klines, 20 coin, 2023-01 -> 2026-08.

Mum ici sirayi 1 dk ile cozmek ve cok pariteli kripto testleri icin.

Aylik parca, sikistirilmis CSV: data/cache_binance_1m/<SEMBOL>/<YYYY-MM>.csv.gz
Yarida kesilirse kaldigi aydan devam eder (biten ay dosyasi varsa atlanir).
Binance ucretsiz ve anahtar istemez; istek basina 1000 mum, ~0,2 sn ara.

Kullanim:  python -m data.kripto_1dk            (hepsi)
           python -m data.kripto_1dk SOL ADA    (secili)
           yukle("SOL") -> 1 dk DataFrame (UTC, open/high/low/close/volume)
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache_binance_1m")
URL = "https://api.binance.com/api/v3/klines"
BAS, SON = "2023-01", "2026-08"
COINLER = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT",
           "LTC", "TRX", "NEAR", "ATOM", "APT", "ARB", "OP", "SUI", "INJ", "UNI"]
ARA = 0.2


def _ay_indir(sym: str, ay: pd.Timestamp) -> pd.DataFrame:
    t0 = int(ay.timestamp() * 1000)
    t1 = int((ay + pd.offsets.MonthBegin(1)).timestamp() * 1000)
    rows = []
    while t0 < t1:
        q = f"{URL}?symbol={sym}&interval=1m&startTime={t0}&endTime={t1 - 1}&limit=1000"
        for deneme in range(8):
            try:
                req = urllib.request.Request(q, headers={"User-Agent": "Mozilla/5.0"})
                batch = json.load(urllib.request.urlopen(req, timeout=30))
                break
            except Exception:
                time.sleep(2 ** deneme)
        else:
            raise RuntimeError(f"{sym} {ay:%Y-%m}: indirilemedi")
        if not batch:
            break
        rows.extend(b[:6] for b in batch)
        t0 = batch[-1][0] + 60_000
        time.sleep(ARA)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df


def indir(coin: str) -> int:
    sym = f"{coin}USDT"
    klasor = os.path.join(CACHE, sym)
    os.makedirs(klasor, exist_ok=True)
    toplam = 0
    for ay in pd.date_range(BAS, SON, freq="MS", tz="UTC"):
        yol = os.path.join(klasor, f"{ay:%Y-%m}.csv.gz")
        if os.path.exists(yol):
            continue
        df = _ay_indir(sym, ay)
        df.to_csv(yol + ".tmp", index=False, compression="gzip")
        os.replace(yol + ".tmp", yol)
        toplam += len(df)
    return toplam


def yukle(coin: str) -> pd.DataFrame:
    klasor = os.path.join(CACHE, f"{coin}USDT")
    parts = [pd.read_csv(os.path.join(klasor, f)) for f in sorted(os.listdir(klasor)) if f.endswith(".csv.gz")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df.astype(float)[["open", "high", "low", "close", "volume"]]


def main(coinler):
    bitti = []
    for c in coinler:
        t = time.time()
        try:
            n = indir(c)
            bitti.append(c)
            print(f"{c}: {n:,} yeni mum, {time.time() - t:.0f} sn", flush=True)
        except Exception as exc:
            print(f"{c}: HATA {exc}", flush=True)
    print(f"bitti: {len(bitti)}/{len(coinler)} coin", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or COINLER)
