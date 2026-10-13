"""
Model enstrümana mi ozel, yoksa genel mi?

NAS100'de guclu sonuc cikti. Ama tek bir enstrümanda iyi calisan bir kural,
o enstrümanin o donemdeki huyuna uydurulmus da olabilir. SMC/ICT iddiasi
piyasa MIKROYAPISI hakkindadir - likidite avi, emir kumeleri, seans yapisi -
ve bu iddia dogruysa model baska likit enstrümanlarda da calismali.

Bu, "NAS100 ve forex daha kesin sonuc verir" hipotezinin
dogrudan sinavi.

Karsilastirma YUZDE bazli maliyet ve yuzde bazli risk esigi ile yapiliyor:
puan cinsinden esikler yalnizca tek bir enstrüman icin anlamlidir (4 puan
NAS100'de ~%0.014, EURUSD 1.16'da saçma bir rakam).
"""
import pandas as pd

from backtest.evaluate import evaluate, max_drawdown_r, summarize
from data.histdata import download_range, resample
from strategy.setup import find_setups

# Gercekci gidis-donus maliyet varsayimlari (fiyatin yuzdesi).
# NAS100 icin olculdu: spread 0.93-1.43 puan / ~29.000 = %0.003-0.005,
# gidis-donus + kayma ~%0.014 (4 puan). Digerleri tipik perakende
# spread'lerinden turetilmis temkinli degerler.
MARKETS = {
    "NAS100": {"start": "2023-01", "cost_pct": 0.014},
    "SP500": {"start": "2023-01", "cost_pct": 0.014},
    "EURUSD": {"start": "2023-01", "cost_pct": 0.004},
    "XAUUSD": {"start": "2023-01", "cost_pct": 0.010},
}


def measure(pair: str, start: str, cost_pct: float, rule: str = "5min") -> dict:
    minutes = download_range(pair, start, "2026-08", verbose=False)
    bars = resample(minutes, rule)
    setups = find_setups(bars)
    if not setups:
        return {"enstrüman": pair, "bar": len(bars), "kurulum": 0}

    trades = evaluate(bars, setups, cost_pct=cost_pct)
    s = summarize(trades)
    return {
        "enstrüman": pair,
        "bar": len(bars),
        "kurulum": s["islem"],
        "kazanma_%": s["kazanma_orani"],
        "ortalama_R": s["ortalama_R"],
        "toplam_R": s["toplam_R"],
        "maks_dusus_R": round(max_drawdown_r(trades), 1),
    }


def main() -> None:
    rows = []
    for pair, cfg in MARKETS.items():
        print(f"{pair} olculuyor...", flush=True)
        try:
            rows.append(measure(pair, cfg["start"], cfg["cost_pct"]))
        except Exception as exc:
            print(f"  {pair}: HATA {str(exc)[:70]}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    print("MODEL FARKLI ENSTRUMANLARDA (2023-2026, 5 dakikalik, yuzde bazli maliyet)")
    print("=" * 76)
    print(df.to_string(index=False))

    if "ortalama_R" in df.columns:
        pozitif = (df["ortalama_R"] > 0).sum()
        print(f"\n  {pozitif}/{len(df)} enstrümanda pozitif beklenen deger")
        if pozitif == len(df):
            print("  -> Model enstrümana ozel DEGIL. SMC'nin mikroyapi iddiasi")
            print("     bu veride destekleniyor.")
        elif pozitif == 1:
            print("  -> UYARI: yalnizca bir enstrümanda calisiyor. Sonuc o")
            print("     enstrümanin donemsel huyuna uydurulmus olabilir.")


if __name__ == "__main__":
    main()
