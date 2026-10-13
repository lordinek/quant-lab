"""
Islem maliyeti duyarliligi ve es zamanli pozisyon sayisi.

Iki soru:

1. MALIYET. HistData barlari spread icermiyor, bu yuzden 2 puanlik bir
   maliyet VARSAYDIK. Varsayimin kendisi sonucu belirliyorsa strateji
   degil, tahminimiz olculuyor demektir. BTC projesinde komisyon
   modellenince grid stratejisi olmustu - ayni sinav burada da gecilmeli.

2. ES ZAMANLI POZISYON. R cinsinden olcum, ayni anda kac islemin acik
   oldugunu gizler. On islem ayni anda acilirsa gercek bir hesapta bu
   sermaye kisiti (ve ayni yonde on kat risk) demektir.
"""
import numpy as np
import pandas as pd

from backtest.evaluate import evaluate, max_drawdown_r, summarize
from backtest.run import load
from strategy.setup import find_setups


def cost_sweep(bars: pd.DataFrame, setups, costs=(0, 1, 2, 4, 6, 10)) -> pd.DataFrame:
    rows = []
    for c in costs:
        trades = evaluate(bars, setups, cost_points=c)
        s = summarize(trades)
        rows.append({
            "maliyet_puan": c,
            "islem": s["islem"],
            "kazanma_%": s["kazanma_orani"],
            "toplam_R": s["toplam_R"],
            "ortalama_R": s["ortalama_R"],
            "maks_dusus_R": round(max_drawdown_r(trades), 1),
        })
    return pd.DataFrame(rows)


def overlap_stats(bars: pd.DataFrame, setups) -> dict:
    """Ayni anda kac islem acik kaliyor."""
    trades = evaluate(bars, setups)
    n = len(bars)
    open_count = np.zeros(n, dtype=int)
    for t in trades:
        open_count[t.setup.entry_index: t.exit_index + 1] += 1

    active = open_count[open_count > 0]
    return {
        "ayni_anda_ort": round(float(active.mean()), 2) if len(active) else 0,
        "ayni_anda_maks": int(open_count.max()),
        "piyasada_kalma_%": round(len(active) / n * 100, 1),
    }


def main() -> None:
    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    setups = find_setups(bars)
    print(f"  {len(bars):,} bar | {len(setups)} kurulum\n", flush=True)

    print("MALIYET DUYARLILIGI (puan / islem, gidis-donus)")
    print(cost_sweep(bars, setups).to_string(index=False))

    print("\nES ZAMANLI POZISYON")
    for k, v in overlap_stats(bars, setups).items():
        print(f"  {k:20} {v}")


if __name__ == "__main__":
    main()
