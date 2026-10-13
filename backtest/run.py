"""
IFVG modelinin ilk gercek olcumu: NAS100, 5.5 yil.

Kullanim:
    python -m backtest.run                 # 5 dakikalik, varsayilan ayarlar
    python -m backtest.run 15min           # baska zaman dilimi
"""
import sys

import pandas as pd

from backtest.evaluate import (by_year, evaluate, max_drawdown_r, summarize)
from data.histdata import download_range, resample
from strategy.setup import DEFAULT_CONFIG, find_setups, to_frame


def load(pair: str = "NAS100", start: str = "2021-01", end: str = "2026-08",
         rule: str = "5min") -> pd.DataFrame:
    minutes = download_range(pair, start, end, verbose=False)
    bars = resample(minutes, rule)
    # spread kolonu HistData'da yok; simulator sabit maliyet kullaniyor.
    return bars


def main(rule: str = "5min") -> None:
    print(f"Veri yukleniyor ({rule})...", flush=True)
    bars = load(rule=rule)
    print(f"  {len(bars):,} bar | {bars.index[0]} -> {bars.index[-1]}\n", flush=True)

    print("Kurulumlar araniyor...", flush=True)
    setups = find_setups(bars)
    print(f"  {len(setups)} kurulum bulundu\n", flush=True)

    if not setups:
        print("Hic kurulum yok - filtreler fazla dar olabilir.")
        return

    trades = evaluate(bars, setups)
    stats = summarize(trades)

    print("=" * 52)
    print(f"IFVG modeli - NAS100 {rule}")
    print("=" * 52)
    for k, v in stats.items():
        print(f"  {k:16} {v}")
    print(f"  {'maks_dusus_R':16} {max_drawdown_r(trades):.1f}")

    print("\nYillara gore:")
    print(by_year(trades).to_string())

    frame = to_frame(setups)
    print("\nSupurulen seviyelere gore kurulum sayisi:")
    print(frame["swept"].value_counts().to_string())

    print("\nYon dagilimi:")
    print(frame["direction"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5min")
