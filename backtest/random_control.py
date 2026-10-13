"""
Rastgele giris kontrolu - modelin gercek bir edge'i var mi?

$150 -> $2,36 milyon ve 28/28 karli pencere cikinca sorulmasi gereken tek
soru su: bu sonuc SINYALDEN mi geliyor, yoksa cikis mantigindan mi?

2R hedef / 1R stop bir kurulumda, rastgele yuruyuste beklenen kazanma orani
kabaca %33'tur (hedefe ulasmak iki kat uzaga gitmeyi gerektirir). Bizim
olctugumuz %53. Bu ya gercek bir edge, ya da olcumde bir hata.

Test: kurulumlarin ZAMANLARINI ve YONLERINI bozup, stop/hedef geometrisini
aynen koruyoruz. Sinyal degerliyse gercek kurulumlar rastgeleyi acik ara
yenmeli. Yenmiyorsa, olctugumuz sey stratejinin kendisi degil cikis
kurallarinin veya olcum yonteminin urettigi bir yanilsamadir.

Uc kontrol grubu:
  1. rastgele_bar   : ayni yon, ayni risk, rastgele bar
  2. rastgele_yon   : ayni bar, ayni risk, yon yazi-tura
  3. tam_rastgele   : ikisi de rastgele
"""
import copy
import random

import numpy as np
import pandas as pd

from backtest.evaluate import evaluate, summarize
from strategy.setup import Setup

SEED = 42


def _clone_with(setup: Setup, entry_index: int, direction: str,
                bars: pd.DataFrame) -> Setup | None:
    """Ayni risk mesafesiyle, baska bar/yonde bir kurulum uretir.

    Risk (stop mesafesi) ve hedef orani KORUNUYOR - degisen tek sey girisin
    ne zaman ve hangi yonde oldugu. Boylece cikis geometrisinin katkisi
    sabit tutulup yalnizca sinyalin katkisi olculur.
    """
    s = copy.copy(setup)
    risk = setup.risk
    if risk <= 0:
        return None

    entry_price = float(bars["close"].iloc[entry_index])
    rr = setup.reward / risk if risk else 2.0

    s.entry_index = entry_index
    s.entry_time = bars.index[entry_index]
    s.direction = direction
    s.entry_price = entry_price
    if direction == "long":
        s.stop = entry_price - risk
        s.target = entry_price + risk * rr
    else:
        s.stop = entry_price + risk
        s.target = entry_price - risk * rr
    return s


def build_controls(bars: pd.DataFrame, setups: list[Setup],
                   seed: int = SEED) -> dict[str, list[Setup]]:
    rng = random.Random(seed)
    n = len(bars)
    margin = 500      # sona cok yakin girisler cozulemez

    out: dict[str, list[Setup]] = {"rastgele_bar": [], "rastgele_yon": [],
                                   "tam_rastgele": []}

    for s in setups:
        rb = rng.randrange(margin, n - margin)
        flip = rng.choice(["long", "short"])
        rb2 = rng.randrange(margin, n - margin)
        flip2 = rng.choice(["long", "short"])

        for key, idx, direction in (
            ("rastgele_bar", rb, s.direction),
            ("rastgele_yon", s.entry_index, flip),
            ("tam_rastgele", rb2, flip2),
        ):
            clone = _clone_with(s, idx, direction, bars)
            if clone is not None:
                out[key].append(clone)

    return out


def main() -> None:
    from backtest.run import load
    from strategy.setup import find_setups

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    setups = find_setups(bars)
    print(f"  {len(bars):,} bar | {len(setups)} kurulum\n", flush=True)

    rows = []

    real = evaluate(bars, setups)
    s = summarize(real)
    rows.append({"grup": "GERCEK MODEL", **_pick(s)})

    for name, ctrl in build_controls(bars, setups).items():
        t = evaluate(bars, ctrl)
        rows.append({"grup": name, **_pick(summarize(t))})

    df = pd.DataFrame(rows)
    print("=" * 70)
    print("RASTGELE GIRIS KONTROLU")
    print("=" * 70)
    print(df.to_string(index=False))

    real_r = s["ortalama_R"]
    ctrl_best = max(r["ortalama_R"] for r in rows[1:])
    print(f"\n  gercek ortalama_R : {real_r}")
    print(f"  en iyi kontrol    : {ctrl_best}")
    if real_r > ctrl_best * 2 and real_r > 0.05:
        print("  -> Model kontrolleri acik ara yeniyor: sinyalin katkisi gercek.")
    elif real_r > ctrl_best:
        print("  -> Model onde ama fark kucuk: edge zayif, dikkatli olunmali.")
    else:
        print("  -> UYARI: model rastgeleyi yenemiyor. Olculen sey sinyal degil,")
        print("     cikis geometrisi veya olcum yontemi olabilir.")


def _pick(s: dict) -> dict:
    return {
        "islem": s["islem"],
        "kazanma_%": s["kazanma_orani"],
        "ortalama_R": s["ortalama_R"],
        "toplam_R": s["toplam_R"],
        "hedef": s["hedef"],
        "stop": s["stop"],
        "zaman_asimi": s["zaman_asimi"],
    }


if __name__ == "__main__":
    main()
