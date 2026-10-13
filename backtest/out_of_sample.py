"""
Out-of-sample dogrulama: model gormedigi donemde de calisiyor mu?

Iki ayri soru olculuyor:

  A) SABIT PARAMETRE, GORULMEMIS DONEM
     Varsayilan ayarlar 2021-2023'te ne yapiyor, 2024-2026'da ne yapiyor?
     Ayarlar hicbir donemde optimize edilmedi, ama yine de "gelistirme
     sirasinda bakilan" donem ile "hic bakilmayan" donemi ayirmak dogru.

  B) OPTIMIZASYON YARDIM EDIYOR MU?
     Egitim doneminde en iyi parametreyi sec, test doneminde uygula.
     Sabit varsayilanla karsilastir.

     BTC projesinde bunun cevabi netti: her yil parametre optimize etmek
     sabit MA'da durmaktan DAHA KOTU sonuc verdi (%43,4 vs %47,1 CAGR).
     Optimizer egitim verisinde parlayan degerleri seciyor, sonra test
     doneminde kirbaçlaniyor. Ayni sinav burada da yapiliyor - cevap
     "optimizasyon yardim etmiyor" cikarsa, parametrelere DOKUNMAMAK
     bilincli bir karar haline gelir.
"""
import pandas as pd

from backtest.evaluate import evaluate, max_drawdown_r, summarize
from strategy.setup import DEFAULT_CONFIG, find_setups

COST = 4.0

# Egitim / test ayrimi. Egitim donemi gelistirme sirasinda cokca bakilan
# donem; test donemi karara katilmayan donem.
TRAIN = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-01")

# Optimizasyon denemesi icin taranacak parametreler (her biri tek tek).
SEARCH = {
    "rr_target": [1.0, 1.5, 2.0, 2.5, 3.0],
    "sweep_to_choch": [20, 40, 60, 90, 120],
    "choch_to_ifvg": [20, 40, 60, 90, 120],
    "min_risk_pct": [0.02, 0.05, 0.08, 0.15],
    "min_gap_pct": [0.0, 0.01, 0.02, 0.05],
}


def slice_period(bars: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start, end = period
    return bars.loc[start:end]


def measure(bars: pd.DataFrame, cfg: dict) -> dict:
    setups = find_setups(bars, cfg)
    if not setups:
        return {"islem": 0, "kazanma_orani": 0.0, "ortalama_R": 0.0,
                "toplam_R": 0.0, "maks_dusus_R": 0.0}
    trades = evaluate(bars, setups, cost_points=COST)
    s = summarize(trades)
    s["maks_dusus_R"] = round(max_drawdown_r(trades), 1)
    return s


def optimize(bars: pd.DataFrame) -> dict:
    """Egitim doneminde koordinat inisi ile 'en iyi' parametreleri sec.

    Tam grid degil tek tek tarama (altin projesindeki yaklasimin aynisi) -
    amac zaten en iyiyi bulmak degil, optimizasyonun YARDIM EDIP ETMEDIGINI
    olcmek.
    """
    cfg = dict(DEFAULT_CONFIG)
    for key, values in SEARCH.items():
        best_v, best_score = cfg[key], -1e9
        for v in values:
            trial = {**cfg, key: v}
            score = measure(bars, trial)["toplam_R"]
            if score > best_score:
                best_v, best_score = v, score
        cfg[key] = best_v
    return cfg


def main() -> None:
    from backtest.run import load

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    train = slice_period(bars, TRAIN)
    test = slice_period(bars, TEST)
    print(f"  egitim: {len(train):,} bar ({TRAIN[0]} -> {TRAIN[1]})")
    print(f"  test  : {len(test):,} bar ({TEST[0]} -> {TEST[1]})\n", flush=True)

    print("=" * 62)
    print("A) SABIT VARSAYILAN AYARLAR")
    print("=" * 62)
    rows = []
    for name, part in (("egitim donemi", train), ("TEST donemi", test)):
        s = measure(part, DEFAULT_CONFIG)
        rows.append({"donem": name, "islem": s["islem"],
                     "kazanma_%": s["kazanma_orani"], "ortalama_R": s["ortalama_R"],
                     "toplam_R": s["toplam_R"], "maks_dusus_R": s["maks_dusus_R"]})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 62)
    print("B) EGITIMDE OPTIMIZE EDILMIS AYARLAR")
    print("=" * 62)
    print("Egitim doneminde parametre araniyor...", flush=True)
    tuned = optimize(train)
    changed = {k: (DEFAULT_CONFIG[k], tuned[k]) for k in SEARCH
               if DEFAULT_CONFIG[k] != tuned[k]}
    print(f"  degisen parametreler: {changed if changed else 'yok'}\n")

    rows = []
    for name, part in (("egitim donemi", train), ("TEST donemi", test)):
        s = measure(part, tuned)
        rows.append({"donem": name, "islem": s["islem"],
                     "kazanma_%": s["kazanma_orani"], "ortalama_R": s["ortalama_R"],
                     "toplam_R": s["toplam_R"], "maks_dusus_R": s["maks_dusus_R"]})
    print(pd.DataFrame(rows).to_string(index=False))

    # Karsilastirma: test doneminde hangisi daha iyi?
    fixed_test = measure(test, DEFAULT_CONFIG)
    tuned_test = measure(test, tuned)
    print("\n" + "=" * 62)
    print("KARAR: TEST DONEMINDE HANGISI?")
    print("=" * 62)
    print(f"  sabit varsayilan : ortalama_R {fixed_test['ortalama_R']:+.3f} | "
          f"toplam_R {fixed_test['toplam_R']:+.1f}")
    print(f"  optimize edilmis : ortalama_R {tuned_test['ortalama_R']:+.3f} | "
          f"toplam_R {tuned_test['toplam_R']:+.1f}")

    if tuned_test["ortalama_R"] <= fixed_test["ortalama_R"]:
        print("\n  -> Optimizasyon YARDIM ETMIYOR. Parametrelere dokunmamak")
        print("     bilincli karar olmali (BTC projesindeki sonucun aynisi).")
    else:
        diff = tuned_test["ortalama_R"] - fixed_test["ortalama_R"]
        print(f"\n  -> Optimize edilmis ayarlar test doneminde {diff:+.3f}R daha iyi.")
        print("     Yine de dikkat: tek bir bolunme tek bir denemedir.")


if __name__ == "__main__":
    main()
