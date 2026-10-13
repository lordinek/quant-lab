"""
Parametre saglamligi: model tek bir sansli ayara mi bagli, yoksa genis bir
platoda mi calisiyor?

BTC projesindeki en degerli test buydu. Orada MA periyodu 40-60 araliginda
Sharpe 1,14-1,34 cikmisti - yani tepe nokta degil GENIS PLATO, fikir saglam
demekti. Tersi olsaydi (sadece tek bir degerde iyi), sonuc veriye uydurulmus
bir tesadufden ibaret olurdu.

Ayni sinav burada: her parametreyi tek tek kaydirip sonucun nasil degistigine
bakiyoruz. Aranan sey en yuksek sayi DEGIL, komsulariyla birlikte calisan
bir bolge.

Uyari: bu bir OPTIMIZASYON aracı degil. BTC projesinde walk-forward,
"her yil en iyi parametreyi sec" yaklasiminin sabit parametreden DAHA KOTU
oldugunu gosterdi. Burada amac en iyiyi bulmak degil, mevcut secimin bir
ucurumun kenarinda durup durmadigini anlamak.
"""
import pandas as pd

from backtest.evaluate import evaluate, max_drawdown_r, summarize
from strategy.setup import DEFAULT_CONFIG, find_setups

# Taranan parametreler ve denenecek degerler.
GRID = {
    "rr_target": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "min_risk_pct": [0.01, 0.03, 0.05, 0.08, 0.12, 0.2],
    "sweep_to_choch": [20, 40, 60, 90, 120],
    "choch_to_ifvg": [20, 40, 60, 90, 120],
    "ifvg_to_entry": [20, 40, 60, 90, 120],
    "swing_right": [1, 2, 3, 4],
    "min_gap_pct": [0.0, 0.01, 0.02, 0.05, 0.1],
    "sweep_min_pen_pct": [0.0, 0.005, 0.01, 0.02, 0.05],
    "stop_buffer_pct": [0.0, 0.01, 0.02, 0.05, 0.1],
}


def scan_one(bars: pd.DataFrame, key: str, values: list,
             cost_points: float = 4.0) -> pd.DataFrame:
    """Tek bir parametreyi kaydirip sonuclari toplar (digerleri varsayilan)."""
    rows = []
    for v in values:
        cfg = {**DEFAULT_CONFIG, key: v}
        setups = find_setups(bars, cfg)
        if not setups:
            rows.append({key: v, "islem": 0, "kazanma_%": 0.0,
                         "ortalama_R": 0.0, "toplam_R": 0.0, "maks_dusus_R": 0.0})
            continue
        trades = evaluate(bars, setups, cost_points=cost_points)
        s = summarize(trades)
        rows.append({
            key: v,
            "islem": s["islem"],
            "kazanma_%": s["kazanma_orani"],
            "ortalama_R": s["ortalama_R"],
            "toplam_R": s["toplam_R"],
            "maks_dusus_R": round(max_drawdown_r(trades), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    from backtest.run import load

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    print(f"  {len(bars):,} bar\n", flush=True)

    # Maliyet olculen spread'e yakin, kaymayi da katan temkinli bir deger.
    cost = 4.0
    print(f"(tum taramalarda islem maliyeti {cost} puan varsayildi)\n")

    for key, values in GRID.items():
        print("=" * 62)
        print(f"PARAMETRE: {key}   (varsayilan: {DEFAULT_CONFIG[key]})")
        print("=" * 62)
        df = scan_one(bars, key, values, cost_points=cost)
        print(df.to_string(index=False))

        pos = (df["ortalama_R"] > 0).sum()
        print(f"  -> {pos}/{len(df)} deger pozitif ortalama_R veriyor")
        if pos == len(df):
            print("     GENIS PLATO: sonuc bu parametreye asiri duyarli degil.")
        elif pos <= 1:
            print("     UYARI: yalnizca tek deger calisiyor - uydurma riski.")
        print()


if __name__ == "__main__":
    main()
