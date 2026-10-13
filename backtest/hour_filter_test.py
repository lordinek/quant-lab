"""
Saat filtresi gercek mi, yoksa veriye uydurma mi?

Islem analizi carpici bir sey gosterdi: New York saatiyle 09-10 (borsa
acilisi) modelin EN KOTU pencereleri (%53 kazanma, +0.07R), gece saatleri
ise en iyisi (%70-74). Bu SMC literaturunun soyledigi seyin tersi - ki
tam da bu yuzden supheli.

Bir filtreyi tum veride kesfedip yine tum veride dogrulamak kendini
kandirmaktir. Dogru sinav:

    1. Kotu saatleri YALNIZCA egitim doneminde bul (2021-2023).
    2. O saatleri test doneminde (2024-2026) disla.
    3. Filtresiz haliyle karsilastir.

Filtre gercek bir olguya dayaniyorsa test doneminde de yardim etmeli.
Sadece egitim verisinin gurultusuyse test doneminde ya yardim etmez ya
zarar verir.

BTC projesinde ayni sinif sinav "parametre optimizasyonu kendini kandirmaktan
ibaret" sonucunu vermisti. Buradaki cevap ne cikarsa ciksin, filtre ancak
bu testi gecerse varsayilan olur.
"""
import pandas as pd

from backtest.evaluate import evaluate, max_drawdown_r, summarize
from strategy.setup import DEFAULT_CONFIG, find_setups

COST = 4.0
TRAIN = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-01")

# Egitim doneminde bir saati "kotu" saymak icin esikler.
MIN_TRADES = 100        # az ornekli saatler gurultudur
MAX_AVG_R = 0.12        # genel ortalamanin belirgin altinda


def hour_table(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    setups = find_setups(bars, cfg)
    trades = evaluate(bars, setups, cost_points=COST)
    rows = []
    for t in trades:
        ny = t.setup.entry_time.tz_convert("America/New_York")
        rows.append({"saat": ny.hour, "R": t.r_multiple})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = df.groupby("saat").agg(islem=("R", "size"), ortalama_R=("R", "mean"))
    out["ortalama_R"] = out["ortalama_R"].round(3)
    return out


def measure(bars: pd.DataFrame, cfg: dict) -> dict:
    setups = find_setups(bars, cfg)
    if not setups:
        return {"islem": 0, "kazanma_orani": 0.0, "ortalama_R": 0.0,
                "toplam_R": 0.0, "maks_dusus_R": 0.0}
    trades = evaluate(bars, setups, cost_points=COST)
    s = summarize(trades)
    s["maks_dusus_R"] = round(max_drawdown_r(trades), 1)
    return s


def main() -> None:
    from backtest.out_of_sample import slice_period
    from backtest.run import load

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    train = slice_period(bars, TRAIN)
    test = slice_period(bars, TEST)
    print(f"  egitim {len(train):,} bar | test {len(test):,} bar\n", flush=True)

    # 1. Kotu saatleri YALNIZCA egitim doneminde bul.
    print("=" * 64)
    print("1) EGITIM DONEMINDE SAAT TABLOSU (2021-2023)")
    print("=" * 64)
    table = hour_table(train, DEFAULT_CONFIG)
    print(table.to_string())

    bad = sorted(table[(table["islem"] >= MIN_TRADES) &
                       (table["ortalama_R"] < MAX_AVG_R)].index.tolist())
    print(f"\n  Egitimde 'kotu' bulunan saatler (NY): {bad}")
    if not bad:
        print("  Filtrelenecek saat yok - test burada biter.")
        return

    # 2. Test doneminde uygula.
    print("\n" + "=" * 64)
    print("2) TEST DONEMINDE (2024-2026) - filtre gormedigi veride")
    print("=" * 64)

    plain = measure(test, DEFAULT_CONFIG)
    filtered = measure(test, {**DEFAULT_CONFIG, "exclude_entry_hours": bad})

    rows = [
        {"ayar": "filtresiz", **{k: plain[k] for k in
         ("islem", "kazanma_orani", "ortalama_R", "toplam_R", "maks_dusus_R")}},
        {"ayar": "saat filtreli", **{k: filtered[k] for k in
         ("islem", "kazanma_orani", "ortalama_R", "toplam_R", "maks_dusus_R")}},
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 64)
    print("KARAR")
    print("=" * 64)
    d_avg = filtered["ortalama_R"] - plain["ortalama_R"]
    d_total = filtered["toplam_R"] - plain["toplam_R"]
    print(f"  ortalama_R farki : {d_avg:+.3f}")
    print(f"  toplam_R farki   : {d_total:+.1f}")
    print(f"  atilan islem     : {plain['islem'] - filtered['islem']}")

    if d_avg > 0.02 and d_total > 0:
        print("\n  -> Filtre test doneminde de YARDIM EDIYOR: hem islem basina")
        print("     beklenti hem toplam yukseldi. Varsayilan yapilabilir.")
    elif d_avg > 0 and d_total < 0:
        print("\n  -> Karma sonuc: islem basina beklenti artti ama TOPLAM dustu.")
        print("     Filtre iyi islemleri de kesiyor. Kucuk hesapta islem basina")
        print("     kalite daha onemli olabilir - yine de otomatik benimsenmemeli.")
    else:
        print("\n  -> Filtre YARDIM ETMIYOR. Egitim verisinde gorulen fark")
        print("     gurultuymus. Filtre eklenmemeli (BTC projesindeki dersin aynisi).")


if __name__ == "__main__":
    main()
