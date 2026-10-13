"""
Koruma esiklerinin taranmasi - ve egitim/test ayrimiyla sinanmasi.

Bir koruma acmak bir PARAMETRE secmektir. Bu projede parametre, egitim
doneminde parlak gorundugu icin secilmez: saat filtresi tam olarak bunu yapmis,
kagit uzerinde harika gorunmus, egitim/test ayrilinca gecememis ve EKLENMEMISTI.
Ayni sinav burada da kuruluyor.

Sorulan iki soru:

  A) Her koruma tek basina egitim doneminde ne yapiyor?
     (digerleri kapali, yalnizca o acik)

  B) Egitimde iyi gorunen degerler TEST doneminde de iyi mi?
     Degilse o koruma eklenmez - ne kadar mantikli gorunurse gorunsun.

OLCUT: koruma katmaninin isi daha cok kazanmak DEGIL, daha az batmak. O yuzden
tek basina son bakiye yaniltici olur; "yuzde dusus basina kac kat buyume"
(buyume_kat / maks_dusus_%) birlikte raporlaniyor. Bir koruma bakiyeyi biraz
dusurup dususu cok azaltiyorsa ISE YARIYOR demektir.

NOT: 5 dakikalik bir modelde stop'lar seyrek degil - gunde 2-3 kurulum, bunlarin
%38'i stop. Cok siki bir koruma (or. 3 ardisik stop, 24 saat kilit) botu ayda
haftalarca kapali tutabilir. Tarama bunu da gosteriyor: `koruma_engeli` sutunu
korumanin kac kurulumu iptal ettigi.
"""
import sys

import pandas as pd

from backtest.evaluate import evaluate
from backtest.portfolio import simulate
from backtest.run import load
from strategy.protections import ProtectionConfig
from strategy.setup import find_setups

# out_of_sample.py ile AYNI ayrim - iki dosyanin farkli sinir kullanmasi
# karsilastirmayi anlamsiz kilar.
TRAIN = ("2021-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-01")

START_BALANCE = 150.0
RISK_PCT = 1.0
MAX_CONCURRENT = 3

# Her koruma TEK BASINA taraniyor; 0 = kapali (referans).
SEARCH: dict[str, list] = {
    "stop_streak": [0, 3, 4, 5, 6, 8],
    "stop_count": [0, 4, 5, 6, 8, 10],
    "max_drawdown_pct": [0, 10, 15, 20, 25, 30],
    "cooldown_hours": [0, 0.5, 1.0, 2.0, 4.0],
}


def config_for(name: str, value) -> ProtectionConfig:
    """Yalnizca bir korumasi acik config uret."""
    if name == "stop_count":
        # Pencere ve kilit suresi sabit tutuluyor: her ikisini de taramak
        # uzayi patlatir ve asil soru "kac stop" sorusu.
        return ProtectionConfig(stop_count=value, stop_count_hours=24.0,
                                stop_count_lock_hours=12.0)
    return ProtectionConfig(**{name: value})


def score(result) -> float:
    """Yuzde dusus basina buyume kati. Dusus yoksa buyumenin kendisi."""
    growth = result.final_balance / result.starting_balance
    dd = max(result.max_drawdown_pct, 1.0)
    return growth / dd


def run_period(bars, trades, period, protection=None) -> dict:
    start, end = period
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    subset = [t for t in trades if lo <= t.setup.entry_time <= hi]
    if not subset:
        return {}

    r = simulate(bars, subset, starting_balance=START_BALANCE,
                 risk_pct=RISK_PCT, max_concurrent=MAX_CONCURRENT,
                 compound=False, protection=protection)
    return {
        "bakiye": round(r.final_balance, 0),
        "maks_dusus_%": r.max_drawdown_pct,
        "alinan": r.trades_taken,
        "koruma_engeli": r.trades_blocked,
        "kilit": r.locks,
        "skor": round(score(r), 3),
    }


def scan(bars, trades) -> dict[str, list]:
    """A) Her korumayi egitim doneminde tek tek tara."""
    best: dict[str, list] = {}

    for name, values in SEARCH.items():
        print(f"\n--- {name} (egitim donemi) ---")
        rows = []
        for v in values:
            cfg = None if v in (0, 0.0) else config_for(name, v)
            row = run_period(bars, trades, TRAIN, cfg)
            row["deger"] = v
            rows.append(row)
            flag = "  <- kapali (referans)" if cfg is None else ""
            print(f"  {name}={v:<5} bakiye ${row['bakiye']:>7,.0f} | "
                  f"dusus %{row['maks_dusus_%']:>5.1f} | alinan {row['alinan']:>4} | "
                  f"engel {row['koruma_engeli']:>4} | skor {row['skor']:.3f}{flag}")

        baseline = rows[0]
        winners = [r for r in rows[1:] if r["skor"] > baseline["skor"]]
        winners.sort(key=lambda r: r["skor"], reverse=True)
        best[name] = winners
        if winners:
            print(f"  -> egitimde referansi gecen: "
                  f"{[w['deger'] for w in winners]}")
        else:
            print("  -> egitimde hicbir deger referansi gecmedi")

    return best


def verify(bars, trades, best: dict[str, list]) -> None:
    """B) Egitimde kazananlari TEST doneminde sina."""
    print("\n" + "=" * 70)
    print("TEST DONEMI DOGRULAMASI (2024-01 -> 2026-09)")
    print("=" * 70)

    base = run_period(bars, trades, TEST, None)
    print(f"  korumasiz referans: bakiye ${base['bakiye']:,.0f} | "
          f"dusus %{base['maks_dusus_%']:.1f} | skor {base['skor']:.3f}\n")

    survivors = []
    for name, winners in best.items():
        if not winners:
            continue
        top = winners[0]["deger"]
        row = run_period(bars, trades, TEST, config_for(name, top))
        gecti = row["skor"] > base["skor"]
        print(f"  {name}={top:<5} bakiye ${row['bakiye']:>7,.0f} | "
              f"dusus %{row['maks_dusus_%']:>5.1f} | engel {row['koruma_engeli']:>4} | "
              f"skor {row['skor']:.3f} | {'GECTI' if gecti else 'gecemedi'}")
        if gecti:
            survivors.append((name, top, row))

    print()
    if not survivors:
        print("SONUC: hicbir koruma egitim+test sinavini gecmedi.")
        print("Korumalar KAPALI kalir - saat filtresiyle ayni karar.")
        return

    print("SONUC: su korumalar iki donemde de referansi gecti:")
    for name, value, row in survivors:
        print(f"  {name} = {value}  (test: ${row['bakiye']:,.0f}, "
              f"dusus %{row['maks_dusus_%']:.1f})")
    print("\nBirlesik hallerini olcmek ayri bir adim: korumalar birbirinin")
    print("tetikleyicisini yiyebilir, toplamlari parcalarinin toplami degildir.")


def main() -> None:
    print("Veri yukleniyor...", flush=True)
    bars = load()
    setups = find_setups(bars)
    trades = evaluate(bars, setups)
    print(f"  {len(bars):,} bar | {len(trades)} islem", flush=True)

    best = scan(bars, trades)
    verify(bars, trades, best)


if __name__ == "__main__":
    main()
