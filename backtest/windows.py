"""
Kaydirmali pencere testi - altin ve BTC projelerindeki metodolojinin aynisi.

Tek koşunun soyleyemedigi sey: sonuc birkac sansli donemden mi geliyor, yoksa
strateji farkli rejimlerde ayakta mi kaliyor? "Her yil pozitif" iyi bir isaret
ama yil siniri keyfi bir kesimdir; pencereler kaydirilinca resim degisebilir.

Olculen sey pencere BASINA sonuc dagilimidir:
  - patlama orani  : hesabi bitiren pencere yuzdesi
  - karli pencere  : pozitif biten pencere yuzdesi
  - medyan / ortalama ayrimi: getiri kumelenmis mi, esit mi dagilmis

Altin ve BTC'de asil bilgi hep medyan ile ortalama arasindaki ucurumda cikti.
"""
import numpy as np
import pandas as pd

from backtest.evaluate import evaluate
from backtest.portfolio import simulate
from strategy.setup import _bar_minutes, find_setups

# Ceyrek yillik pencereler, yarim ceyrek kaydirarak (ust uste binen pencereler
# daha fazla ornek verir; bagimsiz degiller ama rejim cesitliligini gosterir).
DEFAULT_WINDOW_DAYS = 90
DEFAULT_STEP_DAYS = 45


def _bars_per_day(bars: pd.DataFrame) -> float:
    """Pencere GUN cinsinden tanimli, bar cinsinden DEGIL.

    Onceden burada `BARS_PER_DAY_5MIN = 288` sabiti vardi ve 5 dakikalik veri
    varsayiyordu. 15 dakikalik veride ayni sabit pencereyi sessizce UC KATINA
    cikariyordu: "90 gunluk pencere" aslinda 270 gun oluyor, getiriler daha uzun
    bilesikleme yuzunden sisiyor ve pencere sayisi 28'den 8'e duserek testin
    kendisini zayiflatiyordu. Zaman dilimleri karsilastirildiginda bu fark
    tamamen olcum artefakti olarak gorunur.

    Ayni hata bu projede daha once de yasandi (pencereler bar cinsindeyken
    "15 dakikalik daha iyi" olculup sonra geri alinmisti). Cozum orada ne idiyse
    burada da o: birimi zamana baglamak ve bar sayisini verinin kendi
    araligindan turetmek.
    """
    return 24 * 60 / _bar_minutes(bars)


def run_windows(bars: pd.DataFrame, setups, trades,
                window_days: int = DEFAULT_WINDOW_DAYS,
                step_days: int = DEFAULT_STEP_DAYS,
                starting_balance: float = 150.0,
                risk_pct: float = 1.0,
                max_concurrent: int = 3) -> pd.DataFrame:
    """Her pencerede portfoyu bagimsiz olarak bastan calistirir.

    Her pencere KENDI baslangic bakiyesiyle basliyor - amac stratejinin o
    donemde ne yaptigini gormek, onceki pencerelerin birikimini tasimak degil.
    """
    per_day = _bars_per_day(bars)
    window_bars = max(1, int(round(window_days * per_day)))
    step_bars = max(1, int(round(step_days * per_day)))
    n = len(bars)

    by_entry = {}
    for t in trades:
        by_entry.setdefault(t.setup.entry_index, []).append(t)

    rows = []
    start = 0
    while start + window_bars <= n:
        end = start + window_bars
        window = bars.iloc[start:end]

        # Bu pencerede BASLAYAN islemler (bitisi pencereyi asabilir - gercek
        # hayatta da oyle olur).
        picked = [t for i, group in by_entry.items() if start <= i < end
                  for t in group]

        if picked:
            # Indeksleri pencereye goreli hale getir.
            shifted = []
            for t in picked:
                t2 = _shift(t, start, len(window))
                if t2 is not None:
                    shifted.append(t2)
            res = simulate(window, shifted, starting_balance=starting_balance,
                           risk_pct=risk_pct, max_concurrent=max_concurrent)
            ret = (res.final_balance / starting_balance - 1) * 100
            rows.append({
                "baslangic": window.index[0].date(),
                "bitis": window.index[-1].date(),
                "islem": res.trades_taken,
                "getiri_%": round(ret, 1),
                "maks_dusus_%": res.max_drawdown_pct,
                "bitis_bakiye": round(res.final_balance, 2),
            })

        start += step_bars

    return pd.DataFrame(rows)


def _shift(trade, offset: int, window_len: int):
    """Islemi pencereye goreli indekslere tasir; pencereyi asan cikisi kirpar."""
    import copy
    t = copy.copy(trade)
    t.setup = copy.copy(trade.setup)
    t.setup.entry_index -= offset
    t.exit_index -= offset
    if t.setup.entry_index < 0:
        return None
    if t.exit_index >= window_len:
        t.exit_index = window_len - 1
    return t


def summarize(windows: pd.DataFrame) -> dict:
    if windows.empty:
        return {"pencere": 0}

    r = windows["getiri_%"].to_numpy(dtype=float)
    return {
        "pencere": len(windows),
        "patlama_%": round(float((windows["bitis_bakiye"] <= 1.0).mean() * 100), 1),
        "karli_pencere_%": round(float((r > 0).mean() * 100), 1),
        "medyan_getiri_%": round(float(np.median(r)), 1),
        "ortalama_getiri_%": round(float(r.mean()), 1),
        "en_iyi_%": round(float(r.max()), 1),
        "en_kotu_%": round(float(r.min()), 1),
        "ort_maks_dusus_%": round(float(windows["maks_dusus_%"].mean()), 1),
        "ort_islem": round(float(windows["islem"].mean()), 1),
    }


def main() -> None:
    from backtest.run import load

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    setups = find_setups(bars)
    trades = evaluate(bars, setups)
    years = (bars.index[-1] - bars.index[0]).days / 365.25
    print(f"  {len(bars):,} bar | {len(setups)} kurulum | {years:.1f} yil\n", flush=True)

    # Once tek koşu: tum donem, gercek hesap.
    print("=" * 58)
    print("TEK KOSU - $150 hesap, islem basina %1 risk, azami 3 pozisyon")
    print("=" * 58)
    from backtest.portfolio import summarize as psum
    res = simulate(bars, trades, starting_balance=150.0, risk_pct=1.0,
                   max_concurrent=3)
    for k, v in psum(res, years).items():
        print(f"  {k:20} {v}")

    # Es zamanli pozisyon siniri duyarliligi.
    print("\nES ZAMANLI POZISYON SINIRI ETKISI")
    rows = []
    for limit in (1, 2, 3, 5, 10, 16):
        r = simulate(bars, trades, starting_balance=150.0, risk_pct=1.0,
                     max_concurrent=limit)
        s = psum(r, years)
        rows.append({"sinir": limit, "bitis_$": s["bitis"], "CAGR_%": s["CAGR_%"],
                     "maks_dusus_%": s["maks_dusus_%"],
                     "alinan": s["alinan_islem"], "atlanan": s["atlanan_islem"]})
    print(pd.DataFrame(rows).to_string(index=False))

    # Kaydirmali pencereler.
    print("\n" + "=" * 58)
    print("KAYDIRMALI PENCERE (90 gunluk, 45 gun kaydirmali)")
    print("=" * 58)
    windows = run_windows(bars, setups, trades)
    for k, v in summarize(windows).items():
        print(f"  {k:20} {v}")

    print("\nPencere detayi:")
    print(windows.to_string(index=False))


if __name__ == "__main__":
    main()
