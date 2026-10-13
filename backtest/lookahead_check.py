"""
Butunsel gelecege bakma denetimi.

Birim testler her katmani ayri ayri kontrol ediyor, ama gercek tehlike
katmanlar BIRLESTIGINDE dogar: bir yerde bir olayin YASANDIGI bar ile
BILINDIGI bar karisir ve zincir boyunca sessizce yayilir.

Test cok basit ve aldatilmasi zor:
    Veriyi ilk N bara kes, kurulumlari bul.
    Ayni veriyi TAMAMIYLA ver, kurulumlari bul.
    Ilk N bar icinde kalan kurulumlar BIRE BIR AYNI olmali.

Farklilik varsa model gelecegi goruyor demektir. Bu hata sessizdir:
backtest'i muhtesem gosterir, canlida hicbir sey tutturmaz. Altin ve BTC
projelerinde de ayni sinif test (`test_no_lookahead`) tutuldu.
"""
import sys

import pandas as pd

from data.histdata import download_range, resample
from strategy.setup import find_setups


def check(bars: pd.DataFrame, cut: int, margin: int = 500) -> bool:
    """`cut` barina kadar olan kurulumlar, tum veriyle ayni cikiyor mu.

    `margin`: kesim noktasinin hemen oncesindeki kurulumlar dogal olarak
    farkli olabilir - orada bir kurulumun tamamlanmasi icin gereken gelecek
    barlar henuz yok. O yuzden kesime yakin bolge karsilastirma disi.
    """
    partial = find_setups(bars.iloc[:cut])
    full = find_setups(bars)

    limit = cut - margin
    a = [(s.entry_index, s.direction, round(s.entry_price, 4), round(s.stop, 4))
         for s in partial if s.entry_index < limit]
    b = [(s.entry_index, s.direction, round(s.entry_price, 4), round(s.stop, 4))
         for s in full if s.entry_index < limit]

    print(f"  kesim {cut:,} bar | kismi: {len(a)} kurulum | tam: {len(b)} kurulum")

    if a == b:
        print("  ESLESTI - gelecege bakma belirtisi yok\n")
        return True

    print("  UYUSMAZLIK - model gelecegi goruyor olabilir!")
    only_a = [x for x in a if x not in b]
    only_b = [x for x in b if x not in a]
    for x in only_a[:5]:
        print(f"    sadece kismi veride: {x}")
    for x in only_b[:5]:
        print(f"    sadece tam veride  : {x}")
    print()
    return False


def main() -> None:
    print("Veri yukleniyor...", flush=True)
    minutes = download_range("NAS100", "2025-01", "2026-08", verbose=False)
    bars = resample(minutes, "5min")
    print(f"  {len(bars):,} bar\n", flush=True)

    ok = True
    for frac in (0.4, 0.6, 0.8):
        ok &= check(bars, int(len(bars) * frac))

    print("SONUC:", "temiz" if ok else "GELECEGE BAKMA VAR")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
