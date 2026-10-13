"""
Ozyinelemeli (recursive) sapma denetimi.

Kardes dosya `lookahead_check.py` verinin SONUNU keser ve sorar:
    "gelecegi silince gecmisteki kararlar degisiyor mu?"   (gelecege bakma)

Bu dosya verinin BASINI keser ve tersini sorar:
    "gecmisi kisaltinca BUGUNKU karar degisiyor mu?"       (ozyinelemeli sapma)

Neden onemli: backtest tum veriyi bir kerede gorur (5,5 yil, yuz binlerce
bar). Canli bot ise her turda yalnizca son 2000 bari gorur
(`live/paper_trader.py` LOOKBACK). Ikisi AYNI kodu calistirir ama AYNI
girdiyi gormez. Bir hesap, gordugu gecmisin uzunluguna duyarliysa backtest'te
olculen sey canlida calisacak sey DEGILDIR - ve bu fark hicbir hata vermez,
hicbir log satiri dusurmez. Sadece canli sonuc backtest'i tutturmaz.

SOMUT SUPHELI - tahmin degil, kodun kendisi:
`signals/structure.py: find_breaks()` bir kirilmanin BOS mu CHoCH mu
oldugunu, o ana kadar BIRIKMIS `trend` durumuna gore belirliyor:

    kind = "choch" if trend == "bear" else "bos"

Pencerenin ilk kirilmasinda `trend` daima None'dir, yani ilk kirilma DAIMA
"bos" olur. Pencere baska bir yerden baslarsa ayni bardaki ayni kirilma
"choch" olabilir. Strateji yalnizca CHoCH'lardan giriyor
(`strategy/setup.py: _first_choch`), yani bu fark dogrudan islem uretir ya da
uretmez. Ayni sekilde `used_high` / `used_low` de birikimli durumdur.

Test bunu tartismiyor, olcuyor.
"""
import sys
import time

import pandas as pd

from data.histdata import download_range, resample
from signals.structure import find_breaks, find_swings
from strategy.setup import DEFAULT_CONFIG, find_setups

# Canli botun her turda gordugu pencere (live/paper_trader.py: LOOKBACK).
LIVE_LOOKBACK = 2000

# Zincirin azami omru: supurme -> CHoCH (10sa) -> IFVG (10sa) -> giris (5sa)
# = 25 saat, 5dk barda ~300 bar. Bir pencerenin ilk bu kadar bari ISINMA
# bolgesidir: orada zincirin bir halkasi pencerenin disinda kalmis olabilir ve
# fark gormek dogaldir, hata degil. Karsilastirma daima bu bolgenin disinda.
WARMUP_BARS = 400


def _key(setup) -> tuple:
    """Bir kurulumun kimligi.

    `entry_index` KULLANILMAZ: o sayi pencereye gorelidir, pencere kayinca ayni
    kurulum baska bir sayi alir ve iki liste sirf bu yuzden farkli gorunur.
    Zaman damgasi mutlaktir, pencereden bagimsizdir.
    """
    return (setup.entry_time, setup.direction,
            round(setup.entry_price, 4), round(setup.stop, 4))


def scan(bars: pd.DataFrame, lookbacks: list, compare_bars: int = 600) -> bool:
    """Farkli hafiza uzunluklarini ortak zaman araliginda karsilastirir.

    Referans: TUM veri. Her hafiza uzunlugu onunla olculur. Karsilastirma
    yalnizca son `compare_bars` bar icinde yapilir - orasi her pencerenin
    isinma bolgesinin disidir, yani adil kiyas alanidir.
    """
    cutoff = bars.index[-compare_bars]
    print(f"Karsilastirma alani: son {compare_bars:,} bar "
          f"({cutoff:%Y-%m-%d %H:%M} sonrasi)\n")

    reference = None
    ok = True

    for lb in lookbacks:
        window = bars if lb is None else bars.iloc[-lb:]
        label = "TUM VERI" if lb is None else f"{lb:,} bar"

        if lb is not None and lb <= compare_bars + WARMUP_BARS:
            print(f"  {label:>12} | atlandi (isinma payi yetmiyor)")
            continue

        found = sorted(_key(s) for s in find_setups(window, DEFAULT_CONFIG)
                       if s.entry_time >= cutoff)

        if reference is None:
            reference = found
            print(f"  {label:>12} | {len(found)} kurulum | REFERANS")
            continue

        if found == reference:
            print(f"  {label:>12} | {len(found)} kurulum | ayni")
            continue

        ok = False
        missing = [x for x in reference if x not in found]
        extra = [x for x in found if x not in reference]
        print(f"  {label:>12} | {len(found)} kurulum | FARKLI "
              f"(-{len(missing)} / +{len(extra)})")
        for x in missing[:3]:
            print(f"                 kacirilan: {x[0]:%Y-%m-%d %H:%M} {x[1]} @ {x[2]}")
        for x in extra[:3]:
            print(f"                 fazladan : {x[0]:%Y-%m-%d %H:%M} {x[1]} @ {x[2]}")

    print()
    return ok


def diagnose_breaks(bars: pd.DataFrame, lookbacks: list,
                    compare_bars: int = 600) -> None:
    """Fark varsa nereden geldigini soyler: kirilmalarin TIPI kayiyor mu.

    Supurme, bosluk ve swing tespiti yereldir (her bar kendi komsularina
    bakar), bu yuzden pencereden etkilenmemeleri beklenir. Kirilma tipi ise
    birikimli `trend` durumundan gelir - beklenen kirilma noktasi burasidir.
    Bu fonksiyon sucluyu isaretler, sadece "bir yerde fark var" demez.
    """
    cutoff = bars.index[-compare_bars]
    print("Teshis - ayni bardaki kirilmalarin tipi (bos/choch):")

    reference = None
    for lb in lookbacks:
        window = bars if lb is None else bars.iloc[-lb:]
        label = "TUM VERI" if lb is None else f"{lb:,} bar"

        if lb is not None and lb <= compare_bars + WARMUP_BARS:
            continue

        swings = find_swings(window, DEFAULT_CONFIG["swing_left"],
                             DEFAULT_CONFIG["swing_right"])
        breaks = {window.index[b.index]: b.kind
                  for b in find_breaks(window, swings)
                  if window.index[b.index] >= cutoff}

        if reference is None:
            reference = breaks
            n_choch = sum(1 for v in breaks.values() if v == "choch")
            print(f"  {label:>12} | {len(breaks)} kirilma, {n_choch} CHoCH | REFERANS")
            continue

        flipped = [(t, reference[t], breaks[t]) for t in breaks
                   if t in reference and reference[t] != breaks[t]]
        n_choch = sum(1 for v in breaks.values() if v == "choch")

        if not flipped:
            print(f"  {label:>12} | {len(breaks)} kirilma, {n_choch} CHoCH | tip kaymasi yok")
        else:
            print(f"  {label:>12} | {len(breaks)} kirilma, {n_choch} CHoCH | "
                  f"{len(flipped)} KIRILMANIN TIPI DEGISTI")
            for t, a, b in flipped[:3]:
                print(f"                 {t:%Y-%m-%d %H:%M}: {a} -> {b}")
    print()


def live_sim(bars: pd.DataFrame, steps: int = 400,
             lookback: int = LIVE_LOOKBACK) -> bool:
    """Nihai soru: canli botun gordugu kurulum, backtest'inkiyle ayni mi.

    `live/paper_trader.py` her turda son `lookback` bari find_setups'a verir ve
    yalnizca SON barda tetiklenen kurulumu alir (`entry_index == last_index`).
    Backtest ise tum veriyi tek seferde tarar. Bu fonksiyon o dongunun bire bir
    taklidi - yavastir ama olctugu sey tam olarak canlida yasanacak sey.
    """
    print(f"Canli simulasyon: son {steps} bar, {lookback:,} barlik pencere "
          f"kaydirilarak...", flush=True)

    full = {_key(s) for s in find_setups(bars, DEFAULT_CONFIG)}
    n = len(bars)
    started = time.time()

    live: set = set()
    for i in range(n - steps, n):
        window = bars.iloc[max(0, i - lookback + 1): i + 1]
        last = len(window) - 1
        for s in find_setups(window, DEFAULT_CONFIG):
            if s.entry_index == last:
                live.add(_key(s))

    horizon = bars.index[n - steps]
    backtest = {k for k in full if k[0] >= horizon}

    missed = sorted(backtest - live)
    phantom = sorted(live - backtest)

    print(f"  {time.time() - started:.0f} saniye surdu")
    print(f"  backtest {len(backtest)} kurulum | canli {len(live)} kurulum")

    if not missed and not phantom:
        print("  BIRE BIR AYNI - canli bot backtest'in gordugunu goruyor\n")
        return True

    print(f"  UYUSMAZLIK: canlinin kacirdigi {len(missed)}, "
          f"canlida fazladan cikan {len(phantom)}")
    for x in missed[:5]:
        print(f"    kaciran : {x[0]:%Y-%m-%d %H:%M} {x[1]} @ {x[2]}")
    for x in phantom[:5]:
        print(f"    fazladan: {x[0]:%Y-%m-%d %H:%M} {x[1]} @ {x[2]}")
    print()
    return False


def main() -> None:
    print("Veri yukleniyor...", flush=True)
    minutes = download_range("NAS100", "2025-01", "2026-08", verbose=False)
    bars = resample(minutes, "5min")
    print(f"  {len(bars):,} bar\n", flush=True)

    # Canli botun kullandigi 2000 listede: asil sorulan soru "2000 yeterli mi".
    lookbacks = [None, 16000, 8000, 4000, LIVE_LOOKBACK]

    print("1) Hafiza uzunluguna duyarlilik")
    ok = scan(bars, lookbacks)

    if not ok:
        diagnose_breaks(bars, lookbacks)

    print("2) Canli dongunun bire bir taklidi")
    ok &= live_sim(bars)

    print("SONUC:", "temiz - canli ile backtest ayni sey"
          if ok else "SAPMA VAR - canli bot backtest'ten farkli davranacak")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
