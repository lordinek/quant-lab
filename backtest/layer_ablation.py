"""
Katman ayristirma testi: dort katmanli zincirin hangi parcasi deger katiyor?

Baglam (2026-09-12): stop yon hatasi duzeltildikten sonra tam zincir
(supurme -> CHoCH -> IFVG -> giris) negatife dondu (-0.027R). Soru: model
yanlis mi, yoksa fazla mi kisitlayici? Uc katman kombinasyonu AYNI risk/hedef
kurallariyla (stop_source=sweep, rr_target, min_risk_pct, killzone) olculuyor
- degisen tek sey GIRIS SINYALININ hangi katmanlarda durdugu:

  1. yalniz_supurme  : likidite alindi -> bir sonraki barin acilisinda ters yone gir
  2. supurme_choch   : + yapi ters yonde kirilana kadar bekle, CHoCH sonraki barda gir
  3. tam_zincir      : + ters cevrilmis bosluk ve onun retesti (mevcut model)

Iki katman pozitifse IFVG sarti gereksiz bir filtredir - daha basit ve
calisan bir model demektir. Hepsi negatifse SMC'nin bu kurgusu NAS100 5
dakikalikta calismiyor demektir. Bilincli olarak parametre TARANMIYOR: model
supheliyken parametre aramak kendini kandirmaktir (bkz. min_risk taramasindaki
tek pozitif satir, 14 kurulumdan geliyordu - gurultu).

Giris fiyati olarak sinyal barinin KAPANISI degil, BIR SONRAKI BARIN ACILISI
kullaniliyor: sinyal ancak kendi barinin kapanisinda bilinir, ayni barin
kapanisina emir koymak gelecege bakmaktir. Tam zincirde bu sorun yok (retest
zaten sonraki barlarda araniyor), ama 1. ve 2. varyantlar icin bu ayrim
kritik.
"""
from dataclasses import dataclass

import pandas as pd

from backtest.evaluate import by_year, evaluate, max_drawdown_r, summarize
from signals.liquidity import build_levels, find_sweeps
from signals.sessions import in_killzone
from signals.structure import find_breaks, find_swings
from strategy.setup import (DEFAULT_CONFIG, _bar_minutes, _first_choch,
                            _stop_from_sweep, find_setups)


@dataclass
class AblationSetup:
    """evaluate() yalnizca bu alanlara bakar - Setup'in tam kimligi gerekmiyor."""
    entry_index: int
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    stop: float
    target: float

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry_price)


def _valid_directional_stop(direction: str, entry_price: float, stop: float) -> bool:
    """Stop girisin DOGRU tarafinda mi - tam zincirdeki ayni kontrol (setup.py).

    Bu kontrol olmadan short'ta stop asagida / long'ta yukarida kalabilir,
    o zaman zarar yonunde hicbir cikis olmaz ve risk olculemez hale gelir.
    """
    if direction == "short":
        return stop > entry_price
    return stop < entry_price


def _make_setup(bars: pd.DataFrame, cfg: dict, entry_index: int,
                direction: str, stop: float) -> AblationSetup | None:
    if entry_index >= len(bars):
        return None
    entry_price = float(bars["open"].iloc[entry_index])
    if not _valid_directional_stop(direction, entry_price, stop):
        return None

    risk = abs(entry_price - stop)
    if risk < entry_price * cfg["min_risk_pct"] / 100.0:
        return None

    target = (entry_price + risk * cfg["rr_target"] if direction == "long"
              else entry_price - risk * cfg["rr_target"])
    return AblationSetup(entry_index, bars.index[entry_index], direction,
                         entry_price, stop, target)


def build_layer1(bars: pd.DataFrame, cfg: dict | None = None) -> list[AblationSetup]:
    """Yalniz supurme: likidite alindi -> hemen ters yone gir. CHoCH/IFVG yok."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    levels = build_levels(bars)
    min_pen = bars["close"].to_numpy(dtype=float) * cfg["sweep_min_pen_pct"] / 100.0
    sweeps = find_sweeps(bars, levels, min_penetration=min_pen)
    kz = in_killzone(bars.index, cfg["killzones"]).to_numpy()

    out = []
    for sweep in sweeps:
        if not kz[sweep.index]:
            continue
        want = "short" if sweep.side == "high" else "long"
        stop = _stop_from_sweep(sweep, want, cfg["stop_buffer_pct"])
        setup = _make_setup(bars, cfg, sweep.index + 1, want, stop)
        if setup is not None:
            out.append(setup)
    return out


def build_layer2(bars: pd.DataFrame, cfg: dict | None = None) -> list[AblationSetup]:
    """Supurme + CHoCH: yapi ters yonde kirilana kadar bekle, sonra gir. IFVG yok."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    bar_minutes = _bar_minutes(bars)
    win_choch = max(1, int(round(cfg["sweep_to_choch_hours"] * 60 / bar_minutes)))

    swings = find_swings(bars, cfg["swing_left"], cfg["swing_right"])
    breaks = find_breaks(bars, swings)
    levels = build_levels(bars)
    min_pen = bars["close"].to_numpy(dtype=float) * cfg["sweep_min_pen_pct"] / 100.0
    sweeps = find_sweeps(bars, levels, min_penetration=min_pen)
    kz = in_killzone(bars.index, cfg["killzones"]).to_numpy()

    out = []
    for sweep in sweeps:
        if not kz[sweep.index]:
            continue
        want = "short" if sweep.side == "high" else "long"
        choch = _first_choch(breaks, after=sweep.index, within=win_choch, direction=want)
        if choch is None:
            continue
        stop = _stop_from_sweep(sweep, want, cfg["stop_buffer_pct"])
        setup = _make_setup(bars, cfg, choch.index + 1, want, stop)
        if setup is not None:
            out.append(setup)
    return out


def main() -> None:
    from backtest.run import load

    print("Veri yukleniyor...", flush=True)
    bars = load(rule="5min")
    print(f"  {len(bars):,} bar\n", flush=True)

    cfg = DEFAULT_CONFIG
    variants = {
        "1_yalniz_supurme": build_layer1(bars, cfg),
        "2_supurme_choch": build_layer2(bars, cfg),
        "3_tam_zincir": find_setups(bars, cfg),
    }

    rows = []
    trades_by_variant = {}
    for name, setups in variants.items():
        trades = evaluate(bars, setups)
        trades_by_variant[name] = trades
        s = summarize(trades)
        rows.append({
            "katman": name,
            "islem": s.get("islem", 0),
            "kazanma_%": s.get("kazanma_orani", 0.0),
            "ortalama_R": s.get("ortalama_R", 0.0),
            "toplam_R": s.get("toplam_R", 0.0),
            "maks_dusus_R": round(max_drawdown_r(trades), 1) if trades else 0.0,
        })

    df = pd.DataFrame(rows)
    print("=" * 72)
    print("KATMAN AYRISTIRMA TESTI")
    print("=" * 72)
    print(df.to_string(index=False))

    print("\nYillara gore tutarlilik (kac yil ortalama_R > 0):")
    for name, trades in trades_by_variant.items():
        by = by_year(trades)
        pos = int((by["ortalama_R"] > 0).sum()) if not by.empty else 0
        total = len(by)
        print(f"  {name:20} {pos}/{total} yil pozitif")

    print()
    positives = [r for r in rows if r["ortalama_R"] > 0]
    chain_positive = rows[-1]["ortalama_R"] > 0
    if positives and not chain_positive:
        print("-> Tam zincir negatif ama daha basit bir katman pozitif:")
        print("   IFVG/retest sarti gereksiz bir kisitlama olabilir - daha basit,")
        print("   calisan bir model bu verilerin arasinda duruyor olabilir.")
    elif not positives:
        print("-> Hicbir katman kombinasyonu pozitif degil: SMC'nin bu kurgusu")
        print("   NAS100 5 dakikalikta calismiyor gibi gorunuyor.")
    else:
        print("-> Tum katmanlar pozitif; ek katmanlar sinyali bozmuyor.")

    print("\nUYARI: bu bir parametre taramasi degil. Pozitif cikan varyant")
    print("varsa canliya/karara gecmeden ONCE random_control ve lookahead_check")
    print("ile ayni disiplinle sinanmali - burada yalniz kesif yapiliyor.")


if __name__ == "__main__":
    main()
