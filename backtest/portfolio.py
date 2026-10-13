"""
Gercek hesap simulasyonu: R cinsinden olcumu paraya cevirir.

R metrigi strateji KALITESINI olcer ama iki gercegi gizler:

  1. AYNI ANDA kac pozisyon acik oldugunu. Olcum 16 es zamanli pozisyon
     gosterdi; $150'lik bir hesapta bu tasinamaz.
  2. POZISYON BUYUKLUGUNU. Bir CFD pozisyonunun degeri hesaptan buyuk
     olabilir - altin botunda tam da bu sessizce 30x kaldirac acmisti.

Burada ikisi de acikca modelleniyor: slot doluysa kurulum ATLANIR (gercek
hayatta olan budur), ve kaldirac tavani asilirsa pozisyon kucultulur.

Bilesik buyume var: risk her zaman O ANKI bakiyenin yuzdesi. Bu kazanirken
hizlandirir, kaybederken yavaslatir - sabit lot ile taban tabana zit bir
risk profili, ve $150'den buyutme hedefinin dogal karsiligi.
"""
from dataclasses import dataclass

import pandas as pd

from backtest.evaluate import Trade
from strategy.protections import Guard, ProtectionConfig

# NAS100 CFD'de tipik perakende kaldirac tavani (1:20 -> 20x).
DEFAULT_MAX_LEVERAGE = 20.0


@dataclass
class Position:
    trade: Trade
    size: float          # kac birim (CFD "lot" karsiligi)
    risk_cash: float     # bu islemde riske edilen para
    entry_index: int
    exit_index: int


@dataclass
class Result:
    starting_balance: float
    final_balance: float
    trades_taken: int
    trades_skipped: int
    max_drawdown_pct: float
    equity: pd.Series
    peak_concurrent: int
    # Koruma katmani yuzunden atlananlar `trades_skipped`ten AYRI sayilir:
    # slot dolulugu ile "bot kendini durdurdu" ayni sey degil.
    trades_blocked: int = 0
    locks: int = 0


def simulate(bars: pd.DataFrame, trades: list[Trade],
             starting_balance: float = 150.0,
             risk_pct: float = 1.0,
             max_concurrent: int = 3,
             max_leverage: float = DEFAULT_MAX_LEVERAGE,
             compound: bool = True,
             max_notional: float | None = None,
             protection: ProtectionConfig | None = None) -> Result:
    """Kurulumlari zaman sirasinda gercek bir hesapta oynatir.

    `max_concurrent`: ayni anda tasinabilecek azami pozisyon. Dolu oldugunda
    yeni kurulum ATLANIR - sonradan "en iyilerini secmek" gelecege bakmaktir,
    o an hangisinin kazanacagi bilinmiyor.

    `compound=False`: risk her zaman BASLANGIC bakiyesinin yuzdesi. Bilesik
    modda $150 -> $872 MILYON cikti; matematik dogru ama gercek degil, cunku
    o buyuklukte pozisyonlar piyasayi kendileri hareket ettirir ve model
    5,7 yil boyunca bozulmadan calismaz. Bilesiksiz mod stratejinin saf
    performansini gosterir - abartisiz bir taban.

    `max_notional`: tek pozisyonun azami buyuklugu ($). Likidite gercegi:
    hesap ne kadar buyurse buyusun, 5 dakikalik bir kurulumda sinirsiz
    buyuklukte emir gecirilemez.
    """
    ordered = sorted(trades, key=lambda t: t.setup.entry_index)
    balance = starting_balance
    open_positions: list[Position] = []
    taken = skipped = blocked = 0
    peak_concurrent = 0
    guard = Guard(protection, starting_balance=starting_balance)

    curve_idx, curve_val = [], []

    for t in ordered:
        entry_i = t.setup.entry_index

        # Bu kurulum acilmadan ONCE kapanmis pozisyonlari tahsil et.
        # Kapanislar KRONOLOJIK sirayla islenir: koruma katmani "arka arkaya N
        # stop" gibi siraya bagli sorular soruyor ve listenin kendi sirasi
        # cikis sirasi degil.
        closing = sorted((p for p in open_positions if p.exit_index <= entry_i),
                         key=lambda p: p.exit_index)
        open_positions = [p for p in open_positions if p.exit_index > entry_i]
        for p in closing:
            balance += p.risk_cash * p.trade.r_multiple
            closed_at = bars.index[p.exit_index]
            curve_idx.append(closed_at)
            curve_val.append(balance)
            guard.record_close(closed_at, p.trade.outcome, balance)

        if len(open_positions) >= max_concurrent:
            skipped += 1
            continue

        if balance <= 0:
            break

        # Koruma katmani: bot kendini durdurmus olabilir. Kontrol pozisyon
        # acmadan ONCE ve yalnizca o ana kadar KAPANMIS islemlere dayanarak.
        if guard.blocked(bars.index[entry_i]):
            blocked += 1
            continue

        base = balance if compound else starting_balance
        risk_cash = base * risk_pct / 100.0
        stop_distance = t.setup.risk
        if stop_distance <= 0:
            skipped += 1
            continue

        size = risk_cash / stop_distance
        notional = size * t.setup.entry_price

        # Kaldirac tavani: pozisyon degeri hesabin katini asamaz. Asiyorsa
        # kuculturuz - islemi atlamak yerine, gercek bir broker da boyle
        # davranir (ya da emri reddeder; kuculmek daha temkinli varsayim).
        if notional > balance * max_leverage:
            size = balance * max_leverage / t.setup.entry_price
            risk_cash = size * stop_distance
            notional = size * t.setup.entry_price

        # Likidite tavani: emrin kendisi piyasayi hareket ettirecek kadar
        # buyuyemez. Bilesik modda bu sinir olmadan sonuc astronomik cikiyor
        # ve sayi anlamsizlasiyor.
        if max_notional is not None and notional > max_notional:
            size = max_notional / t.setup.entry_price
            risk_cash = size * stop_distance

        open_positions.append(Position(trade=t, size=size, risk_cash=risk_cash,
                                       entry_index=entry_i, exit_index=t.exit_index))
        taken += 1
        peak_concurrent = max(peak_concurrent, len(open_positions))

    # Kalan acik pozisyonlari tahsil et.
    for p in sorted(open_positions, key=lambda p: p.exit_index):
        balance += p.risk_cash * p.trade.r_multiple
        curve_idx.append(bars.index[p.exit_index])
        curve_val.append(balance)

    equity = pd.Series(curve_val, index=curve_idx).sort_index()
    if equity.empty:
        max_dd = 0.0
    else:
        peak = equity.cummax()
        max_dd = float(((peak - equity) / peak * 100).max())

    return Result(starting_balance=starting_balance, final_balance=balance,
                  trades_taken=taken, trades_skipped=skipped,
                  max_drawdown_pct=round(max_dd, 1), equity=equity,
                  peak_concurrent=peak_concurrent,
                  trades_blocked=blocked, locks=len(guard.lock_log))


def summarize(result: Result, years: float) -> dict:
    growth = result.final_balance / result.starting_balance
    cagr = (growth ** (1 / years) - 1) * 100 if years > 0 and growth > 0 else float("nan")
    return {
        "baslangic": round(result.starting_balance, 2),
        "bitis": round(result.final_balance, 2),
        "buyume_kat": round(growth, 2),
        "CAGR_%": round(cagr, 1),
        "maks_dusus_%": result.max_drawdown_pct,
        "alinan_islem": result.trades_taken,
        "atlanan_islem": result.trades_skipped,
        "koruma_engeli": result.trades_blocked,
        "kilit_sayisi": result.locks,
        "azami_es_zamanli": result.peak_concurrent,
    }
