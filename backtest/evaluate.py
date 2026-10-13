"""
Kurulumlarin sonucunu olcer: stop mu once vuruldu, hedef mi?

Iki kural bu dosyanin durustlugunu belirliyor:

1. AYNI BARDA IKISI DE VURULURSA STOP SAYILIR. Bar ici yolu bilmiyoruz;
   dakikalik barda bile fiyatin once nereye gittigi belirsiz. Belirsizligi
   kendi lehimize cozmek backtest'i sistematik olarak iyimser yapar - bu
   hata sessizdir ve sonuclari kullanilamaz hale getirir.

2. ISLEM MALIYETI HER ISLEMDEN DUSULUR. HistData barlari spread icermiyor,
   bu yuzden sabit ve KONSERVATIF bir maliyet varsayiyoruz. BTC projesinde
   komisyon modellenince stratejinin oldugu gorulmustu; day trade'de islem
   sayisi cok daha yuksek oldugu icin etki daha da buyuk.

Sonuclar R cinsinden olculuyor (1R = bir islemde riske edilen mesafe).
Boylece pozisyon buyuklugunden bagimsiz, saf strateji kalitesi gorulur.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.news import in_news_window
from strategy.setup import Setup, _bar_minutes

# Islem BASINA (gidis-donus) toplam maliyet.
#
# 2026-09-13'te 2.0'dan YUKSELTILDI. Onceki deger bir tahmindi ("tipik spread
# ~1 puan"); `backtest/cost_reality.py` Dukascopy'den inen gercek tick
# verisini MODELIN ISLEM YAPTIGI killzone saatlerine filtreleyip olctu:
#
#   killzone ortalama spread : 1.304 puan (tek yon)
#   gidis-donus              : 2.61 puan
#   + kayma payi 0.5         : 3.11 puan
#
# Yani 2.0 varsayimi gercegin %55 altindaydi. 3.5'e cikarilmasinin sebebi
# olculen degerin de bir ALT SINIR olmasi: ornek yalnizca 5 SAKIN GUNU
# kapsiyor, haber gunleri (NFP/TUFE/FOMC) yok ve spread tam da o anlarda
# aciliyor. Aradaki 0.4 puan o bilinen eksigin payi - bir tahmin, ama
# yonu bilinen bir tahmin.
DEFAULT_COST_POINTS = 3.5

# Bir kurulum sonsuza kadar acik kalmaz: bir islem seansi (8 saat) sonunda
# kapatilir. SAAT cinsinden tanimli, bar cinsinden DEGIL - bar sayisi zaman
# dilimine gore anlam degistirir (96 bar 5dk'da 8 saat, 15dk'da 24 saat) ve
# bu sessizce farkli bir strateji olur. Onceki deger 480 bardi (40 saat),
# "day trade, pozisyon gecelemez" yazilmasina ragmen.
DEFAULT_MAX_HOURS = 8.0


@dataclass
class Trade:
    setup: Setup
    outcome: str        # 'target' | 'stop' | 'timeout'
    exit_index: int
    exit_time: pd.Timestamp
    exit_price: float
    r_multiple: float


def evaluate(bars: pd.DataFrame, setups: list[Setup],
             cost_points: float = DEFAULT_COST_POINTS,
             max_hours: float = DEFAULT_MAX_HOURS,
             cost_pct: float | None = None,
             stop_slip_points: float | None = None,
             news_exit: bool = False) -> list[Trade]:
    """Her kurulumu giris barindan itibaren ileri oynatir.

    `cost_pct` verilirse maliyet fiyatin yuzdesi olarak hesaplanir ve
    `cost_points` yok sayilir. Puan cinsinden maliyet yalnizca tek bir
    enstrüman icin anlamlidir: 4 puan NAS100'de ~%0.014, EURUSD'de saçma
    bir rakam. Enstrümanlar arasi karsilastirmada daima yuzde kullanilmali.

    `stop_slip_points` verilirse KAYMA AYRI MODELLENIR: `cost_points` yalnizca
    spread sayilir, kayma SADECE stop ile kapanan islemlere eklenir.

    Neden: varsayilan model kaymayi her isleme yukluyor. Ama hedef bir LIMIT
    emridir - fiyat oraya gelmeden dolmaz, dolarsa da istenen fiyattan dolar,
    yani kaymaz. Kayma esasen STOP'ta olusur (islemlerin ~%32'si). Kaymayi
    kazanan islemlere de yuklemek modeli haksiz cezalandiriyordu; buna karsilik
    stop'taki gercek kayma 0,5 puandan buyuk olabilir. Iki hata birbirini
    maskeliyordu - bu parametre ikisini ayirir.

    `news_exit=True`: acik pozisyon, haber penceresinin ACILDIGI bardan BIR
    BAR ONCE kapatilir ('news' sonucu). Bu gelecege bakma DEGIL - haber saati
    takvimden onceden bilinir, fiyattan degil. Gerekcesi olculdu: haber
    aninda stop kaymasi 88 puana cikiyor (bkz. `signals/news.py`).
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    news = in_news_window(bars.index).to_numpy() if news_exit else None
    # Zaman siniri bar sayisina, verinin kendi bar araligina gore cevriliyor.
    max_bars = max(1, int(round(max_hours * 60 / _bar_minutes(bars))))

    trades: list[Trade] = []
    for s in setups:
        risk = s.risk
        if risk <= 0:
            continue

        outcome, exit_i, exit_px = "timeout", min(n - 1, s.entry_index + max_bars), None
        # GIRIS BARI DAHIL. Once atlanmisti; o hal "girdigin barda stop
        # yiyemezsin" demekti ve sonuclari sistematik olarak iyimser yapiyordu.
        # Girisin ayni barda vurulmasi gercek hayatta en sik yasanan seydir.
        for i in range(s.entry_index, min(n, s.entry_index + max_bars + 1)):
            # Haber penceresi BIR SONRAKI barda aciliyorsa, bu barin
            # kapanisinda cikilir. Kontrol stop/hedeften ONCE: gercekte de
            # pozisyon o bar baslamadan kapanmis olur.
            if news is not None and i + 1 < n and news[i + 1]:
                outcome, exit_i, exit_px = "news", i, float(close[i])
                break

            if s.direction == "long":
                hit_stop = low[i] <= s.stop
                hit_target = high[i] >= s.target
            else:
                hit_stop = high[i] >= s.stop
                hit_target = low[i] <= s.target

            # Ikisi ayni barda: STOP. Bar ici yolu bilmiyoruz, belirsizligi
            # kendi lehimize cozmek yasak.
            if hit_stop:
                outcome, exit_i, exit_px = "stop", i, s.stop
                break
            if hit_target:
                outcome, exit_i, exit_px = "target", i, s.target
                break

        if exit_px is None:
            exit_px = close[exit_i]

        gross = (exit_px - s.entry_price) if s.direction == "long" else (s.entry_price - exit_px)
        cost = (s.entry_price * cost_pct / 100.0) if cost_pct is not None else cost_points
        # Kayma yalnizca stop'ta: limit emir istenen fiyattan dolar, stop
        # emri hareketin icinde tetiklenir ve bir sonraki fiyattan dolar.
        if stop_slip_points is not None and outcome == "stop":
            cost += stop_slip_points
        net = gross - cost
        trades.append(Trade(setup=s, outcome=outcome, exit_index=exit_i,
                            exit_time=bars.index[exit_i], exit_price=float(exit_px),
                            r_multiple=float(net / risk)))

    return trades


def summarize(trades: list[Trade]) -> dict:
    """Strateji kalitesinin ozet tablosu."""
    if not trades:
        return {"islem": 0}

    r = np.array([t.r_multiple for t in trades])
    wins = r[r > 0]
    losses = r[r <= 0]

    return {
        "islem": len(trades),
        "kazanma_orani": round(len(wins) / len(r) * 100, 1),
        "toplam_R": round(float(r.sum()), 1),
        "ortalama_R": round(float(r.mean()), 3),
        "medyan_R": round(float(np.median(r)), 3),
        "ort_kazanc_R": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "ort_kayip_R": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "en_iyi_R": round(float(r.max()), 2),
        "en_kotu_R": round(float(r.min()), 2),
        "hedef": sum(1 for t in trades if t.outcome == "target"),
        "stop": sum(1 for t in trades if t.outcome == "stop"),
        "zaman_asimi": sum(1 for t in trades if t.outcome == "timeout"),
    }


def equity_curve(trades: list[Trade]) -> pd.Series:
    """R cinsinden kumulatif getiri (zaman sirali)."""
    if not trades:
        return pd.Series(dtype=float)
    ordered = sorted(trades, key=lambda t: t.exit_time)
    return pd.Series(np.cumsum([t.r_multiple for t in ordered]),
                     index=[t.exit_time for t in ordered])


def max_drawdown_r(trades: list[Trade]) -> float:
    """R cinsinden en buyuk tepe-dip dususu."""
    curve = equity_curve(trades)
    if curve.empty:
        return 0.0
    return float((curve.cummax() - curve).max())


def by_year(trades: list[Trade]) -> pd.DataFrame:
    """Yillara gore dagilim.

    Tek bir toplam sayi yaniltir: strateji birkac iyi yildan mi yasiyor,
    yoksa her rejimde ayakta mi? Altin ve BTC projelerinde asil bilgi hep
    bu ayrimda cikti.
    """
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "yil": t.exit_time.year, "R": t.r_multiple,
        "kazanc": 1 if t.r_multiple > 0 else 0,
    } for t in trades])
    out = df.groupby("yil").agg(
        islem=("R", "size"), toplam_R=("R", "sum"),
        ortalama_R=("R", "mean"), kazanma_orani=("kazanc", "mean"))
    out["toplam_R"] = out["toplam_R"].round(1)
    out["ortalama_R"] = out["ortalama_R"].round(3)
    out["kazanma_orani"] = (out["kazanma_orani"] * 100).round(1)
    return out
