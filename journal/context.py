"""
Botun bir andaki gorusunu okur.

Bu dosya gunlugu siradan bir islem tablosundan ayiran parca. Bot zaten her
5 dakikada bir ne dusundugunu `live/paper_history.jsonl`'a yaziyor: fiyat,
bakiye, acik pozisyon sayisi ve o barda olan olaylar ("ACILDI long @ ...").
Elle acilan bir islemi o kayitla eslestirince su olculebilir hale geliyor:

    botun onayladigi islemlerde mi kaybediyorum, onaylamadiklarinda mi?

DURUSTLUK NOTU: bot yalnizca CALISIRKEN kayit tutar. Bot kapaliyken yapilmis
bir islem icin "bot onaylamadi" DEMEK YANLISTIR - bot o an bir sey demedi.
Bu ayrim `available` alaniyla korunuyor; rapor, bot susmus islemleri ayri
sayiyor. Sessizligi ret saymak, olcumu sessizce kullanicinin aleyhine egerdi.
"""
import json
import os

import pandas as pd

HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "live", "paper_history.jsonl")

# Bir islemin bot kaydiyla eslestigi kabul edilen azami zaman farki.
# 5 dakikalik barlarda calisiyoruz; 10 dakika iki bar demek, makul tolerans.
MATCH_MINUTES = 10.0

# Botun "yakin zamanda ayni yonde islem acti" penceresi. Elle giris ile botun
# girisi ayni saniyede olmaz; ayni FIKRI paylasip paylasmadiklarina bakiyoruz.
AGREEMENT_MINUTES = 45.0


def _load_history(path: str = HISTORY_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _opened_direction(events: list[str]) -> str | None:
    """Bir bardaki olaylardan botun actigi pozisyonun yonu."""
    for e in events:
        if e.startswith("ACILDI"):
            parts = e.split()
            if len(parts) > 1:
                return parts[1]
    return None


def bot_view(when: pd.Timestamp, path: str = HISTORY_PATH) -> dict:
    """`when` aninda bot ne goruyordu.

    Doner:
      available   : bot o an calisiyor muydu (kayit var mi)
      price       : botun gordugu fiyat
      balance     : bot bakiyesi
      open_positions : botun tasidigi pozisyon sayisi
      signal      : bot o barda pozisyon acti mi (yon) / None
      agreed      : +-45 dk icinde bot ayni yonde bir pozisyon acti mi
      agreed_dir  : o pencerede botun actigi yon(ler)
    """
    rows = _load_history(path)
    if not rows:
        return {"available": False, "reason": "bot kaydi yok"}

    when = pd.Timestamp(when)
    if when.tzinfo is None:
        when = when.tz_localize("UTC")

    stamps = [pd.Timestamp(r["ts"]) for r in rows]
    deltas = [abs((s - when).total_seconds()) / 60.0 for s in stamps]
    best = min(range(len(deltas)), key=lambda i: deltas[i])

    if deltas[best] > MATCH_MINUTES:
        return {"available": False,
                "reason": f"bot o an kapaliydi (en yakin kayit "
                          f"{deltas[best]:.0f} dk uzakta)"}

    row = rows[best]
    window = [r for r, s in zip(rows, stamps)
              if abs((s - when).total_seconds()) / 60.0 <= AGREEMENT_MINUTES]
    dirs = {d for r in window if (d := _opened_direction(r.get("events", [])))}

    return {
        "available": True,
        "ts": row["ts"],
        "price": row.get("price"),
        "balance": row.get("balance"),
        "open_positions": row.get("open_positions", 0),
        "signal": _opened_direction(row.get("events", [])),
        "agreed_dirs": sorted(dirs),
    }


def verdict(view: dict, direction: str) -> str:
    """Botun bu isleme dair goruşunu tek kelimeye indirger.

    'onayladi'   - bot ayni pencerede AYNI yonde pozisyon acti
    'ters'       - bot ayni pencerede TERS yonde pozisyon acti
    'sustu'      - bot calisiyordu ama o pencerede islem acmadi
    'bilinmiyor' - bot kapaliydi; bu bir ret DEGIL
    """
    if not view.get("available"):
        return "bilinmiyor"
    dirs = view.get("agreed_dirs") or []
    if direction in dirs:
        return "onayladi"
    if dirs:
        return "ters"
    return "sustu"
