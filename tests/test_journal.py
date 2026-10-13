"""
Gunluk testleri.

En kritik iki davranis:
  1. Bot KAPALIYKEN yapilmis islem "bot onaylamadi" sayilmamali. Sessizligi
     ret saymak olcumu sessizce carpitir.
  2. Az veriyle istatistik uretilmemeli - uc islemden kazanma orani cikarmak
     bilgi degil gurultudur.
"""
import json

import pandas as pd
import pytest

from journal.context import bot_view, verdict
from journal.report import MIN_SAMPLE, basics, by_bot_agreement, emotional_markers, render
from journal.store import Entry, load_entries, to_utc


def entry(opened="2026-09-10 12:00", closed="2026-09-10 13:00", direction="long",
          pnl=10.0, size=1.0, feeling="", followed=True, bot=None) -> Entry:
    return Entry(
        opened_at=pd.Timestamp(opened, tz="UTC").isoformat(),
        closed_at=pd.Timestamp(closed, tz="UTC").isoformat(),
        direction=direction, entry_price=29500.0, exit_price=29550.0,
        size=size, pnl=pnl, plan="", feeling=feeling,
        followed_plan=followed, bot=bot if bot is not None else {},
    )


# --- zaman ----------------------------------------------------------------

def test_local_time_is_stored_as_utc():
    """Diskte UTC, ekranda yerel. Karisik zaman dilimi bu projede daha once
    sessiz hata uretmisti (killzone'lar)."""
    utc = to_utc("2026-09-12 14:30")          # Istanbul = UTC+3
    assert utc.tz_convert("UTC").hour == 11


def test_bare_time_means_today():
    utc = to_utc("09:00")
    assert utc.tzinfo is not None


# --- botun gorusu ---------------------------------------------------------

def test_bot_silence_is_not_a_rejection():
    """Bot kapaliyken 'onaylamadi' DENMEZ - o an bir sey demedi."""
    assert verdict({"available": False, "reason": "kapali"}, "long") == "bilinmiyor"


def test_bot_agreement_needs_same_direction():
    view = {"available": True, "agreed_dirs": ["long"]}
    assert verdict(view, "long") == "onayladi"
    assert verdict(view, "short") == "ters"


def test_bot_running_but_idle_is_sustu():
    assert verdict({"available": True, "agreed_dirs": []}, "long") == "sustu"


def test_bot_view_matches_nearest_bar(tmp_path):
    path = tmp_path / "hist.jsonl"
    rows = [
        {"ts": "2026-09-10 11:55:00+00:00", "price": 29400, "balance": 150,
         "open_positions": 0, "events": []},
        {"ts": "2026-09-10 12:00:00+00:00", "price": 29420, "balance": 150,
         "open_positions": 1, "events": ["ACILDI long @ 29420.0 stop ... hedef ..."]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    view = bot_view(pd.Timestamp("2026-09-10 12:01", tz="UTC"), path=str(path))

    assert view["available"] is True
    assert view["signal"] == "long"
    assert verdict(view, "long") == "onayladi"


def test_bot_view_far_from_any_bar_is_unavailable(tmp_path):
    path = tmp_path / "hist.jsonl"
    path.write_text(json.dumps(
        {"ts": "2026-09-10 12:00:00+00:00", "price": 1, "balance": 1,
         "open_positions": 0, "events": []}), encoding="utf-8")

    view = bot_view(pd.Timestamp("2026-09-10 18:00", tz="UTC"), path=str(path))
    assert view["available"] is False


# --- rapor ----------------------------------------------------------------

def test_basics_counts_wins_and_losses():
    s = basics([entry(pnl=10.0), entry(pnl=-5.0), entry(pnl=20.0)])
    assert s["islem"] == 3
    assert s["kazanma_%"] == pytest.approx(66.7, abs=0.1)
    assert s["toplam_$"] == 25.0


def test_small_sample_refuses_breakdowns():
    """Kirilimlar esigin altinda acilmaz - dogru cevap 'bilmiyorum'."""
    out = render([entry() for _ in range(MIN_SAMPLE - 1)])
    assert "Kirilimlar" in out
    assert "BOTUN GORUSUNE GORE" not in out


def test_breakdowns_open_above_threshold():
    out = render([entry(bot={"available": True, "agreed_dirs": ["long"]})
                  for _ in range(MIN_SAMPLE)])
    assert "BOTUN GORUSUNE GORE" in out


def test_revenge_trade_detected_by_time_gap():
    """Kayiptan 10 dk sonra acilan islem 'kayiptan hemen sonra' sayilir."""
    a = entry(opened="2026-09-10 12:00", closed="2026-09-10 12:30", pnl=-20.0)
    b = entry(opened="2026-09-10 12:40", closed="2026-09-10 13:00", pnl=-15.0)

    m = emotional_markers([a, b])

    assert m["kayiptan_hemen_sonra"]["islem"] == 1
    assert m["sogukkanli"]["islem"] == 0


def test_calm_trade_after_long_gap():
    a = entry(opened="2026-09-10 12:00", closed="2026-09-10 12:30", pnl=-20.0)
    b = entry(opened="2026-09-10 20:00", closed="2026-09-10 21:00", pnl=5.0)

    m = emotional_markers([a, b])

    assert m["kayiptan_hemen_sonra"]["islem"] == 0
    assert m["sogukkanli"]["islem"] == 1


def test_off_plan_trades_are_grouped():
    m = emotional_markers([entry(followed=True), entry(followed=False)])
    assert m["plan_disi"]["islem"] == 1
    assert m["plana_uygun"]["islem"] == 1


def test_agreement_groups_are_separate():
    entries = [
        entry(bot={"available": True, "agreed_dirs": ["long"]}),
        entry(bot={"available": False}),
    ]
    g = by_bot_agreement(entries)
    assert g["onayladi"]["islem"] == 1
    assert g["bilinmiyor"]["islem"] == 1


def test_empty_journal_gives_instructions():
    assert "python -m journal ekle" in render([])


def test_corrupt_line_does_not_kill_the_file(tmp_path):
    """JSONL secmenin sebebi: bozuk bir satir digerlerini goturmemeli."""
    path = tmp_path / "entries.jsonl"
    good = json.dumps({
        "opened_at": "2026-09-10T12:00:00+00:00",
        "closed_at": "2026-09-10T13:00:00+00:00",
        "direction": "long", "entry_price": 1.0, "exit_price": 2.0,
        "size": 1.0, "pnl": 5.0,
    })
    path.write_text(good + "\n{bozuk satir\n" + good + "\n", encoding="utf-8")

    assert len(load_entries(str(path))) == 2
