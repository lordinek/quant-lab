"""
Gunluk komut satiri.

    python -m journal ekle     islem ekle (soru-cevap)
    python -m journal rapor    desen raporu
    python -m journal liste    son islemler

Tasarim kurali: EKLEMEK KOLAY OLMALI. Kullanilmayan gunluk, olmayan gunluktur.
Bu yuzden her soru tek satir, cogu opsiyonel, Enter gecer.
"""
import sys

from journal.context import bot_view, verdict
from journal.report import render
from journal.store import Entry, append_entry, load_entries, to_local, to_utc


def ask(soru: str, default: str = "") -> str:
    ek = f" [{default}]" if default else ""
    cevap = input(f"  {soru}{ek}: ").strip()
    return cevap or default


def ask_float(soru: str, default: str = "") -> float:
    while True:
        raw = ask(soru, default).replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("    (sayi girmelisin, or. 29500.5)")


def ask_time(soru: str) -> str:
    while True:
        raw = ask(soru)
        if not raw:
            print("    (bos birakilamaz - or. '14:30' ya da '2026-09-12 14:30')")
            continue
        try:
            return to_utc(raw).isoformat()
        except Exception:
            print("    (anlasilmadi - or. '14:30' ya da '2026-09-12 14:30')")


def cmd_ekle() -> None:
    print("\nYeni islem (yerel saat, Enter = varsayilan)\n")

    yon = ""
    while yon not in ("long", "short"):
        raw = ask("Yon (l=long / s=short)").lower()
        yon = {"l": "long", "long": "long", "s": "short", "short": "short"}.get(raw, "")

    opened = ask_time("Giris saati")
    closed = ask_time("Cikis saati")
    entry_price = ask_float("Giris fiyati")
    exit_price = ask_float("Cikis fiyati")
    size = ask_float("Buyukluk (lot)", "1")

    move = (exit_price - entry_price) if yon == "long" else (entry_price - exit_price)
    print(f"    -> fiyat hareketi: {move:+.1f} puan ({'lehine' if move > 0 else 'aleyhine'})")

    pnl = ask_float("Gerceklesen kar/zarar ($, zarar icin eksi)")
    plan = ask("Girmeden once plan neydi?")
    feeling = ask("O an ne hissettin? (tek kelime: sabirsiz, emin, korkmus...)")
    followed = ask("Planina uydun mu? (e/h)", "e").lower().startswith("e")

    view = bot_view(opened)
    entry = Entry(opened_at=opened, closed_at=closed, direction=yon,
                  entry_price=entry_price, exit_price=exit_price, size=size,
                  pnl=pnl, plan=plan, feeling=feeling, followed_plan=followed,
                  bot=view)
    append_entry(entry)

    print("\n  kaydedildi.")
    v = verdict(view, yon)
    mesaj = {
        "onayladi": "Bot da ayni yonde islem acmisti.",
        "ters": "Bot TERS yonde islem acmisti.",
        "sustu": "Bot o sirada calisiyordu ama islem acmadi.",
        "bilinmiyor": f"Bot o an kapaliydi - gorusu yok "
                      f"({view.get('reason', '')}).",
    }[v]
    print(f"  Botun gorusu: {v} - {mesaj}")

    n = len(load_entries())
    print(f"  Gunlukte {n} islem var. Rapor: python -m journal rapor\n")


def cmd_rapor() -> None:
    print()
    print(render(load_entries()))
    print()


def cmd_liste(limit: int = 15) -> None:
    entries = load_entries()
    if not entries:
        print("Gunluk bos.")
        return
    print(f"\nSon {min(limit, len(entries))} islem:\n")
    for e in entries[-limit:]:
        t = to_local(e.opened_at).strftime("%d.%m %H:%M")
        v = verdict(e.bot, e.direction)
        print(f"  {t}  {e.direction:5} {e.entry_price:>9,.1f} -> {e.exit_price:>9,.1f} "
              f"| ${e.pnl:>8,.2f} | bot: {v}")
    print()


def main() -> None:
    komut = sys.argv[1] if len(sys.argv) > 1 else "rapor"
    if komut in ("ekle", "add"):
        cmd_ekle()
    elif komut in ("liste", "list"):
        cmd_liste()
    elif komut in ("rapor", "report"):
        cmd_rapor()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
