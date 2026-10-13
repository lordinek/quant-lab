"""
Gunluk raporu: ne oldugu degil, NEDEN oldugu.

Broker ekrani "ne kadar kaybettin" sorusunu zaten cevapliyor. Burada aranan
sey desen: hangi kosulda kaybediyorsun.

Uc kadran var:
  1. Temel  - kazanma orani, ortalama kazanc/kayip, beklenen deger.
  2. Bot ile uyum - botun onayladigi islemler mi kazandiriyor, digerleri mi.
  3. Duygusal izler - intikam islemi, plan disi giris, his etiketleri.

AZ VERIYLE ISTATISTIK YAPILMAZ. Her kadran kendi esigini kontrol ediyor ve
yetersizse sayiyi sunmak yerine "henuz yeterli veri yok" diyor. Uc islemden
"kazanma oranin %33" cikarmak bilgi degil gurultudur ve yanlis karara goturur;
bu projede ayni ilke strateji tarafinda da uygulandi (bkz. saat filtresi).
"""
from journal.context import verdict
from journal.store import Entry, to_local

# Bir grup hakkinda konusmak icin gereken en az islem sayisi.
MIN_SAMPLE = 8
# Bir "intikam islemi" suphesi icin: onceki kayiptan sonra bu kadar sure
# icinde acilan islem.
REVENGE_MINUTES = 30.0


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _stats(entries: list[Entry]) -> dict:
    if not entries:
        return {"islem": 0}
    wins = [e for e in entries if e.won]
    losses = [e for e in entries if not e.won]
    total = sum(e.pnl for e in entries)
    return {
        "islem": len(entries),
        "kazanma_%": round(_pct(len(wins), len(entries)), 1),
        "toplam_$": round(total, 2),
        "islem_basi_$": round(total / len(entries), 2),
        "ort_kazanc_$": round(sum(e.pnl for e in wins) / len(wins), 2) if wins else 0.0,
        "ort_kayip_$": round(sum(e.pnl for e in losses) / len(losses), 2) if losses else 0.0,
    }


def basics(entries: list[Entry]) -> dict:
    return _stats(entries)


def by_bot_agreement(entries: list[Entry]) -> dict[str, dict]:
    """Botun goruşune gore gruplama.

    'bilinmiyor' grubu ayri duruyor ve bir yargi DEGIL: bot o an kapaliydi,
    susmasi ret sayilmaz.
    """
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        groups.setdefault(verdict(e.bot, e.direction), []).append(e)
    return {k: _stats(v) for k, v in sorted(groups.items())}


def emotional_markers(entries: list[Entry]) -> dict[str, dict]:
    """Duygusal islem izleri.

    Her isaret bir IKILI karsilastirma: isaretin oldugu islemler vs olmadigi.
    Tek basina "intikam islemlerinin kazanma orani %40" bir sey soylemez;
    digerlerinin %65 oldugunu gorunce soyler.
    """
    if len(entries) < 2:
        return {}

    ordered = sorted(entries, key=lambda e: e.opened_at)
    revenge: list[Entry] = []
    calm: list[Entry] = []
    bigger_after_loss: list[Entry] = []
    normal_after_loss: list[Entry] = []

    avg_size = sum(e.size for e in ordered) / len(ordered)

    for prev, cur in zip(ordered, ordered[1:]):
        gap_min = (cur.opened - prev.closed).total_seconds() / 60.0
        if not prev.won and 0 <= gap_min <= REVENGE_MINUTES:
            revenge.append(cur)
        else:
            calm.append(cur)

        if not prev.won:
            (bigger_after_loss if cur.size > avg_size
             else normal_after_loss).append(cur)

    out = {
        "kayiptan_hemen_sonra": _stats(revenge),
        "sogukkanli": _stats(calm),
        "kayiptan_sonra_buyuttu": _stats(bigger_after_loss),
        "kayiptan_sonra_buyutmedi": _stats(normal_after_loss),
    }

    off_plan = [e for e in entries if not e.followed_plan]
    on_plan = [e for e in entries if e.followed_plan]
    out["plan_disi"] = _stats(off_plan)
    out["plana_uygun"] = _stats(on_plan)
    return out


def by_feeling(entries: list[Entry]) -> dict[str, dict]:
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        key = (e.feeling or "-").strip().lower()
        groups.setdefault(key, []).append(e)
    return {k: _stats(v) for k, v in sorted(groups.items())}


def strongest_signal(entries: list[Entry]) -> str | None:
    """Kazanma oranini en cok ayiran ikili hangisi.

    "Duygusal islem yaptigimi ilk ele veren metrik" sorusunun veriye dayali
    cevabi. Yeterli ornek yoksa None doner - tahmin yurutulmez.
    """
    pairs = [
        ("botun onayladigi vs onaylamadigi", "onayladi", "sustu"),
    ]
    agreement = by_bot_agreement(entries)
    best: tuple[str, float] | None = None

    for label, a, b in pairs:
        sa, sb = agreement.get(a, {}), agreement.get(b, {})
        if sa.get("islem", 0) >= MIN_SAMPLE and sb.get("islem", 0) >= MIN_SAMPLE:
            diff = abs(sa["kazanma_%"] - sb["kazanma_%"])
            if best is None or diff > best[1]:
                best = (label, diff)

    markers = emotional_markers(entries)
    for label, a, b in [
        ("kayiptan hemen sonra girmek", "kayiptan_hemen_sonra", "sogukkanli"),
        ("kayiptan sonra buyutmek", "kayiptan_sonra_buyuttu", "kayiptan_sonra_buyutmedi"),
        ("plan disina cikmak", "plan_disi", "plana_uygun"),
    ]:
        sa, sb = markers.get(a, {}), markers.get(b, {})
        if sa.get("islem", 0) >= MIN_SAMPLE and sb.get("islem", 0) >= MIN_SAMPLE:
            diff = abs(sa["kazanma_%"] - sb["kazanma_%"])
            if best is None or diff > best[1]:
                best = (label, diff)

    if best is None:
        return None
    return f"{best[0]} ({best[1]:.0f} puan fark)"


def render(entries: list[Entry]) -> str:
    """Raporun tamamini metin olarak uretir."""
    if not entries:
        return ("Gunluk bos. Ilk islemi eklemek icin:\n"
                "    python -m journal ekle")

    lines = []
    add = lines.append

    first = to_local(entries[0].opened_at).strftime("%d.%m.%Y")
    last = to_local(entries[-1].opened_at).strftime("%d.%m.%Y")
    add(f"ELLE ISLEM GUNLUGU  ({first} - {last}, {len(entries)} islem)")
    add("=" * 62)

    b = basics(entries)
    add(f"  kazanma orani   %{b['kazanma_%']}")
    add(f"  toplam          ${b['toplam_$']:+,.2f}")
    add(f"  islem basina    ${b['islem_basi_$']:+,.2f}")
    add(f"  ort. kazanc     ${b['ort_kazanc_$']:+,.2f}")
    add(f"  ort. kayip      ${b['ort_kayip_$']:+,.2f}")

    if len(entries) < MIN_SAMPLE:
        add("")
        add(f"  (Kirilimlar {MIN_SAMPLE} islemden sonra acilir - daha az sayiyla")
        add("   yapilan cikarim bilgi degil gurultudur.)")
        return "\n".join(lines)

    add("")
    add("BOTUN GORUSUNE GORE")
    add("-" * 62)
    aciklama = {
        "onayladi": "bot ayni yonde islem acti",
        "ters": "bot TERS yonde islem acti",
        "sustu": "bot calisiyordu, islem acmadi",
        "bilinmiyor": "bot kapaliydi (yargi degil)",
    }
    for key, s in by_bot_agreement(entries).items():
        if not s.get("islem"):
            continue
        note = aciklama.get(key, "")
        add(f"  {key:12} {s['islem']:>3} islem | %{s['kazanma_%']:>5.1f} kazanma | "
            f"${s['toplam_$']:>9,.2f}   {note}")

    add("")
    add("DUYGUSAL IZLER")
    add("-" * 62)
    for key, s in emotional_markers(entries).items():
        if not s.get("islem"):
            continue
        add(f"  {key:26} {s['islem']:>3} islem | %{s['kazanma_%']:>5.1f} kazanma | "
            f"${s['toplam_$']:>9,.2f}")

    feelings = {k: v for k, v in by_feeling(entries).items()
                if v.get("islem", 0) >= 3 and k != "-"}
    if feelings:
        add("")
        add("HIS ETIKETINE GORE (en az 3 islem)")
        add("-" * 62)
        for key, s in feelings.items():
            add(f"  {key:26} {s['islem']:>3} islem | %{s['kazanma_%']:>5.1f} kazanma | "
                f"${s['toplam_$']:>9,.2f}")

    add("")
    signal = strongest_signal(entries)
    if signal:
        add(f"EN COK AYIRAN ETKEN: {signal}")
    else:
        add(f"EN COK AYIRAN ETKEN: henuz belirlenemedi - bir kirilimin iki")
        add(f"tarafinda da en az {MIN_SAMPLE} islem birikmesi gerekiyor.")

    return "\n".join(lines)
