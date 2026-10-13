"""
Gunluk kaydi: tek bir JSONL dosyasi.

Neden JSONL ve neden veritabani degil: kayit elle okunabilir olmali. Kullanici bu
dosyayi Not Defteri'nde acip gecmisini gorebilmeli, bir arac gerekmemeli.
Satir basina bir islem, bozulan bir satir digerlerini goturmez.

Zaman daima UTC saklanir, gosterimde yerele cevrilir. Karisik zaman dilimi
bu projede daha once sessiz hata uretti (killzone'lar UTC'ye sabitlenmisti,
yilin yarisinda bir saat kayiyordu) - ayni tuzaga girmemek icin kural net:
DISKTE UTC, EKRANDA YEREL.
"""
import json
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "entries.jsonl")

# kullanicinin yerel saati. Girilen saatler bu dilimde yorumlanir.
LOCAL_TZ = "Europe/Istanbul"


@dataclass
class Entry:
    """Elle yapilmis tek bir islem."""
    opened_at: str              # ISO 8601, UTC
    closed_at: str              # ISO 8601, UTC
    direction: str              # 'long' | 'short'
    entry_price: float
    exit_price: float
    size: float                 # lot / birim
    pnl: float                  # gerceklesmis kar-zarar ($)
    plan: str = ""              # girmeden ONCEKI niyet
    feeling: str = ""           # o anki duygu
    followed_plan: bool = True  # kendi kuralina uydu mu
    bot: dict = field(default_factory=dict)   # botun o andaki gorusu

    @property
    def opened(self) -> pd.Timestamp:
        return pd.Timestamp(self.opened_at)

    @property
    def closed(self) -> pd.Timestamp:
        return pd.Timestamp(self.closed_at)

    @property
    def won(self) -> bool:
        return self.pnl > 0

    @property
    def hours_held(self) -> float:
        return (self.closed - self.opened).total_seconds() / 3600.0


def to_utc(text: str) -> pd.Timestamp:
    """Yerel saatte yazilmis bir zamani UTC'ye cevirir.

    Kabul edilen bicimler: '2026-09-12 14:30', '14:30' (bugun varsayilir).
    """
    text = text.strip()
    if len(text) <= 5 and ":" in text:
        today = pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
        text = f"{today} {text}"
    naive = pd.Timestamp(text)
    if naive.tzinfo is None:
        naive = naive.tz_localize(LOCAL_TZ)
    return naive.tz_convert("UTC")


def to_local(ts: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ)


def load_entries(path: str = JOURNAL_PATH) -> list[Entry]:
    if not os.path.exists(path):
        return []
    out: list[Entry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Entry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                # Bozuk satir digerlerini goturmemeli - JSONL secmenin sebebi bu.
                print(f"  (atlanan bozuk satir: {exc})")
    out.sort(key=lambda e: e.opened_at)
    return out


def append_entry(entry: Entry, path: str = JOURNAL_PATH) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
