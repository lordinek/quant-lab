"""
Koruma katmani: botun "dur" diyebilmesi.

Bu dosyaya kadar uc botun hicbirinde dur mekanizmasi yoktu. Arka arkaya alti
stop yese hicbiri durmuyordu - model bozuldugunda, rejim degistiginde ya da
veri kaynagi sacmaladiginda bot ayni hizla kaybetmeye devam ediyordu.

Tasarim freqtrade'in `plugins/protections/` katmanindan alindi ama tek
enstrumana sadelestirildi (orada cift bazli kilitler de var, bizde tek sembol
oldugu icin anlamsiz).

DORT KORUMA, hepsi varsayilan olarak KAPALI (deger 0 = kapali):

  stop_streak      arka arkaya N stop yenirse dur.  "Ust uste kaybediyorum."
  stop_count       X saatte N stop yenirse dur.     "Bugun cok kaybettim."
  max_drawdown_pct tepe bakiyeden %N dususte dur.   "Hesap eriyor."
  cooldown_hours   her kapanistan sonra bekle.      "Intikam islemi yapma."

Hepsi kapali dogdu cunku bir korumayi acmak bir PARAMETRE secmektir ve bu
projede parametreler taranip plato aranmadan secilmez (bkz. saat filtresi:
kagit uzerinde harikaydi, egitim/test ayrilinca gecemedi, eklenmedi).
Taramasi: backtest/protection_scan.py

NEDENSELLIK - bu dosyanin en kritik kurali:
Koruma yalnizca KAPANMIS islemlere ve o ana kadarki bakiyeye bakar. Acik
pozisyonun anlik zarari, bir sonraki barin sonucu, "bugunun toplami" gibi
seyler burada YASAK. Bu botun tum hikayesi zaten bir gelecege bakma hatasini
kovalamakla gecti (bkz. _allocate_gaps); koruma katmani ayni tuzagin en kolay
kurulacagi yerdir, cunku "su ana kadar ne kadar kaybettim" sorusu masumca
gorunur ama yanlis yerden sorulunca gelecegi icerir.

Ayni Guard hem backtest'te (backtest/portfolio.py) hem canlida
(live/paper_trader.py) kullanilir. Ikisinin ayni sinifi kullanmasi sart: farkli
iki uygulama yazmak, backtest'in olctugu botun canlidakinden baska bir bot
olmasi demektir.
"""
from dataclasses import dataclass

import pandas as pd

# Bir pozisyonun kapanis sebepleri (evaluate.py ve paper_trader.py ayni
# kelimeleri kullaniyor; koruma yalnizca 'stop' ile ilgileniyor).
STOP = "stop"


@dataclass
class ProtectionConfig:
    """Koruma esikleri. 0 = o koruma kapali."""

    # Arka arkaya kac stop. Zaman penceresi yok: "son N islemin hepsi stop".
    stop_streak: int = 0
    stop_streak_lock_hours: float = 12.0

    # Kayan zaman penceresinde kac stop.
    stop_count: int = 0
    stop_count_hours: float = 24.0
    stop_count_lock_hours: float = 12.0

    # Tepe bakiyeden yuzde kac dusus. Bakiye TEPESI takip edilir, baslangic
    # degil: $150 -> $400 -> $340 bir dusustur, baslangicin uzerinde olsa bile.
    max_drawdown_pct: float = 0.0
    drawdown_lock_hours: float = 24.0

    # Her kapanistan sonra beklenecek sure (intikam islemi frenlemesi).
    cooldown_hours: float = 0.0


# Canlida ve olcumde kullanilan ayar. Backtest ile canlinin AYNI korumayi
# kullanmasi sart; ikisine farkli deger vermek, olctugumuz botun calisan bottan
# baska bir bot olmasi demektir.
#
# Neden yalnizca dusus korumasi ve neden %20:
# `backtest/protection_scan.py` dort korumayi egitim/test ayrimiyla taradi.
# Sonuc: stop_streak, stop_count ve cooldown'un HEPSI zararli - islem sayisini
# kesiyorlar, bakiyeyi dusuruyorlar ve dususu AZALTMIYORLAR (bazen artiriyorlar).
# Sebep anlasilir: model pozitif beklenen degerli ve ardisik stop'lar bir
# bozulma sinyali degil, normal dagilimin parcasi. Islemi engellemek kari
# engellemek oluyor.
#
# %20'lik dusus korumasi ise 5,5 yilda BIR KEZ BILE tetiklenmedi (olculdu:
# bilesik modda bile tepe dusus %13,4). Yani olculen maliyeti TAM SIFIR -
# gecmis veride tek bir islemi bile engellemedi.
#
# O halde neden duruyor: bu bir optimizasyon degil SIGORTA. Gecmiste felaket
# olmadigi icin gecmis veride faydasi da olcülemez; isi, henuz yasanmamis
# senaryoyu karsilamak - model bozulur, rejim degisir, veri kaynagi sacmalar.
# Maliyeti sifir olan bir guvenlik agini reddetmek icin sebep yok.
LIVE_PROTECTION = ProtectionConfig(
    max_drawdown_pct=20.0,
    drawdown_lock_hours=24.0,
)


@dataclass
class Lock:
    """Yururlukteki bir kilit - neden kilitlendigi kayitli kalir."""
    until: pd.Timestamp
    reason: str


class Guard:
    """Islem acilabilir mi sorusuna cevap veren durum makinesi.

    Kullanim (backtest ve canli ayni):
        guard = Guard(config)
        ...
        guard.record_close(kapanis_zamani, "stop", yeni_bakiye)
        ...
        if guard.blocked(simdi):
            # islemi atla
    """

    def __init__(self, config: ProtectionConfig | None = None,
                 starting_balance: float | None = None) -> None:
        self.cfg = config or ProtectionConfig()
        self._outcomes: list[tuple[pd.Timestamp, str]] = []
        # Tepe, ILK kapanistan degil BASLANGICTAN itibaren takip edilir.
        # Aksi halde hic kazanmadan dusen bir hesapta tepe yanlis yerden
        # baslar ve dusus oldugundan kucuk olculur: $150 -> $145 -> $115
        # gercekte %23,3 dususken, tepe $145 sanilirsa %20,7 gorunur.
        self._peak_balance: float | None = starting_balance
        self._last_close: pd.Timestamp | None = None
        self._lock: Lock | None = None
        self.lock_log: list[Lock] = []

    # --- sorgu ------------------------------------------------------------

    def blocked(self, now: pd.Timestamp) -> bool:
        """Su an yeni pozisyon acmak yasak mi."""
        return self.reason(now) is not None

    def reason(self, now: pd.Timestamp) -> str | None:
        """Yasaksa gerekcesi, degilse None."""
        if self._lock is not None and now < self._lock.until:
            return self._lock.reason

        cd = self.cfg.cooldown_hours
        if cd > 0 and self._last_close is not None:
            if now < self._last_close + pd.Timedelta(hours=cd):
                return f"cooldown ({cd:g}sa)"

        return None

    # --- durum guncelleme -------------------------------------------------

    def record_close(self, when: pd.Timestamp, outcome: str,
                     balance: float) -> str | None:
        """Bir pozisyon kapandi; korumalari yeniden degerlendir.

        `balance`: bu kapanis ISLENDIKTEN SONRAKI bakiye.
        Doner: bu kapanis yeni bir kilit tetiklediyse gerekcesi, yoksa None.
        """
        self._outcomes.append((when, outcome))
        self._last_close = when

        if self._peak_balance is None or balance > self._peak_balance:
            self._peak_balance = balance

        triggered = (self._check_streak()
                     or self._check_count(when)
                     or self._check_drawdown(balance))
        if triggered is None:
            return None

        reason, hours = triggered
        until = when + pd.Timedelta(hours=hours)
        # Yururlukteki kilit daha uzunsa uzatilmaz, kisaltilmaz.
        if self._lock is None or until > self._lock.until:
            self._lock = Lock(until=until, reason=reason)
            self.lock_log.append(self._lock)
        return reason

    # --- tek tek korumalar ------------------------------------------------

    def _check_streak(self) -> tuple[str, float] | None:
        n = self.cfg.stop_streak
        if n <= 0 or len(self._outcomes) < n:
            return None
        son = self._outcomes[-n:]
        if all(o == STOP for _, o in son):
            return f"{n} ardisik stop", self.cfg.stop_streak_lock_hours
        return None

    def _check_count(self, now: pd.Timestamp) -> tuple[str, float] | None:
        n = self.cfg.stop_count
        if n <= 0:
            return None
        since = now - pd.Timedelta(hours=self.cfg.stop_count_hours)
        stops = sum(1 for t, o in self._outcomes if o == STOP and t > since)
        if stops >= n:
            return (f"{self.cfg.stop_count_hours:g}saatte {stops} stop",
                    self.cfg.stop_count_lock_hours)
        return None

    def _check_drawdown(self, balance: float) -> tuple[str, float] | None:
        limit = self.cfg.max_drawdown_pct
        if limit <= 0 or not self._peak_balance:
            return None
        dd = (self._peak_balance - balance) / self._peak_balance * 100.0
        if dd >= limit:
            return f"tepeden %{dd:.1f} dusus", self.cfg.drawdown_lock_hours
        return None
