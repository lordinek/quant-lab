"""
Prop firma sinav simulatoru: model $50.000'lik bir sinavdan geciyor mu?

NEDEN AYRI BIR OLCUM GEREKIYOR. Backtest TOPLAM getiriye bakar, prop firmasi
YOLCULUGUN EN KOTU ANINA bakar. +244 R'lik bir model tek bir gunde gunluk
zarar limitini delerse sinav biter ve o 244 R hic yasanmaz. Iki metrik
birbirinden bagimsiz - biri digerini ima etmez. Bu dosya ikincisini olcer.

Fikrin kaynagi: Lil Fish (Luke) ucretli sinavlari arka arkaya patlattiktan
sonra prop kurallarini koda dokup stratejiyi once sahte sinava sokuyor.
Ayni seyi yapiyoruz - sinav ucreti odemeden once cevabi biliyoruz.

DORT KONSERVATIF VARSAYIM (hepsi bizim aleyhimize, bilerek):

  1. Acik pozisyonun en kotu ani (MAE) gercek zamanli equity sayilir. Prop
     firmalari tam boyle olcuyor: gerceklesmemis zarar da hesaba girer.
  2. Ayni anda acik pozisyonlarin en kotu anlari AYNI ANDA yasanmis kabul
     edilir. Gercekte ust uste binmeyebilir; bilmedigimiz icin en kotuyu
     varsayiyoruz (`evaluate.py`'deki "ayni barda ikisi de vurulursa stop"
     disiplininin aynisi).
  3. Tepe equity icin en IYI an (MFE) sayilir. Apex'te bu ALEYHE calisir:
     gerceklesmemis kar da trailing esigini yukari iter ve yastigi kalici
     olarak tuketir.
  4. Islem maliyeti CFD spread'i (3,5 puan) olarak kaliyor. Futures'ta
     (MNQ) round-turn komisyon + spread daha UCUZ - yani gercek sonuc bu
     olcumden biraz daha iyi olmali.

BILINEN EKSIK: kurallar 2026-09-21'de firma yardim merkezlerinden alindi.
Prop firmalari kural degistirir. Sinav ucreti odenmeden once RULES
sozlugundeki degerler tekrar dogrulanmali.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest.evaluate import DEFAULT_COST_POINTS, Trade


@dataclass(frozen=True)
class PropRules:
    """Bir prop firmasinin sinav kurallari.

    Tum para degerleri $ cinsinden ve MUTLAK (yuzde degil) - firmalar da
    boyle yaziyor ve yuzdeye cevirmek hesap buyukluge gore sessizce
    farkli kural uretir.
    """
    name: str
    account: float
    profit_target: float
    max_loss: float            # baslangic yastigi ($)
    trailing: str              # "none" | "eod" | "intraday"
    trail_lock: float | None   # tabanin kilitlendigi deger; None = kilit yok
    daily_loss: float | None   # gunluk zarar limiti; None = yok
    daily_loss_kills: bool     # True: hesap yanar. False: sadece o gun biter
    min_trading_days: int
    max_calendar_days: int | None
    point_value: float         # 1 puan = kac $ (MNQ 2.0, CFD tarafinda 1.0)
    contract_step: float       # 0 = serbest lot (CFD), 1 = tam kontrat
    reset_tz: str
    reset_hour: int
    consistency_pct: float | None  # en iyi gun / kar hedefi tavani


# Kurallar 2026-09-21'de dogrulandi. Kaynaklar:
#   Topstep : help.topstep.com (gunluk zarar limiti) + 2026 kural ozetleri
#   Apex    : apextraderfunding.com/help-center (intraday trailing drawdown)
#   FTMO    : ftmo.com/en/trading-objectives
# FTMO gunu 00:00 CE(S)T'de doner (ftmo.com/en/trading-objectives, 2026-09-25
# tekrar okundu: "recalculated daily at 00:00 CE(S)T"). Ilk surumde New York
# yaziyordu - 6 saat kayma, Londra seansinin hangi gune dustugunu degistirir.
FTMO_TZ = "Europe/Prague"

RULES = {
    # Gunluk zarar limiti Topstep'te SECMELI bir eklenti ve ihlal sayilmaz:
    # vurulunca pozisyonlar likide edilir, o seans biter, hesap yasar.
    # Burada acik tutuldu cunku asil sorumuz tam da bu limitin ne yaptigi.
    # Max Loss Limit tabani yalnizca GUN SONUNDA yukselir ama gercek zamanli
    # izlenir; baslangic bakiyesine ulasinca kalici kilitlenir.
    "topstep50k": PropRules(
        name="Topstep 50K Combine", account=50_000, profit_target=3_000,
        max_loss=2_000, trailing="eod", trail_lock=50_000,
        daily_loss=1_000, daily_loss_kills=False,
        min_trading_days=2, max_calendar_days=None,
        point_value=2.0, contract_step=1.0,
        reset_tz="America/Chicago", reset_hour=17, consistency_pct=0.5),

    # Apex'in farki: gunluk zarar limiti YOK ama trailing esik gercek
    # zamanli ve gerceklesmemis kari da sayiyor. Yastik $2.500, esik
    # $50.100'de kilitleniyor. Sinav 30 gun.
    "apex50k": PropRules(
        name="Apex 50K Evaluation", account=50_000, profit_target=3_000,
        max_loss=2_500, trailing="intraday", trail_lock=50_100,
        daily_loss=None, daily_loss_kills=False,
        min_trading_days=0, max_calendar_days=30,
        point_value=2.0, contract_step=1.0,
        reset_tz="America/New_York", reset_hour=18, consistency_pct=None),

    # Apex'in ikinci secenegi: ayni yastik, ama taban gun sonunda hareket
    # ediyor (Topstep modeli). Ucuncu secenek statik drawdown - 100K hesapta
    # yastik yalnizca $625 oldugu icin bizim risk boyumuzda anlamsiz, o
    # yuzden modellenmedi.
    #
    # NOT: EOD varyantinin kendi yastik ve gunluk limit degerleri resmi
    # kaynaktan DOGRULANMADI; burada intraday ile ayni ($2.500) varsayildi.
    # Amac firma kiyasi degil, TEK BASINA trailing modelinin ne kadar fark
    # ettigini olcmek - o yuzden tek degisken degistirildi.
    "apex50k_eod": PropRules(
        name="Apex 50K (EOD trailing varyanti)", account=50_000, profit_target=3_000,
        max_loss=2_500, trailing="eod", trail_lock=50_100,
        daily_loss=None, daily_loss_kills=False,
        min_trading_days=0, max_calendar_days=30,
        point_value=2.0, contract_step=1.0,
        reset_tz="America/New_York", reset_hour=18, consistency_pct=None),

    # FTMO CFD tarafinda: taban SABIT (45.000), trailing yok. Buna karsilik
    # gunluk %5 limiti ihlal sayilir - vurursan hesap yanar. Hedef %10.
    #
    # IKI ASAMA. ftmo.com/trading-objectives dogrudan okundu: 2-step surecte
    # once Challenge (%10 hedef), gecince Verification (%5 hedef), sonra
    # funded. Ilk surum tek asama modelliyordu ve FTMO'yu haksiz yere kolay
    # gosteriyordu. Asgari islem gunu de 10 degil 4.
    "ftmo50k": PropRules(
        name="FTMO 50K Challenge (1. asama)", account=50_000, profit_target=5_000,
        max_loss=5_000, trailing="none", trail_lock=None,
        daily_loss=2_500, daily_loss_kills=True,
        min_trading_days=4, max_calendar_days=None,
        point_value=1.0, contract_step=0.0,
        reset_tz=FTMO_TZ, reset_hour=0, consistency_pct=None),

    # Verification: ayni limitler, hedef yari (%5). Ayri bir sinav degil,
    # ayni sinavin ikinci ayagi - ucreti tekrar odenmez ama gecilmesi sart.
    "ftmo50k_p2": PropRules(
        name="FTMO 50K Verification (2. asama)", account=50_000, profit_target=2_500,
        max_loss=5_000, trailing="none", trail_lock=None,
        daily_loss=2_500, daily_loss_kills=True,
        min_trading_days=4, max_calendar_days=None,
        point_value=1.0, contract_step=0.0,
        reset_tz=FTMO_TZ, reset_hour=0, consistency_pct=None),

    # En kucuk FTMO hesabi. Kurallar 50K ile
    # ayni YUZDELER: %10 / %5 hedef, %5 gunluk, %10 toplam (sabit), 4 gun.
    "ftmo10k": PropRules(
        name="FTMO 10K Challenge (1. asama)", account=10_000, profit_target=1_000,
        max_loss=1_000, trailing="none", trail_lock=None,
        daily_loss=500, daily_loss_kills=True,
        min_trading_days=4, max_calendar_days=None,
        point_value=1.0, contract_step=0.0,
        reset_tz=FTMO_TZ, reset_hour=0, consistency_pct=None),
    "ftmo10k_p2": PropRules(
        name="FTMO 10K Verification (2. asama)", account=10_000, profit_target=500,
        max_loss=1_000, trailing="none", trail_lock=None,
        daily_loss=500, daily_loss_kills=True,
        min_trading_days=4, max_calendar_days=None,
        point_value=1.0, contract_step=0.0,
        reset_tz=FTMO_TZ, reset_hour=0, consistency_pct=None),
}


# Sinav ucretleri (2026-09-21). Prop firmalari surekli indirim yapiyor -
# Apex'in liste fiyati neredeyse hic odenmiyor (%80 kupon standart). Buradaki
# rakamlar LISTE fiyati, yani ust sinir.
#   aylik      : her ay yenilenen abonelik ($) - 0 ise tek seferlik
#   pesin      : sinav ucreti ($)
#   aktivasyon : sinav gecilince bir kez odenen ($)
# Iki asamali sinavlar: birinci ayagi gecen hesap ikinci ayaga gecer.
# Ucret yalnizca bir kez odenir; ikinci ayak patlarsa her sey bastan.
PHASE2 = {"ftmo50k": "ftmo50k_p2", "ftmo10k": "ftmo10k_p2"}


def main_keys() -> list:
    """Kampanya kurulabilen firmalar - ikinci asama anahtarlari haric.

    `RULES` uzerinde dogrudan donen kod ikinci asamayi ayri bir firma sanir
    ve `COSTS[...]` aramasinda patlar. Donguler bunu kullanmali.
    """
    return [k for k in RULES if k not in PHASE2.values()]

COSTS = {
    "topstep50k": {"aylik": 49.0, "pesin": 0.0, "aktivasyon": 149.0},
    "apex50k": {"aylik": 0.0, "pesin": 131.0, "aktivasyon": 79.0},
    "apex50k_eod": {"aylik": 0.0, "pesin": 131.0, "aktivasyon": 79.0},
    # FTMO tek seferlik ~EUR345; kur 1.08 varsayildi. DOGRULANACAK.
    "ftmo50k": {"aylik": 0.0, "pesin": 373.0, "aktivasyon": 0.0},
    # 10K: kaynaklar EUR89-155 arasi celisiyor (resmi fiyat sayfasi JS, okunamadi).
    # Ust sinir 155 x 1.08. DOGRULANACAK.
    "ftmo10k": {"aylik": 0.0, "pesin": 167.0, "aktivasyon": 0.0},
}


def expected_cost(df: pd.DataFrame, cost: dict) -> dict:
    """Sinavi gecmenin BEKLENEN maliyeti ve suresi.

    Tek bir sinavin gecme orani yariyi bilgi verir; digeri su: patlayinca
    tekrar deniyorsun ve her deneme hem para hem ay yiyor. Yuksek risk
    gecme oranini dusurur ama denemeyi KISALTIR - aylik abonelikte bu ikisi
    ters yonde caliser ve kazanani ancak hesap soyler.
    """
    if df.empty:
        return {}
    passed = df[df["sonuc"] == "gecti"]
    p = len(passed) / len(df)
    if p == 0:
        return {"gecme_%": 0.0, "bek_deneme": None, "bek_ay": None, "bek_$": None}

    failed = df[df["sonuc"] != "gecti"]
    ay_gecen = float(passed["takvim_gun"].mean()) / 30.44
    ay_patlayan = float(failed["takvim_gun"].mean()) / 30.44 if len(failed) else 0.0

    denemeler = 1 / p                       # geometrik dagilim
    toplam_ay = (denemeler - 1) * ay_patlayan + ay_gecen
    # Aylik abonelikte her BASLANAN ay odenir (yukari yuvarlama).
    para = cost["pesin"] * denemeler + cost["aylik"] * np.ceil(toplam_ay) + cost["aktivasyon"]
    return {
        "gecme_%": round(p * 100, 1),
        "bek_deneme": round(denemeler, 1),
        "bek_ay": round(toplam_ay, 1),
        "bek_$": round(para, 0),
    }


def excursions(bars: pd.DataFrame, trades: list[Trade],
               cost_points: float = DEFAULT_COST_POINTS) -> pd.DataFrame:
    """Her islemin en kotu (MAE) ve en iyi (MFE) anini R cinsinden cikarir.

    Kapanis R'si islemin SONUCUDUR; prop firmasi ise islemin ICINDE ne
    olduguna bakar. -0,2R ile kapanan bir islem yolda -0,9R'ye inmisse
    gunluk limiti o an deler.

    MAE 1R'de kirpilir: stop oradadir, pozisyon zaten kapanmistir.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    rows = []
    for t in trades:
        s = t.setup
        i, j = s.entry_index, t.exit_index
        seg_hi = high[i:j + 1].max()
        seg_lo = low[i:j + 1].min()
        if s.direction == "long":
            adverse = (s.entry_price - seg_lo) / s.risk
            favorable = (seg_hi - s.entry_price) / s.risk
        else:
            adverse = (seg_hi - s.entry_price) / s.risk
            favorable = (s.entry_price - seg_lo) / s.risk

        cost_r = cost_points / s.risk
        mae_r = min(max(adverse, 0.0), 1.0) + cost_r   # stop'un otesine gecemez
        mfe_r = max(favorable, 0.0) - cost_r

        rows.append({
            "entry_time": s.entry_time, "exit_time": t.exit_time,
            "entry_index": s.entry_index, "exit_index": t.exit_index,
            "direction": s.direction, "entry_price": s.entry_price,
            "risk_points": s.risk, "r": t.r_multiple,
            "mae_r": mae_r, "mfe_r": mfe_r, "outcome": t.outcome,
        })

    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    return df


def _prop_day(ts: pd.Timestamp, tz: str, reset_hour: int):
    """Prop firmasinin islem gunu. Gun 00:00'da baslamaz.

    Topstep gunu 17:00 CT'de doner, Apex 18:00 ET'de, FTMO 00:00 ET'de.
    UTC gunune gore gruplamak bir NY seansini ikiye boler ve gunluk zarar
    limitini oldugundan gevsek gosterir.
    """
    local = ts.tz_convert(tz)
    return (local - pd.Timedelta(hours=reset_hour)).date()


@dataclass(frozen=True)
class FundedRules:
    """Sinav gecildikten SONRAKI hesabin kurallari.

    Sinavi gecmek isin yarisi. Funded hesapta ayni dusus kurallari devam
    eder ama hedef yoktur - yerine PARA CEKME kurallari vardir ve asil
    kazanc/kayip orada belirlenir.
    """
    name: str
    min_winning_days: int      # odeme icin gereken nitelikli gun sayisi
    winning_day_min: float     # bir gunun "kazanan gun" sayilmasi icin asgari $
    consistency_pct: float | None  # tek gun, son odemeden beri karin en fazla %X'i
    payout_cap: float          # tek seferde cekilebilecek azami $
    min_payout: float
    profit_split: float        # esik sonrasi trader payi
    split_threshold: float     # bu tutara kadar kar %100 trader'in
    payout_pct: float          # karin ne kadari cekilir (kalani tampon)
    monthly_fee: float         # funded asamada aylik ucret
    inactivity_days: int | None  # bu kadar gun nitelikli islem yoksa hesap kapanir
    min_days_between: int = 0  # iki odeme arasindaki asgari takvim gunu

    # --- olcekleme (2026-09-23 eklendi, su an yalnizca FTMO) -------------
    # Firmalar tutarli trader'i buyutuyor. Modellemezsek uzun vadeli kazanci
    # SISTEMATIK olarak dusuk olceriz - ve bu hata hep ayni yone bakar, yani
    # gurultu degil YANLILIK uretir. Ilk surumde FTMO'ya sabit %80 yazilmisti.
    scale_days: int | None = None     # olcek penceresi (takvim gunu)
    scale_profit_pct: float = 0.0     # pencerede hesap boyutunun %X'i net kar
    scale_min_payouts: int = 0        # pencerede islenmis asgari odeme sayisi
    scale_boost: float = 0.0          # hesap buyume orani (0.25 = %25)
    scale_split: float | None = None  # olcek sonrasi trader payi


# Odeme kurallari 2026-09-21'de dogrulandi (help.topstep.com/payout-policy,
# support.apextraderfunding.com PA & Compliance, ftmo.com).
FUNDED = {
    # Topstep standart yol: 5 adet $150+ kazanan gun. Bolusme 90/10.
    # Cekim tavani 50K kademesinde $3.000, asgari $125.
    # Cekim tavani help.topstep.com/express-funded-account-parameters'tan
    # dogrudan okundu: "50% of balance, up to $5,000". Ilk surumde $3.000
    # yazilmisti (ikincil kaynaktan).
    "topstep50k": FundedRules(
        name="Topstep 50K Express Funded", min_winning_days=5,
        winning_day_min=150.0, consistency_pct=None,
        payout_cap=5_000.0, min_payout=125.0,
        profit_split=0.90, split_threshold=0.0, payout_pct=0.5,
        monthly_fee=0.0, inactivity_days=None),

    # Apex PA: 5 nitelikli gun + %50 tutarlilik (tek gun, son odemeden beri
    # karin yarisini gecemez). Ilk $25.000 %100 trader'in. 30 gun nitelikli
    # islem yoksa hesap KALICI kapanir.
    "apex50k": FundedRules(
        name="Apex 50K PA", min_winning_days=5,
        winning_day_min=50.0, consistency_pct=0.5,
        payout_cap=2_000.0, min_payout=500.0,
        profit_split=0.90, split_threshold=25_000.0, payout_pct=0.5,
        monthly_fee=0.0, inactivity_days=30),

    # EOD varyanti ayni odeme kurallarini kullaniyor - tek degisken taban
    # modeli olsun diye odeme tarafi bilerek aynen kopyalandi.
    "apex50k_eod": FundedRules(
        name="Apex 50K PA (EOD)", min_winning_days=5,
        winning_day_min=50.0, consistency_pct=0.5,
        payout_cap=2_000.0, min_payout=500.0,
        profit_split=0.90, split_threshold=25_000.0, payout_pct=0.5,
        monthly_fee=0.0, inactivity_days=30),

    # FTMO: talep uzerine odeme, %80 trader payi, tutarlilik kurali yok.
    #
    # OLCEKLEME (ftmo.com/en/reward-growth-and-scaling-plan, 2026-09-23'te
    # dogrudan okundu). Uc sart birden: (1) en az 4 ay FTMO Trader olarak
    # (veya son olceklemeden beri), (2) o 4 ayda baslangic bakiyesinin uzerine
    # en az %10 net kar, (3) ayni donemde en az 2 islenmis odeme. Saglaninca
    # hesap %25 buyuyor ve pay %80 -> %90'a cikiyor (yalnizca 2 asamali urun).
    #
    # DIKKAT - VARSAYIM: hesap buyuyunce zarar limitlerinin de ayni oranda
    # buyudugu varsayildi (%10 toplam zarar "baslangic bakiyesinin" yuzdesi
    # olarak tanimli, hesap yeniden boyutlaniyor). FTMO bunu sayfasinda ACIKCA
    # yazmiyor. Limitler sabit kalsaydi olcekleme hesabi daha kirilgan yapardi;
    # yani bu varsayim bizim LEHIMIZE. Dogrulanmali.
    "ftmo50k": FundedRules(
        name="FTMO 50K Funded", min_winning_days=4,
        winning_day_min=0.01, consistency_pct=None,
        payout_cap=1e9, min_payout=0.0,
        profit_split=0.80, split_threshold=0.0, payout_pct=1.0,
        # ILK SURUMDE FAZLA COMERTTI: "4 karli gun sonra karin tamamini cek",
        # yani pratikte haftada birkac odeme. Gercekte dongu 14 gunluk.
        monthly_fee=0.0, inactivity_days=None, min_days_between=14,
        scale_days=122, scale_profit_pct=0.10, scale_min_payouts=2,
        scale_boost=0.25, scale_split=0.90),
}
# 10K funded: ayni kurallar (hepsi yuzde ya da gun), yalnizca ad farkli.
FUNDED["ftmo10k"] = replace(FUNDED["ftmo50k"], name="FTMO 10K Funded")


@dataclass
class ExamResult:
    outcome: str          # "gecti" | "dusus" | "gunluk" | "sure" | "bitmedi"
    started: pd.Timestamp
    ended: pd.Timestamp
    calendar_days: int
    trading_days: int
    trades: int
    final_balance: float
    peak_balance: float
    locked_days: int      # gunluk limit yuzunden kapanan gun sayisi
    end_pos: int = 0      # veri icinde nerede kaldik (kampanya zinciri icin)


def run_exam(rows: pd.DataFrame, rules: PropRules, risk_pct: float,
             start_pos: int = 0, max_concurrent: int = 1,
             end_time: pd.Timestamp | None = None) -> ExamResult:
    """Tek bir sinav denemesini bastan sona oynatir.

    Olay bazli: her islem acilisinda once o ana kadar KAPANMIS pozisyonlar
    tahsil edilir, sonra o an acik olan her seyin en kotu hali ile equity
    kontrol edilir. Kontrol sirasi onemli - once tahsil, sonra kontrol,
    sonra yeni pozisyon.
    """
    acc = rules.account
    balance = acc
    floor = acc - rules.max_loss
    peak_equity = acc            # intraday trailing icin
    peak_eod = acc               # eod trailing icin
    open_pos: list[dict] = []
    taken = 0
    day = None
    day_open_balance = acc
    day_locked = False
    day_pnl: dict = {}
    trading_days = 0
    locked_days = 0
    outcome = "bitmedi"

    start_time = rows.at[start_pos, "entry_time"]
    now = start_time

    def realized_floor_update():
        """Gun sonu tabani (Topstep modeli): taban yalnizca gun sonu yukselir."""
        nonlocal floor, peak_eod
        if rules.trailing != "eod":
            return
        peak_eod = max(peak_eod, balance)
        new_floor = peak_eod - rules.max_loss
        if rules.trail_lock is not None:
            new_floor = min(new_floor, rules.trail_lock)
        floor = max(floor, new_floor)

    def settle(up_to: pd.Timestamp):
        """up_to anina kadar kapanmis pozisyonlari tahsil et (kronolojik)."""
        nonlocal balance, open_pos
        closing = sorted((p for p in open_pos if p["exit_time"] <= up_to),
                         key=lambda p: p["exit_time"])
        open_pos = [p for p in open_pos if p["exit_time"] > up_to]
        for p in closing:
            balance += p["pnl"]

    def live_equity():
        """Acik pozisyonlarin en kotu / en iyi hali ile anlik equity araligi."""
        worst = sum(p["worst"] for p in open_pos)
        best = sum(p["best"] for p in open_pos)
        return balance + worst, balance + best

    def check() -> str | None:
        """Ihlal kontrolu. Pozisyon ACILDIKTAN SONRA da cagrilmali.

        Ilk surumde yalnizca yeni islem acilirken kontrol ediliyordu ve bu
        sessiz bir iyimserlikti: islemlerin %54'u giris barinda kapaniyor,
        yani o islemler acilip kapandiktan sonraki ilk kontrolde ZATEN
        tahsil edilmis oluyordu - en kotu anlari hic olculmedi. Prop
        firmasi o ani gorur ve hesabi orada kapatir.
        """
        nonlocal floor, peak_equity, day_locked, locked_days
        eq_min, eq_max = live_equity()
        peak_equity = max(peak_equity, eq_max)

        if rules.trailing == "intraday":
            new_floor = peak_equity - rules.max_loss
            if rules.trail_lock is not None:
                new_floor = min(new_floor, rules.trail_lock)
            floor = max(floor, new_floor)

        if eq_min <= floor:
            return "dusus"

        if rules.daily_loss is not None and not day_locked:
            if day_open_balance - eq_min >= rules.daily_loss:
                if rules.daily_loss_kills:
                    return "gunluk"
                day_locked = True
                locked_days += 1
        return None

    for pos in range(start_pos, len(rows)):
        row = rows.iloc[pos]
        now = row["entry_time"]

        settle(now)

        # --- gun degisimi ---
        key = _prop_day(now, rules.reset_tz, rules.reset_hour)
        if key != day:
            if day is not None:
                realized_floor_update()
                # Gun sonunda hedef + asgari gun sarti birlikte kontrol edilir.
                if _passed(balance, acc, rules, trading_days, day_pnl):
                    return ExamResult("gecti", start_time, now,
                                      (now - start_time).days, trading_days,
                                      taken, balance, peak_equity, locked_days,
                                      end_pos=pos)
            day = key
            day_open_balance = balance
            day_locked = False

        if rules.max_calendar_days is not None:
            if (now - start_time).days > rules.max_calendar_days:
                outcome = "sure"
                break

        # Kampanya zincirinde sinav sonsuza kadar suremez: butce penceresi
        # dolduysa deneme burada kesilir.
        if end_time is not None and now > end_time:
            outcome = "sure"
            break

        # --- ihlal kontrolu (acik pozisyonlarin en kotu hali dahil) ---
        hit = check()
        if hit:
            outcome = hit
            break

        if day_locked or len(open_pos) >= max_concurrent:
            continue

        # --- pozisyon boyutu ---
        risk_cash = acc * risk_pct / 100.0
        stop_pts = row["risk_points"]
        if stop_pts <= 0:
            continue
        size = risk_cash / (stop_pts * rules.point_value)
        if rules.contract_step > 0:
            # Tam kontrat: asagi yuvarlanir ama en az bir mikro kontrat alinir -
            # gercek hayatta trader islemi atlamaz, en kucuk boyutla girer.
            size = max(rules.contract_step,
                       np.floor(size / rules.contract_step) * rules.contract_step)
        if size <= 0:
            continue

        real_risk = size * stop_pts * rules.point_value   # yuvarlama sonrasi GERCEK risk
        pnl = row["r"] * real_risk
        worst = min(pnl, -row["mae_r"] * real_risk)
        best = max(pnl, row["mfe_r"] * real_risk)

        open_pos.append({"exit_time": row["exit_time"], "pnl": pnl,
                         "worst": worst, "best": best})
        taken += 1
        if key not in day_pnl:      # 0.0 kar eden gun de islem gunudur
            trading_days += 1
            day_pnl[key] = 0.0
        day_pnl[key] += pnl

        # Pozisyon ACIK haldeyken ihlal olur mu? Islemlerin yarisi giris
        # barinda kapandigi icin bu kontrol buraya sarttir.
        hit = check()
        if hit:
            outcome = hit
            break

    else:
        # Veri bitti: kalan pozisyonlari tahsil et, son bir kez bak.
        settle(pd.Timestamp.max.tz_localize("UTC"))
        realized_floor_update()
        if _passed(balance, acc, rules, trading_days, day_pnl):
            outcome = "gecti"

    if outcome == "bitmedi":
        settle(now)

    return ExamResult(outcome, start_time, now, (now - start_time).days,
                      trading_days, taken, balance, peak_equity, locked_days,
                      end_pos=min(pos + 1, len(rows) - 1))


def _passed(balance: float, acc: float, rules: PropRules,
            trading_days: int, day_pnl: dict) -> bool:
    """Hedefe ulasildi mi? Tutarlilik kurali hedefi YUKSELTEBILIR."""
    if trading_days < rules.min_trading_days:
        return False
    target = rules.profit_target
    if rules.consistency_pct and day_pnl:
        best_day = max(day_pnl.values())
        if best_day > target * rules.consistency_pct:
            target = best_day / rules.consistency_pct
    return balance >= acc + target


def rolling_exams(rows: pd.DataFrame, rules: PropRules, risk_pct: float,
                  every_days: int = 14, max_concurrent: int = 1) -> pd.DataFrame:
    """Farkli tarihlerden baslayan bagimsiz sinavlar.

    TEK bir sinav sonucu tesadufdur - hangi gun basladigin sonucu belirler.
    Asil soru "bu sinav gecildi mi" degil, "bu sinava girsem gecme olasiligim
    ne" - o yuzden iki haftada bir yeni sinav baslatilip dagilimina bakiliyor.
    """
    starts = []
    cursor = rows.at[0, "entry_time"]
    last = rows.at[len(rows) - 1, "entry_time"]
    step = pd.Timedelta(days=every_days)
    while cursor <= last:
        pos = int(rows["entry_time"].searchsorted(cursor, side="left"))
        if pos < len(rows):
            starts.append(pos)
        cursor += step

    out = []
    for p in sorted(set(starts)):
        r = run_exam(rows, rules, risk_pct, start_pos=p,
                     max_concurrent=max_concurrent)
        out.append({
            "baslangic": r.started, "sonuc": r.outcome,
            "takvim_gun": r.calendar_days, "islem_gunu": r.trading_days,
            "islem": r.trades, "bitis_$": round(r.final_balance, 0),
            "kilitli_gun": r.locked_days,
        })
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"sinav": 0}
    n = len(df)
    passed = df[df["sonuc"] == "gecti"]
    return {
        "sinav": n,
        "gecme_%": round(len(passed) / n * 100, 1),
        "dusus_%": round((df["sonuc"] == "dusus").mean() * 100, 1),
        "gunluk_%": round((df["sonuc"] == "gunluk").mean() * 100, 1),
        "sure_%": round((df["sonuc"] == "sure").mean() * 100, 1),
        "bitmedi_%": round((df["sonuc"] == "bitmedi").mean() * 100, 1),
        "medyan_gun": round(float(passed["takvim_gun"].median()), 0) if len(passed) else None,
        "ort_islem": round(float(df["islem"].mean()), 0),
        # Gunluk limit ihlal sayilmayan firmalarda (Topstep) hesap yanmaz ama
        # o gun kapanir. Sifir olmayan bir sayi "limit vuruluyor" demektir.
        "kilit_gun": round(float(df["kilitli_gun"].mean()), 1),
    }


@dataclass
class FundedResult:
    outcome: str          # "yasiyor" | "dusus" | "gunluk" | "hareketsizlik" | "sure"
    payouts: list         # trader'a NET gecen tutarlar ($)
    days: int
    trades: int
    end_pos: int
    ended: pd.Timestamp
    final_balance: float
    scales: int = 0       # kac kez olceklendi (FTMO disinda hep 0)


def run_funded(rows: pd.DataFrame, rules: PropRules, funded: FundedRules,
               risk_pct: float, start_pos: int, end_time: pd.Timestamp,
               max_concurrent: int = 1, paid_before: float = 0.0) -> FundedResult:
    """Sinav gecildikten sonraki hesabi oynatir.

    Sinavdan iki farki var:
      1. Hedef yok - hesap patlayana ya da sure dolana kadar devam eder.
      2. Kar kendiliginden para degil. Cekmek icin nitelikli gun sayisi,
         tutarlilik kurali ve tavan var; cekilen para bakiyeden DUSER ve
         taban kilitli kaldigi icin **her odeme tamponu daraltir.** Odeme
         almak hesabi olume yaklastirir - modelin gostermesi gereken sey bu.
    """
    acc = rules.account
    max_loss = rules.max_loss      # olceklenince buyur, o yuzden yerel
    split = funded.profit_split    # olceklenince %90'a cikar, o yuzden yerel
    balance = acc
    floor = acc - max_loss
    peak_equity = acc
    open_pos: list[dict] = []
    taken = 0
    day = None
    day_open_balance = acc
    day_locked = False
    # Olcek penceresi: ne zaman basladi, o pencerede ne kadar kar uretildi,
    # kac odeme islendi. Olcekleme olunca ucu de sifirlanir ("son
    # olceklemeden beri" kurali).
    scale_window_start = None
    scale_window_gross = 0.0
    scale_window_payouts = 0
    scale_count = 0
    # Son odemeden BERI gunluk kar/zarar - tutarlilik ve nitelikli gun sayimi
    # her odemeden sonra sifirlanir (Topstep'te de Apex'te de boyle).
    day_pnl: dict = {}
    qualifying_log: list = []      # (gun, kar) - hareketsizlik kontrolu icin
    payouts: list = []
    paid_total = paid_before
    last_payout_at = None          # odeme dongusu kontrolu icin
    start_time = rows.at[start_pos, "entry_time"]
    now = start_time
    scale_window_start = start_time
    outcome = "yasiyor"
    pos = start_pos

    def settle(up_to):
        nonlocal balance, open_pos
        closing = sorted((p for p in open_pos if p["exit_time"] <= up_to),
                         key=lambda p: p["exit_time"])
        open_pos = [p for p in open_pos if p["exit_time"] > up_to]
        for p in closing:
            balance += p["pnl"]

    def check():
        nonlocal floor, peak_equity, day_locked
        worst = balance + sum(p["worst"] for p in open_pos)
        best = balance + sum(p["best"] for p in open_pos)
        peak_equity = max(peak_equity, best)
        if rules.trailing == "intraday":
            nf = peak_equity - max_loss
            if rules.trail_lock is not None:
                nf = min(nf, rules.trail_lock)
            floor = max(floor, nf)
        if worst <= floor:
            return "dusus"
        if rules.daily_loss is not None and not day_locked:
            if day_open_balance - worst >= rules.daily_loss:
                if rules.daily_loss_kills:
                    return "gunluk"
                day_locked = True
        return None

    def try_payout():
        """Gun sonunda odeme sartlari olustuysa cek. Trader'a NET geceni dondurur."""
        nonlocal balance, paid_total, day_pnl, last_payout_at
        nonlocal scale_window_gross, scale_window_payouts
        profit = balance - acc
        if profit <= 0 or not day_pnl:
            return
        # Odeme dongusu: firmalar her karli gunde para vermiyor.
        if funded.min_days_between and last_payout_at is not None:
            if (now - last_payout_at).days < funded.min_days_between:
                return
        winning = [v for v in day_pnl.values() if v >= funded.winning_day_min]
        if len(winning) < funded.min_winning_days:
            return
        if funded.consistency_pct is not None:
            gains = sum(v for v in day_pnl.values() if v > 0)
            if gains > 0 and max(day_pnl.values()) > gains * funded.consistency_pct:
                return          # tek gun fazla buyuk - kural odemeyi engeller
        amount = min(profit * funded.payout_pct, funded.payout_cap)
        if amount < funded.min_payout:
            return
        # Bolusme: esige kadar %100, sonrasi profit_split.
        remaining_full = max(0.0, funded.split_threshold - paid_total)
        full_part = min(amount, remaining_full)
        net = full_part + (amount - full_part) * split
        balance -= amount            # cekilen para hesaptan cikar, taban kalir
        paid_total += amount
        payouts.append(net)
        last_payout_at = now
        day_pnl = {}                 # tutarlilik saati sifirlanir
        # Olcek penceresinin "uretilen kar" ve "islenmis odeme" sayaclari.
        scale_window_gross += amount
        scale_window_payouts += 1

    def try_scale():
        """Olcek sartlari olustuysa hesabi buyut ve payi yukselt.

        Uc sart BIRDEN aranir; biri bile eksikse olcekleme olmaz. Pencerede
        uretilen kar = o pencerede CEKILEN tutar + su an hesapta duran fazla.
        Cekilen parayi saymazsak duzenli odeme alan bir trader hic
        olceklenemezdi, ki kuralin amaci tam tersi.
        """
        nonlocal acc, max_loss, split, floor, peak_equity, balance
        nonlocal scale_window_start, scale_window_gross, scale_window_payouts
        nonlocal scale_count
        if funded.scale_days is None or scale_window_start is None:
            return
        if (now - scale_window_start).days < funded.scale_days:
            return
        if scale_window_payouts < funded.scale_min_payouts:
            return
        uretilen = scale_window_gross + max(0.0, balance - acc)
        if uretilen < acc * funded.scale_profit_pct:
            return

        eklenen = acc * funded.scale_boost   # firma hesaba bu kadar sermaye koyar
        acc += eklenen
        balance += eklenen
        # VARSAYIM (yukarida FUNDED icinde gerekcesi yazili): zarar limiti de
        # ayni oranda buyur. Sabit kalsaydi olcekleme hesabi kirilganlastirirdi.
        max_loss *= 1.0 + funded.scale_boost
        # FTMO tabani SABIT ve baslangic bakiyesine gore; yeni baslangic
        # bakiyesi buyumus hesap boyutu.
        floor = acc - max_loss
        peak_equity = max(peak_equity, balance)
        if funded.scale_split is not None:
            split = funded.scale_split
        scale_count += 1
        scale_window_start = now
        scale_window_gross = 0.0
        scale_window_payouts = 0

    for pos in range(start_pos, len(rows)):
        row = rows.iloc[pos]
        now = row["entry_time"]
        if now > end_time:
            outcome = "sure"
            break

        settle(now)

        key = _prop_day(now, rules.reset_tz, rules.reset_hour)
        if key != day:
            if day is not None:
                if rules.trailing == "eod":
                    nf = balance - max_loss
                    if rules.trail_lock is not None:
                        nf = min(nf, rules.trail_lock)
                    floor = max(floor, nf)
                try_payout()
                # Olcek kontrolu odemeden SONRA: sartlardan biri "islenmis
                # odeme sayisi", yani bugunun odemesi de sayilmali.
                try_scale()
                # Hareketsizlik: Apex'te 30 gun nitelikli islem yoksa hesap
                # KALICI kapanir. Nitelikli gun = $50+ net kar.
                if funded.inactivity_days is not None:
                    recent = [d for d, v in qualifying_log
                              if (now.date() - d).days <= funded.inactivity_days
                              and v >= 50.0]
                    elapsed = (now - start_time).days
                    if elapsed > funded.inactivity_days and len(recent) < 2:
                        outcome = "hareketsizlik"
                        break
            day = key
            day_open_balance = balance
            day_locked = False

        hit = check()
        if hit:
            outcome = hit
            break

        if day_locked or len(open_pos) >= max_concurrent:
            continue

        stop_pts = row["risk_points"]
        if stop_pts <= 0:
            continue
        risk_cash = acc * risk_pct / 100.0
        size = risk_cash / (stop_pts * rules.point_value)
        if rules.contract_step > 0:
            size = max(rules.contract_step,
                       np.floor(size / rules.contract_step) * rules.contract_step)
        if size <= 0:
            continue

        real_risk = size * stop_pts * rules.point_value
        pnl = row["r"] * real_risk
        open_pos.append({"exit_time": row["exit_time"], "pnl": pnl,
                         "worst": min(pnl, -row["mae_r"] * real_risk),
                         "best": max(pnl, row["mfe_r"] * real_risk)})
        taken += 1
        day_pnl[key] = day_pnl.get(key, 0.0) + pnl
        qualifying_log.append((key, pnl))

        hit = check()
        if hit:
            outcome = hit
            break

    return FundedResult(outcome, payouts, (now - start_time).days, taken,
                        min(pos + 1, len(rows) - 1), now, balance, scale_count)


@dataclass
class CampaignResult:
    net: float            # cekilen - odenen ($)
    gross_payouts: float
    fees: float
    exams_taken: int
    exams_passed: int
    blowups: int
    payout_count: int
    months_survived: float


def run_campaign(rows: pd.DataFrame, key: str, risk_pct: float,
                 start_pos: int, months: float = 12.0,
                 max_concurrent: int = 1) -> CampaignResult:
    """Bir yillik prop KAMPANYASI: sinav -> funded -> odeme -> patlarsa bastan.

    Tek bir sinavin gecilmesi bir sey ifade etmiyor. Asil soru su: bir yil
    boyunca bu isi yaparsam cebime ne girer, cebimden ne cikar? Sinav ucreti,
    aktivasyon, aylik abonelik CIKAR; odemeler GIRER; hesap patlayinca
    bastan baslanir ve ucret yeniden odenir.
    """
    rules, funded, cost = RULES[key], FUNDED[key], COSTS[key]
    start_time = rows.at[start_pos, "entry_time"]
    horizon = start_time + pd.Timedelta(days=months * 30.44)

    pos = start_pos
    fees = gross = 0.0
    exams = passed = blowups = n_payouts = 0
    paid_total = 0.0

    while pos < len(rows) - 1 and rows.at[pos, "entry_time"] < horizon:
        exam = run_exam(rows, rules, risk_pct, start_pos=pos,
                        max_concurrent=max_concurrent, end_time=horizon)
        exams += 1
        fees += cost["pesin"]
        fees += cost["aylik"] * max(1, np.ceil(exam.calendar_days / 30.44))
        pos = max(exam.end_pos, pos + 1)

        if exam.outcome != "gecti":
            if exam.outcome in ("sure", "bitmedi") and exam.ended >= horizon:
                break
            continue

        # Iki asamali sinavlar (FTMO): birinci ayak gecildi, ikincisi de
        # gecilmeden funded hesap YOK. Ucret tekrar odenmez ama patlarsa
        # bastan baslanir - yani ilk ayagin parasi da gitmis olur.
        second = PHASE2.get(key)
        if second is not None:
            p2 = run_exam(rows, RULES[second], risk_pct, start_pos=pos,
                          max_concurrent=max_concurrent, end_time=horizon)
            pos = max(p2.end_pos, pos + 1)
            if p2.outcome != "gecti":
                if p2.outcome in ("sure", "bitmedi") and p2.ended >= horizon:
                    break
                continue

        passed += 1
        fees += cost["aktivasyon"]
        f = run_funded(rows, rules, funded, risk_pct, start_pos=pos,
                       end_time=horizon, max_concurrent=max_concurrent,
                       paid_before=paid_total)
        gross += sum(f.payouts)
        paid_total += sum(f.payouts)
        n_payouts += len(f.payouts)
        fees += funded.monthly_fee * max(1, np.ceil(f.days / 30.44))
        pos = max(f.end_pos, pos + 1)
        if f.outcome in ("dusus", "gunluk", "hareketsizlik"):
            blowups += 1
        elif f.outcome == "sure":
            break

    survived = (min(rows.at[pos, "entry_time"], horizon) - start_time).days / 30.44
    return CampaignResult(net=round(gross - fees, 0), gross_payouts=round(gross, 0),
                          fees=round(fees, 0), exams_taken=exams,
                          exams_passed=passed, blowups=blowups,
                          payout_count=n_payouts, months_survived=round(survived, 1))
