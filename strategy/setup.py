"""
IFVG kurulumu: dort sinyal katmanini tek bir islem kararina baglar.

Modelin hikayesi (ICT'nin klasik dizilimi):

    1. LIKIDITE SUPURMESI - fiyat belirgin bir tepe/dibin otesine fitille
       uzanip geri doner. Oradaki stop emirleri toplanmistir.
    2. CHoCH - yapi ters yonde kirilir. Supurme tesadufi degildi, karakter
       degisti.
    3. IFVG - supurmenin yonunde bir bosluk TERS CEVRILIR. Onceki yonu
       savunan emirler yenilmistir.
    4. GIRIS - fiyat ters cevrilmis boslugu yeniden test ettiginde girilir.

Dizilim ZORUNLU ve siralidir: supurme -> CHoCH -> IFVG -> giris. Sirasi
bozulan bir birlesim ayni parcalardan olusur ama bambaska bir sey anlatir.

Her adim kendi teyit gecikmesine saygi duyar (swing'ler `confirmed_index`,
bosluklar `inverted_index`). Bu dosyada hicbir yerde bir olayin YASANDIGI bar
ile BILINDIGI bar birbirinin yerine kullanilmaz - o karisiklik gelecege bakma
hatasinin en sik girdigi yerdir.

RISK: bu bot onceki iki bottan yapisal olarak farklidir. Altin botu stopsuz
grid/martingale idi (zarar edince buyutur). Burada HER islemin sabit stop'u
var, kayip kucuk ve kesin. Kazanma orani dusuk olacak (~%40-50 beklenir),
kar kazanclarin kayiplardan buyuk olmasindan gelir.
"""
from dataclasses import dataclass, field

import pandas as pd

from signals.fvg import Gap, find_gaps, track_lifecycle
from signals.liquidity import Sweep, build_levels, find_sweeps
from signals.sessions import in_killzone
from signals.structure import (StructureBreak, find_breaks, find_swings)

# Bir zincirin kac bosluga aday olabilecegi. Ilk bosluk baska bir zincire
# gidebildigi icin yedek gerekiyor, ama sinirsiz degil: bes bosluk da baskasina
# gittiyse o bolge zaten doymustur, altinci bosluk artik ayni hikayenin parcasi
# degil. Sinir ayni zamanda maliyeti sabit tutuyor (her aday icin ayri retest
# taramasi yapiliyor).
MAX_GAP_CANDIDATES = 5

DEFAULT_CONFIG = {
    "killzones": ["london", "newyork"],
    "swing_left": 2,
    "swing_right": 2,
    "min_gap_pct": 0.02,        # boslugun fiyata orani (%) - gurultu filtresi
    "gap_max_age_hours": 25.0,   # bosluk omru (5dk'da 300 bar karsiligi)
    "sweep_min_pen_pct": 0.01,  # supurmenin seviyeyi asma miktari (%)
    # Zincir pencereleri SAAT cinsinden. Once bar cinsindendi ve bu sessiz bir
    # tuzakti: 120 bar 5 dakikalik veride 10 saat, 15 dakikalikta 30 saat
    # demek - zaman dilimi degisince parametrenin ANLAMI degisiyordu ama
    # hicbir yerde hata cikmiyordu. Bar sayisina cevrim `find_setups` icinde,
    # verinin kendi bar araligina gore yapiliyor.
    #
    # Olculdu: pencereler 10 saatten sonra DOYUYOR - daha uzun pencereler
    # kagit uzerinde biraz daha iyi (80 saat -> +1456R) ama o artik day trade
    # degil, supurme ile giris arasinda uc gun gecen gevsek bir korelasyon.
    # Sayi daha iyi diye modelin hikayesi terk edilmedi.
    "sweep_to_choch_hours": 10.0,
    "choch_to_ifvg_hours": 10.0,
    "ifvg_to_entry_hours": 5.0,
    "stop_buffer_pct": 0.02,    # stop'u referans noktanin biraz otesine koy
    # Stop NEREDEN turuyor:
    #   "sweep" - supurme ekstremumunun otesi (ilk tasarim)
    #   "gap"   - ters cevrilmis boslugun KARSI kenarinin otesi
    #
    # "sweep" tasarimin ilk halidir ve iki sorunu var. Birincisi: supurme
    # seviyesi girisden cok uzakta kalabiliyor, stop uzak olunca pozisyon
    # kuculuyor ve getiri eziliyor. Ikincisi ve daha kotusu: giris supurme
    # seviyesinin USTUNDE kalirsa stop yanlis tarafa dusuyordu (kurulumlarin
    # %32,5'i boyleydi, artik eleniyor ama o kurulumlar tamamen kayboluyor).
    #
    # "gap" modelin kendi hikayesine daha sadik: bosluk ters cevrildi, fiyat
    # ona geri dondu, girildi. Fiyat boslugu TEKRAR gecerse hikaye
    # yanlislanmis demektir - cikilacak yer orasidir. Ayrica giris boslugun
    # bir kenari, stop obur kenari oldugu icin stop'un yanlis tarafa dusmesi
    # YAPISAL OLARAK imkansiz.
    "stop_source": "sweep",
    # Hedef = risk x bu kat. 1.0-4.0 araliginin TAMAMI ayni beklenen degeri
    # veriyor (ortalama_R 0.191-0.203) - yani edge giris zamanlamasindan
    # geliyor, hedef seciminden degil. Degisen tek sey kazanma orani /
    # buyukluk takasi. 1.5 secildi cunku ayni getiriyi daha dusuk dususle
    # veriyor (28R vs 34R) ve kazanma orani %57 (2.0'da %53). Bu bir
    # optimizasyon degil risk tercihi: tepe nokta yok, plato var.
    "rr_target": 1.5,
    # Stop cok yakinsa o kurulum GERCEKTE ALINAMAZ: mesafe spread ve kayma
    # mertebesine duser, emir daha dolmadan stop olur. Ilk olcumde bu filtre
    # yoktu ve tek bir islem -200R gosterdi - imkansiz bir sayi, cunku 2
    # puanlik sabit maliyet 0.01 puanlik riske bolunuyordu. Filtre bir
    # "iyilestirme" degil, gercekligin kendisi.
    #
    # YUZDE olarak tanimli, mutlak puan olarak DEGIL. Once 15 puan yazilmisti
    # ama o sayi yalnizca NAS100 icin anlamli: EURUSD 1.16 fiyatinda 15 puan
    # absurt, altinda bambaska bir sey demek. Enstrümandan bagimsiz bir model
    # istiyorsak esikler fiyata goreli olmali.
    # NAS100 ~29.000'de %0.05 = ~14.5 puan (onceki degerin karsiligi).
    "min_risk_pct": 0.05,
    # GIRIS saatine gore dislama (New York yerel saati). Bos liste = filtre yok.
    # Killzone filtresi SUPURME barina bakiyor; giris baska bir saatte olabilir.
    # Bu ayri bir kadran ve varsayilani BOS: veri bazi saatlerin belirgin kotu
    # oldugunu soyluyor ama bir filtreyi tum veride kesfedip yine tum veride
    # dogrulamak kendini kandirmaktir. `backtest/hour_filter_test.py` bunu
    # egitim/test ayirarak sinar - filtre ancak orada gecerse varsayilan olur.
    "exclude_entry_hours": [],
}


def _bar_minutes(bars: pd.DataFrame) -> float:
    """Verinin bar araligi (dakika). Medyan kullaniliyor: seans bosluklari
    ve hafta sonlari tek tek farklar arasinda uc degerler uretir, ortalama
    onlardan etkilenir."""
    if len(bars) < 2:
        return 5.0
    deltas = pd.Series(bars.index[1:]) - pd.Series(bars.index[:-1])
    minutes = deltas.dt.total_seconds().median() / 60.0
    return float(minutes) if minutes and minutes > 0 else 5.0


@dataclass
class Setup:
    """Tetiklenmis bir islem kurulumu ve onu ureten olay zinciri."""
    entry_index: int
    entry_time: pd.Timestamp
    direction: str          # 'long' | 'short'
    entry_price: float
    stop: float
    target: float
    sweep: Sweep
    choch: StructureBreak
    gap: Gap

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry_price)


def find_setups(bars: pd.DataFrame, config: dict | None = None) -> list[Setup]:
    """Veri uzerinde tum IFVG kurulumlarini sirayla bulur.

    Tek yonlu tarama: her supurme icin ondan SONRAKI CHoCH, ondan sonraki
    ters cevrilmis bosluk ve ondan sonraki giris testi aranir. Zaman geriye
    akmaz.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if len(bars) < 3:
        return []

    bar_minutes = _bar_minutes(bars)
    to_bars = lambda hours: max(1, int(round(hours * 60 / bar_minutes)))
    win_choch = to_bars(cfg["sweep_to_choch_hours"])
    win_gap = to_bars(cfg["choch_to_ifvg_hours"])
    win_entry = to_bars(cfg["ifvg_to_entry_hours"])

    swings = find_swings(bars, cfg["swing_left"], cfg["swing_right"])
    breaks = find_breaks(bars, swings)
    gaps = track_lifecycle(bars, find_gaps(bars, cfg["min_gap_pct"]), to_bars(cfg["gap_max_age_hours"]))

    levels = build_levels(bars)
    # Esik her barin KENDI fiyatindan turetiliyor, veri setinin ortalamasindan
    # degil. Onceki hal (`bars["close"].mean()`) tum veriyi - yani gelecegi -
    # tek bir sayiya katiyordu: ayni bar, veri setinin uzunluguna gore farkli
    # esik goruyordu. Butunsel denetim (backtest/lookahead_check.py) bunu
    # yakaladi; birim testler yakalayamamisti cunku hata katmanlarin
    # BIRLESIMINDE duruyordu.
    min_pen = bars["close"].to_numpy(dtype=float) * cfg["sweep_min_pen_pct"] / 100.0
    sweeps = find_sweeps(bars, levels, min_penetration=min_pen)

    kz = in_killzone(bars.index, cfg["killzones"]).to_numpy()
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    # Ters cevrilmis bosluklari, cevrildikleri bara gore sirali tutuyoruz.
    inverted = sorted((g for g in gaps if g.inverted_index is not None),
                      key=lambda g: g.inverted_index)

    # Once TUM adaylar toplanir, bosluk tahsisi YAPILMADAN. Tahsis ikinci
    # asamada ve giris sirasina gore yapilir - gerekcesi _allocate_gaps'te.
    candidates: list[Setup] = []

    for sweep in sweeps:
        if not kz[sweep.index]:
            continue

        # Supurme ustteki likiditeyi aldiysa beklenen yon ASAGI.
        want = "short" if sweep.side == "high" else "long"

        choch = _first_choch(breaks, after=sweep.index,
                             within=win_choch, direction=want)
        if choch is None:
            continue

        sweep_stop = _stop_from_sweep(sweep, want, cfg["stop_buffer_pct"])

        # Bu zincirin TUM olasi bosluklari aday yapilir, yalnizca ilki degil:
        # ilk bosluk baska bir zincire gidebilir ve bu zincirin sirada bekleyen
        # bir secenegi olmali. Kimin hangisini aldigi _allocate_gaps'te, giris
        # sirasina gore cozuluyor.
        for gap in _inverted_gaps(inverted, after=choch.index,
                                  within=win_gap, direction=want):
            entry = _first_retest(bars, gap, after=gap.inverted_index,
                                  within=win_entry, direction=want)
            if entry is None:
                continue

            entry_index, entry_price = entry

            stop = (_stop_from_gap(gap, want, cfg["stop_buffer_pct"])
                    if cfg["stop_source"] == "gap" else sweep_stop)

            if cfg["exclude_entry_hours"]:
                ny_hour = bars.index[entry_index].tz_convert("America/New_York").hour
                if ny_hour in cfg["exclude_entry_hours"]:
                    continue

            # Stop, girisin DOGRU tarafinda olmali: short'ta yukarisinda,
            # long'ta asagisinda. Kod bunu hic kontrol etmiyordu ve
            # kurulumlarin %32,5'i (6.999'un 2.273'u) stop'u KAR yonunde
            # tasiyordu.
            #
            # Nasil oluyor: stop supurulen seviyenin otesine konuyor, ama
            # giris noktasi bazen o seviyenin cok USTUNDE kaliyor (supurmeden
            # sonra fiyat yukari gidip orada bir bosluk ters ceviriyor). O
            # durumda "seviyenin ustu" girisin ALTINDA kalir.
            #
            # Neden olumcul: o pozisyonda zarar yonunde hicbir cikis yok.
            # Stop ve hedef ayni tarafta oldugu icin fiyat ters giderse
            # pozisyonu yalnizca zaman asimi kapatir - yani risk olculemez.
            # Ustelik backtest'te "stop" etiketi kar uretiyor ve kazanma
            # oranini sisiriyordu.
            #
            # min_risk_pct MESAFEYI kontrol ediyordu, YONU kimse sormamisti.
            if want == "short" and stop <= entry_price:
                continue
            if want == "long" and stop >= entry_price:
                continue

            risk = abs(entry_price - stop)
            if risk < entry_price * cfg["min_risk_pct"] / 100.0:
                continue

            target = (entry_price + risk * cfg["rr_target"] if want == "long"
                      else entry_price - risk * cfg["rr_target"])

            candidates.append(Setup(
                entry_index=entry_index, entry_time=bars.index[entry_index],
                direction=want, entry_price=entry_price, stop=stop, target=target,
                sweep=sweep, choch=choch, gap=gap,
            ))

    return _allocate_gaps(candidates)


def _allocate_gaps(candidates: list[Setup]) -> list[Setup]:
    """Bir bosluga yalnizca BIR islem baglanir - ve onu ilk GIREN alir.

    Once supuren degil, once GIREN. Fark kritik: eski hal sessiz bir gelecege
    bakma hatasiydi.

    Eskiden tahsis supurme sirasinda yapiliyor, bosluk kurulum TAMAMLANDIGINDA
    "harcandi" isaretleniyordu. Ama bir kurulumun giris ani, baska bir
    kurulumun karar aninin ILERISINDE olabilir:

        supurme A (10:00) -> girisi 12:00'de -> boslugu 12:00'de harcar
        supurme B (11:00) -> ayni bosluga uzanir -> "harcanmis" sayilirdi

    B'nin 11:00'deki karari, 12:00'de olacak bir seye gore veriliyordu. Canli
    bot bunu yapamaz - saat 11:00'de 12:00'yi bilmez - ve ikisi ayrisiyordu:
    4.000 barlik olcumde backtest 46 kurulum gorurken canli 41 goruyordu,
    islemlerin %11'i canlida hic acilmiyordu (hepsi eksik yonde, fazladan
    acilan yok).

    Giris sirasina gore tahsis hem nedensel olarak dogru hem de PENCEREDEN
    BAGIMSIZ: bir bosluga ilk giren aday, veri nereden baslarsa baslasin ayni
    adaydir. Bosluk omru (gap_max_age_hours) canli pencereden cok kisa oldugu
    icin rakip adaylarin ikisi de daima ayni pencerede gorunur.

    Iki tekillik birlikte uygulaniyor:
      bir BOSLUK en fazla bir islem verir  - ayni yerden cift pozisyon olmaz.
      bir SUPURME en fazla bir islem verir - bir zincir tek hikayedir; yedek
      bosluklar (bkz. MAX_GAP_CANDIDATES) o hikayenin alternatifleridir, ek
      pozisyon degil.

    Kimlik olarak id() DEGIL (nesne kimligi cop toplamayla geri donusebilir),
    olaylarin kendi bar damgalari kullaniliyor.

    Dogrulamasi: backtest/recursive_check.py
    """
    order = sorted(candidates, key=lambda s: (s.entry_index, s.sweep.index))
    taken_gaps: set[tuple] = set()
    taken_sweeps: set[tuple] = set()
    chosen: list[Setup] = []

    for s in order:
        # Ayni barda farkli seviyeler supurulebilir; supurme kimligi bar +
        # seviye adidir, tek basina bar degil.
        gap_key = (s.gap.index, s.gap.inverted_index)
        sweep_key = (s.sweep.index, s.sweep.level_name)
        if gap_key in taken_gaps or sweep_key in taken_sweeps:
            continue
        taken_gaps.add(gap_key)
        taken_sweeps.add(sweep_key)
        chosen.append(s)

    return chosen


def _first_choch(breaks: list[StructureBreak], after: int, within: int,
                 direction: str) -> StructureBreak | None:
    """Supurmeden sonraki ilk, DOGRU YONDE karakter degisimi."""
    want = "bull" if direction == "long" else "bear"
    for b in breaks:
        if b.index <= after:
            continue
        if b.index > after + within:
            return None
        if b.kind == "choch" and b.direction == want:
            return b
    return None


def _inverted_gaps(inverted: list[Gap], after: int, within: int,
                   direction: str, limit: int = MAX_GAP_CANDIDATES) -> list[Gap]:
    """CHoCH'tan sonra ters cevrilen uygun bosluklar, zaman sirasinda.

    Ters cevrilmis bir BULLISH bosluk artik direnctir -> short yonu verir.
    Bu yuzden aranan olusum yonu, istenen islem yonunun tersidir.

    Tek bosluk degil LISTE donuyor. Sebep: bir zincirin ilk boslugu baska bir
    zincire gidebilir ve o zaman bu zincirin sirada bekleyen bir secenegi
    olmali. Boslugun "harcanmis" olup olmadigi burada SORULMAZ - o soru bu
    fonksiyonun bilemeyecegi bir seye baglidir, hangi adayin oraya once
    GIRDIGINE. Tekillik _allocate_gaps'te cozuluyor.
    """
    want_origin = "bull" if direction == "short" else "bear"
    out: list[Gap] = []
    for g in inverted:
        if g.inverted_index <= after:
            continue
        if g.inverted_index > after + within:
            break
        if g.direction == want_origin:
            out.append(g)
            if len(out) >= limit:
                break
    return out


def _first_retest(bars: pd.DataFrame, gap: Gap, after: int, within: int,
                  direction: str) -> tuple[int, float] | None:
    """Fiyatin ters cevrilmis bosluga geri donup girisi tetikledigi bar.

    Giris fiyati olarak boslugun YAKIN KENARI kullaniliyor, mid degil:
    gercek bir limit emri orada dolar, ortada degil. Ortayi almak her
    islemi sistematik olarak birkac puan iyi gosterir.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(bars)

    edge = gap.bottom if direction == "short" else gap.top
    for i in range(after + 1, min(n, after + within + 1)):
        if direction == "short" and high[i] >= edge:
            return i, edge
        if direction == "long" and low[i] <= edge:
            return i, edge
    return None


def _stop_from_gap(gap: Gap, direction: str, buffer_pct: float) -> float:
    """Stop, ters cevrilmis boslugun KARSI kenarinin otesinde.

    Mantik dogrudan modelin hikayesinden geliyor: bosluk ters cevrildi
    (orayi savunan taraf yenildi), fiyat geri donup onu test etti, girildi.
    Fiyat boslugu TEKRAR gecerse o hikaye yanlislanmis demektir - cikilacak
    yer orasidir.

    Giris boslugun bir kenari (_first_retest: short'ta bottom, long'ta top),
    stop obur kenarinin otesi. Bu yuzden stop DAIMA girisin zarar tarafinda
    kalir; yanlis tarafa dusmesi aritmetik olarak imkansiz. Onceki tasarimda
    (supurmeye bagli stop) bu garanti yoktu ve kurulumlarin %32,5'i stop'u
    kar yonunde tasiyordu.
    """
    if direction == "short":
        return gap.top + gap.top * buffer_pct / 100.0
    return gap.bottom - gap.bottom * buffer_pct / 100.0


def _stop_from_sweep(sweep: Sweep, direction: str, buffer_pct: float) -> float:
    """Stop, supurmenin ekstremumunun biraz otesinde.

    Mantik: supurme o noktada emirlerin toplandigini soyluyor. Fiyat oraya
    geri donerse kurulumun anlattigi hikaye yanlislanmis demektir - orasi
    islemden cikilacak yerdir.

    Tampon supurulen SEVIYENIN kendisinden olculuyor; veri setinin ortalama
    fiyatindan degil. O sayi gelecegi iceriyordu.
    """
    buffer = sweep.level * buffer_pct / 100.0
    extreme = sweep.level + sweep.penetration
    if direction == "short":
        return extreme + buffer
    return sweep.level - sweep.penetration - buffer


def to_frame(setups: list[Setup]) -> pd.DataFrame:
    if not setups:
        return pd.DataFrame(columns=["entry_time", "direction", "entry_price",
                                     "stop", "target", "risk", "swept"])
    return pd.DataFrame([{
        "entry_time": s.entry_time, "direction": s.direction,
        "entry_price": s.entry_price, "stop": s.stop, "target": s.target,
        "risk": s.risk, "swept": s.sweep.level_name,
    } for s in setups])
