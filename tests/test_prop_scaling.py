"""FTMO olcekleme plani: dogru sartlarda tetiklenir, yanlis sartlarda tetiklenmez.

NEDEN. `prop_eval.py` ilk surumde FTMO'ya sabit %80 pay yaziyordu. Gercekte
(ftmo.com/en/reward-growth-and-scaling-plan) dort ay boyunca %10 net kar ve en
az iki islenmis odeme ureten trader'in hesabi %25 buyuyor ve payi %90'a cikiyor.
Bunu modellemezsek uzun vadeli kazanci SISTEMATIK olarak dusuk olceriz - hata
hep ayni yone baktigi icin gurultu degil YANLILIK uretir.

Ters yon de onemli: olcekleme fazla comert modellenirse kampanya sonuclari
sisirilir. Bu yuzden iki test birden var - tetiklenmesi ve tetiklenmemesi.
"""
import pandas as pd
import pytest

from backtest.prop_eval import FUNDED, RULES, run_funded

BASLANGIC = pd.Timestamp("2026-01-05 14:30", tz="UTC")


def _rows(gun_sayisi: int, gunluk_islem: int = 1, r: float = 1.0) -> pd.DataFrame:
    """Her is gunu ayni sonucu veren yapay islem satirlari.

    Gercek veriye bagli degil: olcekleme kurali takvim + kar + odeme sayisi
    uzerine kurulu, fiyat hareketiyle ilgisi yok. Yapay veri burada dogru
    arac - sartlari tam kontrol edebiliyoruz.
    """
    kayitlar = []
    gun = BASLANGIC
    eklenen = 0
    while eklenen < gun_sayisi:
        if gun.weekday() < 5:
            for k in range(gunluk_islem):
                giris = gun + pd.Timedelta(minutes=30 * k)
                kayitlar.append({
                    "entry_time": giris,
                    "exit_time": giris + pd.Timedelta(minutes=15),
                    "risk_points": 20.0, "r": r,
                    "mae_r": 0.2, "mfe_r": max(r, 0.2),
                })
            eklenen += 1
        gun += pd.Timedelta(days=1)
    return pd.DataFrame(kayitlar).sort_values("entry_time").reset_index(drop=True)


def _calistir(rows: pd.DataFrame, funded, aylar: float = 12.0):
    son = rows.at[0, "entry_time"] + pd.Timedelta(days=int(aylar * 30.44))
    return run_funded(rows, RULES["ftmo50k"], funded, risk_pct=1.0,
                      start_pos=0, end_time=son)


def test_olcekleme_dort_aydan_once_tetiklenmez():
    """Uc aylik kazancli gecmis yeterli degil - takvim sarti gercek."""
    rows = _rows(gun_sayisi=60)          # ~3 ay
    sonuc = _calistir(rows, FUNDED["ftmo50k"], aylar=3.0)
    assert sonuc.scales == 0


def test_olcekleme_sartlar_saglaninca_tetiklenir():
    """Dort ay + %10 kar + iki odeme -> hesap buyur, pay %90 olur."""
    rows = _rows(gun_sayisi=200)         # ~10 ay
    sonuc = _calistir(rows, FUNDED["ftmo50k"])
    assert sonuc.outcome in ("yasiyor", "sure")
    assert len(sonuc.payouts) >= 2
    assert sonuc.scales >= 1
    # Olcekleme gercekten sermaye ekliyor mu: bir olcek = +$12.500.
    assert sonuc.final_balance > 50_000


def test_odeme_sayisi_yetmezse_olcekleme_yok():
    """Kar sarti tutsa bile iki odeme yoksa olcekleme olmamali.

    Odeme dongusunu 400 gune cekiyoruz: kar var, takvim dolu, ama firma
    hic odeme islememis. FTMO'nun uc sartindan biri eksik demektir.
    """
    kurallar = FUNDED["ftmo50k"]
    odemesiz = type(kurallar)(**{**kurallar.__dict__, "min_days_between": 400})
    rows = _rows(gun_sayisi=200)
    sonuc = _calistir(rows, odemesiz)
    assert len(sonuc.payouts) <= 1
    assert sonuc.scales == 0


def test_olcekleme_kapali_firmada_alan_bos():
    """Topstep ve Apex'te olcekleme modellenmiyor - sessizce sizmasin."""
    for anahtar in ("topstep50k", "apex50k", "apex50k_eod"):
        assert FUNDED[anahtar].scale_days is None
        assert FUNDED[anahtar].scale_split is None


def test_ftmo_olcek_parametreleri_kaynakla_uyumlu():
    """Sayilar ftmo.com'dan okundu; kazara degismesin diye kilitli."""
    f = FUNDED["ftmo50k"]
    assert f.profit_split == 0.80          # baslangic payi
    assert f.scale_split == 0.90           # olcek sonrasi
    assert f.scale_boost == 0.25           # %25 hesap buyumesi
    assert f.scale_profit_pct == 0.10      # 4 ayda %10 net kar
    assert f.scale_min_payouts == 2        # ayni donemde 2 islenmis odeme
    assert f.scale_days == pytest.approx(122, abs=3)   # ~4 ay
