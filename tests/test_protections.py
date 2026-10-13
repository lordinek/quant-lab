"""
Koruma katmani testleri.

Bu testlerin isi iki soruyu kilitlemek:
  1. Koruma dogru anda tetikleniyor mu (ne erken ne gec).
  2. Koruma yalnizca GECMISE bakiyor mu - bu dosyanin asil derdi.

Ikinci madde onemli cunku koruma katmani, gelecege bakmanin en kolay
sizacagi yerdir: "su ana kadar ne kaybettim" sorusu masum gorunur ama yanlis
yerden sorulunca acik pozisyonun henuz bilinmeyen sonucunu icerir.
"""
import pandas as pd
import pytest

from strategy.protections import Guard, ProtectionConfig


T0 = pd.Timestamp("2026-09-04 12:00", tz="UTC")


def at(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


# --- varsayilan: hicbir sey kapatmaz ---------------------------------------

def test_default_config_never_blocks():
    """Varsayilan tum korumalar KAPALI dogar - bir koruma acmak parametre
    secmektir, taranmadan secilmez."""
    g = Guard()
    for i in range(20):
        g.record_close(at(i), "stop", 100.0 - i)
    assert not g.blocked(at(21))


# --- ardisik stop ----------------------------------------------------------

def test_stop_streak_locks_after_n_consecutive():
    g = Guard(ProtectionConfig(stop_streak=3, stop_streak_lock_hours=12))

    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(2), "stop", 98.0)
    assert not g.blocked(at(2.5))          # daha iki tane

    g.record_close(at(3), "stop", 97.0)
    assert g.blocked(at(3.5))              # ucuncu geldi
    assert "3 ardisik stop" in g.reason(at(3.5))


def test_streak_resets_on_a_win():
    """Araya kazanc girerse seri kirilir - "son N islemin hepsi" demek bu."""
    g = Guard(ProtectionConfig(stop_streak=3))

    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(2), "hedef", 101.0)
    g.record_close(at(3), "stop", 100.0)
    g.record_close(at(4), "stop", 99.0)

    assert not g.blocked(at(5))


def test_lock_expires():
    g = Guard(ProtectionConfig(stop_streak=2, stop_streak_lock_hours=6))
    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(2), "stop", 98.0)

    assert g.blocked(at(7))          # 2 + 6 = 8'e kadar kilitli
    assert not g.blocked(at(8.1))


# --- pencere ici stop sayisi -----------------------------------------------

def test_stop_count_in_window():
    g = Guard(ProtectionConfig(stop_count=3, stop_count_hours=24,
                               stop_count_lock_hours=12))
    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(5), "hedef", 101.0)
    g.record_close(at(9), "stop", 100.0)
    assert not g.blocked(at(10))

    g.record_close(at(13), "stop", 99.0)
    assert g.blocked(at(14))


def test_old_stops_fall_out_of_window():
    """Pencere KAYAR: dun uc stop yemek bugunu kilitlemez."""
    g = Guard(ProtectionConfig(stop_count=3, stop_count_hours=24))
    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(2), "stop", 98.0)
    # Ucuncusu 30 saat sonra - ilk ikisi artik pencerenin disinda.
    g.record_close(at(32), "stop", 97.0)

    assert not g.blocked(at(33))


# --- dusus -----------------------------------------------------------------

def test_drawdown_measured_from_peak_not_start():
    """$150 -> $400 -> $340 bir dusustur; baslangicin ustunde olmasi
    onemli degil. Tepe takip edilmezse bot kar ederken korumasiz kalir."""
    g = Guard(ProtectionConfig(max_drawdown_pct=15.0))

    g.record_close(at(1), "hedef", 400.0)      # tepe 400
    assert not g.blocked(at(1.5))

    g.record_close(at(2), "stop", 340.0)       # tepeden %15
    assert g.blocked(at(2.5))
    assert "dusus" in g.reason(at(2.5))


def test_peak_starts_at_starting_balance():
    """Hic kazanmadan dusen hesapta tepe BASLANGICTIR.

    Tepe ilk kapanistan itibaren takip edilirse dusus oldugundan kucuk
    olculur ve koruma gec tetiklenir - tam da en cok gerektigi senaryoda.
    """
    g = Guard(ProtectionConfig(max_drawdown_pct=20.0), starting_balance=150.0)
    g.record_close(at(1), "stop", 145.0)
    g.record_close(at(2), "stop", 115.0)       # 150'den %23,3

    assert g.blocked(at(3))
    assert "23" in g.reason(at(3))             # 145'ten olculseydi %20,7 derdi


def test_drawdown_below_threshold_does_not_lock():
    g = Guard(ProtectionConfig(max_drawdown_pct=20.0))
    g.record_close(at(1), "hedef", 200.0)
    g.record_close(at(2), "stop", 170.0)       # %15 - esik %20
    assert not g.blocked(at(3))


# --- cooldown --------------------------------------------------------------

def test_cooldown_blocks_right_after_a_close():
    g = Guard(ProtectionConfig(cooldown_hours=2.0))
    g.record_close(at(1), "hedef", 101.0)

    assert g.blocked(at(2.0))          # 1 + 2 = 3'e kadar
    assert not g.blocked(at(3.1))


def test_cooldown_applies_to_wins_too():
    """Intikam islemi kaybettikten sonra yapilir ama kural kazanca da isler:
    cooldown 'duygu sogusun' kuralidir, 'kayip cezasi' degil."""
    g = Guard(ProtectionConfig(cooldown_hours=1.0))
    g.record_close(at(5), "hedef", 120.0)
    assert g.blocked(at(5.5))


# --- kilit birlesimi -------------------------------------------------------

def test_longer_lock_is_not_shortened():
    """Yeni bir tetik daha kisa kilit onerirse yururlukteki uzun kilit
    kisalmaz - koruma gevsetilmez."""
    g = Guard(ProtectionConfig(stop_streak=2, stop_streak_lock_hours=48,
                               stop_count=3, stop_count_hours=24,
                               stop_count_lock_hours=1))
    g.record_close(at(1), "stop", 99.0)
    g.record_close(at(2), "stop", 98.0)       # 48 saatlik kilit
    g.record_close(at(3), "stop", 97.0)       # kisa kilit onerir

    assert g.blocked(at(40))                  # uzun kilit hala yururlukte


def test_guard_only_sees_what_it_was_told():
    """Nedensellik: Guard'in bildigi tek sey kendisine BILDIRILEN kapanislar.

    Acik pozisyonun anlik zarari, gelecek bar, "gunun toplami" gibi seyleri
    goremez - gorebilseydi backtest canlidan farkli davranirdi.
    """
    g = Guard(ProtectionConfig(stop_streak=2))
    g.record_close(at(1), "stop", 99.0)
    # Ikinci stop HENUZ bildirilmedi (pozisyon acik) -> kilit yok.
    assert not g.blocked(at(1.5))
    g.record_close(at(2), "stop", 98.0)
    assert g.blocked(at(2.5))
