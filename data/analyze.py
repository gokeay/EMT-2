#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Üç grafik gösterir:
1. Toplam Üretim (Solar + Wind)
2. Yük (Load)
3. Net Yük (Load - Toplam Üretim)

Hepsi tek slider ile kontrol edilir.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.dates as mdates

# ─────────────── KULLANICI AYARLARI ───────────────
WINDOW = 200                                   # Aynı anda gösterilecek örnek adedi
CURDIR = os.path.dirname(os.path.abspath(__file__))

# Dosyalar + kullanılacak sütun adları
SOLAR_CSV = "sim_solar_gen_result.csv"         ; SOLAR_COL  = "solar_power_kW"
LOAD_CSV  = "synthetic_load_itu.csv"           ; LOAD_COL   = "load_kw"
WIND_CSV = "sim_wind_gen_result.csv"           ; WIND_COL = "wind_power_kW"
# ──────────────────────────────────────────────────

# Yardımcı tarih biçimleyici
date_fmt = mdates.DateFormatter('%Y-%m-%d')

# --------- 1) Verileri oku ---------
def read_df(path, col):
    full = os.path.join(CURDIR, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{path} bulunamadı!")
    df = pd.read_csv(full, parse_dates=["datetime"])
    print(f"\nDosya okundu: {path}")
    print(f"Sütunlar: {df.columns.tolist()}")
    print(f"İlk 5 değer ({col}):")
    print(df[col].head())
    print(f"Min değer: {df[col].min():.2f}")
    print(f"Max değer: {df[col].max():.2f}")
    return df.sort_values("datetime").reset_index(drop=True)

print("\n=== Veri Okuma ===")
df_solar = read_df(SOLAR_CSV, SOLAR_COL)
df_load  = read_df(LOAD_CSV, LOAD_COL)
df_wind  = read_df(WIND_CSV, WIND_COL)

# Toplam üretimi hesapla
df_solar['total_generation'] = df_solar[SOLAR_COL] + df_wind[WIND_COL]

# Net yükü hesapla (Load - Total Generation)
df_solar['net_load'] = df_load[LOAD_COL] - df_solar['total_generation']

# Ortak uzunluk
data_len = len(df_solar)
print(f"\nVeri boyutları:")
print(f"Solar: {len(df_solar)} satır")
print(f"Load: {len(df_load)} satır")
print(f"Wind: {len(df_wind)} satır")

# --------- 2) Şekil yerleşimi ---------
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(
    nrows=4, ncols=1,
    height_ratios=[4, 4, 4, 0.4],  # Son satır: slider
)

# --- Üst grafik: Toplam Üretim ---
ax_gen = fig.add_subplot(gs[0])
ln_solar, = ax_gen.plot(df_solar["datetime"].iloc[:WINDOW], 
                       df_solar[SOLAR_COL].iloc[:WINDOW], 
                       label='Solar', color='orange', alpha=0.7, linewidth=1.5)
ln_wind, = ax_gen.plot(df_wind["datetime"].iloc[:WINDOW], 
                      df_wind[WIND_COL].iloc[:WINDOW], 
                      label='Wind', color='blue', alpha=0.7, linewidth=1.5)
ln_total, = ax_gen.plot(df_solar["datetime"].iloc[:WINDOW], 
                       df_solar['total_generation'].iloc[:WINDOW], 
                       label='Total', color='green', linewidth=2)
ax_gen.set_title("Toplam Enerji Üretimi", fontsize=14, fontweight='bold', pad=10)
ax_gen.set_ylabel("Güç (kW)", fontsize=12)
ax_gen.grid(True, alpha=.3)
ax_gen.legend(fontsize=10)
ax_gen.xaxis.set_major_formatter(date_fmt)

# --- Orta grafik: Yük Talebi ---
ax_load = fig.add_subplot(gs[1])
ln_load, = ax_load.plot(df_load["datetime"].iloc[:WINDOW],
                       df_load[LOAD_COL].iloc[:WINDOW], 
                       color='red', linewidth=2, label='Load')
ax_load.set_title("Kampüs Yük Talebi", fontsize=14, fontweight='bold', pad=10)
ax_load.set_ylabel("Yük (kW)", fontsize=12)
ax_load.grid(True, alpha=.3)
ax_load.xaxis.set_major_formatter(date_fmt)
ax_load.legend(fontsize=10)

# Y ekseni limitlerini manuel ayarla
y_min = df_load[LOAD_COL].min() * 0.9
y_max = df_load[LOAD_COL].max() * 1.1
ax_load.set_ylim(y_min, y_max)

# --- Alt grafik: Net Yük ---
ax_net = fig.add_subplot(gs[2])
ln_net, = ax_net.plot(df_solar["datetime"].iloc[:WINDOW],
                      df_solar['net_load'].iloc[:WINDOW], 
                      color='purple', linewidth=2, label='Net Load')
ax_net.set_title("Net Yük (Load - Generation)", fontsize=14, fontweight='bold', pad=10)
ax_net.set_ylabel("Net Yük (kW)", fontsize=12)
ax_net.grid(True, alpha=.3)
ax_net.xaxis.set_major_formatter(date_fmt)
ax_net.legend(fontsize=10)

# X-limit'leri pencereye göre ayarla
for ax in (ax_gen, ax_load, ax_net):
    ax.set_xlim(df_solar["datetime"].iloc[0], df_solar["datetime"].iloc[WINDOW-1])
    ax.tick_params(axis='x', rotation=15)

# Y ekseni limitlerini ayarla
ax_gen.set_ylim(0, max(df_solar['total_generation'].max(), 
                       df_solar[SOLAR_COL].max(), 
                       df_wind[WIND_COL].max()) * 1.1)

# --------- 3) Slider ---------
ax_slider = fig.add_subplot(gs[3])
ax_slider.set_facecolor("lightgray")
ax_slider.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
slider = Slider(ax_slider, "Kaydır", 0, data_len-WINDOW, valinit=0, valstep=1)

# --------- 4) Call-back fonksiyonu ---------
def update(val):
    start = int(slider.val)
    end = start + WINDOW
    
    # Üretim grafiğini güncelle
    ln_solar.set_xdata(df_solar["datetime"].iloc[start:end])
    ln_solar.set_ydata(df_solar[SOLAR_COL].iloc[start:end])
    ln_wind.set_xdata(df_wind["datetime"].iloc[start:end])
    ln_wind.set_ydata(df_wind[WIND_COL].iloc[start:end])
    ln_total.set_xdata(df_solar["datetime"].iloc[start:end])
    ln_total.set_ydata(df_solar['total_generation'].iloc[start:end])
    
    # Yük grafiğini güncelle
    ln_load.set_xdata(df_load["datetime"].iloc[start:end])
    ln_load.set_ydata(df_load[LOAD_COL].iloc[start:end])
    
    # Net yük grafiğini güncelle
    ln_net.set_xdata(df_solar["datetime"].iloc[start:end])
    ln_net.set_ydata(df_solar['net_load'].iloc[start:end])
    
    # X ekseni limitlerini güncelle
    for ax in (ax_gen, ax_load, ax_net):
        ax.set_xlim(df_solar["datetime"].iloc[start],
                   df_solar["datetime"].iloc[end-1])
    
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
print("\nGrafik gösteriliyor...")
plt.show()
