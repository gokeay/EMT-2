import pandas as pd
import numpy as np

# Verileri oku
print("\n=== Veri Okuma ===")
df_solar = pd.read_csv('sim_solar_gen_result.csv', parse_dates=['datetime'])
df_load = pd.read_csv('synthetic_load_itu.csv', parse_dates=['datetime'])
df_wind = pd.read_csv('sim_wind_gen_result.csv', parse_dates=['datetime'])

# Toplam üretimi hesapla
total_generation = df_solar['solar_power_kW'] + df_wind['wind_power_kW']

# Net yükü hesapla (Load - Total Generation)
net_load = df_load['load_kw'] - total_generation

# Negatif net yük sayısını bul
negative_count = (net_load < 0).sum()
total_hours = len(net_load)

print(f"\nSonuçlar:")
print(f"Toplam veri sayısı: {total_hours} saat")
print(f"Negatif net yük sayısı: {negative_count} saat")
print(f"Yüzde: {(negative_count/total_hours*100):.2f}%")

print("\nİstatistikler:")
print(f"Net Yük Min: {net_load.min():.2f} kW")
print(f"Net Yük Max: {net_load.max():.2f} kW")
print(f"Net Yük Ortalama: {net_load.mean():.2f} kW")

# En düşük 5 net yük değerini göster
print("\nEn düşük 5 net yük değeri ve tarihleri:")
lowest_5 = pd.DataFrame({
    'datetime': df_load['datetime'],
    'net_load': net_load
}).nsmallest(5, 'net_load')

print(lowest_5.to_string(index=False)) 