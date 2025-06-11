"""
🔋 EMT RL Project - Energy Management Environment
Gymnasium uyumlu RL Environment sınıfı
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging
import yaml
import sys
import os

# Path için parent directory ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_handler import DataHandler

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyEnvironment(gym.Env):
    """
    Energy Management RL Environment
    
    State Space: [load_kw, solar_kw, wind_kw, battery_soc, price_low, price_medium, price_high]
    Action Space (Continuous, Normalized): 
        - Action 1: Grid Connection Tendency [-1, 1] -> Interpreted as 0 (Off) or 1 (On)
        - Action 2: Battery Power Tendency [-1, 1] -> Scaled to [-max_battery_power, +max_battery_power]
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, data_handler=None, config_path: str = "configs/config.yaml", config: Optional[Dict] = None):
        """
        Environment başlatma
        """
        super().__init__()
        
        # Konfigürasyon yükle
        if config:
            self.config = config
        else:
            self.config = self._load_config(config_path)
            
        self.env_config = self.config['environment']
        
        # Battery parameters
        self.battery_config = self.env_config['battery']
        self.battery_capacity = self.battery_config.get('capacity_kwh', 5000)
        self.min_soc = self.battery_config.get('min_soc', 0.1)
        self.max_soc = self.battery_config.get('max_soc', 0.9)
        self.max_battery_power = self.battery_config.get('max_power_kw', 5000)
        self.battery_efficiency = self.battery_config.get('efficiency', 0.92)
        self.initial_soc = self.battery_config.get('initial_soc', 0.8)
        self.battery_soc = self.initial_soc
        
        # Grid parameters from new config structure
        self.grid_config = self.env_config.get('grid', {})
        self.max_grid_power = self.env_config['grid']['max_power_kw']

        # Reward parameters
        self.reward_config = self.config.get('reward', {})
        self.unmet_load_penalty = self.reward_config.get('unmet_load_penalty', -100)
        self.soc_penalty_coef = self.reward_config.get('soc_penalty_coef', -2000)
        self.price_penalty_coef = self.reward_config.get('price_penalty_coef', {'low': -0.01, 'medium': -0.05, 'high': -0.1})
        self.unused_penalty_coef = self.reward_config.get('unused_penalty_coef', -50)
        self.cheap_energy_missed_penalty_coef = self.reward_config.get('cheap_energy_missed_penalty_coef', -50)
        
        # State & Action spaces
        self._define_spaces()
        
        # Data Handler
        self.data_handler = data_handler if data_handler is not None else DataHandler()
        self.episode_data = None
        
        # Episode state
        self.current_step = 0
        self.episode_length = self.config['training'].get('episode_length', 8760)
        
        # Episode metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'soc_violations': 0,
            'renewable_usage_kwh': 0.0,
            'grid_usage_kwh': 0.0,
            'battery_cycles': 0.0
        }
        
        logger.info("🏗️ EnergyEnvironment başlatıldı")
    
    def _load_config(self, config_path: str) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"❌ Config yüklenemedi: {e}")
            raise
    
    def _define_spaces(self):
        """State ve Action space'lerini tanımla"""
        
        # State Space: [load, solar, wind, soc, price_low, price_medium, price_high]
        state_low = np.array([
            0.0,      # load_kw (min)
            0.0,      # solar_kw (min)  
            0.0,      # wind_kw (min)
            0.0,      # battery_soc (min %0)
            0.0,      # price_low (0 or 1)
            0.0,      # price_medium (0 or 1)
            0.0       # price_high (0 or 1)
        ], dtype=np.float32)
        
        state_high = np.array([
            10000.0,  # load_kw (max, geniş)
            5000.0,   # solar_kw (max, geniş)
            3000.0,   # wind_kw (max, geniş)  
            1.0,      # battery_soc (max %100)
            1.0,      # price_low (0 or 1)
            1.0,      # price_medium (0 or 1)
            1.0       # price_high (0 or 1)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=state_low, 
            high=state_high, 
            dtype=np.float32
        )
        
        # Action Space (Normalized Continuous)
        action_low = np.array([-1.0, -1.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high, 
            dtype=np.float32
        )
        
        logger.info("🎯 State & Action spaces tanımlandı (Normalized Continuous)")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Environment'ı reset et"""
        super().reset(seed=seed)
        
        try:
            # Veri yükle (ilk reset'te)
            if self.episode_data is None:
                if not self.data_handler.load_all_data():
                    raise RuntimeError("❌ Veri yüklenemedi!")
            
            # Yeni episode verisi al
            self.episode_data = self.data_handler.get_episode_data(self.episode_length)
            
            # State sıfırla
            self.current_step = 0
            self.battery_soc = self.initial_soc
            
            # Metrics sıfırla
            self.episode_metrics = {
                'total_reward': 0.0,
                'soc_violations': 0,
                'renewable_usage_kwh': 0.0,
                'grid_usage_kwh': 0.0,
                'battery_cycles': 0.0
            }
            
            # İlk observation
            observation = self._get_observation()
            info = self._get_info()
            
            logger.info(f"🔄 Episode reset edildi - {len(self.episode_data)} step")
            return observation, info
            
        except Exception as e:
            logger.error(f"❌ Reset hatası: {e}")
            raise
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Environment'ta bir adım al
        """
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.episode_data) - 1:
            truncated = True
            
        # Mevcut veriyi al
        current_data = self.episode_data.iloc[self.current_step]
        load_kw = float(current_data['load_kw'])
        solar_kw = float(current_data['solar_power_kW'])
        wind_kw = float(current_data['wind_power_kW'])
        renewable_kw = solar_kw + wind_kw
        
        # --- 1. Aksiyonları Yorumla ---
        grid_connection_decision = 1 if action[0] > 0 else 0
        battery_power = float(action[1]) * self.max_battery_power

        # --- 2. Güvenlik ve Fizik Kurallarını Uygula ---
        is_critical_state = (self.battery_soc <= self.min_soc) and (renewable_kw < load_kw)
        if is_critical_state:
            grid_connection = 1
        else:
            grid_connection = grid_connection_decision

        battery_power = self._validate_battery_power(battery_power)

        # --- 3. Enerji Dengesini Hesapla ---
        grid_energy = 0.0
        unmet_load = 0.0

        if grid_connection == 1:
            required_grid_power = load_kw + battery_power - renewable_kw
            grid_energy = max(0, required_grid_power)
            
            if grid_energy > self.max_grid_power:
                unmet_load = grid_energy - self.max_grid_power
                grid_energy = self.max_grid_power
        else:
            balance = renewable_kw - battery_power - load_kw
            if balance < 0:
                unmet_load = abs(balance)

        # --- 4. Ödülü Hesapla ---
        reward, reward_details = self._calculate_reward(load_kw, renewable_kw, grid_energy, battery_power, unmet_load, grid_connection, current_data)
        
        # --- 5. Durumları Güncelle ---
        self._update_battery(battery_power)
        
        # --- 6. Sonraki Adıma Geç ---
        self.current_step += 1
        observation = self._get_observation()
        
        info = {
            'step_details': {
                'load': load_kw,
                'renewable_generation': renewable_kw,
                'grid_energy': grid_energy,
                'battery_power': battery_power,
                'battery_soc': self.battery_soc,
                'unmet_load': unmet_load,
                'reward_details': reward_details
            }
        }
        
        self._update_metrics(reward, grid_energy, renewable_kw, battery_power, unmet_load)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Mevcut state observation'ını döndür"""
        if self.current_step >= len(self.episode_data):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        current_data = self.episode_data.iloc[self.current_step]
        
        price_low = 1.0 if current_data['price_category'] == 'Low' else 0.0
        price_medium = 1.0 if current_data['price_category'] == 'Medium' else 0.0
        price_high = 1.0 if current_data['price_category'] == 'High' else 0.0

        return np.array([
            current_data['load_kw'],
            current_data['solar_power_kW'],
            current_data['wind_power_kW'],
            self.battery_soc,
            price_low,
            price_medium,
            price_high
        ], dtype=np.float32)
    
    def _calculate_reward(self, load_kw: float, renewable_kw: float, grid_energy: float,
                        battery_power: float, unmet_load: float, grid_connection: int, current_data: pd.Series) -> Tuple[float, Dict]:
        """
        💡 YENİ MANTIK: Çelişkili cezalar yerine net maliyet optimizasyonu.
        Ajanın tek hedefi var: Toplam ödülü (yani negatif maliyeti) maksimize etmek.
        """
        rewards = {}
        price_level = current_data.get('price_category', 'medium').lower()
        price_value = current_data.get('price', 0.2) # Gerçek fiyat değerini alalım

        # --- 1. KRİTİK HATA: Karşılanamayan Yük ---
        # Bu en büyük hatadır ve diğer her şeyi geçersiz kılar.
        if unmet_load > 0:
            rewards['unmet_load_penalty'] = unmet_load * self.unmet_load_penalty
            return sum(rewards.values()), rewards

        # --- 2. MALİYET HESAPLAMALARI ---

        # a) Şebeke Enerji Maliyeti: Bu, en doğal cezadır.
        # Kullandığın kadar ödersin. Fiyat katsayılarına gerek yok, gerçek fiyatı kullanalım.
        # config'deki price_penalty_coef'i -1 gibi bir değerle çarpım faktörü yapabiliriz.
        grid_cost_multiplier = -1.5 # Şebeke maliyetini daha belirgin yapmak için
        rewards['grid_cost'] = grid_energy * price_value * grid_cost_multiplier

        # b) Batarya Yıpranma Maliyeti (Degradation Cost):
        # Ajanın bataryayı gereksiz yere kullanmasını engeller. Her kullanımın küçük bir maliyeti olmalı.
        # Bu katsayıyı config'e ekleyebilirsiniz: battery_degradation_penalty: -0.1
        if battery_power != 0:
            rewards['battery_degradation_cost'] = abs(battery_power) * self.config['reward'].get('battery_degradation_penalty', -0.1)

        # --- 3. STRATEJİK FIRSATLAR (ÖDÜLLER VE BÜYÜK CEZALAR) ---

        excess_renewable = renewable_kw - load_kw

        # a) DURUM: Yenilenebilir Enerji Yükten Fazla
        if excess_renewable > 0:
            if grid_connection == 1 and grid_energy > 1: # 1kW gibi bir tolerans bırakalım
                # ❌ KESİN HATA: Bedava enerji varken şebekeyi KULLANMA! Çok büyük ceza.
                rewards['critical_grid_use_penalty'] = grid_energy * self.config['reward'].get('critical_grid_penalty', -100)
            
            if battery_power > 0: # 💡 DOĞRU DAVRANIŞ: Fazla enerjiyle şarj et.
                # Ödülü, şarj ettiği miktarla orantılı yapalım
                rewards['renewable_charge_reward'] = min(battery_power, excess_renewable) * self.config['reward'].get('renewable_charge_reward_coef', 1.0)
            elif self.battery_soc < self.max_soc:
                # ❌ KESİN HATA: Batarya doluyken bedava enerjiyi boşa harcama!
                rewards['renewable_waste_penalty'] = excess_renewable * self.unused_penalty_coef

        # b) DURUM: Düşük Fiyat Fırsatı
        if price_level == 'low' and self.battery_soc < self.max_soc:
            if battery_power > 0 and grid_energy > 0: # 💡 DOĞRU DAVRANIŞ: Ucuzken şebekeden şarj et.
                rewards['cheap_charge_reward'] = battery_power * self.config['reward'].get('cheap_charge_reward_coef', 2.0)
            elif battery_power <= 0:
                # ❌ YANLIŞ DAVRANIŞ: Ucuz şarj fırsatını kaçırma.
                soc_diff = self.max_soc - self.battery_soc
                rewards['missed_cheap_charge_penalty'] = soc_diff * self.cheap_energy_missed_penalty_coef
                
        # c) SOC Sınırları
        if self.battery_soc < self.min_soc:
            rewards['soc_violation_penalty'] = (self.min_soc - self.battery_soc) * self.soc_penalty_coef
        elif self.battery_soc > self.max_soc:
            rewards['soc_violation_penalty'] = (self.battery_soc - self.max_soc) * self.soc_penalty_coef


        # Eğer hiçbir ödül/ceza yoksa, küçük bir "hayatta kalma" ödülü verilebilir.
        if not rewards:
            rewards['time_step_reward'] = 0.1

        return sum(rewards.values()), rewards
    
    def _update_metrics(self, reward: float, grid_energy: float, renewable_kw: float, battery_power: float, unmet_load: float):
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['grid_usage_kwh'] += grid_energy
        self.episode_metrics['unmet_load_kwh'] = self.episode_metrics.get('unmet_load_kwh', 0) + unmet_load
        
        if not (self.min_soc <= self.battery_soc <= self.max_soc):
            self.episode_metrics['soc_violations'] += 1
        
        if battery_power < 0:
            self.episode_metrics['battery_cycles'] += abs(battery_power) / (2 * self.battery_capacity)
    
    def _get_info(self, **kwargs) -> Dict:
        """Get information about the current step"""
        info = {}
        info.update(kwargs)
        return info
    
    def render(self, mode: str = "human"):
        """Environment'ı görselleştir"""
        if self.current_step == 0:
            return
            
        current_data = self.episode_data.iloc[self.current_step - 1]
        
        print(f"Step {self.current_step-1}: "
              f"Load={current_data['load_kw']:.1f}kW, "
              f"Solar={current_data['solar_power_kW']:.1f}kW, "
              f"Wind={current_data['wind_power_kW']:.1f}kW, "
              f"SOC={self.battery_soc:.2%}, "
              f"Price={current_data['price_category']}")
        print(f"  SOC: {self.battery_soc:.2f}, Reward: {self.episode_metrics['total_reward']:.2f}")
    
    def close(self):
        """Environment'ı kapat"""
        logger.info("🔒 Environment kapatıldı")
    
    def _validate_battery_power(self, battery_power: float) -> float:
        """Batarya gücünü SOC durumuna göre doğrular ve ayarlar"""
        if battery_power < 0:
            if self.battery_soc <= self.min_soc:
                return 0.0
            max_discharge_kwh = (self.battery_soc - self.min_soc) * self.battery_capacity
            max_discharge_power = max_discharge_kwh / self.battery_efficiency
            return max(battery_power, -min(self.max_battery_power, max_discharge_power))
        elif battery_power > 0:
            if self.battery_soc >= self.max_soc:
                return 0.0
            max_charge_kwh = (self.max_soc - self.battery_soc) * self.battery_capacity
            max_charge_power = max_charge_kwh * self.battery_efficiency
            return min(battery_power, min(self.max_battery_power, max_charge_power))
        return 0.0

    def _update_battery(self, battery_power: float, time_step_hours: float = 1.0):
        """Batarya SOC'sini güncelle"""
        if battery_power > 0:  # Şarj
            soc_change = (battery_power * time_step_hours * self.battery_efficiency) / self.battery_capacity
        elif battery_power < 0:  # Deşarj
            soc_change = (battery_power * time_step_hours) / self.battery_efficiency / self.battery_capacity
        else:
            soc_change = 0.0
            
        self.battery_soc += soc_change
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0) 