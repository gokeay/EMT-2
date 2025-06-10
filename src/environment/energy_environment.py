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
        
        Args:
            data_handler: DataHandler instance (opsiyonel)
            config_path: Konfigürasyon dosyası yolu (eğer config sözlüğü verilmezse kullanılır)
            config: Hazır konfigürasyon sözlüğü (opsiyonel, testler için)
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
        # Batarya başlangıç SOC'sini tek yerden al ve iki yerde de aynı değeri kullan
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
        
        # Grid düzeltme takibi
        self.grid_adjustments = {
            'violations': 0,
            'increase_count': 0,
            'zero_count': 0,
            'last_log_step': -1000
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
        # Action 1: Grid connection tendency (-1: Off, 1: On)
        # Action 2: Battery power tendency (-1: Full Discharge, 1: Full Charge)
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
            
            # Grid adjustment tracking sıfırla
            self.grid_adjustments = {
                'violations': 0,
                'increase_count': 0,
                'zero_count': 0,
                'last_log_step': -1000
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
        
        Args:
            action: Normalized continuous action [grid_tendency, battery_tendency]
            
        Returns:
            observation, reward, terminated, truncated, info
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
        # Grid bağlantı kararını belirle (continuous -> discrete)
        grid_connection_decision = 1 if action[0] > 0 else 0
        
        # Batarya gücünü belirle (continuous -> scaled)
        battery_power = float(action[1]) * self.max_battery_power

        # --- 2. Güvenlik ve Fizik Kurallarını Uygula ---
        # Kritik durum kontrolü: SOC düşük ve yenilenebilir yetersizse şebekeye bağlanmayı zorunlu kıl
        is_critical_state = (self.battery_soc <= self.min_soc) and (renewable_kw < load_kw)
        if is_critical_state:
            grid_connection = 1  # Ajanın kararını geçersiz kıl
        else:
            grid_connection = grid_connection_decision

        # Batarya gücünü fiziksel limitlere göre ayarla (şarj/deşarj için SOC kontrolü)
        battery_power = self._validate_battery_power(battery_power)

        # --- 3. Enerji Dengesini Hesapla ---
        grid_energy = 0.0
        unmet_load = 0.0

        if grid_connection == 1:
            # Şebeke bağlı: Gerekli enerjiyi hesapla. Pozitif ise şebekeden çek, negatif ise 0 yap (şebekeye satış yok)
            required_grid_power = load_kw + battery_power - renewable_kw
            grid_energy = max(0, required_grid_power)
            
            # Şebeke gücü limiti kontrolü
            if grid_energy > self.max_grid_power:
                unmet_load = grid_energy - self.max_grid_power
                grid_energy = self.max_grid_power
                else:
            # Şebeke bağlı değil: Yenilenebilir ve batarya ile yükü karşılamaya çalış
            balance = renewable_kw - battery_power - load_kw
            if balance < 0:
                unmet_load = abs(balance) # Karşılanamayan yük miktarı

        # --- 5. Ödülü Hesapla ---
        reward, reward_details = self._calculate_reward(load_kw, renewable_kw, grid_energy, battery_power, unmet_load, grid_connection, current_data)
        
        # --- 4. Durumları Güncelle (ÖDÜLDEN SONRA, GÖZLEMDEN ÖNCE) ---
        self._update_battery(battery_power)
        
        # --- 6. Sonraki Adıma Geç ---
            self.current_step += 1
        observation = self._get_observation()
        
        # Info'ya testler ve görselleştirme için detaylı bilgi ekleyelim
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
        """(YENİ DÜZELTME) Ajanın öğrenme kısırdöngüsünü kıran, bütünsel ödül fonksiyonu"""
        
        rewards = {}

        # --- ÖNCELİK 1: Karşılanamayan Yük (KRİTİK HATA) ---
        # Bu durum varsa, başka hiçbir şeye bakılmaz ve ağır ceza verilir.
        if unmet_load > 0:
            rewards['unmet_load'] = unmet_load * self.unmet_load_penalty
            # Karşılanamayan yük en büyük hata olduğu için burada çıkmak mantıklı.
            return sum(rewards.values()), rewards

        # --- ÖNCELİK 2: Diğer Tüm Stratejik ve Operasyonel Cezalar ---
        # Bu cezalar birikir ve ajana daha zengin bir geri bildirim sağlar.

        # a) SOC İhlali (Artık fonksiyondan çıkmıyor)
        if self.battery_soc < self.min_soc:
            soc_penalty = (self.min_soc - self.battery_soc) * self.soc_penalty_coef
            if battery_power > 0: # Düzeltme aksiyonu (şarj) varsa cezayı azalt
                soc_penalty *= 0.5
            rewards['soc_violation'] = soc_penalty
        elif self.battery_soc > self.max_soc:
            soc_penalty = (self.battery_soc - self.max_soc) * self.soc_penalty_coef
            if battery_power < 0: # Düzeltme aksiyonu (deşarj) varsa cezayı azalt
                soc_penalty *= 0.5
            rewards['soc_violation'] = soc_penalty

        # b) Gereksiz Şebeke Kullanımı
        can_operate_off_grid = (renewable_kw >= load_kw) or \
                               (self.battery_soc > self.min_soc and (renewable_kw + ((self.battery_soc - self.min_soc) * self.battery_capacity)) >= load_kw)
        
        if grid_connection == 1 and can_operate_off_grid and grid_energy > 0:
            price_level = current_data.get('price_category', 'medium').lower()
            penalty_coef = self.price_penalty_coef.get(price_level, -1.0)
            rewards['unnecessary_grid'] = penalty_coef * grid_energy

        # c) Yenilenebilir Enerji İsrafı
                excess_renewable = renewable_kw - load_kw
        if excess_renewable > 0 and self.battery_soc < self.max_soc and (battery_power <= 0 or battery_power == 0): # Eğer şarj etmiyorsan
             wasted_power = excess_renewable
             rewards['renewable_waste'] = wasted_power * self.unused_penalty_coef

        # d) Ucuz Şarj Fırsatını Kaçırma
        price_level = current_data.get('price_category', 'medium').lower()
        if self.battery_soc < self.max_soc and (price_level == 'low' or price_level == 'medium') and battery_power <= 0:
            soc_diff = self.max_soc - self.battery_soc
            rewards['missed_cheap_charge'] = soc_diff * self.cheap_energy_missed_penalty_coef
            
        # e) Şebeke kullanımının genel maliyeti (gereksiz olmasa bile)
        if grid_energy > 0:
            price_level = current_data.get('price_category', 'medium').lower()
            price_coef = self.price_penalty_coef.get(price_level, -1.0) # Fiyat katsayıları zaten negatif
            rewards['grid_cost'] = grid_energy * price_coef
            
        # Tüm hesaplanan ödül/ceza bileşenlerini topla
        return sum(rewards.values()), rewards
    
    def _update_metrics(self, reward: float, grid_energy: float, renewable_kw: float, battery_power: float, unmet_load: float):
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['grid_usage_kwh'] += grid_energy
        self.episode_metrics['unmet_load_kwh'] = self.episode_metrics.get('unmet_load_kwh', 0) + unmet_load
        
        if not (self.min_soc <= self.battery_soc <= self.max_soc):
            self.episode_metrics['soc_violations'] += 1
        
        # Batarya döngüsünü yaklaşık olarak hesapla (deşarj miktarına göre)
        if battery_power < 0:
        self.episode_metrics['battery_cycles'] += abs(battery_power) / (2 * self.battery_capacity)
    
    def _get_info(self, **kwargs) -> Dict:
        """Get information about the current step. (Artık doğrudan step içinde oluşturuluyor)"""
        # Bu fonksiyonun içeriği step metoduna taşındı.
        # Gelecekteki kullanımlar için temel yapıyı koruyalım.
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
        """Batarya gücünü SOC durumuna göre doğrular ve ayarlar."""
        # Deşarj durumu
        if battery_power < 0:
            # SOC minimumun altındaysa veya eşitse deşarja izin verme
            if self.battery_soc <= self.min_soc:
                return 0.0
            # İzin verilen maksimum deşarj miktarını hesapla (verimliliği de katarak)
            max_discharge_kwh = (self.battery_soc - self.min_soc) * self.battery_capacity
            max_discharge_power = max_discharge_kwh / self.battery_efficiency # Deşarjda verimlilik paydaya gelir
            # İstenen deşarj, izin verilenin üzerindeyse limiti uygula
            return max(battery_power, -min(self.max_battery_power, max_discharge_power))
        # Şarj durumu
        elif battery_power > 0:
            # SOC maksimumun üzerindeyse şarja izin verme
            if self.battery_soc >= self.max_soc:
                return 0.0
            # İzin verilen maksimum şarj miktarını hesapla
            max_charge_kwh = (self.max_soc - self.battery_soc) * self.battery_capacity
            max_charge_power = max_charge_kwh * self.battery_efficiency # şarj verimi
            # İstenen şarj, izin verilenin üzerindeyse limiti uygula
            return min(battery_power, min(self.max_battery_power, max_charge_power))
        # Batarya power = 0 (idle)
        return 0.0

    def _update_battery(self, battery_power: float, time_step_hours: float = 1.0):
        """Batarya SOC'sini güncelle."""
        
        # Batarya gücü verimlilikle ayarlanır
        if battery_power > 0:  # Şarj
            soc_change = (battery_power * time_step_hours * self.battery_efficiency) / self.battery_capacity
        elif battery_power < 0:  # Deşarj
            soc_change = (battery_power * time_step_hours) / self.battery_efficiency / self.battery_capacity
            else:
            soc_change = 0.0
            
        self.battery_soc += soc_change
        # SOC'yi [0, 1] aralığında tut (nadiren de olsa limit aşımı olabilir)
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0) 