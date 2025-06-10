"""
🔋 EMT RL Project - Data Handler Module
CSV dosyalarını okuma, doğrulama ve preprocessing işlemleri
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import logging

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    CSV verilerini yükler ve RL environment için hazırlar
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Data Handler başlatma
        
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        self.config = self._load_config(config_path)
        self.data_paths = self.config['data']
        self.price_config = self.config['prices']
        
        # Veri depolama
        self.load_data: Optional[pd.DataFrame] = None
        self.solar_data: Optional[pd.DataFrame] = None  
        self.wind_data: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None
        
        logger.info("🔧 DataHandler başlatıldı")
    
    def _load_config(self, config_path: str) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"❌ Config yüklenemedi: {e}")
            raise
    
    def load_all_data(self) -> bool:
        """
        Tüm CSV dosyalarını yükle ve birleştir
        
        Returns:
            bool: Başarılı ise True
        """
        try:
            logger.info("📊 Veri yükleme başladı...")
            
            # CSV dosyalarını yükle
            self.load_data = self._load_csv(
                self.data_paths['load_file'], 
                ['datetime', 'load_kw']
            )
            
            self.solar_data = self._load_csv(
                self.data_paths['solar_file'],
                ['datetime', 'solar_power_kW'] 
            )
            
            self.wind_data = self._load_csv(
                self.data_paths['wind_file'],
                ['datetime', 'wind_power_kW']
            )
            
            # Veri doğrulama
            if not self._validate_data():
                return False
                
            # Verileri birleştir
            self.combined_data = self._combine_data()
            
            # Fiyat kategorilerini ekle
            self.combined_data = self._add_price_categories(self.combined_data)
            
            logger.info(f"✅ Toplam {len(self.combined_data)} satır veri yüklendi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def _load_csv(self, file_path: str, expected_columns: list) -> pd.DataFrame:
        """
        CSV dosyasını yükle ve datetime parse et
        
        Args:
            file_path: Dosya yolu
            expected_columns: Beklenen sütun isimleri
            
        Returns:
            pd.DataFrame: Yüklenmiş veri
        """
        try:
            df = pd.read_csv(file_path)
            
            # Sütun kontrolü
            if not all(col in df.columns for col in expected_columns):
                raise ValueError(f"❌ Eksik sütunlar: {file_path}")
            
            # Datetime dönüşümü
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Negatif değer kontrolü (enerji değerleri)
            numeric_cols = [col for col in df.columns if col != 'datetime']
            for col in numeric_cols:
                if (df[col] < 0).any():
                    logger.warning(f"⚠️ {file_path} dosyasında negatif değerler var!")
                    df[col] = df[col].clip(lower=0)  # Negatif değerleri 0 yap
            
            logger.info(f"📄 {file_path} yüklendi: {len(df)} satır")
            return df
            
        except Exception as e:
            logger.error(f"❌ {file_path} yüklenemedi: {e}")
            raise
    
    def _validate_data(self) -> bool:
        """
        Yüklenen verilerin tutarlılığını kontrol et
        
        Returns:
            bool: Geçerli ise True
        """
        try:
            # Tüm dataframe'ler yüklendi mi?
            if any(df is None for df in [self.load_data, self.solar_data, self.wind_data]):
                logger.error("❌ Bazı veriler yüklenmedi!")
                return False
            
            # Aynı tarih aralığında mı?
            dfs = [self.load_data, self.solar_data, self.wind_data]
            start_dates = [df['datetime'].min() for df in dfs]
            end_dates = [df['datetime'].max() for df in dfs]
            
            if not (all(d == start_dates[0] for d in start_dates) and 
                   all(d == end_dates[0] for d in end_dates)):
                logger.warning("⚠️ Veri tarih aralıkları farklı!")
            
            # Aynı uzunlukta mı?
            lengths = [len(df) for df in dfs]
            if not all(l == lengths[0] for l in lengths):
                logger.error(f"❌ Veri uzunlukları farklı: {lengths}")
                return False
            
            # Missing value kontrolü
            for i, df in enumerate(dfs):
                if df.isnull().any().any():
                    logger.warning(f"⚠️ Dataset {i+1}'de eksik değerler var!")
            
            logger.info("✅ Veri doğrulaması başarılı")
            return True
            
        except Exception as e:
            logger.error(f"❌ Veri doğrulama hatası: {e}")
            return False
    
    def _combine_data(self) -> pd.DataFrame:
        """
        Tüm verileri datetime üzerinden birleştir
        
        Returns:
            pd.DataFrame: Birleştirilmiş veri
        """
        try:
            # Inner join ile birleştir
            combined = self.load_data.merge(
                self.solar_data, on='datetime', how='inner'
            ).merge(
                self.wind_data, on='datetime', how='inner'
            )
            
            # Toplam yenilenebilir enerji
            combined['renewable_total_kW'] = (
                combined['solar_power_kW'] + combined['wind_power_kW']
            )
            
            # Net load (tüketim - yenilenebilir üretim)
            combined['net_load_kW'] = (
                combined['load_kw'] - combined['renewable_total_kW']
            )
            
            # Zaman özellikleri
            combined['hour'] = combined['datetime'].dt.hour
            combined['day_of_week'] = combined['datetime'].dt.dayofweek
            combined['month'] = combined['datetime'].dt.month
            
            logger.info(f"✅ Veriler birleştirildi: {len(combined)} satır")
            return combined
            
        except Exception as e:
            logger.error(f"❌ Veri birleştirme hatası: {e}")
            raise
    
    def _add_price_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fiyat kategorilerini ekle (low/medium/high)
        
        Args:
            df: Veri çerçevesi
            
        Returns:
            pd.DataFrame: Fiyat kategorili veri
        """
        def get_price_category(hour: int) -> str:
            """Saate göre fiyat kategorisi döndür"""
            for period_name, period_info in self.price_config.items():
                if hour in period_info['hours']:
                    return period_info['category']
            return 'medium'  # Default
        
        def get_price_value(hour: int) -> float:
            """Saate göre fiyat değeri döndür"""
            for period_name, period_info in self.price_config.items():
                if hour in period_info['hours']:
                    return period_info['price']
            return self.price_config['day']['price']  # Default
        
        # Kategorik ve sayısal fiyat bilgisi
        df['price_category'] = df['hour'].apply(get_price_category)
        df['price_value'] = df['hour'].apply(get_price_value)
        
        # One-hot encoding for price categories
        df['price_low'] = (df['price_category'] == 'low').astype(int)
        df['price_medium'] = (df['price_category'] == 'medium').astype(int)
        df['price_high'] = (df['price_category'] == 'high').astype(int)
        
        logger.info("✅ Fiyat kategorileri eklendi")
        return df
    
    def get_data_stats(self) -> Dict:
        """
        Veri istatistiklerini döndür
        
        Returns:
            Dict: İstatistik bilgileri
        """
        if self.combined_data is None:
            return {}
        
        stats = {
            'total_records': len(self.combined_data),
            'date_range': {
                'start': self.combined_data['datetime'].min().strftime('%Y-%m-%d'),
                'end': self.combined_data['datetime'].max().strftime('%Y-%m-%d')
            },
            'load_stats': {
                'mean': float(self.combined_data['load_kw'].mean()),
                'min': float(self.combined_data['load_kw'].min()),
                'max': float(self.combined_data['load_kw'].max()),
                'std': float(self.combined_data['load_kw'].std())
            },
            'solar_stats': {
                'mean': float(self.combined_data['solar_power_kW'].mean()),
                'min': float(self.combined_data['solar_power_kW'].min()),
                'max': float(self.combined_data['solar_power_kW'].max()),
                'total_generation': float(self.combined_data['solar_power_kW'].sum())
            },
            'wind_stats': {
                'mean': float(self.combined_data['wind_power_kW'].mean()),
                'min': float(self.combined_data['wind_power_kW'].min()),
                'max': float(self.combined_data['wind_power_kW'].max()),
                'total_generation': float(self.combined_data['wind_power_kW'].sum())
            },
            'renewable_coverage': {
                'mean_coverage_pct': float(
                    (self.combined_data['renewable_total_kW'] / 
                     self.combined_data['load_kw'] * 100).mean()
                ),
                'hours_fully_covered': int(
                    (self.combined_data['renewable_total_kW'] >= 
                     self.combined_data['load_kw']).sum()
                )
            }
        }
        
        return stats
    
    def get_episode_data(self, episode_length: int = 8760) -> pd.DataFrame:
        """
        Episode için veri döndür
        
        Args:
            episode_length: Episode uzunluğu (saat)
            
        Returns:
            pd.DataFrame: Episode verisi
        """
        if self.combined_data is None:
            raise ValueError("❌ Veri henüz yüklenmedi!")
        
        if len(self.combined_data) < episode_length:
            logger.warning(f"⚠️ İstenen episode uzunluğu ({episode_length}) mevcut veriden ({len(self.combined_data)}) büyük!")
            return self.combined_data.copy()
        
        # Random başlangıç noktası seç
        max_start = len(self.combined_data) - episode_length
        start_idx = np.random.randint(0, max_start + 1)
        
        episode_data = self.combined_data.iloc[start_idx:start_idx + episode_length].copy()
        episode_data = episode_data.reset_index(drop=True)
        
        logger.info(f"📊 Episode verisi hazırlandı: {len(episode_data)} satır")
        return episode_data 