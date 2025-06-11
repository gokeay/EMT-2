"""
Data handler for EMT RL project.
Handles loading and preprocessing of energy data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles loading and preprocessing of energy data for the EMT RL environment.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize DataHandler.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        self.logger = logging.getLogger(__name__)
        self.combined_data = None
        
    def load_energy_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load energy data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with energy data
        """
        try:
            # Eğer file_path zaten data/ ile başlıyorsa, sadece o path'i kullan
            if str(file_path).startswith('data/') or str(file_path).startswith('data\\'):
                full_path = Path(file_path)
            else:
                full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
            
            if str(full_path) in self.data_cache:
                return self.data_cache[str(full_path)]
                
            df = pd.read_csv(full_path)
            self.data_cache[str(full_path)] = df
            self.logger.info(f"Loaded data from {full_path}: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_all_data(self) -> bool:
        """
        Load all available energy data files and combine them.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Dosya yollarını kontrol et
            load_file = getattr(self, 'load_file', '/content/EMT-2/data/synthetic_load_itu.csv')
            solar_file = getattr(self, 'solar_file', '/content/EMT-2/data/sim_solar_gen_result.csv')
            wind_file = getattr(self, 'wind_file', '/content/EMT-2/data/sim_wind_gen_result.csv')
            # Verileri yükle
            load_data = self.load_energy_data(load_file)
            solar_data = self.load_energy_data(solar_file)
            wind_data = self.load_energy_data(wind_file)
            
            # Datetime sütunlarını parse et
            load_data['datetime'] = pd.to_datetime(load_data['datetime'])
            solar_data['datetime'] = pd.to_datetime(solar_data['datetime'])
            wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])
            
            # Verileri birleştir
            combined = load_data.merge(solar_data, on='datetime', how='inner')
            combined = combined.merge(wind_data, on='datetime', how='inner')
            
            # Fiyat kategorilerini ekle
            combined = self._add_price_categories(combined)
            
            # Yenilenebilir toplam ekle
            combined['renewable_total_kW'] = combined['solar_power_kW'] + combined['wind_power_kW']
            combined['net_load_kW'] = combined['load_kw'] - combined['renewable_total_kW']
            
            self.combined_data = combined
            self.logger.info(f"✅ Tüm veriler başarıyla yüklendi ve birleştirildi: {len(combined)} satır")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def load_all_data_dict(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available energy data files as dictionary.
        
        Returns:
            Dictionary mapping file names to DataFrames
        """
        data_files = {
            'wind': 'sim_wind_gen_result.csv',
            'solar': 'sim_solar_gen_result.csv',
            'load': 'synthetic_load_itu.csv',
            'weather': 'istanbul_sariyer_tmy-2022_v2.csv'
        }
        
        loaded_data = {}
        for key, filename in data_files.items():
            try:
                loaded_data[key] = self.load_energy_data(filename)
            except Exception as e:
                self.logger.warning(f"Could not load {key} data from {filename}: {e}")
                
        return loaded_data
    
    def _add_price_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat kategorilerini ekle"""
        df = df.copy()
        df['hour'] = df['datetime'].dt.hour
        
        # Fiyat kategorileri
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
        day_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        peak_hours = [18, 19, 20, 21]
        
        # Kategori sütunları
        df['price_low'] = df['hour'].isin(night_hours).astype(int)
        df['price_medium'] = df['hour'].isin(day_hours).astype(int)
        df['price_high'] = df['hour'].isin(peak_hours).astype(int)
        
        # Kategori ismi
        df['price_category'] = 'low'
        df.loc[df['price_medium'] == 1, 'price_category'] = 'medium'
        df.loc[df['price_high'] == 1, 'price_category'] = 'high'
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, 
                       normalize: bool = True,
                       handle_missing: str = 'interpolate') -> pd.DataFrame:
        """
        Preprocess energy data.
        
        Args:
            df: Input DataFrame
            normalize: Whether to normalize numerical columns
            handle_missing: How to handle missing values ('drop', 'interpolate', 'fill')
            
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Handle missing values
        if handle_missing == 'drop':
            df_processed = df_processed.dropna()
        elif handle_missing == 'interpolate':
            df_processed = df_processed.interpolate()
        elif handle_missing == 'fill':
            df_processed = df_processed.fillna(0)
            
        # Normalize numerical columns
        if normalize:
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df_processed[col].std() > 0:
                    df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
                    
        return df_processed
    
    def get_time_series_data(self, data_type: str, 
                           start_idx: int = 0, 
                           length: int = 1000) -> np.ndarray:
        """
        Get time series data for training.
        
        Args:
            data_type: Type of data ('wind', 'solar', 'load', 'weather')
            start_idx: Starting index
            length: Length of time series
            
        Returns:
            Time series data as numpy array
        """
        all_data = self.load_all_data_dict()
        
        if data_type not in all_data:
            self.logger.error(f"❌ Data type {data_type} not available - Mock veri kullanılmayacak!")
            raise ValueError(f"❌ Gerçek veri bulunamadı: {data_type}. Mock veri kesinlikle kullanılmaz!")
            
        df = all_data[data_type]
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            self.logger.error(f"❌ No numerical columns in {data_type} data - Mock veri kullanılmayacak!")
            raise ValueError(f"❌ {data_type} verisinde numerik sütun bulunamadı!")
            
        # Extract time series
        end_idx = min(start_idx + length, len(df))
        data = df[numerical_cols].iloc[start_idx:end_idx].values
        
        # If not enough data, raise error - NO MOCK DATA!
        if data.shape[0] < length:
            self.logger.error(f"❌ Yetersiz gerçek veri: {data.shape[0]} < {length} - Mock veri kullanılmayacak!")
            raise ValueError(f"❌ Yetersiz gerçek veri! İstenen: {length}, Mevcut: {data.shape[0]}")
            
        return data
    
    def generate_mock_data(self, length: int, features: int = 1) -> np.ndarray:
        """
        ❌ MOCK VERİ FONKSİYONU DEVRE DIŞI! 
        
        Bu fonksiyon artık çalışmaz - sadece gerçek veri kullanılır!
        """
        self.logger.error("❌ Mock veri oluşturma denemesi! Bu kesinlikle yasaklandı!")
        raise RuntimeError("❌ MOCK VERİ YASAK! Sadece gerçek veri kullanılabilir!")
    
    def get_data_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all loaded data.
        
        Returns:
            Dictionary with statistics for each data type
        """
        all_data = self.load_all_data_dict()
        stats = {}
        
        for data_type, df in all_data.items():
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                stats[data_type] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'numerical_columns': list(numerical_cols),
                    'mean': df[numerical_cols].mean().to_dict(),
                    'std': df[numerical_cols].std().to_dict(),
                    'min': df[numerical_cols].min().to_dict(),
                    'max': df[numerical_cols].max().to_dict()
                }
            else:
                stats[data_type] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'numerical_columns': [],
                    'note': 'No numerical columns found'
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
            self.logger.warning(f"⚠️ İstenen episode uzunluğu ({episode_length}) mevcut veriden ({len(self.combined_data)}) büyük!")
            return self.combined_data.copy()
        
        # Random başlangıç noktası seç
        max_start = len(self.combined_data) - episode_length
        start_idx = np.random.randint(0, max_start + 1)
        
        episode_data = self.combined_data.iloc[start_idx:start_idx + episode_length].copy()
        episode_data = episode_data.reset_index(drop=True)
        
        self.logger.info(f"📊 Episode verisi hazırlandı: {len(episode_data)} satır")
        return episode_data
    
    def clear_cache(self):
        """Clear data cache."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared") 