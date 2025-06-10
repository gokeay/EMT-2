"""
🧪 EMT RL Project - Data Handler Unit Tests
DataHandler sınıfının tüm fonksiyonlarını test eder
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from datetime import datetime
import sys
sys.path.append('.')

from src.utils.data_handler import DataHandler


class TestDataHandler:
    """DataHandler unit testleri"""
    
    @pytest.fixture
    def sample_config(self):
        """Test için örnek konfigürasyon"""
        return {
            'data': {
                'load_file': 'test_load.csv',
                'solar_file': 'test_solar.csv',
                'wind_file': 'test_wind.csv'
            },
            'prices': {
                'night': {
                    'hours': [22, 23, 0, 1, 2, 3, 4, 5, 6, 7],
                    'price': 0.12123,
                    'category': 'low'
                },
                'day': {
                    'hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                    'price': 0.20428,
                    'category': 'medium'
                },
                'peak': {
                    'hours': [18, 19, 20, 21],
                    'price': 0.30867,
                    'category': 'high'
                }
            },
            'environment': {'battery_capacity_kwh': 5000},
            'training': {'total_episodes': 50}
        }
    
    @pytest.fixture
    def sample_data(self):
        """Test için örnek veriler"""
        dates = pd.date_range('2022-01-01', periods=100, freq='H')
        
        load_data = pd.DataFrame({
            'datetime': dates,
            'load_kw': np.random.uniform(800, 1500, 100)
        })
        
        solar_data = pd.DataFrame({
            'datetime': dates,
            'solar_power_kW': np.random.uniform(0, 1000, 100)
        })
        
        wind_data = pd.DataFrame({
            'datetime': dates,
            'wind_power_kW': np.random.uniform(200, 800, 100)
        })
        
        return load_data, solar_data, wind_data
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Geçici konfigürasyon dosyası"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def temp_csv_files(self, sample_data):
        """Geçici CSV dosyaları"""
        load_data, solar_data, wind_data = sample_data
        
        temp_files = []
        
        # Load file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            load_data.to_csv(f.name, index=False)
            temp_files.append(f.name)
        
        # Solar file  
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            solar_data.to_csv(f.name, index=False)
            temp_files.append(f.name)
        
        # Wind file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            wind_data.to_csv(f.name, index=False)
            temp_files.append(f.name)
        
        yield temp_files
        
        # Cleanup
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_config_loading(self, temp_config_file):
        """Konfigürasyon yükleme testi"""
        handler = DataHandler(temp_config_file)
        
        assert handler.config is not None
        assert 'data' in handler.config
        assert 'prices' in handler.config
        assert handler.data_paths == handler.config['data']
    
    def test_config_loading_invalid_path(self):
        """Geçersiz konfigürasyon yolu testi"""
        with pytest.raises(Exception):
            DataHandler("nonexistent_config.yaml")
    
    def test_load_csv_valid(self, temp_csv_files):
        """CSV yükleme başarı testi"""
        load_file, solar_file, wind_file = temp_csv_files
        
        # Temporary config for this test
        config = {
            'data': {'load_file': load_file},
            'prices': {'day': {'price': 0.2, 'category': 'medium'}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            handler = DataHandler(config_path)
            df = handler._load_csv(load_file, ['datetime', 'load_kw'])
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 100
            assert 'datetime' in df.columns
            assert 'load_kw' in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df['datetime'])
            
        finally:
            os.unlink(config_path)
    
    def test_load_csv_missing_columns(self, temp_csv_files):
        """Eksik sütun testi"""
        load_file = temp_csv_files[0]
        
        config = {
            'data': {'load_file': load_file},
            'prices': {'day': {'price': 0.2, 'category': 'medium'}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            handler = DataHandler(config_path)
            
            with pytest.raises(ValueError):
                handler._load_csv(load_file, ['datetime', 'nonexistent_column'])
                
        finally:
            os.unlink(config_path)
    
    def test_load_csv_negative_values(self, temp_csv_files):
        """Negatif değer işleme testi"""
        # Negatif değerli veri oluştur
        dates = pd.date_range('2022-01-01', periods=10, freq='H')
        data_with_negatives = pd.DataFrame({
            'datetime': dates,
            'load_kw': [-100, 200, -50, 300, 150, -25, 400, 100, -75, 250]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data_with_negatives.to_csv(f.name, index=False)
            temp_file = f.name
        
        config = {
            'data': {'load_file': temp_file},
            'prices': {'day': {'price': 0.2, 'category': 'medium'}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            handler = DataHandler(config_path)
            df = handler._load_csv(temp_file, ['datetime', 'load_kw'])
            
            # Negatif değerler 0 olmalı
            assert (df['load_kw'] >= 0).all()
            assert df['load_kw'].min() == 0
            
        finally:
            os.unlink(config_path)
            os.unlink(temp_file)
    
    def test_price_categories(self, temp_config_file):
        """Fiyat kategorisi testi"""
        handler = DataHandler(temp_config_file)
        
        # Test data
        test_df = pd.DataFrame({
            'hour': [0, 8, 18, 22],  # night, day, peak, night
            'datetime': pd.date_range('2022-01-01', periods=4, freq='H')
        })
        
        result_df = handler._add_price_categories(test_df)
        
        assert 'price_category' in result_df.columns
        assert 'price_low' in result_df.columns
        assert 'price_medium' in result_df.columns
        assert 'price_high' in result_df.columns
        
        # Kategori kontrolü
        assert result_df.iloc[0]['price_category'] == 'low'    # 0 saat
        assert result_df.iloc[1]['price_category'] == 'medium' # 8 saat
        assert result_df.iloc[2]['price_category'] == 'high'   # 18 saat
        assert result_df.iloc[3]['price_category'] == 'low'    # 22 saat
    
    def test_data_validation_success(self, temp_config_file, sample_data):
        """Veri doğrulama başarı testi"""
        handler = DataHandler(temp_config_file)
        
        # Mock data assignment
        handler.load_data, handler.solar_data, handler.wind_data = sample_data
        
        assert handler._validate_data() == True
    
    def test_data_validation_missing_data(self, temp_config_file):
        """Eksik veri doğrulama testi"""
        handler = DataHandler(temp_config_file)
        
        # One dataset missing
        handler.load_data = pd.DataFrame()
        handler.solar_data = None
        handler.wind_data = pd.DataFrame()
        
        assert handler._validate_data() == False
    
    def test_data_validation_different_lengths(self, temp_config_file):
        """Farklı uzunlukta veri testi"""
        handler = DataHandler(temp_config_file)
        
        dates1 = pd.date_range('2022-01-01', periods=100, freq='H')
        dates2 = pd.date_range('2022-01-01', periods=50, freq='H')  # Farklı uzunluk
        
        handler.load_data = pd.DataFrame({
            'datetime': dates1,
            'load_kw': np.random.uniform(800, 1500, 100)
        })
        
        handler.solar_data = pd.DataFrame({
            'datetime': dates2,
            'solar_power_kW': np.random.uniform(0, 1000, 50)
        })
        
        handler.wind_data = pd.DataFrame({
            'datetime': dates1,
            'wind_power_kW': np.random.uniform(200, 800, 100)
        })
        
        assert handler._validate_data() == False
    
    def test_combine_data(self, temp_config_file, sample_data):
        """Veri birleştirme testi"""
        handler = DataHandler(temp_config_file)
        
        handler.load_data, handler.solar_data, handler.wind_data = sample_data
        
        combined = handler._combine_data()
        
        assert isinstance(combined, pd.DataFrame)
        assert len(combined) == 100
        
        # Yeni sütunlar var mı?
        expected_columns = [
            'renewable_total_kW', 'net_load_kW', 
            'hour', 'day_of_week', 'month'
        ]
        for col in expected_columns:
            assert col in combined.columns
        
        # Renewable total doğru mu?
        assert (combined['renewable_total_kW'] == 
                combined['solar_power_kW'] + combined['wind_power_kW']).all()
        
        # Net load doğru mu?
        assert (combined['net_load_kW'] == 
                combined['load_kw'] - combined['renewable_total_kW']).all()
    
    def test_get_data_stats(self, temp_config_file, sample_data):
        """Veri istatistikleri testi"""
        handler = DataHandler(temp_config_file)
        
        handler.load_data, handler.solar_data, handler.wind_data = sample_data
        handler.combined_data = handler._combine_data()
        handler.combined_data = handler._add_price_categories(handler.combined_data)
        
        stats = handler.get_data_stats()
        
        assert isinstance(stats, dict)
        assert 'total_records' in stats
        assert 'load_stats' in stats
        assert 'solar_stats' in stats
        assert 'wind_stats' in stats
        assert 'renewable_coverage' in stats
        
        assert stats['total_records'] == 100
        assert isinstance(stats['load_stats']['mean'], float)
        assert isinstance(stats['renewable_coverage']['mean_coverage_pct'], float)
    
    def test_get_episode_data(self, temp_config_file, sample_data):
        """Episode veri testi"""
        handler = DataHandler(temp_config_file)
        
        handler.load_data, handler.solar_data, handler.wind_data = sample_data
        handler.combined_data = handler._combine_data()
        
        # Normal episode
        episode_data = handler.get_episode_data(50)
        assert len(episode_data) == 50
        
        # Episode çok büyük
        episode_data = handler.get_episode_data(150)
        assert len(episode_data) == 100  # Tüm veri döner
    
    def test_get_episode_data_no_data(self, temp_config_file):
        """Veri olmadan episode testi"""
        handler = DataHandler(temp_config_file)
        
        with pytest.raises(ValueError):
            handler.get_episode_data(50)


# Test çalıştırma konfigürasyonu
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 