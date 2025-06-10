"""
🧪 EMT RL Project - (GÜNCELLENMİŞ) Energy Environment Unit Tests
Yeni aksiyon ve ödül mantığına göre güncellenmiş testler
"""

import pytest
import numpy as np
import pandas as pd
import yaml
import gymnasium as gym
from src.environment.energy_environment import EnergyEnvironment
from src.data.data_handler import DataHandler

# Sabit bir test konfigürasyonu oluşturalım
CONFIG_YAML = """
environment:
  battery:
    capacity_kwh: 1000
    initial_soc: 0.5
    min_soc: 0.2
    max_soc: 0.9
    max_power_kw: 500
    efficiency: 0.95
  grid:
    max_power_kw: 10000

reward:
  unmet_load_penalty: -1000
  soc_penalty_coef: -200
  price_penalty_coef:
    low: -0.1
    medium: -0.5
    high: -1.0
  unused_penalty_coef: -0.5
  cheap_energy_missed_penalty_coef: -10

training:
  episode_length: 100

data:
  load_file: "mock_load.csv"
  solar_file: "mock_solar.csv"
  wind_file: "mock_wind.csv"
"""
    
    @pytest.fixture
def mock_config():
    """Testler için YAML konfigürasyonunu yükler."""
    return yaml.safe_load(CONFIG_YAML)
    
    @pytest.fixture
def mock_data_handler(tmp_path):
    """Geçici CSV dosyaları ile sahte bir DataHandler oluşturur."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq='h'))
    p = tmp_path
    load_path = p / "mock_load.csv"
    solar_path = p / "mock_solar.csv"
    wind_path = p / "mock_wind.csv"

    # Mock data oluştururken testin ihtiyaç duyduğu sütunları ekleyelim
    pd.DataFrame({'datetime': dates, 'load_kw': 100, 'price': 0.2, 'price_category': 'medium'}).to_csv(load_path, index=False)
    pd.DataFrame({'datetime': dates, 'solar_power_kW': 50}).to_csv(solar_path, index=False)
    pd.DataFrame({'datetime': dates, 'wind_power_kW': 20}).to_csv(wind_path, index=False)
    
    # Düzeltme: DataHandler doğru parametre ile başlatılıyor
    handler = DataHandler(data_dir=str(p))
    
    # DataHandler'ın dosya yollarını mock config'den almasını manuel olarak ayarlayalım
    mock_conf = yaml.safe_load(CONFIG_YAML)
    handler.load_file = mock_conf['data']['load_file']
    handler.solar_file = mock_conf['data']['solar_file']
    handler.wind_file = mock_conf['data']['wind_file']

    handler.load_all_data()
    return handler
    
    @pytest.fixture
def env(mock_data_handler, mock_config):
    """Testler için başlatılmış bir EnergyEnvironment örneği."""
    # Düzeltme: EnergyEnvironment'ı config SÖZLÜĞÜ ile başlatıyoruz
    environment = EnergyEnvironment(data_handler=mock_data_handler, config=mock_config)
    environment.reset()
    return environment

class TestUpdatedEnergyEnvironment:

    def test_initialization(self, env, mock_config):
        """Environment'ın doğru parametrelerle başladığını test eder."""
        assert env.battery_capacity == mock_config['environment']['battery']['capacity_kwh']
        assert env.max_battery_power == mock_config['environment']['battery']['max_power_kw']
        assert env.unmet_load_penalty == mock_config['reward']['unmet_load_penalty']

    def test_action_space_definition(self, env):
        """Aksiyon uzayının normalize edilmiş sürekli değerler için doğru tanımlandığını test eder."""
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, np.array([-1.0, -1.0]))
        np.testing.assert_array_equal(env.action_space.high, np.array([1.0, 1.0]))

    def test_step_logic_grid_connection(self, env):
        """Aksiyonun şebeke bağlantısını doğru kontrol ettiğini test eder."""
        # Action[0] > 0 -> grid_connection = 1
        obs, _, _, _, info = env.step(np.array([0.5, 0.0]))
        # `step` fonksiyonu artık info'da 'grid_connection' döndürmüyor, bu bilgiyi içsel olarak kullanıyor.
        # Bunun yerine, şebeke enerjisinin hesaplanıp hesaplanmadığını kontrol edelim.
        # Eğer bağlantı varsa ve gerekirse, grid_energy > 0 olmalı.
        # Bu test senaryosunda yenilenebilir (70) < yük (100) olduğu için şebeke gerekir.
        assert info['grid_energy'] > 0

    def test_step_logic_grid_disconnection(self, env):
        """Aksiyonun şebeke bağlantısını kestiğini test eder."""
        # Action[0] <= 0 -> grid_connection = 0
        obs, _, _, _, info = env.step(np.array([-0.5, 0.0]))
        # Şebeke bağlantısı kesikse, grid_energy her zaman 0 olmalı.
        assert info['grid_energy'] == 0
        # Bu durumda yük karşılanamadığı için (100 > 70), unmet_load olmalı.
        assert info['unmet_load'] > 0, "Yük karşılanamadığı için 'unmet_load' pozitif olmalıydı."

    def test_step_logic_battery_power(self, env):
        """Aksiyonun batarya gücünü doğru ölçeklediğini ve doğruladığını test eder."""
        # Action[1] = 1.0 -> Bataryanın tam güçle şarj olmasını bekle.
        # SOC limitleri (0.5) ve kapasiteye göre max şarj gücü hesaplanır.
        # max_charge_kwh = (0.9 - 0.5) * 1000 = 400
        # max_charge_power = 400 * 0.95 = 380
        # Bu, config'deki max_power_kw (500) değerinden düşük olduğu için, beklenen güç 380'dir.
        obs, _, _, _, info = env.step(np.array([0.0, 1.0]))
        assert np.isclose(info['battery_power'], 380.0), f"Batarya şarj gücü, SOC limitlerine göre ayarlanmalıydı."

    def test_safety_override_critical_soc(self, env):
        """SOC kritik seviyedeyken şebeke bağlantısının zorunlu kılındığını test eder."""
        env.battery_soc = env.min_soc - 0.01
        # .loc kullanarak pandas uyarısını giderelim
        env.episode_data.loc[env.current_step, 'load_kw'] = 1000 
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 0

        # Ajan şebekeyi kesmek istese bile (action[0] < 0)
        obs, _, _, _, info = env.step(np.array([-1.0, 0.0]))
        # Güvenlik önlemi devreye girmeli ve şebekeden enerji çekilmeli.
        assert info['grid_energy'] > 0, "Güvenlik önlemi devreye girip şebekeden enerji çekmeliydi."

    # --- YENİ ÖDÜL/CEZA TESTLERİ ---

    def test_penalty_unnecessary_grid_use(self, env):
        """Gereksiz şebeke kullanımı cezasını test eder (bu senaryoda ceza OLMAMALI)."""
        # Yükten çok daha fazla yenilenebilir enerji var
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 1000
        env.episode_data.loc[env.current_step, 'load_kw'] = 100
        
        # Ajan şebekeye bağlansa bile, `grid_energy` formülü 0 olacağı için ceza uygulanmaz.
        obs, reward, _, _, info = env.step(np.array([1.0, 0.0]))

        assert 'unnecessary_grid' not in info['reward_details'], "Bu senaryoda gereksiz şebeke cezası olmamalıydı."

    def test_penalty_soc_violation(self, env):
        """SOC ihlal cezasını test eder."""
        violation_amount = 0.1
        env.battery_soc = env.min_soc - violation_amount
        # Diğer tüm ceza kaynaklarını izole et
        env.episode_data.loc[env.current_step, 'load_kw'] = 0
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 0
        
        # Hatalı aksiyon: Deşarj etmeye çalış
        obs, reward, _, _, info = env.step(np.array([-1.0, -1.0]))
        
        expected_penalty = violation_amount * env.soc_penalty_coef
        assert 'soc_violation' in info['reward_details'], "SOC ihlal cezası uygulanmalıydı."
        assert np.isclose(info['reward_details']['soc_violation'], expected_penalty)

    def test_penalty_soc_violation_with_correction(self, env):
        """Hatayı düzelten SOC ihlal cezasının hafifletildiğini test eder."""
        violation_amount = 0.1
        env.battery_soc = env.min_soc - violation_amount
        # İzole et: Yük/Yenilenebilir sıfırla VE fiyatı 'high' yap
        # Bu, 'unnecessary_grid' cezasının tetiklenmesini imkansız hale getirir.
        env.episode_data.loc[env.current_step, 'load_kw'] = 0
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'price_category'] = 'high'
        
        # Doğru aksiyon: Şarj etmeye çalış
        obs, reward, _, _, info = env.step(np.array([1.0, 1.0])) # Şebekeden tam güçle şarj et
        
        expected_base_penalty = violation_amount * env.soc_penalty_coef
        expected_mitigated_penalty = expected_base_penalty * 0.5
        
        assert 'soc_violation' in info['reward_details'], "SOC ihlal cezası uygulanmalıydı."
        assert np.isclose(info['reward_details']['soc_violation'], expected_mitigated_penalty)

    def test_penalty_renewable_waste(self, env):
        """Yenilenebilir enerji israfı cezasını test eder."""
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 200
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 20
        env.episode_data.loc[env.current_step, 'load_kw'] = 100
        env.battery_soc = 0.8  # Batarya şarj için çok dolu değil
        
        # Hatalı aksiyon: Şarj etmek yerine deşarj et
        obs, reward, _, _, info = env.step(np.array([-1.0, -0.5]))

        excess_renewable = (200 + 20) - 100
        expected_penalty = excess_renewable * env.unused_penalty_coef

        assert 'renewable_waste' in info['reward_details'], "Yenilenebilir israf cezası uygulanmalıydı."
        assert np.isclose(info['reward_details']['renewable_waste'], expected_penalty)

    def test_penalty_missed_cheap_charge(self, env):
        """Ucuz şarj fırsatını kaçırma cezasını test eder."""
        env.episode_data.loc[env.current_step, 'price_category'] = 'low'
        env.battery_soc = 0.5
        # İzole et
        env.episode_data.loc[env.current_step, 'load_kw'] = 0
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 0
        
        # Hatalı aksiyon: Şarj etme (batarya idle)
        obs, reward, _, _, info = env.step(np.array([-1.0, 0.0]))

        soc_diff = env.max_soc - 0.5
        expected_penalty = soc_diff * env.cheap_energy_missed_penalty_coef
        
        assert 'missed_cheap_charge' in info['reward_details'], "Ucuz şarj fırsatını kaçırma cezası uygulanmalıydı."
        assert np.isclose(info['reward_details']['missed_cheap_charge'], expected_penalty)

    def test_no_penalties_on_good_action(self, env):
        """Doğru bir aksiyon alındığında hiçbir cezanın uygulanmadığını test eder."""
        env.episode_data.loc[env.current_step, 'load_kw'] = 400
        env.episode_data.loc[env.current_step, 'solar_power_kW'] = 500 # Yükten fazla
        env.episode_data.loc[env.current_step, 'wind_power_kW'] = 0
        env.episode_data.loc[env.current_step, 'price_category'] = 'high'
        env.battery_soc = 0.8
        
        # İyi aksiyon: Yenilenebilir fazlasını bataryaya şarj et
        obs, reward, _, _, info = env.step(np.array([-1.0, 1.0])) # Şebekesiz, şarj et
        
        assert not info['reward_details'], "İyi bir aksiyon için hiçbir ceza uygulanmamalıydı."

"""
NOT: Bu testler `gymnasium` ve `pandas` gibi kütüphaneleri gerektirir.
Testleri çalıştırmak için:
pytest tests/test_energy_environment.py
"""
