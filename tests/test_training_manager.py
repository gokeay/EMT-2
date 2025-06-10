"""
🧪 EMT RL Project - Training Manager Tests
TrainingManager sınıfı için unit testler
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch
from datetime import datetime

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import TrainingManager


class TestTrainingManager(unittest.TestCase):
    """TrainingManager test sınıfı"""
    
    def setUp(self):
        """Test setup"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = "configs/config.yaml"
        
        # Mock trainer oluştur
        with patch('src.training.trainer.cuda_manager'):
            self.trainer = TrainingManager(self.config_path)
            
        # Test paths'leri test dizini içine al
        self.trainer.results_dir = os.path.join(self.test_dir, "results/")
        self.trainer.models_dir = os.path.join(self.test_dir, "models/")
        self.trainer.logs_dir = os.path.join(self.test_dir, "logs/")
        
        # Dizinleri oluştur
        self.trainer._ensure_directories()
    
    def tearDown(self):
        """Test cleanup"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Initialization testi"""
        self.assertEqual(self.trainer.config_path, self.config_path)
        self.assertIsNone(self.trainer.training_start_time)
        self.assertEqual(len(self.trainer.training_history), 0)
        self.assertEqual(self.trainer.total_timesteps_trained, 0)
        
        # Dizinler oluşturuldu mu?
        self.assertTrue(os.path.exists(self.trainer.results_dir))
        self.assertTrue(os.path.exists(self.trainer.models_dir))
        self.assertTrue(os.path.exists(self.trainer.logs_dir))
    
    @patch('src.training.trainer.DataHandler')
    @patch('src.training.trainer.EnergyEnvironment')
    @patch('src.training.trainer.PPOAgent')
    def test_setup_training_success(self, mock_ppo, mock_env, mock_data):
        """Başarılı training setup testi"""
        # Mock objects setup
        mock_data_instance = Mock()
        mock_data_instance.combined_data = [{"test": "data"}]
        mock_data.return_value = mock_data_instance
        
        mock_env_instance = Mock()
        mock_env_instance.observation_space.shape = (10,)
        mock_env.return_value = mock_env_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.device = "cpu"
        mock_ppo.return_value = mock_agent_instance
        
        # Setup çalıştır
        result = self.trainer.setup_training()
        
        # Sonuçları kontrol et
        self.assertTrue(result)
        self.assertIsNotNone(self.trainer.data_handler)
        self.assertIsNotNone(self.trainer.environment)
        self.assertIsNotNone(self.trainer.agent)
    
    def test_create_mock_data(self):
        """Mock data oluşturma testi"""
        mock_data_handler = Mock()
        self.trainer.data_handler = mock_data_handler
        
        self.trainer._create_mock_data()
        
        mock_data = mock_data_handler.combined_data
        self.assertEqual(len(mock_data), 8760)  # 1 yıl
        
        required_columns = ['datetime', 'load_kw', 'solar_power_kW', 'wind_power_kW', 'price_category']
        for col in required_columns:
            self.assertIn(col, mock_data.columns)
    
    @patch('src.training.trainer.datetime')
    def test_train_success(self, mock_datetime):
        """Başarılı training testi"""
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 30, 0)
        
        mock_datetime.now.side_effect = [start_time, end_time]
        mock_datetime.side_effect = lambda *args: datetime(*args)
        
        mock_agent = Mock()
        mock_agent.device = "cpu"
        mock_agent.train.return_value = {"status": "success"}
        self.trainer.agent = mock_agent
        
        timesteps = 1000
        result = self.trainer.train(total_timesteps=timesteps)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_timesteps'], timesteps)
        self.assertEqual(result['training_duration_minutes'], 30.0)
        self.assertEqual(len(self.trainer.training_history), 1)
    
    def test_train_no_agent(self):
        """Agent olmadan training testi"""
        self.trainer.agent = None
        
        with self.assertRaises(ValueError):
            self.trainer.train()
    
    def test_evaluate_model_success(self):
        """Başarılı evaluation testi"""
        mock_agent = Mock()
        eval_results = {
            'mean_reward': 100.5,
            'std_reward': 10.2,
            'mean_length': 250
        }
        mock_agent.evaluate.return_value = eval_results
        self.trainer.agent = mock_agent
        
        result = self.trainer.evaluate_model(n_episodes=5, save_results=False)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['mean_reward'], 100.5)
        self.assertIn('evaluation_timestamp', result)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('pandas.Series')
    def test_create_training_visualization(self, mock_series, mock_close, mock_subplots, mock_savefig):
        """Training visualization testi"""
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Mock pandas Series for device counts
        mock_series_instance = Mock()
        mock_series_instance.value_counts.return_value = Mock()
        mock_series_instance.value_counts.return_value.values = [1]
        mock_series_instance.value_counts.return_value.index = ['cpu']
        mock_series.return_value = mock_series_instance
        
        self.trainer.training_history = [
            {
                'total_timesteps': 1000,
                'training_duration_minutes': 10.0,
                'steps_per_second': 100.0,
                'device': 'cpu'
            }
        ]
        
        result = self.trainer.create_training_visualization(save_plots=True)
        
        self.assertIsInstance(result, dict)
        mock_subplots.assert_called_once()
    
    def test_get_training_summary(self):
        """Training summary testi"""
        self.trainer.training_history = [
            {
                'total_timesteps': 1000,
                'training_duration_seconds': 600,
                'device': 'cpu',
                'training_start': '2024-01-01T10:00:00',
                'training_end': '2024-01-01T10:10:00'
            }
        ]
        
        summary = self.trainer.get_training_summary()
        
        self.assertEqual(summary['total_training_sessions'], 1)
        self.assertEqual(summary['total_timesteps'], 1000)
        self.assertEqual(summary['total_duration_minutes'], 10.0)


if __name__ == '__main__':
    unittest.main() 