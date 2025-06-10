"""
🧪 EMT RL Project - PPO Agent Tests
PPOAgent sınıfı için unit testler
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import logging

# Test ortamı için logging'i sessiz yap
logging.disable(logging.CRITICAL)

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ppo_agent import PPOAgent


class TestPPOAgent(unittest.TestCase):
    """PPOAgent sınıfı için test sınıfı"""
    
    def setUp(self):
        """Test setup - Her test öncesi çalışır"""
        # Mock environment oluştur
        self.mock_env = MagicMock()
        self.mock_env.observation_space = MagicMock()
        self.mock_env.action_space = MagicMock()
        self.mock_env.observation_space.shape = (7,)
        self.mock_env.action_space.shape = (2,)
        
        # Mock config
        self.mock_config = {
            'training': {
                'learning_rate': 3e-4,
                'batch_size': 64,
                'gamma': 0.99
            },
            'monitoring': {
                'tensorboard_log': './logs'
            }
        }
        
        # Temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Test cleanup - Her test sonrası çalışır"""
        # Temporary files cleanup
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_init_basic(self, mock_open, mock_yaml):
        """PPOAgent temel başlatma testi"""
        # Mock config yükleme
        mock_yaml.return_value = self.mock_config
        
        # Agent oluştur
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Assertions
        self.assertEqual(agent.env, self.mock_env)
        self.assertEqual(agent.total_timesteps, 0)
        self.assertEqual(len(agent.training_history), 0)
        self.assertIsNone(agent.model)
        self.assertEqual(agent.model_path, "models/")
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_device_detection(self, mock_open, mock_yaml):
        """Device detection testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Device test
        self.assertIsNotNone(agent.device)
        self.assertIsInstance(agent.use_cuda, bool)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_config_loading_error(self, mock_open, mock_yaml):
        """Config yükleme hatası testi"""
        # YAML hatası simülasyonu
        mock_yaml.side_effect = Exception("Config error")
        
        # Hata beklenir
        with self.assertRaises(Exception):
            PPOAgent(self.mock_env, "dummy_config.yaml")
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    def test_create_model_success(self, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Model oluşturma başarı testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        result = agent.create_model()
        
        # Assertions
        self.assertEqual(result, mock_model)
        self.assertEqual(agent.model, mock_model)
        mock_ppo.assert_called_once()
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    def test_create_model_error(self, mock_ppo, mock_open, mock_yaml):
        """Model oluşturma hata testi"""
        mock_yaml.return_value = self.mock_config
        mock_ppo.side_effect = Exception("PPO error")
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        with self.assertRaises(Exception):
            agent.create_model()
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_train_without_model(self, mock_open, mock_yaml):
        """Model olmadan eğitim testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Model yokken eğitim hatası beklenir
        with self.assertRaises(ValueError):
            agent.train(1000)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    @patch('os.makedirs')
    def test_train_success(self, mock_makedirs, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Başarılı eğitim testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.create_model()
        
        # Mock datetime import
        from datetime import datetime
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 1, 40)  # 100 saniye sonra
        
        # Mock eğitim
        with patch('src.agents.ppo_agent.datetime') as mock_datetime:
            mock_datetime.now.side_effect = [start_time, end_time]
            
            result = agent.train(1000)
            
            # Assertions
            self.assertIsInstance(result, dict)
            self.assertEqual(result['total_timesteps'], 1000)
            self.assertEqual(agent.total_timesteps, 1000)
            self.assertEqual(len(agent.training_history), 1)
            mock_model.learn.assert_called_once_with(
                total_timesteps=1000,
                progress_bar=True,
                reset_num_timesteps=False
            )
            mock_model.save.assert_called()
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_predict_without_model(self, mock_open, mock_yaml):
        """Model olmadan prediction testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        observation = np.array([1, 2, 3, 4, 5, 6, 7])
        
        # Model yokken prediction hatası beklenir
        with self.assertRaises(ValueError):
            agent.predict(observation)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    def test_predict_success(self, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Başarılı prediction testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.5, -1000]), None)
        mock_ppo.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.create_model()
        
        observation = np.array([1, 2, 3, 4, 5, 6, 7])
        action, state = agent.predict(observation)
        
        # Assertions
        self.assertIsInstance(action, np.ndarray)
        mock_model.predict.assert_called_once_with(observation, deterministic=True)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_evaluate_without_model(self, mock_open, mock_yaml):
        """Model olmadan evaluation testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Model yokken evaluation hatası beklenir
        with self.assertRaises(ValueError):
            agent.evaluate()
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    def test_evaluate_success(self, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Başarılı evaluation testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        # Mock environment behavior
        self.mock_env.reset.return_value = (np.array([1, 2, 3, 4, 5, 6, 7]), {})
        self.mock_env.step.side_effect = [
            (np.array([1, 2, 3, 4, 5, 6, 7]), 10, True, False, {}),  # Episode 1 ends
            (np.array([1, 2, 3, 4, 5, 6, 7]), 20, True, False, {}),  # Episode 2 ends
        ]
        
        mock_model.predict.return_value = (np.array([0.5, -1000]), None)
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.create_model()
        
        result = agent.evaluate(n_episodes=2)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result['n_episodes'], 2)
        self.assertIn('mean_reward', result)
        self.assertIn('episode_rewards', result)
        self.assertEqual(len(result['episode_rewards']), 2)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_save_model_without_model(self, mock_open, mock_yaml):
        """Model olmadan kaydetme testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Model yokken kaydetme hatası beklenir
        with self.assertRaises(ValueError):
            agent.save_model("test_model")
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    def test_save_model_success(self, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Başarılı model kaydetme testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.create_model()
        
        agent.save_model("test_model")
        
        # Assertions
        mock_model.save.assert_called_with("test_model")
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    @patch('src.agents.ppo_agent.Monitor')
    def test_load_model_success(self, mock_monitor, mock_vec_env, mock_ppo_class, mock_open, mock_yaml):
        """Başarılı model yükleme testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_ppo_class.load.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.load_model("test_model")
        
        # Assertions
        self.assertEqual(agent.model, mock_model)
        mock_ppo_class.load.assert_called()
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    def test_get_model_info_no_model(self, mock_open, mock_yaml):
        """Model olmadan info testi"""
        mock_yaml.return_value = self.mock_config
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        info = agent.get_model_info()
        
        # Assertions
        self.assertEqual(info['status'], 'Model not created')
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.PPO')
    @patch('src.agents.ppo_agent.DummyVecEnv')
    def test_get_model_info_with_model(self, mock_vec_env, mock_ppo, mock_open, mock_yaml):
        """Model ile info testi"""
        mock_yaml.return_value = self.mock_config
        mock_model = MagicMock()
        mock_model.policy = MagicMock()
        mock_model.device = "cpu"
        mock_model.learning_rate = 3e-4
        mock_model.batch_size = 64
        mock_model.gamma = 0.99
        mock_ppo.return_value = mock_model
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        agent.create_model()
        
        info = agent.get_model_info()
        
        # Assertions
        self.assertIn('policy_class', info)
        self.assertIn('device', info)
        self.assertIn('learning_rate', info)
        self.assertEqual(info['total_timesteps_trained'], 0)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.is_cuda_available')
    def test_clear_gpu_cache_no_cuda(self, mock_cuda_available, mock_open, mock_yaml):
        """CUDA olmadan cache temizleme testi"""
        mock_yaml.return_value = self.mock_config
        mock_cuda_available.return_value = False
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Test - Hata çıkmamalı
        agent.clear_gpu_cache()
        
        # Bu başarılı olmalı (uyarı ile)
        self.assertTrue(True)
    
    @patch('src.agents.ppo_agent.yaml.safe_load')
    @patch('builtins.open')
    @patch('src.agents.ppo_agent.is_cuda_available')
    @patch('src.agents.ppo_agent.cuda_manager')
    def test_clear_gpu_cache_with_cuda(self, mock_cuda_manager, mock_cuda_available, mock_open, mock_yaml):
        """CUDA ile cache temizleme testi"""
        mock_yaml.return_value = self.mock_config
        mock_cuda_available.return_value = True
        
        agent = PPOAgent(self.mock_env, "dummy_config.yaml")
        
        # Test - CUDA cache temizlenmeli
        agent.clear_gpu_cache()
        
        # cuda_manager.clear_cache() çağrılmalı
        mock_cuda_manager.clear_cache.assert_called_once()


if __name__ == '__main__':
    # Test suite oluştur ve çalıştır
    unittest.main(verbosity=2) 