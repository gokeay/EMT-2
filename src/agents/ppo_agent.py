"""
🤖 EMT RL Project - PPO Agent
Stable-Baselines3 tabanlı PPO agent implementasyonu
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cuda_utils import get_device, is_cuda_available, cuda_manager

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Energy Management için PPO Agent
    CUDA destekli, Stable-Baselines3 tabanlı
    """
    
    def __init__(self, environment, config_path: str = "configs/config.yaml", model_save_path: str = "models", log_dir: str = "logs"):
        """PPO Agent başlatma"""
        self.env = environment
        self.config = self._load_config(config_path)
        self.training_config = self.config['training']
        self.log_dir = log_dir
        
        # Device management
        self.device = get_device()
        self.use_cuda = is_cuda_available()
        
        # Model parameters
        self.model: Optional[PPO] = None
        self.model_path = model_save_path
        self._ensure_model_directory()
        
        # Training state
        self.total_timesteps = 0
        self.training_history = []
        
        logger.info(f"🤖 PPOAgent başlatıldı - Device: {self.device}")
        if self.use_cuda:
            logger.info(f"🔥 CUDA enabled - GPU: {cuda_manager.get_device_name()}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"❌ Config yüklenemedi: {e}")
            raise
    
    def _ensure_model_directory(self):
        """Model klasörünü oluştur"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            logger.info(f"📁 Model dizini oluşturuldu: {self.model_path}")
    
    def create_model(self, policy: str = "MlpPolicy", **kwargs) -> PPO:
        """PPO model oluştur"""
        try:
            # Environment'ı vectorize et
            vec_env = DummyVecEnv([lambda: Monitor(self.env)])
            
            # Default PPO parametreleri
            ppo_params = {
                'learning_rate': self.training_config.get('learning_rate', 3e-4),
                'n_steps': 2048,
                'batch_size': self.training_config.get('batch_size', 64),
                'n_epochs': 10,
                'gamma': self.training_config.get('gamma', 0.99),
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': self.training_config.get('exploration', {}).get('entropy_coef', 0.01),
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'device': self.device,
                'verbose': 1,
                'tensorboard_log': self.log_dir
            }
            
            # Custom parametreleri merge et
            ppo_params.update(kwargs)
            
            # Model oluştur
            self.model = PPO(
                policy=policy,
                env=vec_env,
                **ppo_params
            )
            
            logger.info(f"✅ PPO model oluşturuldu - Policy: {policy}")
            logger.info(f"📊 Model parametreleri: lr={ppo_params['learning_rate']}, "
                       f"batch_size={ppo_params['batch_size']}, ent_coef={ppo_params['ent_coef']}, device={self.device}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Model oluşturma hatası: {e}")
            raise
    
    def train(self, total_timesteps: int) -> Dict:
        """Model eğitimi"""
        if self.model is None:
            raise ValueError("❌ Model henüz oluşturulmadı! create_model() kullanın.")
        
        try:
            logger.info(f"🎯 PPO eğitimi başladı - {total_timesteps:,} timesteps")
            
            # Memory monitoring (CUDA varsa)
            if self.use_cuda:
                initial_memory = cuda_manager.get_memory_stats()
                logger.info(f"🔥 Initial GPU memory: {initial_memory.get('usage_percent', 0):.1f}%")
            
            # Eğitimi başlat
            start_time = datetime.now()
            
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # Final model kaydet
            final_model_path = os.path.join(self.model_path, "model.zip")
            self.save_model(final_model_path)
            
            # Eğitim sonuçları
            results = {
                'total_timesteps': total_timesteps,
                'training_duration_seconds': training_duration,
                'training_duration_minutes': training_duration / 60,
                'steps_per_second': total_timesteps / training_duration,
                'final_model_path': final_model_path,
                'device': str(self.device)
            }
            
            # GPU memory usage (if CUDA)
            if self.use_cuda:
                final_memory = cuda_manager.get_memory_stats()
                results['gpu_memory_final'] = final_memory.get('usage_percent', 0)
                results['gpu_memory_peak'] = final_memory.get('allocated_mb', 0)
            
            # Training history'ye ekle
            self.training_history.append(results)
            self.total_timesteps += total_timesteps
            
            logger.info(f"✅ Eğitim tamamlandı!")
            logger.info(f"📊 Süre: {training_duration/60:.1f} dakika")
            logger.info(f"⚡ Hız: {results['steps_per_second']:.1f} steps/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Eğitim hatası: {e}")
            raise
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Action prediction"""
        if self.model is None:
            raise ValueError("❌ Model henüz oluşturulmadı veya yüklenmedi!")
        
        try:
            action, state = self.model.predict(observation, deterministic=deterministic)
            return action, state
        except Exception as e:
            logger.error(f"❌ Prediction hatası: {e}")
            raise
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict:
        """Model değerlendirmesi"""
        if self.model is None:
            raise ValueError("❌ Model henüz oluşturulmadı veya yüklenmedi!")
        
        try:
            logger.info(f"📊 Model değerlendirmesi başladı - {n_episodes} episodes")
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(n_episodes):
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Sonuçları hesapla
            results = {
                'n_episodes': n_episodes,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'episode_rewards': episode_rewards
            }
            
            logger.info(f"✅ Değerlendirme tamamlandı!")
            logger.info(f"📊 Ortalama Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Değerlendirme hatası: {e}")
            raise
    
    def save_model(self, filepath: str):
        """Modeli belirtilen yola kaydet"""
        if self.model is None:
            raise ValueError("❌ Kaydedilecek model yok!")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            logger.info(f"💾 Model kaydedildi: {filepath}")
        except Exception as e:
            logger.error(f"❌ Model kaydetme hatası: {e}")
            raise
    
    def load_model(self, filepath: str, env=None):
        """Model yükle"""
        try:
            if env is None:
                env = DummyVecEnv([lambda: Monitor(self.env)])
            
            self.model = PPO.load(filepath, env=env, device=self.device)
            logger.info(f"📂 Model yüklendi: {filepath}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        if self.model is None:
            return {'status': 'Model not created'}
        
        info = {
            'policy_class': str(type(self.model.policy)),
            'device': str(self.model.device),
            'total_timesteps_trained': self.total_timesteps,
            'learning_rate': self.model.learning_rate,
            'batch_size': self.model.batch_size,
            'gamma': self.model.gamma,
            'training_history_count': len(self.training_history)
        }
        
        if self.use_cuda:
            memory_stats = cuda_manager.get_memory_stats()
            info['gpu_memory_usage'] = memory_stats.get('usage_percent', 0)
        
        return info
    
    def clear_gpu_cache(self):
        """GPU cache temizle"""
        if self.use_cuda:
            cuda_manager.clear_cache()
        else:
            logger.warning("⚠️ CUDA available değil") 