"""
🎯 EMT RL Project - Training Manager
PPO Agent eğitimi ve monitoring için ana sınıf
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_handler import DataHandler
from src.environment.energy_environment import EnergyEnvironment
from src.agents.ppo_agent import PPOAgent
from src.utils.cuda_utils import cuda_manager

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingManager:
    """
    PPO Agent eğitimi ve monitoring yönetimi
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Training Manager başlatma"""
        self.config_path = config_path
        self.training_start_time = None
        self.training_end_time = None
        
        # Components
        self.data_handler: Optional[DataHandler] = None
        self.environment: Optional[EnergyEnvironment] = None
        self.agent: Optional[PPOAgent] = None
        
        # Training state
        self.training_history = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.soc_violations = []
        self.renewable_usage = []
        self.current_episode = 0
        self.total_timesteps_trained = 0
        
        # Monitoring
        self.monitoring_data = {
            'timestamps': [],
            'rewards': [],
            'loss_values': [],
            'learning_rates': [],
            'gpu_memory': [],
            'episode_metrics': []
        }
        
        # Paths
        self.results_dir = "results/"
        self.models_dir = "models/"
        self.logs_dir = "logs/"
        self._ensure_directories()
        
        logger.info("🎯 TrainingManager başlatıldı")
        
        self.config = self._load_config(config_path)
        self.data_handler = DataHandler(config_path=config_path)
        self.env = None
        self.agent = None
        self.log_dir = None
        self.model_dir = None
        self.results_dir = None
        self.cuda_manager = cuda_manager
    
    def _ensure_directories(self):
        """Gerekli dizinleri oluştur"""
        for directory in [self.results_dir, self.models_dir, self.logs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"📁 Dizin oluşturuldu: {directory}")
    
    def _load_config(self, config_path: str) -> Dict:
        """YAML config dosyasını yükle"""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def setup_training(self, model_name: Optional[str] = None, description: Optional[str] = None) -> bool:
        """
        Eğitim için gerekli tüm bileşenleri (environment, agent, dizinler) ayarlar.

        Args:
            model_name (str, optional): Model ve log dosyaları için özel isim.
            description (str, optional): Eğitim oturumu için açıklama.

        Returns:
            bool: Kurulum başarılı ise True, değilse False.
        """
        try:
            # Model adı ve yolları belirle
            if not model_name:
                model_name = f'PPO_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            self.log_dir = os.path.join("logs", model_name)
            self.model_dir = os.path.join("results", "models", model_name)
            self.results_dir = os.path.join("results", "plots", model_name)
            
            # Gerekli dizinleri oluştur
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)

            logger.info("📂 Gerekli dizinler oluşturuldu (logs, results)")

            # Environment'ı oluştur
            self.env = self._create_environment()
            
            # Agent'ı oluştur
            self.agent = PPOAgent(
                env=self.env,
                log_dir=self.log_dir,
                model_save_path=self.model_dir,
                config=self.config,
                use_cuda=self.cuda_manager.is_cuda_available()
            )
            
            logger.info(f"🤖 PPO Agent oluşturuldu - Model Adı: {model_name}")

            # Eğitim özetini kaydet
            self._save_training_metadata(model_name, description)
            
            return True
        except Exception as e:
            logger.error(f"❌ Eğitim kurulumu başarısız: {e}", exc_info=True)
            return False
    
    def _save_training_metadata(self, model_name: str, description: Optional[str]):
        """Eğitim oturumunun meta verilerini kaydeder."""
        metadata = {
            'model_name': model_name,
            'description': description,
            'training_start_time': datetime.now().isoformat(),
            'config': self.config
        }
        meta_path = os.path.join(self.log_dir, 'metadata.yaml')
        with open(meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"📝 Eğitim meta verileri şuraya kaydedildi: {meta_path}")

    def _create_environment(self) -> EnergyEnvironment:
        """EnergyEnvironment örneği oluşturur."""
        if self.data_handler is None:
            logger.error("❌ DataHandler setup edilmemiş!")
            raise ValueError("❌ DataHandler setup edilmemiş!")
        
        logger.info("🏗️ Environment oluşturuluyor...")
        self.environment = EnergyEnvironment(self.data_handler, self.config_path)
        logger.info(f"✅ Environment hazır - State: {self.environment.observation_space.shape}")
        return self.environment
    
    def train(self, 
              total_timesteps: int = 50000,
              save_freq: int = 10000,
              eval_freq: int = 5000,
              eval_episodes: int = 5,
              progress_callback: Optional[Callable] = None) -> Dict:
        """
        Ana eğitim fonksiyonu
        
        Args:
            total_timesteps: Toplam eğitim adımı
            save_freq: Model kaydetme sıklığı
            eval_freq: Evaluation sıklığı
            eval_episodes: Evaluation episode sayısı
            progress_callback: Progress callback fonksiyonu
            
        Returns:
            Dict: Eğitim sonuçları
        """
        if not self.agent:
            raise ValueError("❌ Agent setup edilmemiş! setup_training() çalıştırın.")
        
        try:
            logger.info(f"🚀 Training başlıyor - {total_timesteps:,} timesteps")
            self.training_start_time = datetime.now()
            
            # GPU memory başlangıç durumu
            if self.cuda_manager.is_cuda_available():
                initial_memory = self.cuda_manager.get_memory_stats()
                logger.info(f"🔥 GPU Memory: {initial_memory.get('usage_percent', 0):.1f}%")
            
            # Training callbacks oluştur
            callbacks = self._create_training_callbacks(
                save_freq=save_freq,
                eval_freq=eval_freq,
                eval_episodes=eval_episodes,
                progress_callback=progress_callback
            )
            
            # Ana eğitim
            training_results = self.agent.train(total_timesteps)
            
            self.training_end_time = datetime.now()
            training_duration = (self.training_end_time - self.training_start_time).total_seconds()
            
            # Sonuçları topla
            results = {
                'total_timesteps': total_timesteps,
                'training_duration_seconds': training_duration,
                'training_duration_minutes': training_duration / 60,
                'steps_per_second': total_timesteps / training_duration,
                'device': str(self.agent.device),
                'training_start': self.training_start_time.isoformat(),
                'training_end': self.training_end_time.isoformat()
            }
            
            # GPU memory son durum
            if self.cuda_manager.is_cuda_available():
                final_memory = self.cuda_manager.get_memory_stats()
                results['gpu_memory_final'] = final_memory.get('usage_percent', 0)
            
            # Training history'ye ekle
            self.training_history.append(results)
            self.total_timesteps_trained += total_timesteps
            
            # Sonuçları kaydet
            self._save_training_results(results)
            
            logger.info(f"✅ Training tamamlandı!")
            logger.info(f"📊 Süre: {training_duration/60:.1f} dakika")
            logger.info(f"⚡ Hız: {results['steps_per_second']:.1f} steps/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training hatası: {e}")
            raise
    
    def _create_training_callbacks(self, save_freq: int, eval_freq: int, 
                                 eval_episodes: int, progress_callback: Optional[Callable]) -> List:
        """Training callbacks oluştur"""
        callbacks = []
        
        # Progress monitoring callback
        if progress_callback:
            callbacks.append(progress_callback)
        
        # Bu basit implementasyonda callbacks'ler direkt agent.train() içinde
        # Gelişmiş callback sistemi için stable-baselines3 callbacks kullanılabilir
        
        return callbacks
    
    def evaluate_model(self, n_episodes: int = 10, save_results: bool = True) -> Dict:
        """
        Model değerlendirmesi
        
        Args:
            n_episodes: Değerlendirme episode sayısı
            save_results: Sonuçları kaydet
            
        Returns:
            Dict: Değerlendirme sonuçları
        """
        if not self.agent:
            raise ValueError("❌ Agent setup edilmemiş!")
        
        logger.info(f"📊 Model evaluation başlıyor - {n_episodes} episodes")
        
        # Agent evaluation
        eval_results = self.agent.evaluate(n_episodes=n_episodes, deterministic=True)
        
        # Ek metrikler
        eval_results['evaluation_timestamp'] = datetime.now().isoformat()
        eval_results['total_timesteps_trained'] = self.total_timesteps_trained
        
        if save_results:
            self._save_evaluation_results(eval_results)
        
        logger.info(f"✅ Evaluation tamamlandı!")
        logger.info(f"📈 Mean reward: {eval_results['mean_reward']:.2f}")
        logger.info(f"📏 Mean length: {eval_results['mean_length']:.0f}")
        
        return eval_results
    
    def create_training_visualization(self, save_plots: bool = True) -> Dict[str, str]:
        """
        Training görselleştirme
        
        Args:
            save_plots: Grafikleri kaydet
            
        Returns:
            Dict: Kaydedilen grafik dosyaları
        """
        logger.info("📊 Training visualization oluşturuluyor...")
        
        if not self.training_history:
            logger.warning("⚠️ Training history bulunamadı")
            return {}
        
        saved_plots = {}
        
        try:
            # Training progress plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('🎯 EMT RL Training Progress', fontsize=16, fontweight='bold')
            
            # Training duration plot
            durations = [h['training_duration_minutes'] for h in self.training_history]
            timesteps = [h['total_timesteps'] for h in self.training_history]
            
            axes[0, 0].plot(timesteps, durations, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_title('⏱️ Training Duration')
            axes[0, 0].set_xlabel('Total Timesteps')
            axes[0, 0].set_ylabel('Duration (minutes)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Steps per second plot
            speeds = [h['steps_per_second'] for h in self.training_history]
            axes[0, 1].plot(timesteps, speeds, 'g-o', linewidth=2, markersize=6)
            axes[0, 1].set_title('⚡ Training Speed')
            axes[0, 1].set_xlabel('Total Timesteps')
            axes[0, 1].set_ylabel('Steps/Second')
            axes[0, 1].grid(True, alpha=0.3)
            
            # GPU memory usage (if available)
            if any('gpu_memory_final' in h for h in self.training_history):
                gpu_memory = [h.get('gpu_memory_final', 0) for h in self.training_history]
                axes[1, 0].plot(timesteps, gpu_memory, 'r-o', linewidth=2, markersize=6)
                axes[1, 0].set_title('🔥 GPU Memory Usage')
                axes[1, 0].set_xlabel('Total Timesteps')
                axes[1, 0].set_ylabel('Memory Usage (%)')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'GPU data not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('🔥 GPU Memory Usage')
            
            # Device info
            devices = [h.get('device', 'unknown') for h in self.training_history]
            device_counts = pd.Series(devices).value_counts()
            
            axes[1, 1].pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('💻 Device Usage')
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = os.path.join(self.results_dir, f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots['training_progress'] = plot_path
                logger.info(f"📊 Training progress plot kaydedildi: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Visualization hatası: {e}")
        
        return saved_plots
    
    def _save_training_results(self, results: Dict):
        """Training sonuçlarını kaydet"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(self.results_dir, f"training_results_{timestamp}.json")
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Training sonuçları kaydedildi: {results_path}")
            
        except Exception as e:
            logger.error(f"❌ Results kaydetme hatası: {e}")
    
    def _save_evaluation_results(self, results: Dict):
        """Evaluation sonuçlarını kaydet"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
            
            # NumPy arrays'i ve lists'i güvenli şekilde çevir
            results_copy = results.copy()
            
            # episode_rewards listini işle
            if 'episode_rewards' in results_copy:
                if hasattr(results_copy['episode_rewards'], 'tolist'):
                    results_copy['episode_rewards'] = results_copy['episode_rewards'].tolist()
                elif isinstance(results_copy['episode_rewards'], list):
                    # Zaten list ise dokunma
                    pass
                else:
                    # Diğer durumlar için list'e çevir
                    results_copy['episode_rewards'] = list(results_copy['episode_rewards'])
            
            # Tüm NumPy array'leri kontrol et ve çevir
            for key, value in results_copy.items():
                if hasattr(value, 'tolist'):
                    results_copy[key] = value.tolist()
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'tolist'):
                            results_copy[key][sub_key] = sub_value.tolist()
            
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Evaluation sonuçları kaydedildi: {eval_path}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation kaydetme hatası: {e}")
    
    def get_training_summary(self) -> Dict:
        """Training özeti"""
        if not self.training_history:
            return {'status': 'No training completed'}
        
        total_duration = sum(h['training_duration_seconds'] for h in self.training_history)
        total_timesteps = sum(h['total_timesteps'] for h in self.training_history)
        avg_speed = total_timesteps / total_duration if total_duration > 0 else 0
        
        summary = {
            'total_training_sessions': len(self.training_history),
            'total_timesteps': total_timesteps,
            'total_duration_minutes': total_duration / 60,
            'average_speed_steps_per_second': avg_speed,
            'training_start': self.training_history[0]['training_start'] if self.training_history else None,
            'latest_training': self.training_history[-1]['training_end'] if self.training_history else None,
            'device_used': self.training_history[-1].get('device', 'unknown') if self.training_history else 'unknown'
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup ve memory temizleme"""
        logger.info("🧹 Cleanup başlıyor...")
        
        # GPU cache temizle
        if self.agent and self.agent.use_cuda:
            self.agent.clear_gpu_cache()
        
        # Memory references temizle
        self.monitoring_data.clear()
        
        logger.info("✅ Cleanup tamamlandı") 