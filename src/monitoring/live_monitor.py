"""
📊 EMT RL Project - Live Monitor
Real-time training monitoring ve visualization
"""

import os
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.cuda_utils import cuda_manager

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveMonitor:
    """Real-time training monitoring sınıfı"""
    
    def __init__(self, update_interval: float = 5.0, max_data_points: int = 1000):
        """Live Monitor başlatma"""
        self.update_interval = update_interval
        self.max_data_points = max_data_points
        
        # Monitoring data
        self.monitoring_data = {
            'timestamps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'steps_per_second': []
        }
        
        # Threading
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitoring_lock = threading.Lock()
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        
        logger.info("📊 LiveMonitor başlatıldı")
    
    def add_data_point(self, data: Dict):
        """Yeni veri noktası ekle"""
        with self.monitoring_lock:
            timestamp = datetime.now()
            self.monitoring_data['timestamps'].append(timestamp)
            
            # Veri noktalarını ekle
            for key, value in data.items():
                if key in self.monitoring_data:
                    self.monitoring_data[key].append(value)
            
            # Maksimum veri noktası sınırı
            self._trim_data()
            
            # Callbacks'leri çağır
            for callback in self.data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"❌ Callback hatası: {e}")
    
    def _trim_data(self):
        """Veri boyutunu sınırla"""
        if len(self.monitoring_data['timestamps']) > self.max_data_points:
            # Her listeden eski veriyi sil
            for key in self.monitoring_data:
                if isinstance(self.monitoring_data[key], list):
                    self.monitoring_data[key] = self.monitoring_data[key][-self.max_data_points:]
    
    def start_monitoring(self, target_function: Optional[Callable] = None):
        """Live monitoring başlat"""
        if self.is_monitoring:
            logger.warning("⚠️ Monitoring zaten aktif")
            return
        
        self.is_monitoring = True
        
        if target_function:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(target_function,),
                daemon=True
            )
        else:
            self.monitoring_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
        
        self.monitoring_thread.start()
        logger.info("🚀 Live monitoring başladı")
    
    def stop_monitoring(self):
        """Live monitoring durdur"""
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("⏹️ Live monitoring durduruldu")
    
    def _monitoring_loop(self, target_function: Callable):
        """Ana monitoring döngüsü"""
        while self.is_monitoring:
            try:
                # Target function'dan veri al
                data = target_function()
                if data:
                    self.add_data_point(data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop hatası: {e}")
                time.sleep(self.update_interval)
    
    def _system_monitoring_loop(self):
        """Sistem monitoring döngüsü"""
        while self.is_monitoring:
            try:
                # Sistem metriklerini topla
                system_data = self._collect_system_metrics()
                self.add_data_point(system_data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ System monitoring hatası: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> Dict:
        """Sistem metriklerini topla"""
        metrics = {}
        
        try:
            # GPU memory
            if cuda_manager.is_cuda_available():
                gpu_stats = cuda_manager.get_memory_stats()
                metrics['gpu_memory'] = gpu_stats.get('usage_percent', 0)
            else:
                metrics['gpu_memory'] = 0
            
            # CPU usage (basit implementasyon)
            try:
                import psutil
                metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            except ImportError:
                metrics['cpu_usage'] = 0
            
        except Exception as e:
            logger.warning(f"⚠️ System metrics hatası: {e}")
            metrics['gpu_memory'] = 0
            metrics['cpu_usage'] = 0
        
        return metrics
    
    def create_live_plot(self, save_path: Optional[str] = None) -> str:
        """Live plot oluştur"""
        with self.monitoring_lock:
            if not self.monitoring_data['timestamps']:
                logger.warning("⚠️ Monitoring verisi bulunamadı")
                return ""
            
            try:
                # Plot oluştur
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('📊 Live Training Monitor', fontsize=14, fontweight='bold')
                
                timestamps = self.monitoring_data['timestamps']
                
                # Episode rewards
                if self.monitoring_data['episode_rewards']:
                    axes[0, 0].plot(timestamps[:len(self.monitoring_data['episode_rewards'])], 
                                   self.monitoring_data['episode_rewards'], 'b-', linewidth=2)
                    axes[0, 0].set_title('🎯 Episode Rewards')
                    axes[0, 0].set_ylabel('Reward')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # GPU memory
                if self.monitoring_data['gpu_memory']:
                    axes[0, 1].plot(timestamps[:len(self.monitoring_data['gpu_memory'])], 
                                   self.monitoring_data['gpu_memory'], 'r-', linewidth=2)
                    axes[0, 1].set_title('🔥 GPU Memory Usage')
                    axes[0, 1].set_ylabel('Memory (%)')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # CPU usage
                if self.monitoring_data['cpu_usage']:
                    axes[1, 0].plot(timestamps[:len(self.monitoring_data['cpu_usage'])], 
                                   self.monitoring_data['cpu_usage'], 'g-', linewidth=2)
                    axes[1, 0].set_title('💻 CPU Usage')
                    axes[1, 0].set_ylabel('CPU (%)')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Training speed
                if self.monitoring_data['steps_per_second']:
                    axes[1, 1].plot(timestamps[:len(self.monitoring_data['steps_per_second'])], 
                                   self.monitoring_data['steps_per_second'], 'm-', linewidth=2)
                    axes[1, 1].set_title('⚡ Training Speed')
                    axes[1, 1].set_ylabel('Steps/sec')
                    axes[1, 1].grid(True, alpha=0.3)
                
                # X-axis formatting
                for ax_row in axes:
                    for ax in ax_row:
                        ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                
                # Kaydet
                if save_path is None:
                    save_path = f"live_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"📊 Live plot kaydedildi: {save_path}")
                return save_path
                
            except Exception as e:
                logger.error(f"❌ Live plot hatası: {e}")
                return ""
    
    def get_latest_metrics(self) -> Dict:
        """En son metrikleri al"""
        with self.monitoring_lock:
            if not self.monitoring_data['timestamps']:
                return {}
            
            latest_metrics = {}
            for key, values in self.monitoring_data.items():
                if values and key != 'timestamps':
                    latest_metrics[key] = values[-1]
            
            latest_metrics['timestamp'] = self.monitoring_data['timestamps'][-1].isoformat()
            latest_metrics['data_points'] = len(self.monitoring_data['timestamps'])
            
            return latest_metrics
    
    def get_statistics(self) -> Dict:
        """Monitoring istatistikleri"""
        with self.monitoring_lock:
            if not self.monitoring_data['timestamps']:
                return {'status': 'No data'}
            
            stats = {}
            
            for key, values in self.monitoring_data.items():
                if values and key != 'timestamps' and isinstance(values[0], (int, float)):
                    stats[f'{key}_mean'] = np.mean(values)
                    stats[f'{key}_std'] = np.std(values)
                    stats[f'{key}_min'] = np.min(values)
                    stats[f'{key}_max'] = np.max(values)
            
            stats['monitoring_duration_minutes'] = (
                self.monitoring_data['timestamps'][-1] - self.monitoring_data['timestamps'][0]
            ).total_seconds() / 60
            
            stats['total_data_points'] = len(self.monitoring_data['timestamps'])
            
            return stats
    
    def add_callback(self, callback: Callable):
        """Veri callback'i ekle"""
        self.data_callbacks.append(callback)
        logger.info("📞 Callback eklendi")
    
    def export_data(self, file_path: str) -> bool:
        """Monitoring verilerini export et"""
        with self.monitoring_lock:
            try:
                # DataFrame oluştur
                df_data = {}
                max_length = max(len(values) for values in self.monitoring_data.values() if values)
                
                for key, values in self.monitoring_data.items():
                    # Listeleri aynı uzunlukta yap
                    if len(values) < max_length:
                        values.extend([None] * (max_length - len(values)))
                    df_data[key] = values
                
                df = pd.DataFrame(df_data)
                
                # Export format'a göre kaydet
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                elif file_path.endswith('.json'):
                    df.to_json(file_path, orient='records', date_format='iso')
                else:
                    # Default: CSV
                    df.to_csv(file_path + '.csv', index=False)
                
                logger.info(f"💾 Monitoring verisi export edildi: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Export hatası: {e}")
                return False
    
    def clear_data(self):
        """Monitoring verilerini temizle"""
        with self.monitoring_lock:
            for key in self.monitoring_data:
                self.monitoring_data[key].clear()
        
        logger.info("🧹 Monitoring verisi temizlendi")


class TrainingCallback:
    """Training callback için helper sınıf"""
    
    def __init__(self, monitor: LiveMonitor):
        self.monitor = monitor
        self.episode_count = 0
        self.last_update_time = time.time()
    
    def on_episode_end(self, episode_reward: float, episode_length: int):
        """Episode bittiğinde çağrılır"""
        self.episode_count += 1
        
        # Training data ekle
        data = {
            'episode_rewards': episode_reward,
            'episode_lengths': episode_length
        }
        
        self.monitor.add_data_point(data)
    
    def on_training_step(self, step: int, learning_rate: float = None):
        """Training step'inde çağrılır"""
        current_time = time.time()
        
        # Steps per second hesapla
        time_diff = current_time - self.last_update_time
        if time_diff > 0:
            steps_per_second = 1.0 / time_diff
            
            data = {
                'steps_per_second': steps_per_second
            }
            
            if learning_rate is not None:
                data['learning_rate'] = learning_rate
            
            self.monitor.add_data_point(data)
        
        self.last_update_time = current_time 