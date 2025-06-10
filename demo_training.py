"""
🎯 EMT RL Project - Training Demo
Training Manager ve Live Monitor demo script'i
"""

import os
import sys
import time
import logging

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer import TrainingManager
from src.monitoring.live_monitor import LiveMonitor, TrainingCallback

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demo ana fonksiyonu"""
    print("🎯 EMT RL Project - Training Demo")
    print("=" * 50)
    
    try:
        # 1. Training Manager oluştur
        print("\n🔧 Training Manager oluşturuluyor...")
        trainer = TrainingManager()
        
        # 2. Live Monitor oluştur
        print("📊 Live Monitor oluşturuluyor...")
        monitor = LiveMonitor(update_interval=1.0, max_data_points=50)
        callback = TrainingCallback(monitor)
        
        # 3. Training setup
        print("🔧 Training setup...")
        success = trainer.setup_training()
        
        if not success:
            print("❌ Training setup başarısız!")
            return False
        
        print(f"✅ Setup tamamlandı!")
        print(f"   Device: {trainer.agent.device}")
        print(f"   Environment: {trainer.environment.observation_space.shape}")
        print(f"   Data: {len(trainer.data_handler.combined_data)} kayıt")
        
        # 4. Live monitoring başlat
        print("\n📊 Live monitoring başlatılıyor...")
        monitor.start_monitoring()
        
        # 5. Kısa training (demo için)
        print("\n🚀 Demo training başlıyor...")
        print("   Timesteps: 1,000 (demo için kısa)")
        
        # Training başlat
        training_results = trainer.train(total_timesteps=1000)
        
        print(f"\n✅ Training tamamlandı!")
        print(f"   Süre: {training_results['training_duration_minutes']:.1f} dakika")
        print(f"   Hız: {training_results['steps_per_second']:.1f} steps/sec")
        
        # 6. Model evaluation
        print("\n📊 Model evaluation...")
        eval_results = trainer.evaluate_model(n_episodes=3, save_results=False)
        
        print(f"   Mean Reward: {eval_results['mean_reward']:.2f}")
        print(f"   Std Reward: {eval_results['std_reward']:.2f}")
        
        # 7. Monitoring sonuçları
        print("\n📊 Monitoring sonuçları...")
        monitor_stats = monitor.get_statistics()
        
        if 'status' not in monitor_stats:
            print(f"   Data Points: {monitor_stats['total_data_points']}")
            print(f"   Duration: {monitor_stats['monitoring_duration_minutes']:.1f} dakika")
            
            if 'gpu_memory_mean' in monitor_stats:
                print(f"   Avg GPU Memory: {monitor_stats['gpu_memory_mean']:.1f}%")
            if 'cpu_usage_mean' in monitor_stats:
                print(f"   Avg CPU Usage: {monitor_stats['cpu_usage_mean']:.1f}%")
        
        # 8. Visualization oluştur
        print("\n📊 Visualization oluşturuluyor...")
        
        # Training plots
        training_plots = trainer.create_training_visualization(save_plots=True)
        for plot_type, plot_path in training_plots.items():
            print(f"   📊 {plot_type}: {plot_path}")
        
        # Monitoring plot
        monitor_plot = monitor.create_live_plot()
        if monitor_plot:
            print(f"   📊 monitoring: {monitor_plot}")
        
        # 9. Training summary
        print("\n📋 Training Summary:")
        summary = trainer.get_training_summary()
        print(f"   Total Sessions: {summary['total_training_sessions']}")
        print(f"   Total Timesteps: {summary['total_timesteps']:,}")
        print(f"   Total Duration: {summary['total_duration_minutes']:.1f} dakika")
        print(f"   Average Speed: {summary['average_speed_steps_per_second']:.1f} steps/sec")
        
        print("\n🎉 Demo tamamlandı!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo hatası: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if 'monitor' in locals():
                monitor.stop_monitoring()
            
            if 'trainer' in locals():
                trainer.cleanup()
                
            print("\n🧹 Cleanup tamamlandı")
            
        except Exception as e:
            print(f"❌ Cleanup hatası: {e}")


def demo_monitoring_only():
    """Sadece monitoring demo"""
    print("\n📊 Live Monitor Demo")
    print("-" * 30)
    
    try:
        # Monitor oluştur
        monitor = LiveMonitor(update_interval=0.5, max_data_points=20)
        
        # Test callback
        def test_data_generator():
            import random
            return {
                'episode_rewards': random.uniform(50, 150),
                'gpu_memory': random.uniform(30, 80),
                'cpu_usage': random.uniform(20, 60),
                'steps_per_second': random.uniform(50, 200)
            }
        
        # Monitoring başlat
        monitor.start_monitoring(target_function=test_data_generator)
        
        print("📊 Monitoring aktif... (5 saniye)")
        time.sleep(5)
        
        # Monitoring durdur
        monitor.stop_monitoring()
        
        # Sonuçları göster
        latest = monitor.get_latest_metrics()
        print(f"\n📈 Latest Metrics:")
        for key, value in latest.items():
            if key != 'timestamp':
                print(f"   {key}: {value:.2f}")
        
        # Plot oluştur
        plot_path = monitor.create_live_plot()
        if plot_path:
            print(f"\n📊 Plot kaydedildi: {plot_path}")
        
        print("✅ Monitoring demo tamamlandı!")
        
    except Exception as e:
        print(f"❌ Monitoring demo hatası: {e}")


if __name__ == "__main__":
    print("🚀 EMT RL Training Demo")
    print("1. Full Training Demo")
    print("2. Monitoring Only Demo")
    
    choice = input("\nSeçiminiz (1/2): ").strip()
    
    if choice == "1":
        success = main()
        sys.exit(0 if success else 1)
    elif choice == "2":
        demo_monitoring_only()
        sys.exit(0)
    else:
        print("❌ Geçersiz seçim!")
        sys.exit(1) 