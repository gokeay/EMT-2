"""
🚀 EMT RL Project - Main Training Script
PPO Agent eğitimi için ana script
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml

# Path setup
# Projenin ana dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.energy_environment import EnergyEnvironment
from src.agents.ppo_agent import PPOAgent
from src.data.data_handler import DataHandler
from src.utils.cuda_utils import CudaManager
from src.monitoring.live_monitor import LiveMonitor, TrainingCallback

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Command line argümanları"""
    parser = argparse.ArgumentParser(description="EMT RL Project Training")
    
    parser.add_argument('--timesteps', type=int, default=(100000),
                       help='Total training timesteps (default: 100000)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Config file path (default: configs/config.yaml)')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable live monitoring')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Evaluation episodes (default: 10)')
    parser.add_argument('--model-name', type=str, default=f'PPO_{datetime.now().strftime("%Y%m%d_%H%M")}',
                        help='Name for the trained model and log files')
    parser.add_argument('--description', type=str, default='Direct training run',
                        help='A short description for the training run')
    
    return parser.parse_args()


def main():
    """Ana training fonksiyonu (TrainingManager olmadan, doğrudan)"""
    print("🚀 EMT RL Project - Training Started (Direct Mode)")
    print("=" * 60)
    
    args = parse_arguments()
    monitor = None
    
    try:
        # --- 1. Kurulum ---
        model_name = args.model_name
        log_dir = os.path.join("logs", model_name)
        model_dir = os.path.join("results", "models", model_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        data_handler = DataHandler(data_dir="data")
        if not data_handler.load_all_data():
            raise RuntimeError("Veri yüklenemedi!")

        env = EnergyEnvironment(data_handler=data_handler, config_path=args.config)
        
        cuda_manager = CudaManager()
        agent = PPOAgent(
            environment=env,
            config_path=args.config,
            model_save_path=model_dir,
            log_dir=log_dir
        )
        agent.create_model()
        logger.info(f"🤖 Kurulum tamamlandı. Model Adı: {model_name}")

        # --- 2. Monitoring ---
        callback = None
        if not args.no_monitoring:
            monitor = LiveMonitor(update_interval=2.0)
            callback = TrainingCallback(monitor)
            monitor.start_monitoring()

        # --- 3. Eğitim ---
        logger.info(f"🚀 Eğitim başlıyor - {args.timesteps:,} timesteps")
        agent.train(total_timesteps=args.timesteps)
        logger.info("✅ Eğitim tamamlandı!")

        # --- 4. Değerlendirme ---
        logger.info("📊 Model değerlendiriliyor...")
        results = agent.evaluate(n_episodes=args.eval_episodes)
        print(f"\n📈 Değerlendirme Sonuçları: Ortalama Ödül = {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")

        print("\n🎉 Eğitim süreci başarıyla tamamlandı!")

    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}", exc_info=True)
    finally:
        if monitor:
            monitor.stop_monitoring()
        if 'env' in locals() and env:
            env.close()
        logger.info("🧹 Kaynaklar temizlendi.")


if __name__ == "__main__":
    print("--- SCRIPT BAŞLADI ---")
    main()
    print("--- SCRIPT TAMAMLANDI ---") 