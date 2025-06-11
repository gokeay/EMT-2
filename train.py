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
from src.utils.model_id_manager import model_id_manager

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
    parser.add_argument('--config', type=str, default='/content/EMT-2/configs/config.yaml',
                       help='Config file path (default: /content/EMT-2/configs/config.yaml)')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable live monitoring')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Evaluation episodes (default: 10)')
    parser.add_argument('--model-name', type=str, default='',
                        help='Custom model name (otomatik ID sistemi kullanılır eğer boşsa)')
    parser.add_argument('--description', type=str, default='',
                        help='Model açıklaması (ID oluştururken kullanılır)')
    parser.add_argument('--auto-id', action='store_true', default=True,
                        help='Otomatik ID sistemi kullan (default: True)')
    
    return parser.parse_args()


def main():
    """Ana training fonksiyonu (TrainingManager olmadan, doğrudan)"""
    print("🚀 EMT RL Project - Training Started (Direct Mode)")
    print("=" * 60)
    
    args = parse_arguments()
    monitor = None
    
    try:
        # --- 1. Model İsimlendirme ve Kurulum ---
        if args.auto_id and not args.model_name:
            # Otomatik ID sistemi kullan
            model_id, model_name = model_id_manager.generate_model_name(
                base_name="PPO",
                description=args.description,
                timesteps=args.timesteps
            )
            logger.info(f"🆔 Otomatik ID atandı: {model_id} - {model_name}")
        else:
            # Manuel model ismi kullan
            model_name = args.model_name or f'PPO_{datetime.now().strftime("%Y%m%d_%H%M")}'
            model_id = "manual"
        
        # Yeni organize dosya yapısı
        model_dir = os.path.join("models", model_name)
        log_dir = os.path.join(model_dir, "logs")
        eval_dir = os.path.join(model_dir, "evaluation_results")
        
        # Klasörleri oluştur
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

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
        
        # Config dosyasını model klasörüne kopyala (train sırasında)
        config_copy_path = os.path.join(model_dir, "config.yaml")
        with open(args.config, 'r', encoding='utf-8') as source:
            config_content = source.read()
        with open(config_copy_path, 'w', encoding='utf-8') as target:
            target.write(config_content)
        logger.info(f"⚙️ Config dosyası kopyalandı: {config_copy_path}")
        
        logger.info(f"🤖 Kurulum tamamlandı. Model Adı: {model_name}")
        logger.info(f"📁 Model klasörü: {model_dir}")

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

        # --- 5. Model Kaydını Sisteme Ekle ---
        if args.auto_id and model_id != "manual":
            model_id_manager.register_model(
                model_id=model_id,
                model_name=model_name,
                description=args.description,
                timesteps=args.timesteps,
                config_path=args.config
            )
            logger.info(f"🆔 Model sisteme kaydedildi: {model_id}")

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