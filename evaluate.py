r"""
__ __ ___ _ __ ___  _ __   __ _  _ __ _   _
| '_ ` _ \| '_ ` _ \| '_ \ / _` || '__| | | |
| | | | | | | | | | | |_) | (_| || |  | |_| |
|_| |_| |_|_| |_| |_| .__/ \__,_||_|   \__, |
                    | |                 __/ |
                    |_|                |___/

🚀 Model Değerlendirme Script'i
Kaydedilmiş bir PPO modelini yükler ve performansını değerlendirir.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.energy_environment import EnergyEnvironment
from src.data.data_handler import DataHandler
from stable_baselines3.common.evaluation import evaluate_policy

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Komut satırı argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="EMT RL Projesi - Model Değerlendirme")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Değerlendirilecek modelin .zip dosya yolu.')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Konfigürasyon dosyasının yolu.')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Değerlendirme için çalıştırılacak bölüm (episode) sayısı.')
    parser.add_argument('--no-plot', action='store_true',
                        help='Sonuç grafiğini gösterme.')
    parser.add_argument('--save-csv', action='store_true',
                        help='Değerlendirme bölümünün detaylı verilerini CSV olarak kaydet.')
    return parser.parse_args()


def plot_evaluation_results(df: pd.DataFrame, model_name: str, save_csv: bool = False, config_path: str = None):
    """Bir episode'un sonuçlarını görselleştirir ve isteğe bağlı olarak CSV'ye kaydeder."""
    logger.info("📊 Sonuçlar görselleştiriliyor...")
    
    plot_dir = "evaluation_results"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Modeline ait config dosyasını sonuçlar klasörüne kopyala
    if config_path and os.path.exists(config_path):
        try:
            config_dest_path = os.path.join(plot_dir, f"{model_name}_config_{timestamp}.yaml")
            with open(config_path, 'r', encoding='utf-8') as src:
                with open(config_dest_path, 'w', encoding='utf-8') as dest:
                    dest.write(src.read())
            logger.info(f"💾 Yapılandırma dosyası kopyalandı: {config_dest_path}")
        except Exception as e:
            logger.warning(f"⚠️  Yapılandırma dosyası kopyalanamadı: {e}")

    # CSV Kaydetme
    if save_csv:
        csv_path = os.path.join(plot_dir, f"{model_name}_eval_data_{timestamp}.csv")
        try:
            df.to_csv(csv_path, index=False, float_format='%.4f')
            logger.info(f"💾 Değerlendirme verileri CSV olarak kaydedildi: {csv_path}")
        except Exception as e:
            logger.error(f"❌ CSV kaydetme hatası: {e}")

    try:
        # Plotting
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Model Değerlendirme Sonuçları: {model_name}', fontsize=16)

        # 1. Enerji Akışı
        axes[0].plot(df['load'], label='Yük (kW)', color='r')
        axes[0].plot(df['renewable_generation'], label='Yenilenebilir Üretim (kW)', color='g')
        axes[0].plot(df['grid_energy'], label='Şebeke Enerjisi (kW)', color='b', linestyle='--')
        axes[0].set_ylabel('Güç (kW)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title('Enerji Akışı')

        # 2. Batarya Durumu
        axes[1].plot(df['battery_soc'], label='Batarya Şarj Seviyesi (%)', color='m')
        axes[1].axhline(y=0.2, color='r', linestyle='--', label='Min SOC (%20)')
        axes[1].axhline(y=0.9, color='orange', linestyle='--', label='Max SOC (%90)')
        axes[1].set_ylabel('SOC (%)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title('Batarya Durumu')

        # 3. Batarya Gücü
        axes[2].plot(df['battery_power'], label='Batarya Gücü (kW)', color='c')
        axes[2].axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        axes[2].set_ylabel('Güç (kW)')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_title('Batarya Şarj/Deşarj Gücü')

        # 4. Saatlik Ödül
        axes[3].plot(df['reward'], label='Saatlik Ödül', color='y', marker='o', markersize=2, linestyle='None')
        axes[3].set_xlabel('Zaman Adımı (Saat)')
        axes[3].set_ylabel('Ödül')
        axes[3].legend()
        axes[3].grid(True)
        axes[3].set_title('Saatlik Ödül Dağılımı')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Grafik dosyasını kaydet
        plot_path = os.path.join(plot_dir, f"{model_name}_eval_plot_{timestamp}.png")
        plt.savefig(plot_path)
        logger.info(f"📈 Grafik kaydedildi: {plot_path}")
        
        plt.show()

    except Exception as e:
        logger.error(f"❌ Grafik çiziminde hata: {e}")


def main():
    """Ana değerlendirme fonksiyonu."""
    args = parse_arguments()
    # Model adını klasör adından al, dosya adından değil
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(model_dir)  # Klasör adını kullan
    model_config_path = os.path.join(model_dir, "config.yaml")

    logger.info(f"🚀 Model Değerlendirme Başladı: {model_name}")
    logger.info(f"Model Yolu: {args.model_path}")
    logger.info(f"Bölüm Sayısı: {args.eval_episodes}")

    try:
        # --- Kurulum ---
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        data_handler = DataHandler(data_dir="data")
        if not data_handler.load_all_data():
            raise RuntimeError("Veri yüklenemedi!")

        # Değerlendirme için environment'ı oluştur
        eval_env = EnergyEnvironment(data_handler=data_handler, config_path=args.config)
        
        # Modeli yükle
        model = PPO.load(args.model_path, env=eval_env)
        logger.info("✅ Model başarıyla yüklendi.")

        # --- Değerlendirme ---
        logger.info("⏳ Değerlendirme çalıştırılıyor...")
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            return_episode_rewards=False # Sadece ortalama ve std al
        )

        print("\n" + "="*50)
        logger.info("🎉 DEĞERLENDİRME TAMAMLANDI 🎉")
        logger.info(f"Ortalama Ödül: {mean_reward:.2f} +/- {std_reward:.2f}")
        print("="*50 + "\n")

        # --- Tek Bir Bölümü Çalıştır ve Veri Topla (Görselleştirme için) ---
        if not args.no_plot:
            logger.info("🔍 Görselleştirme için tek bir bölüm çalıştırılıyor...")
            obs, info = eval_env.reset()
            done = False
            episode_data = []
            
            # eval_env.current_step'i sıfırla (eğer reset'te sıfırlanmıyorsa)
            if hasattr(eval_env, 'current_step'):
                 eval_env.current_step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                # `info` içinde step detayları olduğunu varsayıyoruz
                step_info = info.get('step_details', {})
                step_info['reward'] = reward
                episode_data.append(step_info)

            df = pd.DataFrame(episode_data)
            plot_evaluation_results(df, model_name, args.save_csv, model_config_path)

    except Exception as e:
        logger.error(f"❌ Değerlendirme sırasında kritik bir hata oluştu: {e}", exc_info=True)
    finally:
        if 'eval_env' in locals():
            eval_env.close()
        logger.info("🧹 Kaynaklar temizlendi.")


if __name__ == "__main__":
    main() 