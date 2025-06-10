"""
📋 Model Listesi Script'i
Kayıtlı modellerin geçmişini görüntüler
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.model_id_manager import model_id_manager
import json
from datetime import datetime

def list_models():
    """Tüm modelleri listele"""
    models = model_id_manager.get_model_history()
    
    if not models:
        print("📋 Henüz kayıtlı model yok.")
        return
    
    print("=" * 80)
    print("📋 KAYITLI MODELLER")
    print("=" * 80)
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. MODEL:")
        print(f"   🆔 ID: {model.get('id', 'N/A')}")
        print(f"   📛 İsim: {model.get('name', 'N/A')}")
        print(f"   📝 Açıklama: {model.get('description', 'Açıklama yok')}")
        print(f"   🔢 Timesteps: {model.get('timesteps', 0):,}")
        print(f"   ⚙️  Config: {model.get('config_path', 'N/A')}")
        
        created_at = model.get('created_at', '')
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                print(f"   📅 Oluşturulma: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                print(f"   📅 Oluşturulma: {created_at}")
        
        print("-" * 50)
    
    print(f"\n✅ Toplam {len(models)} model kayıtlı.")

def show_counter_status():
    """Counter dosyası durumunu göster"""
    try:
        with open("configs/model_counter.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n" + "=" * 50)
        print("🔢 COUNTER DURUMU")
        print("=" * 50)
        print(f"Mevcut ID: {data.get('current_id', 0)}")
        print(f"Sonraki ID: {data.get('current_id', 0) + 1:04d}")
        print(f"Son Güncelleme: {data.get('last_updated', 'N/A')}")
        print(f"Oluşturulma: {data.get('created_at', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Counter dosyası okunamadı: {e}")

if __name__ == "__main__":
    print("🚀 Model ID Sistemi - Liste Görüntüleyici")
    
    list_models()
    show_counter_status()
    
    print("\n" + "=" * 80) 