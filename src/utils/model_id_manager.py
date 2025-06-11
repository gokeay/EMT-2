"""
🆔 Model ID Manager
Model isimlendirmesi için otomatik ID sistemi
"""

import os
import json
from datetime import datetime
#from typing import str
import logging

logger = logging.getLogger(__name__)

class ModelIDManager:
    """
    Model ID yönetimi için singleton class
    ID'ler 0001'den başlar ve otomatik artar
    """
    
    def __init__(self, counter_file: str = "configs/model_counter.json"):
        self.counter_file = counter_file
        self._ensure_counter_file()
    
    def _ensure_counter_file(self):
        """Counter dosyasını oluştur (yoksa)"""
        if not os.path.exists(self.counter_file):
            os.makedirs(os.path.dirname(self.counter_file), exist_ok=True)
            initial_data = {
                "current_id": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "models": []
            }
            with open(self.counter_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
            logger.info(f"🆔 Model counter dosyası oluşturuldu: {self.counter_file}")
    
    def get_next_id(self) -> str:
        """Sonraki ID'yi al ve counter'ı artır"""
        try:
            with open(self.counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ID'yi artır
            data["current_id"] += 1
            next_id = data["current_id"]
            
            # Dosyayı güncelle
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.counter_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # 4 haneli string döndür (0001, 0002, ...)
            return f"{next_id:04d}"
            
        except Exception as e:
            logger.error(f"❌ ID alma hatası: {e}")
            # Fallback: timestamp tabanlı ID
            return datetime.now().strftime("%m%d")
    
    def register_model(self, model_id: str, model_name: str, description: str = "", 
                      timesteps: int = 0, config_path: str = ""):
        """Model kaydını counter dosyasına ekle"""
        try:
            with open(self.counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_record = {
                "id": model_id,
                "name": model_name,
                "description": description,
                "timesteps": timesteps,
                "config_path": config_path,
                "created_at": datetime.now().isoformat()
            }
            
            data["models"].append(model_record)
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.counter_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"🆔 Model kaydedildi: {model_id} - {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Model kayıt hatası: {e}")
    
    def generate_model_name(self, base_name: str = "PPO", description: str = "", 
                           timesteps: int = 0) -> tuple[str, str]:
        """
        Model ismi oluştur
        Returns: (model_id, full_model_name)
        """
        model_id = self.get_next_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if description:
            # Açıklama varsa: PPO_0001_30k_v4_final_20250610_1700
            full_name = f"{base_name}_{model_id}_{description}_{timestamp}"
        else:
            # Sadece timesteps: PPO_0001_30k_20250610_1700
            steps_str = f"{timesteps//1000}k" if timesteps >= 1000 else str(timesteps)
            full_name = f"{base_name}_{model_id}_{steps_str}_{timestamp}"
        
        return model_id, full_name
    
    def get_model_history(self) -> list:
        """Tüm model geçmişini döndür"""
        try:
            with open(self.counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("models", [])
        except Exception as e:
            logger.error(f"❌ Model geçmişi alma hatası: {e}")
            return []


# Global instance
model_id_manager = ModelIDManager() 