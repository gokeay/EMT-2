"""
🔥 EMT RL Project - CUDA Utilities
GPU detection, memory monitoring ve device management
"""

import torch
import logging
from typing import Dict, Optional

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CudaManager:
    """CUDA GPU yönetimi için utility sınıfı"""
    
    def __init__(self):
        """CUDA Manager başlatma"""
        self.device = self._detect_device()
        self.gpu_info = self._get_gpu_info() if self.is_cuda_available() else {}
        
        logger.info(f"🔥 CudaManager başlatıldı - Device: {self.device}")
    
    def _detect_device(self) -> torch.device:
        """En uygun device'ı tespit et"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("⚠️ CUDA not available - CPU kullanılacak")
        
        return device
    
    def is_cuda_available(self) -> bool:
        """CUDA kullanılabilir mi?"""
        return torch.cuda.is_available()
    
    def get_device(self) -> torch.device:
        """Aktif device'ı döndür"""
        return self.device
    
    def get_device_name(self) -> str:
        """Device adını döndür"""
        if self.is_cuda_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    
    def _get_gpu_info(self) -> Dict:
        """GPU bilgilerini topla"""
        if not self.is_cuda_available():
            return {}
        
        try:
            info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_reserved': torch.cuda.memory_reserved(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'compute_capability': torch.cuda.get_device_properties(0).major
            }
            return info
        except Exception as e:
            logger.error(f"❌ GPU info alınamadı: {e}")
            return {}
    
    def get_memory_stats(self) -> Dict:
        """GPU memory istatistikleri"""
        if not self.is_cuda_available():
            return {'status': 'CUDA not available'}
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            stats = {
                'allocated_mb': allocated / 1024**2,
                'reserved_mb': reserved / 1024**2,
                'total_mb': total / 1024**2,
                'free_mb': (total - reserved) / 1024**2,
                'usage_percent': (allocated / total) * 100
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Memory stats alınamadı: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """GPU cache'ini temizle"""
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache temizlendi")
        else:
            logger.warning("⚠️ CUDA available değil - cache temizlenemedi")
    
    def print_gpu_info(self):
        """GPU bilgilerini yazdır"""
        print("🔥 GPU Information:")
        print(f"  Device: {self.get_device_name()}")
        
        if self.is_cuda_available():
            info = self.gpu_info
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device Count: {info.get('device_count', 'N/A')}")
            print(f"  Compute Capability: {info.get('compute_capability', 'N/A')}")
            print(f"  Total Memory: {info.get('memory_total', 0) / 1024**3:.1f} GB")
            
            # Memory stats
            memory = self.get_memory_stats()
            print(f"  Memory Usage: {memory.get('usage_percent', 0):.1f}%")
            print(f"  Allocated: {memory.get('allocated_mb', 0):.1f} MB")
            print(f"  Free: {memory.get('free_mb', 0):.1f} MB")
        else:
            print("  Status: CUDA not available")
    
    def test_cuda_operations(self) -> bool:
        """CUDA operasyonlarını test et"""
        try:
            if not self.is_cuda_available():
                logger.warning("⚠️ CUDA test skipped - not available")
                return False
            
            # Simple tensor operations test
            logger.info("🧪 CUDA operations test başladı...")
            
            # CPU'da tensor oluştur
            x_cpu = torch.randn(1000, 1000)
            y_cpu = torch.randn(1000, 1000)
            
            # GPU'ya taşı
            x_gpu = x_cpu.to(self.device)
            y_gpu = y_cpu.to(self.device)
            
            # GPU'da işlem yap
            result_gpu = torch.matmul(x_gpu, y_gpu)
            
            # CPU'ya geri taşı
            result_cpu = result_gpu.cpu()
            
            # Doğrulama
            expected = torch.matmul(x_cpu, y_cpu)
            is_close = torch.allclose(result_cpu, expected, rtol=1e-4)
            
            if is_close:
                logger.info("✅ CUDA operations test başarılı!")
                return True
            else:
                logger.error("❌ CUDA operations test başarısız - sonuçlar eşleşmiyor")
                return False
                
        except Exception as e:
            logger.error(f"❌ CUDA test hatası: {e}")
            return False
    
    def benchmark_device_performance(self, matrix_size: int = 2000, iterations: int = 10) -> Dict:
        """CPU vs GPU performance karşılaştırması"""
        import time
        
        results = {}
        
        # Test data
        x = torch.randn(matrix_size, matrix_size)
        y = torch.randn(matrix_size, matrix_size)
        
        # CPU benchmark
        logger.info(f"🏃 CPU benchmark başladı ({iterations} iterations)...")
        cpu_times = []
        for i in range(iterations):
            start = time.time()
            _ = torch.matmul(x, y)
            cpu_times.append(time.time() - start)
        
        results['cpu'] = {
            'avg_time': sum(cpu_times) / len(cpu_times),
            'min_time': min(cpu_times),
            'max_time': max(cpu_times)
        }
        
        # GPU benchmark (if available)
        if self.is_cuda_available():
            logger.info(f"🔥 GPU benchmark başladı ({iterations} iterations)...")
            x_gpu = x.to(self.device)
            y_gpu = y.to(self.device)
            
            # Warm up
            for _ in range(3):
                _ = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            
            gpu_times = []
            for i in range(iterations):
                torch.cuda.synchronize()
                start = time.time()
                _ = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()
                gpu_times.append(time.time() - start)
            
            results['gpu'] = {
                'avg_time': sum(gpu_times) / len(gpu_times),
                'min_time': min(gpu_times),
                'max_time': max(gpu_times)
            }
            
            # Speedup calculation
            if results['cpu']['avg_time'] > 0:
                results['speedup'] = results['cpu']['avg_time'] / results['gpu']['avg_time']
        
        return results


# Global instance
cuda_manager = CudaManager()


def get_device() -> torch.device:
    """Global device getter"""
    return cuda_manager.get_device()


def is_cuda_available() -> bool:
    """Global CUDA check"""
    return cuda_manager.is_cuda_available()


def print_device_info():
    """Global device info printer"""
    cuda_manager.print_gpu_info()


def check_cuda() -> bool:
    """Global CUDA test"""
    return cuda_manager.test_cuda_operations()


def benchmark_performance(matrix_size: int = 2000, iterations: int = 10) -> Dict:
    """Global performance benchmark"""
    return cuda_manager.benchmark_device_performance(matrix_size, iterations) 