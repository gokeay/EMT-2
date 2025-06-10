"""
🧪 EMT RL Project - CUDA Utilities Tests
CUDA utilities için unit testler
"""

import unittest
import sys
import os
import torch
from unittest.mock import patch, MagicMock
import logging

# Test ortamı için logging'i sessiz yap
logging.disable(logging.CRITICAL)

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.cuda_utils import (
    get_device, 
    is_cuda_available, 
    check_cuda,
    benchmark_performance,
    cuda_manager
)


class TestCudaUtils(unittest.TestCase):
    """CUDA utilities için test sınıfı"""
    
    def test_get_device(self):
        """Device detection testi"""
        device = get_device()
        
        # Device torch.device olmalı
        self.assertIsInstance(device, torch.device)
        
        # Device ya "cuda" ya "mps" ya da "cpu" olmalı
        self.assertIn(str(device), ['cuda', 'mps', 'cpu'])
    
    def test_is_cuda_available(self):
        """CUDA availability testi"""
        result = is_cuda_available()
        
        # Boolean döndürmeli
        self.assertIsInstance(result, bool)
        
        # torch.cuda.is_available() ile tutarlı olmalı
        self.assertEqual(result, torch.cuda.is_available())
    
    @patch('torch.cuda.is_available')
    def test_is_cuda_available_false(self, mock_cuda_available):
        """CUDA olmadığında test"""
        mock_cuda_available.return_value = False
        
        result = is_cuda_available()
        
        self.assertFalse(result)
    
    @patch('torch.cuda.is_available')
    def test_cuda_function_no_cuda(self, mock_cuda_available):
        """CUDA yokken check_cuda testi"""
        mock_cuda_available.return_value = False
        
        result = check_cuda()
        
        # False döndürmeli
        self.assertFalse(result)
    
    @patch('torch.cuda.is_available')
    def test_benchmark_performance_no_cuda(self, mock_cuda_available):
        """CUDA yokken benchmark testi"""
        mock_cuda_available.return_value = False
        
        result = benchmark_performance(matrix_size=100, iterations=2)
        
        # Dictionary döndürmeli
        self.assertIsInstance(result, dict)
        
        # CPU sonuçları olmalı
        self.assertIn('cpu', result)


class TestCudaManager(unittest.TestCase):
    """CudaManager sınıfı için test sınıfı"""
    
    def test_cuda_manager_singleton(self):
        """CudaManager singleton pattern testi"""
        manager1 = cuda_manager
        manager2 = cuda_manager
        
        # Aynı instance olmalı
        self.assertIs(manager1, manager2)
    
    def test_cuda_manager_basic_methods(self):
        """CudaManager temel method testleri"""
        # Methods exist kontrolü
        self.assertTrue(hasattr(cuda_manager, 'is_cuda_available'))
        self.assertTrue(hasattr(cuda_manager, 'get_device'))
        self.assertTrue(hasattr(cuda_manager, 'get_device_name'))
        self.assertTrue(hasattr(cuda_manager, 'get_memory_stats'))
        self.assertTrue(hasattr(cuda_manager, 'clear_cache'))
        self.assertTrue(hasattr(cuda_manager, 'benchmark_device_performance'))
    
    def test_cuda_manager_is_available(self):
        """CudaManager is_cuda_available test"""
        result = cuda_manager.is_cuda_available()
        
        # Boolean döndürmeli
        self.assertIsInstance(result, bool)
        
        # torch.cuda.is_available() ile tutarlı olmalı
        self.assertEqual(result, torch.cuda.is_available())
    
    def test_cuda_manager_get_device(self):
        """CudaManager get_device test"""
        device = cuda_manager.get_device()
        
        # torch.device döndürmeli
        self.assertIsInstance(device, torch.device)
    
    def test_cuda_manager_get_device_name(self):
        """CudaManager get_device_name test"""
        name = cuda_manager.get_device_name()
        
        # String döndürmeli
        self.assertIsInstance(name, str)
        
        # En az bir karakter olmalı
        self.assertGreater(len(name), 0)
    
    def test_cuda_manager_get_memory_stats(self):
        """CudaManager get_memory_stats test"""
        stats = cuda_manager.get_memory_stats()
        
        # Dictionary döndürmeli
        self.assertIsInstance(stats, dict)
    
    def test_cuda_manager_clear_cache(self):
        """CudaManager clear_cache test"""
        # Hata çıkmamalı
        cuda_manager.clear_cache()
        
        # Test başarılı olmalı
        self.assertTrue(True)
    
    def test_cuda_manager_benchmark_device_method_exists(self):
        """CudaManager benchmark_device method var mı test"""
        # Method varlığını kontrol et
        self.assertTrue(hasattr(cuda_manager, 'benchmark_device_performance'))
        
        # Test başarılı olmalı
        self.assertTrue(True)


if __name__ == '__main__':
    # Test suite oluştur ve çalıştır
    unittest.main(verbosity=2) 