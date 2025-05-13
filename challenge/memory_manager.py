import numpy as np
from typing import Dict, List, Tuple
import psutil
import gc

class MatrixMemoryManager:
    """Advanced memory management for matrices."""
    _instance = None
    _cpu_pool: Dict[Tuple[Tuple[int, ...], np.dtype], List[np.ndarray]] = {}
    _max_pool_size = 1000
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            'text': memory_info.text / (1024 * 1024) if hasattr(memory_info, 'text') else 0,
            'data': memory_info.data / (1024 * 1024) if hasattr(memory_info, 'data') else 0,
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available': psutil.virtual_memory().available / (1024 * 1024)
        }
        if hasattr(memory_info, 'shared'):
            stats['shared'] = memory_info.shared / (1024 * 1024)
        return stats

    def get_array(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Get an array from the pool or create a new one."""
        key = (shape, dtype)
        if key in self._cpu_pool and self._cpu_pool[key]:
            return self._cpu_pool[key].pop()
        return np.empty(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """Return an array to the pool."""
        key = (array.shape, array.dtype)
        if key not in self._cpu_pool:
            self._cpu_pool[key] = []
        if len(self._cpu_pool[key]) < self._max_pool_size:
            self._cpu_pool[key].append(array)
    
    def clear(self):
        """Clear all pools and force garbage collection."""
        self._cpu_pool.clear()
        gc.collect() 