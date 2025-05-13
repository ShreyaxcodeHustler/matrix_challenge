import numpy as np
from typing import Union, List, Tuple
from functools import lru_cache
import multiprocessing
from multiprocessing import Pool
from memory_manager import MatrixMemoryManager
from optimizations import _optimized_matrix_multiply

class Matrix:
    """
    A highly optimized 2D Matrix class supporting basic operations and broadcasting.
    Uses NumPy and Numba for maximum performance.
    """

    __slots__ = ['_data', '_shape', '_cached_transpose', '_memory_manager']

    def __init__(self, data: Union[List, np.ndarray]):
        """Initialize a Matrix with optimized memory management."""
        self._memory_manager = MatrixMemoryManager.get_instance()
        self._data = np.array(data, dtype=np.float64)
        self._shape = self._data.shape
        self._cached_transpose = None

    def __str__(self) -> str:
        """Return a formatted string representation of the matrix."""
        return str(self._data)

    def __repr__(self) -> str:
        """Return a detailed string representation of the matrix."""
        return f"Matrix({self._data})"

    def format_matrix(self) -> str:
        """Return a nicely formatted string representation of the matrix."""
        rows, cols = self._shape
        result = []
        for i in range(rows):
            row = [f"{self._data[i,j]:.2f}" for j in range(cols)]
            result.append("  ".join(row))
        return "\n".join(result)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_data'):
            self._memory_manager.return_array(self._data)
        if hasattr(self, '_cached_transpose') and self._cached_transpose is not None:
            self._memory_manager.return_array(self._cached_transpose._data)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the matrix."""
        return self._shape

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices with broadcasting support."""
        if isinstance(other, Matrix):
            result = self._data + other._data
            return Matrix(result)
        return Matrix(self._data + np.array(other))

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices with broadcasting support."""
        if isinstance(other, Matrix):
            result = self._data - other._data
            return Matrix(result)
        return Matrix(self._data - np.array(other))

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        """Element-wise multiplication with broadcasting support."""
        if isinstance(other, Matrix):
            result = self._data * other._data
            return Matrix(result)
        return Matrix(self._data * np.array(other))

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication with parallel processing."""
        if isinstance(other, Matrix):
            if self._data.size > 1000000:  # For large matrices, use parallel processing
                return self._parallel_matmul(other)
            result = _optimized_matrix_multiply(self._data, other._data)
            return Matrix(result)
        return Matrix(_optimized_matrix_multiply(self._data, np.array(other)))

    def _parallel_matmul(self, other: 'Matrix') -> 'Matrix':
        """Parallel matrix multiplication."""
        def chunk_multiply(chunk):
            return _optimized_matrix_multiply(chunk, other._data)

        chunk_size = max(1, self._data.shape[0] // (multiprocessing.cpu_count() * 2))
        chunks = [self._data[i:i + chunk_size] for i in range(0, self._data.shape[0], chunk_size)]

        with Pool() as pool:
            results = pool.map(chunk_multiply, chunks)

        return Matrix(np.vstack(results))

    def __pow__(self, power: int) -> 'Matrix':
        """Raise each element to the specified power."""
        result = np.power(self._data, power)
        return Matrix(result)

    @lru_cache(maxsize=128)
    def transpose(self) -> 'Matrix':
        """Return the transpose of the matrix with caching."""
        if self._cached_transpose is None:
            self._cached_transpose = Matrix(np.transpose(self._data))
        return self._cached_transpose

    def clear_cache(self):
        """Clear cached computations."""
        self._cached_transpose = None
        self.transpose.cache_clear()

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics for this matrix."""
        stats = self._memory_manager.get_memory_stats()
        matrix_size = self._data.nbytes / (1024 * 1024)  # Size in MB
        stats['matrix_size'] = matrix_size
        return stats 