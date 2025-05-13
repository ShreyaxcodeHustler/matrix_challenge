import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def _optimized_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication using Numba."""
    m, n = a.shape
    n, p = b.shape
    result = np.zeros((m, p), dtype=np.float64)
    for i in prange(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += a[i, k] * b[k, j]
    return result

@jit(nopython=True, parallel=True, fastmath=True)
def _optimized_element_wise(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """Optimized element-wise operations using Numba."""
    if op == 'add':
        return a + b
    elif op == 'sub':
        return a - b
    elif op == 'mul':
        return a * b
    return np.empty_like(a) 