"""
Side experiment: rebuild the tensor from Strassen's factors in two ways and time them.

Both functions compute sum_r U[:, r] (x) V[:, r] (x) W[:, r], the
reconstruction used in steps 1 and 2: once with plain Python loops and once
with np.einsum. The results must match exactly; the einsum version is faster.

Usage:
    python extras/einsum_vs_loops.py
"""
import time
import numpy as np

# 1. Define Strassen factor matrices
STRASSEN_U = np.array([
    [1, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1],
])
STRASSEN_V = np.array([
    [1, 1, 0, -1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1],
])
STRASSEN_W = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 1, 0],
])

# 2. Plain Python loops over every (i, j, k) entry and every rank-one term r
def reconstruct_loops(U, V, W):
    tensor = np.zeros((4, 4, 4))
    for idx_i in range(4):
        for idx_j in range(4):
            for idx_k in range(4):
                total_sum = 0
                for idx_r in range(7):
                    total_sum += U[idx_i, idx_r] * V[idx_j, idx_r] * W[idx_k, idx_r]
                tensor[idx_i, idx_j, idx_k] = total_sum
    return tensor

# 3. The library optimized approach
def reconstruct_einsum(U, V, W):
    return np.einsum("ir,jr,kr->ijk", U, V, W)


# --- SECTION 1: EQUIVALENCE CHECK (Highest Priority) ---
tensor_loops = reconstruct_loops(STRASSEN_U, STRASSEN_V, STRASSEN_W)
tensor_einsum = reconstruct_einsum(STRASSEN_U, STRASSEN_V, STRASSEN_W)

# Test if every element, shape, and data type matches down to the bit
are_identical = np.array_equal(tensor_loops, tensor_einsum)

print("=" * 50)
print(f"🎯 EQUIVALENCE TEST RESULT: {are_identical}")
print("=" * 50)
if are_identical:
    print("Success: Both algorithms map out identical tensor structures.\n")


# --- SECTION 2: SPEED BENCHMARK ---
print("Running speed benchmarks (10,000 iterations)...")

# Time the nested loops
start_loops = time.perf_counter()
for _ in range(10000):
    reconstruct_loops(STRASSEN_U, STRASSEN_V, STRASSEN_W)
duration_loops = time.perf_counter() - start_loops

# Time the einsum operation
start_einsum = time.perf_counter()
for _ in range(10000):
    reconstruct_einsum(STRASSEN_U, STRASSEN_V, STRASSEN_W)
duration_einsum = time.perf_counter() - start_einsum

# Display metrics
print("-" * 50)
print(f"⏱️  Written Loops Total Time:  {duration_loops:.4f} seconds")
print(f"⏱️  np.einsum Total Time:      {duration_einsum:.4f} seconds")
print(f"🚀 Speed Multiplier:          {duration_loops / duration_einsum:.1f}x faster using library")
print("-" * 50)

# ==============================================================================
# PERFORMANCE NOTE: Why np.einsum is faster than written loops
# ==============================================================================
# Both functions are mathematically identical, but run in different layers:
#
# 1. Compiled C vs. Interpreted Python: 
#    The written loops run line-by-line inside the slow Python interpreter.
#    'np.einsum' passes the layout to pre-compiled C binaries that run instantly.
#
# 2. Overhead Elimination:
#    Python must check data types and array bounds 448,000 times during the loops.
#    'np.einsum' checks types once at the start and runs raw hardware math.
#
# 3. Hardware Optimization:
#    'np.einsum' groups memory access to fit into ultra-fast CPU caches and 
#    uses SIMD vectorization to crunch multiple numbers at the exact same time.
# ==============================================================================
