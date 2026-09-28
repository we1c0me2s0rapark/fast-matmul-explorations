"""
Step 1: Strassen's algorithm and the tensor view of matrix multiplication.

Part A  Count scalar multiplications for the naive method and Strassen,
        and watch the exponent move from 3 towards log2(7) = 2.807.
Part B  Write 2x2 matrix multiplication as a 4x4x4 tensor and verify that
        Strassen's 7 products form a rank-7 decomposition of it.

Usage:
    python step1_count_and_tensor.py
"""
import numpy as np


# ---------------------------------------------------------------------------
# Part A: counting multiplications
# ---------------------------------------------------------------------------
class MultiplicationCounter:
    """Counts scalar multiplications performed by the algorithms below."""

    def __init__(self) -> None:
        self.count = 0


def naive_matmul(A: np.ndarray, B: np.ndarray, counter: MultiplicationCounter) -> np.ndarray:
    """Schoolbook multiplication: C[i, j] = sum_k A[i, k] * B[k, j]."""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n): # row of A
        for j in range(n): # column of B
            for k in range(n): # column of A, row of B
                C[i, j] += A[i, k] * B[k, j]
                counter.count += 1
    return C


def strassen_matmul(A: np.ndarray, B: np.ndarray, counter: MultiplicationCounter,
                    cutoff: int = 1) -> np.ndarray:
    """Recursive Strassen multiplication. The size n must be a power of 2.

    Blocks of size <= cutoff fall back to the naive method, as practical
    implementations do, because Strassen's extra additions cost more than
    they save on small blocks.
    """
    n = A.shape[0]
    if n <= cutoff: # stop recursion because the block is small enough
        return naive_matmul(A, B, counter)

    h = n // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]

    # Seven half-size products instead of eight.
    # Each is (combination of A blocks) x (combination of B blocks), in that
    # order, so the formulas remain valid when the entries are matrices.
    m1 = strassen_matmul(a11 + a22, b11 + b22, counter, cutoff)
    m2 = strassen_matmul(a21 + a22, b11, counter, cutoff)
    m3 = strassen_matmul(a11, b12 - b22, counter, cutoff)
    m4 = strassen_matmul(a22, b21 - b11, counter, cutoff)
    m5 = strassen_matmul(a11 + a12, b22, counter, cutoff)
    m6 = strassen_matmul(a21 - a11, b11 + b12, counter, cutoff)
    m7 = strassen_matmul(a12 - a22, b21 + b22, counter, cutoff)

    C = np.empty((n, n))
    C[:h, :h] = m1 + m4 - m5 + m7
    C[:h, h:] = m3 + m5
    C[h:, :h] = m2 + m4
    C[h:, h:] = m1 - m2 + m3 + m6
    return C


def part_a(max_power: int = 6, seed: int = 0) -> None:
    print("PART A: scalar multiplications, naive vs Strassen")
    print(f"{'n':>5} {'naive':>10} {'strassen':>10} {'ratio':>7} "
          f"{'exp(naive)':>11} {'exp(strassen)':>14}")
    rng = np.random.default_rng(seed)
    for p in range(1, max_power + 1):
        n = 2 ** p
        A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        naive_counter, strassen_counter = MultiplicationCounter(), MultiplicationCounter()
        C_naive = naive_matmul(A, B, naive_counter)
        C_strassen = strassen_matmul(A, B, strassen_counter)
        assert np.allclose(C_naive, A @ B), "naive result is wrong"
        assert np.allclose(C_strassen, A @ B), "Strassen result is wrong"

        # Empirical exponent: count = n^e, so e = log2(count) / log2(n)
        # log2(n) counts the halving levels; the base cancels in the ratio, so e is the same in any base
        e_naive = np.log2(naive_counter.count) / np.log2(n)
        e_strassen = np.log2(strassen_counter.count) / np.log2(n)
        # Fraction of the naive multiplications that Strassen needs (below 1 means fewer)
        ratio = strassen_counter.count / naive_counter.count
        print(f"{n:>5} {naive_counter.count:>10} {strassen_counter.count:>10} {ratio:>7.3f} "
              f"{e_naive:>11.3f} {e_strassen:>14.3f}")
    print(f"Theory: naive exponent = 3, Strassen exponent = log2(7) = {np.log2(7):.4f}")
    print("Both methods match numpy's A @ B.\n")


# ---------------------------------------------------------------------------
# Part B: the tensor view
# ---------------------------------------------------------------------------
# A 2x2 matrix is flattened as [x11, x12, x21, x22], so entry (row, col)
# sits at index row * n + col.

def matmul_tensor(n: int = 2) -> np.ndarray:
    """T[i, j, k] = 1 if A-entry i times B-entry j contributes to C-entry k.

    Row-times-column rule: C[r, s] = sum_m A[r, m] * B[m, s].
    """
    size = n * n
    T = np.zeros((size, size, size), dtype=int) # cube of zeros
    for r in range(n): # row of C (and of A)
        for s in range(n): # column of C (and of B)
            for m in range(n): # position along the dot product
                i = r * n + m # index of A[r, m]
                j = m * n + s # index of B[m, s]
                k = r * n + s # index of C[r, s]
                T[i, j, k] = 1
                # print(f"i={i:>2} (A[{r},{m}]), j={j:>2} (B[{m},{s}]), k={k:>2} (C[{r},{s}]) -> T[{i},{j},{k}] = 1")
    return T


def reconstruct(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Sum of R rank-one tensors: sum over r of U[:, r] (x) V[:, r] (x) W[:, r]."""
    return np.einsum("ir,jr,kr->ijk", U, V, W)


# Strassen's factor matrices. Column r describes product m_{r+1}:
#   U: coefficients of a11, a12, a21, a22 in the A factor
#   V: coefficients of b11, b12, b21, b22 in the B factor
#   W: how the product is added into c11, c12, c21, c22
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
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * (b11)
    # m3 = (a11) * (b12 - b22)
    # m4 = (a22) * (b21 - b11)
    # m5 = (a11 + a12) * (b22)
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)

    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

def naive_decomposition(T: np.ndarray):
    """One rank-one term per nonzero entry of T: the schoolbook algorithm."""
    entries = np.argwhere(T == 1)
    size, rank = T.shape[0], len(entries)
    print(f"Naive decomposition: {rank} rank-one terms, one per nonzero entry of T; {T.shape} tensor")
    U, V, W = (np.zeros((size, rank), dtype=int) for _ in range(3))
    for r, (i, j, k) in enumerate(entries):
        U[i, r] = V[j, r] = W[k, r] = 1
        # print(f"r={r:>2}; (i, j, k)=({i:>2}, {j:>2}, {k:>2}) -> U[{i},{r}] = V[{j},{r}] = W[{k},{r}] = 1")
    return U, V, W


def part_b(seed: int = 1) -> None:
    print("PART B: matrix multiplication as a tensor")
    T = matmul_tensor(2)
    print(f"Tensor shape {T.shape}, number of 1s = {T.sum()} (one per naive product)")

    print("\n[ Naive decomposition ]")
    U, V, W = naive_decomposition(T)
    naive_reconstructed = reconstruct(U, V, W)
    naive_rank = U.shape[1]
    print(f"Naive rank-{naive_rank} decomposition reproduces T:",
          np.array_equal(naive_reconstructed, T))
    
    print("\n[ Strassen decomposition ]")
    strassen_reconstructed = reconstruct(STRASSEN_U, STRASSEN_V, STRASSEN_W)
    strassen_rank = STRASSEN_U.shape[1]
    print(f"Strassen rank-{strassen_rank} decomposition reproduces T:",
          np.array_equal(strassen_reconstructed, T))

    print("\n[ Strassen as an algorithm ]")
    rng = np.random.default_rng(seed)
    A, B = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    products = (STRASSEN_U.T @ A.flatten()) * (STRASSEN_V.T @ B.flatten())  # multiplications
    C = (STRASSEN_W @ products).reshape(2, 2)                               # additions only
    print("Decomposition used as an algorithm matches A @ B:", np.allclose(C, A @ B))

    print("\nKey idea: fewer rank-one terms means fewer multiplications.")
    print(f"Rank R for an n x n block gives exponent log_n(R): "
          f"log_2({strassen_rank}) = {np.log2(strassen_rank):.3f}")


if __name__ == "__main__":
    part_a()
    part_b()