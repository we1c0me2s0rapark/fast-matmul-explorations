"""
Side experiment: trace Strassen's recursion step by step on small integer matrices.

Shows the three phases at every level:
  DOWN   split into 2x2 blocks and form sums/differences (additions only)
  LEAF   multiply two single numbers (the only multiplications)
  UP     combine m1..m7 into C (additions only)

Usage:
    python extras/trace_strassen.py          # 2x2 example, 7 multiplications
    python extras/trace_strassen.py --n 4    # 4x4 example, 49 multiplications
"""
import argparse

import numpy as np

multiplications = 0


def fmt(M: np.ndarray) -> str:
    """Compact one-line matrix, e.g. [[1, 2], [3, 4]]."""
    return str(M.tolist()) if M.size > 1 else str(M.item())


def strassen_trace(A: np.ndarray, B: np.ndarray, label: str = "top", depth: int = 0) -> np.ndarray:
    global multiplications
    pad = "    " * depth
    n = A.shape[0]

    if n == 1:
        multiplications += 1
        product = A * B
        print(f"multiplication #{multiplications:>3} > "
              f"{pad}{label}: LEAF  {A.item()} x {B.item()} = {product.item()}")
        return product

    h = n // 2
    a11, a12, a21, a22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    b11, b12, b21, b22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    print(f"{pad}{label}: DOWN  {n}x{n}, split into four {h}x{h} blocks, 7 calls")

    inputs = [
        ("m1", a11 + a22, b11 + b22),
        ("m2", a21 + a22, b11),
        ("m3", a11, b12 - b22),
        ("m4", a22, b21 - b11),
        ("m5", a11 + a12, b22),
        ("m6", a21 - a11, b11 + b12),
        ("m7", a12 - a22, b21 + b22),
    ]
    m = {}
    for name, left, right in inputs:
        sub_label = name if depth == 0 else f"{label}.{name}"
        m[name] = strassen_trace(left, right, sub_label, depth + 1)

    C = np.empty((n, n), dtype=A.dtype)
    C[:h, :h] = m["m1"] + m["m4"] - m["m5"] + m["m7"]
    C[:h, h:] = m["m3"] + m["m5"]
    C[h:, :h] = m["m2"] + m["m4"]
    C[h:, h:] = m["m1"] - m["m2"] + m["m3"] + m["m6"]
    print(f"{pad}{label}: UP    combine m1..m7 -> {fmt(C)}")
    return C


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, choices=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.n == 2:
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
    else:
        rng = np.random.default_rng(args.seed)
        A = rng.integers(-3, 4, (args.n, args.n))
        B = rng.integers(-3, 4, (args.n, args.n))

    print(f"A = {fmt(A)}\nB = {fmt(B)}\n")
    C = strassen_trace(A, B)
    print(f"\nStrassen result: {fmt(C)}")
    print(f"numpy A @ B:     {fmt(A @ B)}")
    print(f"Match: {np.array_equal(C, A @ B)}")
    print(f"Multiplications: {multiplications} (naive would use {args.n ** 3})")


if __name__ == "__main__":
    main()