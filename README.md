# Fast matrix multiplication explorations

Small, self-contained experiments on why matrix multiplication can be done with fewer than n³ multiplications. They start with Strassen's algorithm and end with rediscovering it from random noise using gradient descent.

I started this after reading [*Improving the matrix multiplication exponent with modern optimization and AlphaEvolve*](https://arxiv.org/abs/2608.16884) (Dupont et al., 2026). At the time I was studying machine learning as an online MCS student at UIUC, and I wanted to see how far the optimisation tools from the course (gradient descent, regularisation, non-convex objectives) could go on this problem at toy scale.

This is a learning project. Every result here is already known (Strassen, 1969). The aim is to make the ideas concrete and runnable, not to find new algorithms.

## The steps

Run them in order. Each script is independent and prints its own explanation.

| Step | Script | What it shows |
|---|---|---|
| 1 | [`step1_count_and_tensor.py`](step1_count_and_tensor.py) | Counts multiplications for naive vs Strassen and shows the exponent falling from 3 to log₂7 ≈ 2.807. Then it rewrites 2×2 matrix multiplication as a 4×4×4 tensor, where an algorithm is a decomposition of that tensor into rank-one terms. |
| 2 | [`step2_gradient_search.py`](step2_gradient_search.py) | Searches for rank-8, rank-7 and rank-6 decompositions from random starting points with gradient descent, then rounds to {-1, 0, 1} and checks whether the result is an exact algorithm. |

### Side experiments

Small scripts I wrote to understand individual pieces. They aren't needed for the main steps.

| Script | What it shows |
|---|---|
| [`extras/trace_strassen.py`](extras/trace_strassen.py) | Strassen's recursion traced line by line: the only multiplications happen at the leaves, and everything else is additions. |
| [`extras/einsum_vs_loops.py`](extras/einsum_vs_loops.py) | Rebuilds the tensor from Strassen's factors with plain Python loops and with `np.einsum`, checks that the two match exactly, and times them. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Step 1 and the side experiments need only NumPy. Step 2 also needs PyTorch.

## Key idea

Multiplying two 2×2 matrices can be written as a 4×4×4 tensor T of 0s and 1s. Any way of writing T as a sum of R rank-one terms

```
T = Σ_r  U[:, r] ⊗ V[:, r] ⊗ W[:, r]
```

is an algorithm that uses R multiplications. Column r of U says which entries of A to add up, column r of V does the same for B, and column r of W says where the product goes in C. The schoolbook method uses R = 8, and Strassen found R = 7. Applied recursively to blocks, R = 7 gives O(n^log₂7) ≈ O(n^2.807).

Finding faster algorithms is therefore a search for low-rank tensor decompositions. That is the problem the AlphaEvolve paper attacks at much larger sizes.

## Example output

Step 1, part A: the exponent:

```
$ python step1_count_and_tensor.py
    n      naive   strassen   ratio  exp(naive)  exp(strassen)
    2          8          7   0.875       3.000          2.807
    4         64         49   0.766       3.000          2.807
   ...
   64     262144     117649   0.449       3.000          2.807
```

Step 2: gradient descent rediscovers a rank-7 algorithm from random noise:

```
$ python step2_gradient_search.py
Seed 0, 200 restarts, allowed values [-1, 0, 1]

rank  fit ok (<1e-4)  best fit loss  exact after rounding
   8       184/200         2.11e-13              46/200
   7       128/200         3.17e-12              80/200
   6         0/200         1.00e+00               0/200

An exact rank-7 algorithm found from random noise:
  M1 = -a12 * (b12+b22)
  M2 = (a21-a22) * b21
  M3 = (a12-a21) * (-b12+b21)
  M4 = (-a11+a21) * (b11+b12)
  M5 = (a11-a12) * -b12
  M6 = -a21 * (b11+b21)
  M7 = (a12-a22) * (b21+b22)
  c11 = M3-M4+M5-M6
  c12 = -M1-M5
  c21 = -M2-M6
  c22 = -M1+M2+M3-M7
Works on random matrices: True
```

The algorithm it found isn't Strassen's exact formulas. It's a different rank-7 decomposition that is just as valid.

Side experiments:

```
$ python extras/trace_strassen.py
...
Strassen result: [[19, 22], [43, 50]]
numpy A @ B:     [[19, 22], [43, 50]]
Match: True
Multiplications: 7 (naive would use 8)
```

In `extras/einsum_vs_loops.py`, `np.einsum` was about 37× faster than plain loops on my machine. Your numbers will differ.

## How step 2 connects to the ML course

- **Phase 1** is ordinary gradient descent (Adam) on a squared-error loss: ‖Σ_r U_r ⊗ V_r ⊗ W_r − T‖².
- **Phase 2** adds a penalty that grows over time, Π_v (x − v)², which is zero only when every entry is one of the allowed values. It plays the same role as L1/L2 regularisation, except that it pulls entries towards a set of values instead of towards zero.
- **The loss is non-convex,** so the script runs many random restarts in parallel as one batch. Rank 8 fits most easily (184/200), rank 7 fits less often (128/200), and rank 6 never fits, because 7 multiplications is proven to be the minimum for 2×2 matrices. Surprisingly, rank 8 gives *fewer* exact algorithms after rounding (46 vs 80). A likely reason: with a spare rank-one term, there are many continuous solutions that fit well but don't lie near whole numbers. Rank 7 has less slack, so its solutions tend to already be close to {-1, 0, 1}.

## Licence

MIT. See [LICENSE](LICENSE).
