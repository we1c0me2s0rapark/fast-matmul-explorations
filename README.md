# Fast matrix multiplication explorations

Small, self-contained experiments on why matrix multiplication can be done with fewer than n³ multiplications. They start with Strassen's algorithm, rediscover it from random noise using gradient descent, and then use that search to compare optimisers, empirical risk minimisation (ERM) and regularisation.

I started this after reading [*Improving the matrix multiplication exponent with modern optimization and AlphaEvolve*](https://arxiv.org/abs/2608.16884) (Dupont et al., 2026). At the time I was studying machine learning as an online MCS student at UIUC, and I wanted to see how far the optimisation tools from the course (gradient descent, regularisation, non-convex objectives) could go on this problem at toy scale.

This is a learning project. Every result here is already known (Strassen, 1969). The aim is to make the ideas concrete and runnable, not to find new algorithms.

## The steps

Run them in order. Each script is independent and prints its own explanation.

| Step | Script | What it shows |
|---|---|---|
| 1 | [`step1_count_and_tensor.py`](step1_count_and_tensor.py) | Counts multiplications for naive vs Strassen and shows the exponent falling from 3 to log₂7 ≈ 2.807. Then it rewrites 2×2 matrix multiplication as a 4×4×4 tensor, where an algorithm is a decomposition of that tensor into rank-one terms. |
| 2 | [`step2_gradient_search.py`](step2_gradient_search.py) | Searches for rank-8, rank-7 and rank-6 decompositions from random starting points with gradient descent, then rounds to {-1, 0, 1} and checks whether the result is an exact algorithm. |
| 3 | [`step3_optimiser_comparison.py`](step3_optimiser_comparison.py) | Runs the rank-7 search with plain gradient descent, momentum, RMSprop and Adam, each over a range of learning rates and from the same starting points. Compares how many exact algorithms each finds and saves their loss curves. |
| 4 | [`step4_erm_regularisation_noise.py`](step4_erm_regularisation_noise.py) | Three experiments on the rank-7 search. A: empirical risk minimisation (ERM) on a fixed set of N random matrix pairs, with and without the whole-number regulariser; below N = 16 it overfits. B: the same comparison on the true risk (the exact tensor loss). C: whether gradient noise from a fresh minibatch every step helps each optimiser find exact algorithms. Saves `step4_overfitting.png` and `step4_noise.png`. |

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

Step 1 and the side experiments need only NumPy. Steps 2–4 also need PyTorch, and steps 3 and 4 use matplotlib for their charts. On a 20-core CPU, step 3 takes about 1.5 minutes and step 4 about 3 minutes.

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

## What steps 3 and 4 found

All runs search for rank-7 algorithms from 200 random starting points. "Exact" means the rounded U, V, W rebuild the target tensor exactly, so the algorithm is correct for every pair of 2×2 matrices. Ranges cover the random seeds that were checked (2–3 per result).

**Regularisation decides whether you get an algorithm at all.** Without the whole-number penalty, no run found an exact algorithm, with or without ERM, on any seed. Many real-valued solutions fit perfectly (scale one column up and another down), so plain fitting stops at messy points that rounding breaks.

| Training loss | Unregularised | Regularised |
|---|---|---|
| ERM, N < 16 pairs | 0 | 0 |
| ERM, N = 32 or 64 pairs | 0 | 53–80 |
| True risk (no ERM) | 0 | 68–83 |

**ERM overfits below 16 pairs.** With 4–12 training pairs, up to 175 of 200 unregularised runs fitted the training pairs perfectly, yet none worked on new matrices (seed 0). The error is a bilinear map with 16 degrees of freedom per output entry, so fewer pairs can't pin down the algorithm. With enough pairs, ERM performs about as well as the true risk.

**RMSprop finds the most algorithms, and the most precise fit doesn't win.** With the regularised true risk, each at its best learning rate:

| Optimiser | Best lr | Exact (exact gradient) | Exact (fresh minibatch of 128) |
|---|---|---|---|
| RMSprop | 0.03 | 177–187 | 198 |
| Adam | 0.1 | 167–170 | 197–199 |
| Momentum | 0.003 | 95–110 | 93 |
| Plain GD | 0.1 | 91–98 | diverges |

Momentum fits the loss most precisely but finds about half as many algorithms as RMSprop, which never fits below 1e-4. Gradient noise helps the adaptive optimisers and breaks plain gradient descent. Why RMSprop does best is still an open question; one guess is that its evenly sized steps keep it from settling at messy points.

### How steps 3 and 4 connect to the ML course

- **MLE and MAP.** Squared error is the negative log-likelihood under Gaussian noise, so Phase 1 is MLE. The whole-number penalty is the negative log of a prior with peaks at −1, 0 and 1, so Phase 2 is MAP. Here the MLE isn't unique, so the prior does the real work.
- **Why not Ridge or Lasso.** Both pull weights towards 0 only. A valid algorithm needs −1, 0 and 1, so the penalty (x+1)²·x²·(x−1)² has a valley at each.
- **ERM and the true risk.** For random matrices with standard normal entries, the expected squared error of an algorithm equals the tensor loss exactly, so steps 2 and 3 minimise the true risk. Step 4 replaces it with an average over sampled pairs, which is ERM.
- **Optimisers.** Plain gradient descent, momentum, RMSprop and Adam differ only in how they turn gradients into steps, and that changes which solution they land on, not just how fast.

## Licence

MIT. See [LICENSE](LICENSE).
