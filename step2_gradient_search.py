"""
Step 2: rediscover Strassen's algorithm with gradient descent.

The factor matrices U, V, W are hidden. Starting from random numbers, the
search looks for factors whose rank-one terms rebuild the 2x2 matrix
multiplication tensor T.

Phase 1 (fit)         minimise || sum_r U_r (x) V_r (x) W_r - T ||^2
Phase 2 (discretise)  add a slowly growing penalty that pulls every entry
                      towards an allowed value (default -1, 0, 1), then
                      round and check whether the result is EXACT.
                      An exact rounded solution is a genuine algorithm.

Many random restarts run in parallel as a batch, for rank 8 (easy),
rank 7 (Strassen's rank, harder) and rank 6 (proven impossible).

Usage:
    python step2_gradient_search.py
    python step2_gradient_search.py --seed 3 --restarts 50
    python step2_gradient_search.py --disc-steps 0          # skip the discretise phase
    python step2_gradient_search.py --allowed 0 1           # forbid negative coefficients
    python step2_gradient_search.py --ranks 7               # search one rank only
"""
import argparse

import numpy as np
import torch

A_NAMES = ["a11", "a12", "a21", "a22"]
B_NAMES = ["b11", "b12", "b21", "b22"]
C_NAMES = ["c11", "c12", "c21", "c22"]


def matmul_tensor(n: int = 2) -> torch.Tensor:
    """T[i, j, k] = 1 if A-entry i times B-entry j contributes to C-entry k."""
    size = n * n
    T = torch.zeros(size, size, size)
    for r in range(n):
        for s in range(n):
            for m in range(n):
                T[r * n + m, m * n + s, r * n + s] = 1.0
    return T


T = matmul_tensor(2)


def reconstruct(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Batched sum of rank-one terms.

    U, V, W have shape (batch, N_ENTRIES, rank).
    Returns shape (batch, N_ENTRIES, N_ENTRIES, N_ENTRIES): one rebuilt tensor per restart.
    """
    return torch.einsum("bir,bjr,bkr->bijk", U, V, W)


def fit_loss(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Squared error between each rebuilt tensor and the target T.

    U, V, W have shape (restarts, N_ENTRIES, rank).
    Returns shape (restarts,): one loss value per restart.
    A loss of exactly 0 means that restart's U, V, W form a correct algorithm.
    """
    rebuilt = reconstruct(U, V, W)      # (restarts, N_ENTRIES, N_ENTRIES, N_ENTRIES)
    error = rebuilt - T                 # T is broadcast across all restarts
    squared = error ** 2                # every cell's error becomes positive
    return squared.sum(dim=(1, 2, 3))   # sum over all cells, keep the restart axis


def discrete_penalty(X: torch.Tensor, allowed: list) -> torch.Tensor:
    """Zero exactly when every entry equals one of the allowed values."""
    penalty = torch.ones_like(X)
    for value in allowed:
        penalty = penalty * (X - value) ** 2
    return penalty.sum(dim=(1, 2))


def round_to_allowed(X: torch.Tensor, allowed: list[float]) -> torch.Tensor:
    """Snap every entry to the nearest allowed value.

    Unlike torch.round, this respects any allowed set, e.g. [0, 1].
    X has any shape; the result has the same shape as X.
    """
    values = torch.tensor(allowed, dtype=X.dtype)   # allowed values as a tensor, same number type as X

    distances = (X.unsqueeze(-1) - values).abs()    # add an axis at the end, then compare every entry
                                                    # with every allowed value: shape (*X.shape, len(allowed))
    nearest = distances.argmin(dim=-1)              # position of the closest allowed value for each entry

    return values[nearest]                          # swap each position for its allowed value


def search(
        rank: int,
        restarts: int,
        fit_steps: int,
        disc_steps: int,
        lr: float,
        allowed: list
    ):
    """Run all restarts in parallel. Returns fit losses, exactness flags and rounded factors."""
    N_ENTRIES = T.shape[0]  # entries per matrix (n * n): rows of U, V and W
    N_FACTORS = 3           # one factor matrix each for A, B and C: U, V, W
    INIT_SCALE = 0.7        # starting size of the random factors; tuned by trial
    params = [(torch.randn(restarts, N_ENTRIES, rank) * INIT_SCALE).requires_grad_() for _ in range(N_FACTORS)]
    optimiser = torch.optim.Adam(params, lr=lr)

    # Phase 1: plain fit
    # Adjust U, V, W so that each restart's rebuilt tensor moves closer to T.
    for _ in range(fit_steps):
        optimiser.zero_grad()               # clear gradients from the previous step (they accumulate otherwise)
        loss = fit_loss(*params).sum()      # forward pass: one loss per restart, summed into a scalar
                                            # summing is safe: each restart only affects its own loss
        loss.backward()                     # compute the gradient of the loss for every value in U, V, W
        optimiser.step()                    # the optimiser updates every value using its gradient

    # Record each restart's loss at the end of phase 1 (reported as "fit ok" and "best fit loss").
    # Gradients are switched off: this is only a measurement, not a training step.
    with torch.no_grad():
        fit = fit_loss(*params)             # shape (restarts,): one loss per restart

    # Phase 2: keep fitting while slowly strengthening the pull towards allowed values
    for step in range(disc_steps):
        strength = 0.01 * (1 + step / 100)       # penalty weight, grows over time
        optimiser.zero_grad()

        current_fit = fit_loss(*params)          # (restarts,): how wrong each recipe is
        penalty = sum(discrete_penalty(p, allowed) for p in params)  # (restarts,): distance from allowed values, summed over U, V, W
        loss = current_fit + strength * penalty  # (restarts,): balance correctness and clean values

        loss.sum().backward()
        optimiser.step()

    # Round and check exactness
    # Gradients are switched off: this is only a measurement, not a training step.
    with torch.no_grad():
        # Snap every value in U, V and W to its nearest allowed value (e.g. 0.97 -> 1)
        rounded = [round_to_allowed(p, allowed) for p in params]   # [U, V, W], each (restarts, N_ENTRIES, rank)

        # Rebuild each restart's tensor from its rounded recipe and compare with T.
        # == is safe here: rounded values are exact whole numbers, so no tolerance is needed.
        matches = reconstruct(*rounded) == T                       # (restarts, N_ENTRIES, N_ENTRIES, N_ENTRIES): True/False per cell
        exact = matches.all(dim=(1, 2, 3))                         # (restarts,): True only if all cells match

    # fit:     phase 1 loss per restart            -> "fit ok" and "best fit loss" columns
    # exact:   True where rounding gave an exact algorithm -> "exact after rounding" column
    # rounded: the rounded U, V, W                  -> used to print an example algorithm
    return fit, exact, rounded


def works_on_random_matrices(U: np.ndarray, V: np.ndarray, W: np.ndarray, trials: int = 5) -> bool:
    """Use a decomposition as an algorithm and compare with numpy."""
    rng = np.random.default_rng()
    for _ in range(trials):
        A, B = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
        products = (U.T @ A.flatten()) * (V.T @ B.flatten())
        if not np.allclose((W @ products).reshape(2, 2), A @ B):
            return False
    return True


def format_combination(coeffs, names) -> str:
    """Turn coefficients like [1, 0, 0, -1] into '(a11-a22)'."""
    terms = []
    for c, name in zip(coeffs, names):
        c = int(c)
        if c == 0:
            continue
        sign = "+" if c > 0 else "-"
        size = "" if abs(c) == 1 else str(abs(c))
        terms.append(f"{sign}{size}{name}")
    text = "".join(terms).lstrip("+")
    return f"({text})" if len(terms) > 1 else text


def print_algorithm(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> None:
    rank = U.shape[1]
    product_names = [f"M{r + 1}" for r in range(rank)]
    for r in range(rank):
        print(f"  M{r + 1} = {format_combination(U[:, r], A_NAMES)} * "
              f"{format_combination(V[:, r], B_NAMES)}")
    for k, name in enumerate(C_NAMES):
        print(f"  {name} = {format_combination(W[k], product_names).strip('()')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for low-rank 2x2 matmul decompositions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 7, 6])
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--disc-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--allowed", type=float, nargs="+", default=[-1, 0, 1])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Seed {args.seed}, {args.restarts} restarts, allowed values {args.allowed}\n")
    print(f"{'rank':>4} "
          f"{'fit ok (<1e-4)':>15} "
          f"{'best fit loss':>14} "
          f"{'exact after rounding':>21}")

    example = None
    for rank in args.ranks:
        fit, exact, rounded = search(
            rank, args.restarts, args.fit_steps, args.disc_steps, args.lr, args.allowed
        )
        fitted = (fit < 1e-4).sum().item()
        print(f"{rank:>4} "
              f"{fitted:>9}/{args.restarts:<5} "
              f"{fit.min().item():>14.2e} "
              f"{exact.sum().item():>15}/{args.restarts}")
        if rank == 7 and exact.any():
            i = exact.nonzero()[0].item()
            example = tuple(x[i].numpy() for x in rounded)

    if example is not None:
        U, V, W = example
        print("\nAn exact rank-7 algorithm found from random noise:")
        print_algorithm(U, V, W)
        print("Works on random matrices:", works_on_random_matrices(U, V, W))
    elif 7 in args.ranks:
        print("\nNo exact rank-7 solution this run. Try another seed or more restarts.")


if __name__ == "__main__":
    main()