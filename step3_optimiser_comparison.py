"""
Step 3: compare optimisers on the rank-7 search from step 2.

Runs the same two-phase search (fit, then discretise) with four optimisers:

  gd        plain gradient descent        theta -= lr * grad
  momentum  gradient descent + momentum    keeps a running average of past gradients (0.9)
  rmsprop   RMSprop                        divides each step by a running size of that entry's gradients
  adam      Adam                           momentum and RMSprop's per-entry step size combined

For a fair comparison:
  * every run starts from the same random U, V, W (the seed is reset before each run)
  * each optimiser is tried with several learning rates, and is judged at its best one,
    because a single shared learning rate would mostly measure which optimiser it suits

Reports, per optimiser and learning rate, how many restarts fit TARGET_TENSOR after phase 1,
how many are exact algorithms after rounding, and how many diverged (loss became
inf or NaN). Also saves the median fit loss over the steps for each optimiser at
its best learning rate.

Usage:
    python step3_optimiser_comparison.py
    python step3_optimiser_comparison.py --lrs 0.01 0.03 --restarts 50
    python step3_optimiser_comparison.py --optimisers gd adam
    python step3_optimiser_comparison.py --plot ""          # skip the plot
"""
import argparse
from functools import partial

import torch

from step2_gradient_search import search

OPTIMISERS = {
    "gd": torch.optim.SGD,
    "momentum": partial(torch.optim.SGD, momentum=0.9),
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
}
RECORD_EVERY = 10 # steps between points on the loss curve


def run(name: str, lr: float, args) -> dict:
    """One full search with one optimiser and learning rate. Returns counts and the loss curve."""
    curve_steps, curve_median = [], []

    def record(step: int, fits: torch.Tensor) -> None:
        if step % RECORD_EVERY == 0:
            curve_steps.append(step)
            curve_median.append(fits.nanmedian().item()) # median over restarts; NaN restarts are skipped

    torch.manual_seed(args.seed) # same starting U, V, W for every run
    fit, exact, _ = search(
        args.rank, args.restarts, args.fit_steps, args.disc_steps, lr, args.allowed,
        make_optimiser=OPTIMISERS[name], on_step=record,
    )
    finite = torch.isfinite(fit)
    return {
        "name": name,
        "lr": lr,
        "fit_ok": (fit < 1e-4).sum().item(),
        "exact": exact.sum().item(),
        "diverged": (~finite).sum().item(),
        "best_fit": fit[finite].min().item() if finite.any() else float("inf"),
        "curve": (curve_steps, curve_median),
    }


def plot_curves(best: list, args) -> None:
    import matplotlib
    matplotlib.use("Agg") # write a file; no window needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for result in best:
        steps, median = result["curve"]
        ax.plot(steps, median, label=f"{result['name']} (lr={result['lr']:g})")
    ax.axvline(args.fit_steps, color="grey", linestyle="--", linewidth=1)
    ax.text(args.fit_steps, ax.get_ylim()[1], " discretise starts", va="top", fontsize=9, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("median fit loss over restarts")
    ax.set_title(f"Rank-{args.rank} search, {args.restarts} restarts, best learning rate per optimiser")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nLoss curves saved to {args.plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optimisers on the 2x2 matmul decomposition search.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--disc-steps", type=int, default=3000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--optimisers", nargs="+", default=list(OPTIMISERS), choices=list(OPTIMISERS))
    parser.add_argument("--allowed", type=float, nargs="+", default=[-1, 0, 1])
    parser.add_argument("--plot", default="step3_loss_curves.png", help='output image; "" to skip')
    args = parser.parse_args()

    print(f"Rank {args.rank}, seed {args.seed}, {args.restarts} restarts, "
          f"{args.fit_steps} fit + {args.disc_steps} discretise steps\n")
    print(f"{'optimiser':<9} {'lr':>6} {'fit ok (<1e-4)':>15} {'exact':>9} {'diverged':>9} {'best fit loss':>14}")

    results = []
    for name in args.optimisers:
        for lr in args.lrs:
            r = run(name, lr, args)
            results.append(r)
            print(f"{name:<9} {lr:>6g} "
                  f"{r['fit_ok']:>9}/{args.restarts:<5} "
                  f"{r['exact']:>5}/{args.restarts:<3} "
                  f"{r['diverged']:>9} "
                  f"{r['best_fit']:>14.2e}")
        print()

    # Best learning rate per optimiser: most exact algorithms, then most good fits
    best = [max((r for r in results if r["name"] == name), key=lambda r: (r["exact"], r["fit_ok"]))
            for name in args.optimisers]
    print("Best learning rate per optimiser:")
    for r in sorted(best, key=lambda r: -r["exact"]):
        print(f"  {r['name']:<9} lr={r['lr']:<6g} exact {r['exact']}/{args.restarts}")

    if args.plot:
        plot_curves(best, args)


if __name__ == "__main__":
    main()
