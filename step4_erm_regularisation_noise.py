"""
Step 4: empirical risk minimisation (ERM), regularisation and gradient noise.

Steps 2 and 3 train on fit_loss, the squared error over the 64 cells of TARGET_TENSOR.
For random matrices A, B with standard normal entries, that is exactly the expected
error of the algorithm on new inputs (the true risk):

    E_{A,B} || ALG(A, B) - A @ B ||^2  =  sum_ijk (rebuilt - TARGET_TENSOR)^2  =  fit_loss

So steps 2 and 3 minimise the true risk, computed exactly. The regulariser is the
phase 2 whole-number penalty; "unregularised" runs replace phase 2 with more plain
fitting, so both versions train for the same number of steps.

Experiment A  ERM on a fixed training set of N pairs, unregularised vs regularised
              The error is a bilinear map with 16 degrees of freedom per output entry,
              so with fewer than 16 pairs U, V, W can reach zero training loss and still
              be wrong (overfitting). With 16 or more random pairs, zero training loss
              forces exactness.

Experiment B  the true risk (no ERM), unregularised vs regularised
              The same comparison with the exact loss used in steps 2 and 3.

Experiment C  regularised true risk with gradient noise: a fresh minibatch every step
              (stochastic gradient descent). Tests whether noise raises the exact rate,
              as RMSprop's jitter seemed to in step 3. Each optimiser runs at its best
              learning rate from step 3.

Every run starts from the same random U, V, W, and every result is judged against the
exact TARGET_TENSOR.

Usage:
    python step4_erm_regularisation_noise.py
    python step4_erm_regularisation_noise.py --experiments a b
    python step4_erm_regularisation_noise.py --seeds 0                 # one seed, about 3 minutes
    python step4_erm_regularisation_noise.py --sizes 8 16 --batch-sizes 32 --restarts 50
"""
import argparse

import torch

from step2_gradient_search import fit_loss, search
from step3_optimiser_comparison import OPTIMISERS

# Best learning rate per optimiser in step 3 (seeds 0, 1 and 2 agreed)
BEST_LR = {"gd": 0.1, "momentum": 0.003, "rmsprop": 0.03, "adam": 0.1}
N_ENTRIES = 4   # entries per 2x2 matrix


def sample_pairs(count: int, generator: torch.Generator):
    """Random 2x2 pairs (A, B), flattened to rows of 4, with the true products C = A @ B."""
    A = torch.randn(count, 2, 2, generator=generator)
    B = torch.randn(count, 2, 2, generator=generator)
    return A.reshape(count, N_ENTRIES), B.reshape(count, N_ENTRIES), (A @ B).reshape(count, N_ENTRIES)


def sampled_loss(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, a, b, c) -> torch.Tensor:
    """Squared error of every restart's algorithm on the given pairs, averaged over pairs.

    U, V, W have shape (restarts, N_ENTRIES, rank); a, b, c have shape (pairs, N_ENTRIES).
    Returns shape (restarts,). Its expected value over random pairs equals fit_loss.
    """
    # Run each restart's U, V, W as an algorithm on every pair, exactly as step 1 part B does for one pair
    products = torch.einsum("ni,bir->bnr", a, U) * torch.einsum("nj,bjr->bnr", b, V)  # (restarts, pairs, rank): one multiplication per term
    predicted = torch.einsum("bnr,bkr->bnk", products, W)                              # (restarts, pairs, N_ENTRIES): combine into C
    return ((predicted - c) ** 2).sum(dim=2).mean(dim=1)                               # sum over the 4 entries of C, average over pairs


def run(args, make_optimiser, lr: float, loss_fn=None, regularised: bool = True) -> dict:
    """One full search. Reports fits at the end of training (before rounding) and exactness after rounding."""
    if loss_fn is None:
        loss_fn = fit_loss
    final = {}

    # search() returns only the rounded U, V, W, so keep a handle on the unrounded ones
    # to measure their true risk after training
    def tracked_loss(U, V, W):
        final["params"] = (U, V, W)         # the live U, V, W, updated in place by the optimiser
        return loss_fn(U, V, W)

    def record(step: int, losses: torch.Tensor) -> None:
        final["train"] = losses             # overwritten every step: ends as the last training loss

    torch.manual_seed(args.seed)            # same starting U, V, W for every run
    phase1_fit, exact, _ = search(
        args.rank, args.restarts, args.fit_steps, args.disc_steps, lr, args.allowed,
        make_optimiser=make_optimiser, on_step=record, loss_fn=tracked_loss,
        penalty_scale=1.0 if regularised else 0.0,
    )
    with torch.no_grad():
        true_risk = fit_loss(*final["params"])   # (restarts,): expected error on new random matrices, at the end of training
    return {
        "train_ok": (final["train"] < 1e-4).sum().item(),
        "true_risk_ok": (true_risk < 1e-4).sum().item(),
        "phase1_true_risk_ok": (phase1_fit < 1e-4).sum().item(),
        "exact": exact.sum().item(),
        "diverged": (~torch.isfinite(true_risk)).sum().item(),
    }


def print_header(first_column: str) -> None:
    print(f"\n{'':>10} {'------- unregularised -------':>30} {'-------- regularised --------':>30}")
    print(f"{first_column:>10} {'train ok':>10} {'true-risk ok':>12} {'exact':>7} "
          f"{'train ok':>10} {'true-risk ok':>12} {'exact':>7}")


def print_row(label: str, plain: dict, reg: dict) -> None:
    def cells(r):
        return f"{r['train_ok']:>10} {r['true_risk_ok']:>12} {r['exact']:>7}"
    print(f"{label:>10} {cells(plain)} {cells(reg)}")


def experiment_a(args) -> list:
    print(f"EXPERIMENT A: ERM on a fixed training set of N pairs ({args.ab_optimiser}, lr={args.ab_lr:g})")
    print(f"  counts out of {args.restarts} restarts, at the end of training:")
    print("  train ok:     training loss < 1e-4")
    print("  true-risk ok: error on new matrices < 1e-4")
    print("  exact:        an exact algorithm after rounding to the allowed values")
    print_header("N")

    make_optimiser = OPTIMISERS[args.ab_optimiser]
    rows = []
    for n in args.sizes:
        # The training set: N pairs drawn once and reused at every step, which is what makes this ERM.
        # Its own generator keeps the starting U, V, W unchanged, and both columns train on the same pairs.
        a, b, c = sample_pairs(n, torch.Generator().manual_seed(args.seed + 1000))

        # The training loss: error on these N pairs only (a=a etc. fixes this N's pairs inside the function)
        def erm_loss(U, V, W, a=a, b=b, c=c):
            return sampled_loss(U, V, W, a, b, c)

        # Same pairs, same starting point, same optimiser; only the phase 2 penalty differs
        plain = run(args, make_optimiser, args.ab_lr, erm_loss, regularised=False)
        reg = run(args, make_optimiser, args.ab_lr, erm_loss, regularised=True)
        print_row(str(n), plain, reg)
        rows.append((n, plain, reg))
    print()
    return rows


def experiment_b(args) -> tuple:
    print(f"EXPERIMENT B: the true risk, no ERM ({args.ab_optimiser}, lr={args.ab_lr:g})")
    print("  same columns as experiment A; the training loss here is the true risk itself")
    print_header("")

    make_optimiser = OPTIMISERS[args.ab_optimiser]
    plain = run(args, make_optimiser, args.ab_lr, regularised=False)
    reg = run(args, make_optimiser, args.ab_lr, regularised=True)
    print_row("true risk", plain, reg)
    print()
    return plain, reg


def experiment_c(args) -> dict:
    print("EXPERIMENT C: regularised true risk, exact gradient vs a fresh minibatch every step (SGD)")
    print("  each optimiser at its best lr from step 3; true-risk ok is measured after phase 1")
    print(f"\n{'optimiser':<9} {'lr':>6} {'gradient from':>16} {'true-risk ok':>14} {'exact':>9} {'diverged':>9}")

    results = {}
    for name in args.optimisers:
        lr = BEST_LR[name]
        r = run(args, OPTIMISERS[name], lr)
        results[name] = [("exact loss", r)]
        print(f"{name:<9} {lr:>6g} {'exact loss':>16} {r['phase1_true_risk_ok']:>10}/{args.restarts:<3} "
              f"{r['exact']:>5}/{args.restarts:<3} {r['diverged']:>9}")
        for size in args.batch_sizes:
            generator = torch.Generator().manual_seed(args.seed + 2000)

            def fresh_batch_loss(U, V, W, size=size, generator=generator):
                return sampled_loss(U, V, W, *sample_pairs(size, generator))

            r = run(args, OPTIMISERS[name], lr, fresh_batch_loss)
            label = f"batch of {size}"
            results[name].append((label, r))
            print(f"{'':<9} {'':>6} {label:>16} {r['phase1_true_risk_ok']:>10}/{args.restarts:<3} "
                  f"{r['exact']:>5}/{args.restarts:<3} {r['diverged']:>9}")
        print()
    return results


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
# Colours: the first three slots of a colour-blind-safe categorical palette for the
# line chart, and three steps of one blue ramp (light = small batch) for the bars.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BATCH_BLUES = ["#86b6ef", "#3987e5", "#1c5cab"]
OPTIMISER_NAMES = {"gd": "Plain GD", "momentum": "Momentum", "rmsprop": "RMSprop", "adam": "Adam"}


def new_axes(figsize):
    import matplotlib
    matplotlib.use("Agg")                   # write a file; no window needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, title: str, subtitle: str, path: str, top: float) -> str:
    """Title and subtitle above the legend, left-aligned with the plot, then save."""
    fig.subplots_adjust(top=top)
    x0 = ax.get_position().x0
    fig.text(x0, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(x0, 0.91, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2, wrap=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    return path


def spread(values: list) -> tuple:
    """Mean, minimum and maximum of one count across seeds."""
    return sum(values) / len(values), min(values), max(values)


def seed_text(seeds: list, range_mark: str) -> str:
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return f"mean of seeds {', '.join(map(str, seeds))} ({range_mark}: min to max)"


def plot_overfitting(rows_by_seed: list, b_by_seed: list, args) -> str:
    """Experiments A and B: training fit vs true fit vs exact algorithms, by training set size."""
    fig, ax = new_axes((8, 4.8))
    sizes = [n for n, _, _ in rows_by_seed[0]]
    x = list(range(len(sizes)))
    series = [   # (label, how to read the count from (plain, reg), marker, marker size)
        ("Fits the N training pairs (unregularised)", lambda plain, reg: plain["train_ok"], "o", 9),
        ("Works on new matrices (unregularised)", lambda plain, reg: plain["true_risk_ok"], "s", 8),
        ("Exact algorithm after rounding (regularised)", lambda plain, reg: reg["exact"], "D", 6),
    ]
    for (label, read, marker, size), colour in zip(series, SERIES):
        stats = [spread([read(rows[i][1], rows[i][2]) for rows in rows_by_seed]) for i in range(len(sizes))]
        means = [m for m, _, _ in stats]
        if len(rows_by_seed) > 1:
            ax.fill_between(x, [lo for _, lo, _ in stats], [hi for _, _, hi in stats],
                            color=colour, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, means, color=colour, linewidth=2, marker=marker, markersize=size,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)

    labels = [str(n) for n in sizes]
    if b_by_seed:                           # the true risk: the N -> infinity reference, set apart
        x_true = len(sizes) + 0.6
        ax.axvline(len(sizes) - 0.2, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
        for k, ((_, read, marker, size), colour) in enumerate(zip(series, SERIES)):
            mean, lo, hi = spread([read(plain, reg) for plain, reg in b_by_seed])
            xk = x_true + (k - 1) * 0.12    # small sideways offset so close values stay visible
            if len(b_by_seed) > 1:
                ax.plot([xk, xk], [lo, hi], color=colour, linewidth=1.5, alpha=0.5, zorder=2)
            ax.plot([xk], [mean], color=colour, marker=marker, markersize=size,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=3)
        x = x + [x_true]
        labels = labels + ["true risk\n(no ERM)"]

    below_16 = [i for i, n in enumerate(sizes) if n < 16]
    if below_16:                            # fewer than 16 pairs cannot pin down the algorithm
        edge = below_16[-1] + 0.5
        ax.axvspan(-0.5, edge, color=GRID, alpha=0.35, linewidth=0, zorder=0)
        ax.text((edge - 0.5) / 2, args.restarts * 0.32, "fewer than 16 pairs:\nperfect fit possible,\nwrong algorithm",
                ha="center", va="center", fontsize=9, color=INK_2)

    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(-8, args.restarts * 1.05)
    ax.set_xlabel("training pairs N (fixed training set, reused every step)", color=INK_2)
    ax.set_ylabel(f"restarts (out of {args.restarts})", color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, frameon=False, fontsize=9,
              labelcolor=INK_2, borderaxespad=0)
    return finish(fig, ax, "Fitting the training data is not the same as finding the algorithm",
                  f"Rank-{args.rank} search, {OPTIMISER_NAMES[args.ab_optimiser]} (lr {args.ab_lr:g}), "
                  f"{seed_text(args.seeds, 'bands and whiskers')}.\nUnregularised runs find no exact algorithm at any N.",
                  f"{args.plot}_overfitting.png", top=0.70)


def plot_noise(results_by_seed: list, args) -> str:
    """Experiment C: exact algorithms with the exact gradient vs fresh minibatches, per optimiser."""
    fig, ax = new_axes((8, 4.6))
    names = list(results_by_seed[0])
    labels = [label for label, _ in results_by_seed[0][names[0]]]
    colours = [MUTED] + BATCH_BLUES[:len(labels) - 1]
    width = 0.8 / len(labels)

    for i, (label, colour) in enumerate(zip(labels, colours)):
        offsets = [g + (i - (len(labels) - 1) / 2) * width for g in range(len(names))]
        stats = [spread([results[name][i][1]["exact"] for results in results_by_seed]) for name in names]
        legend = "exact gradient (no noise)" if label == "exact loss" else label.replace("batch of", "minibatch of")
        ax.bar(offsets, [m for m, _, _ in stats], width=width, color=colour, edgecolor=SURFACE, linewidth=2,
               label=legend, zorder=3)
        if len(results_by_seed) > 1:        # whiskers: min to max across seeds
            ax.vlines(offsets, [lo for _, lo, _ in stats], [hi for _, _, hi in stats],
                      color=INK_2, linewidth=1, zorder=4)
        for x, name, (mean, _, hi) in zip(offsets, names, stats):   # explain zero bars instead of leaving them blank
            if hi == 0:
                diverged = sum(results[name][i][1]["diverged"] for results in results_by_seed) / len(results_by_seed)
                note = "diverged" if diverged > args.restarts / 2 else "0"
                ax.text(x, 4, note, rotation=90 if note == "diverged" else 0, ha="center", va="bottom",
                        fontsize=8, color=INK_2)

    ax.set_xticks(range(len(names)), [f"{OPTIMISER_NAMES[n]}\n(lr {BEST_LR[n]:g})" for n in names])
    ax.set_ylim(0, args.restarts * 1.05)
    ax.set_ylabel(f"exact algorithms (out of {args.restarts})", color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(labels), frameon=False, fontsize=9,
              labelcolor=INK_2, borderaxespad=0, handlelength=1, columnspacing=1.2)
    return finish(fig, ax, "Gradient noise helps adaptive optimisers and breaks plain GD",
                  f"Rank-{args.rank} search, regularised true risk, each optimiser at its best learning rate "
                  f"from step 3,\n{seed_text(args.seeds, 'whiskers')}.", f"{args.plot}_noise.png", top=0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description="ERM vs true risk, with and without regularisation, and gradient noise.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="random seeds; every experiment runs once per seed and the charts show the mean")
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--disc-steps", type=int, default=3000)
    parser.add_argument("--allowed", type=float, nargs="+", default=[-1, 0, 1])
    parser.add_argument("--experiments", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 15, 16, 32, 64],
                        help="training set sizes N for experiment A")
    parser.add_argument("--ab-optimiser", default="adam", choices=list(OPTIMISERS),
                        help="optimiser for experiments A and B; one that fits precisely shows overfitting best")
    parser.add_argument("--ab-lr", type=float, default=0.03)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32, 128],
                        help="minibatch sizes for experiment C")
    parser.add_argument("--optimisers", nargs="+", default=list(OPTIMISERS), choices=list(OPTIMISERS),
                        help="optimisers for experiment C")
    parser.add_argument("--plot", default="step4", help='prefix for the chart files; "" to skip')
    args = parser.parse_args()

    rows_by_seed, b_by_seed, c_by_seed = [], [], []
    for seed in args.seeds:
        args.seed = seed                    # run() and the samplers read args.seed
        print(f"===== Rank {args.rank}, seed {seed}, {args.restarts} restarts, "
              f"{args.fit_steps} fit + {args.disc_steps} discretise steps =====\n")
        if "a" in args.experiments:
            rows_by_seed.append(experiment_a(args))
        if "b" in args.experiments:
            b_by_seed.append(experiment_b(args))
        if "c" in args.experiments:
            c_by_seed.append(experiment_c(args))

    if args.plot:
        if rows_by_seed:
            print(f"Overfitting chart saved to {plot_overfitting(rows_by_seed, b_by_seed, args)}")
        if c_by_seed:
            print(f"Noise chart saved to {plot_noise(c_by_seed, args)}")


if __name__ == "__main__":
    main()
