# COM 307
# Muhammad Abdullah
# Project 3
# Date: Nov 1st 2026


#python experiments.py --data ionosphere.data


import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from data_utils import FEATURE_NAMES, load_ionosphere, pulse_of
from random_forest import (SEED, TEST_FRACTION, RandomForest, accuracy, build_tree,
                           describe_results, predict, stratified_split)



M_VALUES = [1, 2, 3, 4, 6, 8, 12, 17, 25, 34]
N_FOLDS = 5
N_CV_REPEATS = 3             
TREES_FOR_M_SWEEP = 100      
SINGLE_TREES_PER_FOLD = 10    
MAX_TREES = 300               
N_SEEDS_COMPARE = 20          


FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")



BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "font.size": 10, "legend.frameon": False,
})



def stratified_folds(y, n_folds, rng):

    folds = [[] for _ in range(n_folds)]

    for label in (0, 1):

        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)

        for k, chunk in enumerate(np.array_split(rows, n_folds)):
            folds[k].extend(chunk)
    return [np.array(sorted(f)) for f in folds]


def cross_validate(X, y, m, rng):

    forest_scores, tree_scores = [], []
    all_folds = []

    for repeat in range(N_CV_REPEATS):
        all_folds += stratified_folds(y, N_FOLDS, np.random.default_rng(SEED + repeat))


    for val in all_folds:

        fit = np.setdiff1d(np.arange(len(y)), val)

        forest = RandomForest(n_trees=TREES_FOR_M_SWEEP, m=m,
                              seed=int(rng.integers(1_000_000))).fit(X[fit], y[fit])
        forest_scores.append(accuracy(y[val], forest.predict(X[val])))


        singles = [accuracy(y[val], predict(build_tree(X[fit], y[fit], m, rng), X[val]))
                   for _ in range(SINGLE_TREES_PER_FOLD)]
        tree_scores.append(np.mean(singles))

    return forest_scores, tree_scores


def accuracy_as_trees_are_added(forest, X_train, y_train, X_test, y_test):

    k = np.arange(1, len(forest.trees) + 1)[:, None]


    test_votes = np.array([predict(t, X_test) for t in forest.trees])   # (trees, rows)
    good_share = np.cumsum(test_votes, axis=0) / k
    test_acc = np.mean((good_share >= 0.5) == y_test, axis=1)



    train_votes = np.array([predict(t, X_train) for t in forest.trees])
    out_of_bag = ~np.array(forest.in_bag)
    oob_good = np.cumsum(train_votes * out_of_bag, axis=0)
    oob_count = np.cumsum(out_of_bag, axis=0)
    oob_acc = []
    for i in range(len(forest.trees)):
        has_vote = oob_count[i] > 0
        pred = oob_good[i, has_vote] / oob_count[i, has_vote] >= 0.5
        oob_acc.append(np.mean(pred == y_train[has_vote]))
    return test_acc, np.array(oob_acc)



def plot_choosing_m(results, best_m, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in (("forest", BLUE, f"Forest ({TREES_FOR_M_SWEEP} trees)"),
                              ("tree", ORANGE, "Single random tree")):
        means = np.array([np.mean(results[m][key]) for m in M_VALUES]) * 100
        stds = np.array([np.std(results[m][key]) for m in M_VALUES]) * 100
        ax.fill_between(M_VALUES, means - stds, means + stds, color=color, alpha=0.12, linewidth=0)
        ax.plot(M_VALUES, means, color=color, linewidth=2, marker="o", markersize=6,
                label=label, zorder=3)
    best_mean = np.mean(results[best_m]["forest"]) * 100
    ax.scatter([best_m], [best_mean], s=110, facecolor="none", edgecolor=INK,
               linewidth=1.5, zorder=4)
    ax.annotate(f"chosen: m = {best_m}\n({best_mean:.1f}%)", (best_m, best_mean),
                xytext=(0, 22), textcoords="offset points", ha="center",
                fontsize=9, color=INK)
    ax.axvline(np.sqrt(34), color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(np.sqrt(34), ax.get_ylim()[0] + 0.5, "√34 ≈ 5.8 ", ha="right", fontsize=8, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xticks(M_VALUES, [str(m) for m in M_VALUES])
    ax.minorticks_off()
    ax.set_xlabel("m (features drawn at random at each node, log scale)")
    ax.set_ylabel(f"CV accuracy (%), {N_CV_REPEATS}×{N_FOLDS}-fold")
    ax.set_title("Choosing m: cross-validation on the training rows")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_number_of_trees(test_acc, oob_acc, m, out):
    k = np.arange(1, len(test_acc) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k, test_acc * 100, color=BLUE, linewidth=2, label="Test accuracy", zorder=3)
    ax.plot(k, oob_acc * 100, color=ORANGE, linewidth=2, label="Out-of-bag accuracy", zorder=3)
    ax.set_xscale("log")

    ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= len(test_acc)]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Number of trees in the forest (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Adding trees (m = {m})")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tree_vs_forest(tree_scores, forest_scores, baseline, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for x, scores, color, label in ((0, tree_scores, ORANGE, "Single random tree"),
                                    (1, forest_scores, BLUE, "Random forest")):

        values = np.round(np.array(scores) * 100, 6)
        xs = np.zeros(len(values))
        for v in np.unique(values):
            same = np.flatnonzero(values == v)
            xs[same] = (np.arange(len(same)) - (len(same) - 1) / 2) * 0.035
        ax.scatter(x + xs, values, s=36, color=color,
                   edgecolor="white", linewidth=1, zorder=3)
        mean = np.mean(scores) * 100
        ax.hlines(mean, x - 0.25, x + 0.25, color=INK, linewidth=2, zorder=4)
        ax.text(x, values.max() + 1.0, f"mean {mean:.1f}%", ha="center",
                fontsize=9, color=INK)
    ax.axhline(baseline * 100, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(1.45, baseline * 100 + 0.6, "always guess good", ha="right",
            fontsize=8, color=INK_MUTED)
    ax.set_xticks([0, 1], ["Single random tree", "Random forest"])
    ax.set_xlim(-0.5, 1.5)

    ax.set_ylim(top=97)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{N_SEEDS_COMPARE} random seeds each, same test rows")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance, out):
    order = np.argsort(importance)
    colors = [BLUE if pulse_of(j)[1] == "real" else ORANGE for j in order]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([FEATURE_NAMES[j] for j in order], importance[order], color=colors,
            height=0.7, zorder=3)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total Gini reduction across the forest")
    ax.set_title("Which features the forest relies on")
    ax.legend(handles=[Patch(color=BLUE, label="real part (odd A)"),
                       Patch(color=ORANGE, label="imaginary part (even A)")],
              loc="lower right")

    
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Step 6: experiments.")
    parser.add_argument("--data", default="ionosphere.data")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    report = []

    def log(text=""):
        print(text, flush=True)
        report.append(text)

    X, y, _ = load_ionosphere(args.data)
    train, test = stratified_split(y, TEST_FRACTION, np.random.default_rng(SEED))
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    rng = np.random.default_rng(SEED)
    start = time.time()

    log("STEP 6: EXPERIMENTS")
    log("=" * 50)
    log(f"Same split as random_forest.py: {len(train)} train / {len(test)} test rows")


    log("")
    log(f"1. CHOOSING m ({N_CV_REPEATS} x {N_FOLDS}-fold CV on the {len(train)} training rows)")
    log(f"   {'m':>3}   {'forest':>15}   {'single tree':>15}")
    results = {}
    for m in M_VALUES:
        forest_scores, tree_scores = cross_validate(X_train, y_train, m, rng)
        results[m] = {"forest": forest_scores, "tree": tree_scores}
        log(f"   {m:>3}   {np.mean(forest_scores):6.1%} ± {np.std(forest_scores):4.1%}"
            f"   {np.mean(tree_scores):6.1%} ± {np.std(tree_scores):4.1%}")

    top_m = max(M_VALUES, key=lambda m: np.mean(results[m]["forest"]))
    top = np.mean(results[top_m]["forest"])
    close = [m for m in M_VALUES if top - np.mean(results[m]["forest"]) < 0.01]
    best_m = min(close, key=lambda m: abs(m - np.sqrt(34)))
    log(f"   highest forest CV accuracy: m = {top_m} ({top:.1%})")
    log(f"   m values within 1 point of it: {close}")
    log(f"   -> chosen m = {best_m} (the one in that group closest to sqrt(34))")
    plot_choosing_m(results, best_m, FIG_DIR / "step6_choosing_m.png")


    log("")
    log(f"2. NUMBER OF TREES (m = {best_m}, up to {MAX_TREES} trees)")
    big_forest = RandomForest(n_trees=MAX_TREES, m=best_m, seed=SEED).fit(X_train, y_train)
    test_acc, oob_acc = accuracy_as_trees_are_added(big_forest, X_train, y_train, X_test, y_test)
    for k in (1, 5, 10, 25, 50, 100, 200, 300):
        if k <= MAX_TREES:
            log(f"   {k:>3} trees: test {test_acc[k - 1]:.1%}   out-of-bag {oob_acc[k - 1]:.1%}")
    plot_number_of_trees(test_acc, oob_acc, best_m, FIG_DIR / "step6_number_of_trees.png")


    log("")
    log(f"3. SINGLE TREE vs FOREST ({N_SEEDS_COMPARE} seeds each, m = {best_m}, test rows)")
    tree_scores = [accuracy(y_test, predict(build_tree(X_train, y_train, best_m,
                                                        np.random.default_rng(s)), X_test))
                   for s in range(N_SEEDS_COMPARE)]
    forest_scores = [accuracy(y_test, RandomForest(m=best_m, seed=s)
                              .fit(X_train, y_train).predict(X_test))
                     for s in range(N_SEEDS_COMPARE)]
    for name, scores in (("single tree", tree_scores), ("forest", forest_scores)):
        log(f"   {name:>11}: mean {np.mean(scores):.1%}, std {np.std(scores):.1%}, "
            f"range {min(scores):.1%} – {max(scores):.1%}")
    plot_tree_vs_forest(tree_scores, forest_scores, y_test.mean(),
                        FIG_DIR / "step6_tree_vs_forest.png")


    log("")
    log(f"4. FINAL MODEL: forest of 100 trees, m = {best_m}, on the test rows")
    final = RandomForest(m=best_m, seed=SEED).fit(X_train, y_train)
    log(describe_results(y_test, final.predict(X_test)))
    log(f"  out-of-bag accuracy: {final.oob_accuracy(X_train, y_train):.1%}")

    importance = final.feature_importance()
    real_share = sum(importance[j] for j in range(len(importance)) if pulse_of(j)[1] == "real")
    log(f"  importance from real parts: {real_share:.1%}, imaginary parts: {1 - real_share:.1%}")
    plot_feature_importance(importance, FIG_DIR / "step6_feature_importance.png")

    try:
        from sklearn.ensemble import RandomForestClassifier
        sk = [accuracy(y_test, RandomForestClassifier(n_estimators=100, max_features=best_m,
                                                      random_state=s)
                       .fit(X_train, y_train).predict(X_test))
              for s in range(N_SEEDS_COMPARE)]
        log(f"  scikit-learn forest, same settings, {N_SEEDS_COMPARE} seeds: "
            f"mean {np.mean(sk):.1%}, std {np.std(sk):.1%}   (sanity check only)")
    except ImportError:
        log("  scikit-learn not installed; skipped the sanity check.")

    #  1989 split
    log("")
    log("5. ORIGINAL 1989 SPLIT (train rows 1-200, test rows 201-351)")
    old_forest = RandomForest(m=best_m, seed=SEED).fit(X[:200], y[:200])
    log(describe_results(y[200:], old_forest.predict(X[200:])))
    log("  published on this split: nearest neighbor 92.1%, C4 94.0%,")
    log("  linear perceptron 90.7%, backprop ~96%, IB3 96.7%")


    log("")
    log(f"Finished in {time.time() - start:.0f} s")
    (RESULTS_DIR / "experiments_summary.txt").write_text("\n".join(report) + "\n")
    print(f"Summary written to {(RESULTS_DIR / 'experiments_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
