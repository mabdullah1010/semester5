"""
step1_explore.py
----------------
Step 1 of COM 307 Project 3: load the Ionosphere data and look at it before
building any trees.

What this script checks, and why it matters for later steps:
  * class balance        -> the "always guess good" baseline we have to beat
  * missing / duplicates -> whether any cleaning is needed
  * constant features    -> a split search must skip a feature with one value
  * binary features      -> threshold search still has to work on 0/1 columns
  * value range          -> whether scaling is needed (trees don't need it)
  * row order            -> whether "first N rows = training set" is safe
  * per-feature separation between classes -> a first look at which pulses
    carry signal (a screening statistic only, not used by the model)

Outputs:
  figures/step1_class_balance.png
  figures/step1_pulse_profiles.png
  figures/step1_top_feature_histograms.png
  figures/step1_correlation_heatmap.png
  results/step1_summary.txt

Usage:
  python step1_explore.py --data ionosphere.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from data_utils import CLASS_NAMES, FEATURE_NAMES, N_PULSES, load_ionosphere, pulse_of

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")
N_TOP_FEATURES = 6          # how many features to show in the histogram grid
SPLIT_ROW = 200             # the 1989 paper trained on the first 200 rows

# Colors: one fixed color per class, used in every figure.
COLOR = {1: "#2a78d6", 0: "#eb6834"}   # good = blue, bad = orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def class_counts(y: np.ndarray) -> dict[str, int]:
    """Number of rows in each class, keyed by class name."""
    return {CLASS_NAMES[c]: int((y == c).sum()) for c in (1, 0)}


def find_constant_features(X: np.ndarray) -> list[int]:
    """Indices of columns that hold a single value (useless for splitting)."""
    return [j for j in range(X.shape[1]) if np.unique(X[:, j]).size == 1]


def find_binary_features(X: np.ndarray) -> list[int]:
    """Indices of columns that hold exactly two distinct values."""
    return [j for j in range(X.shape[1]) if np.unique(X[:, j]).size == 2]


def separation_scores(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Standardized difference in class means for every feature.

    score_j = |mean_good - mean_bad| / pooled standard deviation

    Larger means the two classes sit further apart on that feature. It only
    looks at means, so it can miss features that split well in other ways.
    It is used here for exploration, not by the tree.
    """
    good, bad = X[y == 1], X[y == 0]
    n_g, n_b = len(good), len(bad)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        m_g, m_b = good[:, j].mean(), bad[:, j].mean()
        v_g, v_b = good[:, j].var(ddof=1), bad[:, j].var(ddof=1)
        pooled_sd = np.sqrt(((n_g - 1) * v_g + (n_b - 1) * v_b) / (n_g + n_b - 2))
        score = abs(m_g - m_b) / pooled_sd if pooled_sd > 0 else 0.0
        pulse, part = pulse_of(j)
        rows.append({"feature": name, "pulse": pulse, "part": part,
                     "mean_good": m_g, "mean_bad": m_b, "separation": score})
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_class_balance(y: np.ndarray, out: Path) -> None:
    """Share of good/bad overall and in the two row blocks of the file.

    Shows both the class imbalance and that the file is not shuffled.
    """
    groups = {
        f"All rows\n(n={len(y)})": y,
        f"Rows 1–{SPLIT_ROW}\n(n={SPLIT_ROW})": y[:SPLIT_ROW],
        f"Rows {SPLIT_ROW + 1}–{len(y)}\n(n={len(y) - SPLIT_ROW})": y[SPLIT_ROW:],
    }
    labels = list(groups)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for offset, cls in ((-width / 2 - 0.01, 1), (width / 2 + 0.01, 0)):
        shares = [100 * (g == cls).mean() for g in groups.values()]
        counts = [int((g == cls).sum()) for g in groups.values()]
        bars = ax.bar(x + offset, shares, width, color=COLOR[cls],
                      label=CLASS_NAMES[cls], zorder=3)
        for bar, share, count in zip(bars, shares, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{share:.0f}%\n({count})", ha="center", va="bottom",
                    fontsize=9, color=INK)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of rows (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_title("Class balance overall and by position in the file")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pulse_profiles(X: np.ndarray, y: np.ndarray, out: Path) -> None:
    """Mean value per class at each of the 17 pulses, real and imaginary parts.

    Uses the dataset's structure (17 pulses x 2 parts) instead of treating the
    34 columns as unrelated numbers.
    """
    pulses = np.arange(1, N_PULSES + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, part, start in ((axes[0], "Real part (odd attributes)", 0),
                            (axes[1], "Imaginary part (even attributes)", 1)):
        cols = np.arange(start, X.shape[1], 2)
        for cls in (1, 0):
            means = X[y == cls][:, cols].mean(axis=0)
            ax.plot(pulses, means, color=COLOR[cls], linewidth=2,
                    marker="o", markersize=5, label=CLASS_NAMES[cls], zorder=3)
            ax.annotate(CLASS_NAMES[cls], (pulses[-1], means[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_title(part)
        ax.set_xlabel("Pulse number")
        ax.set_xticks(pulses)
        ax.set_xlim(0.5, N_PULSES + 1.8)
    axes[0].set_ylabel("Mean value")
    axes[0].legend(loc="lower left")
    fig.suptitle("Average signal per pulse, by class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_top_feature_histograms(X: np.ndarray, y: np.ndarray,
                                scores: pd.DataFrame, out: Path) -> None:
    """Class-wise histograms for the most separating features."""
    top = scores.head(N_TOP_FEATURES)
    bins = np.linspace(-1, 1, 21)
    ncols = 3
    nrows = int(np.ceil(N_TOP_FEATURES / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        j = FEATURE_NAMES.index(row["feature"])
        for cls in (1, 0):
            ax.hist(X[y == cls, j], bins=bins, color=COLOR[cls], alpha=0.55,
                    edgecolor="white", linewidth=1, label=CLASS_NAMES[cls], zorder=3)
        ax.set_title(f"{row['feature']}  (pulse {row['pulse']}, {row['part']})  "
                     f"sep = {row['separation']:.2f}", fontsize=10)
        ax.set_xlim(-1.05, 1.05)
    for ax in list(axes.flat)[N_TOP_FEATURES:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rows")
    fig.suptitle(f"The {N_TOP_FEATURES} features whose class means differ most",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(X: np.ndarray, constant: list[int], out: Path) -> None:
    """Pearson correlation between features (constant columns left out)."""
    keep = [j for j in range(X.shape[1]) if j not in constant]
    corr = np.corrcoef(X[:, keep], rowvar=False)
    names = [FEATURE_NAMES[j] for j in keep]

    # Diverging map: orange (negative) -> neutral gray -> blue (positive).
    cmap = LinearSegmentedColormap.from_list(
        "orange_gray_blue", ["#b2440f", "#f2b48f", "#e8e7e3", "#86b6ef", "#184f95"])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation")
    ax.set_title("Correlation between features "
                 f"({', '.join(FEATURE_NAMES[j] for j in constant)} omitted: constant)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: explore the Ionosphere data.")
    parser.add_argument("--data", default="ionosphere.data",
                        help="path to ionosphere.data (default: ./ionosphere.data)")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y, df = load_ionosphere(args.data)
    n, p = X.shape

    counts = class_counts(y)
    baseline = max(counts.values()) / n
    constant = find_constant_features(X)
    binary = find_binary_features(X)
    n_missing = int(df[FEATURE_NAMES].isna().sum().sum())
    n_dupes = int(df.duplicated(subset=FEATURE_NAMES + ["class"]).sum())
    first_block = class_counts(y[:SPLIT_ROW])
    second_block = class_counts(y[SPLIT_ROW:])
    scores = separation_scores(X, y)

    # Highly correlated feature pairs (|r| > 0.7), constant columns excluded.
    keep = [j for j in range(p) if j not in constant]
    corr = np.corrcoef(X[:, keep], rowvar=False)
    strong_pairs = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            if abs(corr[a, b]) > 0.7:
                strong_pairs.append((FEATURE_NAMES[keep[a]], FEATURE_NAMES[keep[b]], corr[a, b]))

    lines = [
        "STEP 1 — DATA EXPLORATION SUMMARY",
        "=" * 50,
        f"Rows: {n}    Features: {p}",
        f"Class counts: {counts}",
        f"Majority-class baseline accuracy: {baseline:.1%}",
        f"Missing values: {n_missing}",
        f"Duplicate rows: {n_dupes}",
        f"Value range across all features: [{X.min():.3f}, {X.max():.3f}]",
        "",
        f"Constant features (cannot be split on): "
        f"{[FEATURE_NAMES[j] for j in constant] or 'none'}",
        f"Binary features: "
        f"{[(FEATURE_NAMES[j], sorted(np.unique(X[:, j]).tolist())) for j in binary] or 'none'}",
        "",
        "Row order check:",
        f"  rows 1–{SPLIT_ROW}:      {first_block}",
        f"  rows {SPLIT_ROW + 1}–{n}:  {second_block}",
        "  -> the file is not shuffled; later steps must shuffle and stratify.",
        "",
        f"Feature pairs with |correlation| > 0.7: {len(strong_pairs)}",
        *[f"  {a} ~ {b}: r = {r:+.2f}" for a, b, r in strong_pairs],
        "",
        "Features ranked by class separation (|mean_good - mean_bad| / pooled SD):",
        scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    ]
    summary = "\n".join(lines)
    print(summary)
    (RESULTS_DIR / "step1_summary.txt").write_text(summary + "\n")

    plot_class_balance(y, FIG_DIR / "step1_class_balance.png")
    plot_pulse_profiles(X, y, FIG_DIR / "step1_pulse_profiles.png")
    plot_top_feature_histograms(X, y, scores, FIG_DIR / "step1_top_feature_histograms.png")
    plot_correlation_heatmap(X, constant, FIG_DIR / "step1_correlation_heatmap.png")
    print(f"\nFigures written to {FIG_DIR.resolve()}")
    print(f"Summary written to {(RESULTS_DIR / 'step1_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
