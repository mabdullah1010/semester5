"""
random_forest.py
----------------
COM 307, Project 3: a random forest written from scratch for the Ionosphere data.

The file is organized in the same order as the algorithm:

    STEP 2  Gini impurity and the best-split search
    STEP 3  Building one decision tree (recursively) and predicting with it
    STEP 4  Evaluating a single random tree
    STEP 5  The random forest: bootstrapped trees + majority vote

Labels are 1 = good, 0 = bad.
Split rule everywhere: a row goes LEFT if  row[feature] <= threshold,  else RIGHT.

Usage:
    python random_forest.py --data ionosphere.data
"""

import argparse
from pathlib import Path

import numpy as np

from data_utils import FEATURE_NAMES, load_ionosphere

# ---------------------------------------------------------------------------
# Settings (all the choices in one place)
# ---------------------------------------------------------------------------
M = 6                    # features drawn at random at each node (sqrt(34) ~ 5.8)
MAX_DEPTH = None         # None = grow until the node is pure
MIN_SAMPLES_SPLIT = 2    # a node with fewer rows than this becomes a leaf
N_TREES = 100            # trees in the forest
TEST_FRACTION = 0.30     # share of rows held out for testing
N_REPEATS = 20           # how many single trees to build when measuring variability
SEED = 307               # makes every run give the same results

RESULTS_DIR = Path("results")


# ===========================================================================
# STEP 2: Gini impurity and the best split
# ===========================================================================
def gini(n_good, n_total):
    """Gini impurity of a group with n_good good rows out of n_total.

    Gini = 1 - p_good^2 - p_bad^2
         = 0    when the group is pure (all good or all bad)
         = 0.5  when the group is half good, half bad (the worst case)
    """
    if n_total == 0:
        return 0.0
    p_good = n_good / n_total
    p_bad = 1 - p_good
    return 1 - p_good ** 2 - p_bad ** 2


def best_split(X, y, features):
    """Find the best (feature, threshold) among the given features.

    For each candidate feature:
      1. sort the rows by that feature
      2. walk through the sorted rows, moving one row at a time from the
         right group to the left group, and keep a running count of how many
         good rows are on the left
      3. at every point where the value changes, a cut is possible; measure
         how pure the two groups would be (weighted Gini)
    The cut with the lowest weighted Gini (largest gain) wins.

    Returns (feature, threshold, gain), or (None, None, 0.0) if no cut helps.
    """
    n = len(y)
    total_good = int(y.sum())
    parent_gini = gini(total_good, n)

    best_feature, best_threshold, best_gain = None, None, 0.0

    for feature in features:
        order = np.argsort(X[:, feature])
        values = X[order, feature].tolist()
        labels = y[order].tolist()

        left_good = 0
        for i in range(n - 1):
            left_good += labels[i]      # row i has just moved to the left group

            # We can only cut between two DIFFERENT values; otherwise equal
            # values would end up on opposite sides of the threshold.
            if values[i] == values[i + 1]:
                continue

            n_left = i + 1
            n_right = n - n_left
            right_good = total_good - left_good

            weighted_gini = (n_left * gini(left_good, n_left)
                             + n_right * gini(right_good, n_right)) / n
            gain = parent_gini - weighted_gini

            # Strictly greater: on ties the first one found is kept.
            # The tiny margin stops floating-point noise from counting as a gain.
            if gain > best_gain + 1e-12:
                best_feature = feature
                best_threshold = (values[i] + values[i + 1]) / 2   # midpoint
                best_gain = gain

    return best_feature, best_threshold, best_gain


# ===========================================================================
# STEP 3: One decision tree
# ===========================================================================
class Node:
    """One node of a tree.

    Every node stores how many good/bad training rows reached it and what it
    would predict. A leaf stops there. An internal node also stores its split
    and its two children.
    """

    def __init__(self, n_good, n_bad):
        self.n_good = n_good
        self.n_bad = n_bad
        self.prediction = 1 if n_good >= n_bad else 0   # ties go to "good"

        # filled in only if the node gets split
        self.feature = None
        self.threshold = None
        self.gain = 0.0
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.feature is None


def build_tree(X, y, m, rng, max_depth=MAX_DEPTH,
               min_samples_split=MIN_SAMPLES_SPLIT, depth=0):
    """Grow a random decision tree on (X, y) and return its root Node.

    At every node, m features are drawn at random and only those are
    considered for the split. Then the same thing happens again, separately,
    in the left and right children (recursion).

    The node becomes a leaf when:
      - it is pure (all good or all bad), or
      - it has fewer than min_samples_split rows, or
      - it has reached max_depth, or
      - none of the m drawn features gives a useful split.
    """
    n_good = int(y.sum())
    n_bad = len(y) - n_good
    node = Node(n_good, n_bad)

    # --- stopping rules ---
    if n_good == 0 or n_bad == 0:
        return node
    if len(y) < min_samples_split:
        return node
    if max_depth is not None and depth >= max_depth:
        return node

    # --- draw m random features ---
    # Only features that still vary at this node are eligible. A feature that
    # has one value here (A2 always, others deep in the tree) can't split
    # anything, so drawing it would just waste one of the m picks.
    usable = [f for f in range(X.shape[1]) if X[:, f].min() < X[:, f].max()]
    if not usable:
        return node
    features = rng.choice(usable, size=min(m, len(usable)), replace=False)

    # --- find the best split among them ---
    feature, threshold, gain = best_split(X, y, features)
    if feature is None:
        return node

    # --- split the rows and recurse ---
    node.feature = int(feature)
    node.threshold = threshold
    node.gain = gain
    goes_left = X[:, feature] <= threshold
    node.left = build_tree(X[goes_left], y[goes_left], m, rng,
                           max_depth, min_samples_split, depth + 1)
    node.right = build_tree(X[~goes_left], y[~goes_left], m, rng,
                            max_depth, min_samples_split, depth + 1)
    return node


def predict_one(node, row):
    """Follow the splits from the root down to a leaf and return its label."""
    while not node.is_leaf():
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node.prediction


def predict(tree, X):
    """Predict a label for every row of X."""
    return np.array([predict_one(tree, row) for row in X])


def tree_depth(node):
    """Number of splits on the longest path from the root to a leaf."""
    if node.is_leaf():
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_leaves(node):
    if node.is_leaf():
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def tree_to_text(node, indent=""):
    """Readable text drawing of a tree, one line per node."""
    counts = f"[{node.n_good} good / {node.n_bad} bad]"
    if node.is_leaf():
        label = "GOOD" if node.prediction == 1 else "BAD"
        return f"{indent}-> {label} {counts}\n"
    name = FEATURE_NAMES[node.feature]
    text = f"{indent}{name} <= {node.threshold:.4f}  {counts}\n"
    text += tree_to_text(node.left, indent + "    ")
    text += f"{indent}{name} >  {node.threshold:.4f}\n"
    text += tree_to_text(node.right, indent + "    ")
    return text


# ===========================================================================
# STEP 4: Evaluating a single random tree
# ===========================================================================
def stratified_split(y, test_fraction, rng):
    """Shuffle and split row indices so train and test keep the same
    good/bad mix as the full data. (The file itself is NOT shuffled: its
    first 200 rows are ~50/50 and the rest are 82% good.)
    """
    train_idx, test_idx = [], []
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)
        n_test = round(len(rows) * test_fraction)
        test_idx.extend(rows[:n_test])
        train_idx.extend(rows[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Counts of (actual, predicted), as a dict for easy reading."""
    return {
        "good predicted good": int(np.sum((y_true == 1) & (y_pred == 1))),
        "good predicted bad":  int(np.sum((y_true == 1) & (y_pred == 0))),
        "bad predicted bad":   int(np.sum((y_true == 0) & (y_pred == 0))),
        "bad predicted good":  int(np.sum((y_true == 0) & (y_pred == 1))),
    }


def describe_results(y_true, y_pred):
    """Accuracy, confusion matrix, and accuracy within each class."""
    cm = confusion_matrix(y_true, y_pred)
    n_good = cm["good predicted good"] + cm["good predicted bad"]
    n_bad = cm["bad predicted bad"] + cm["bad predicted good"]
    return (
        f"  accuracy: {accuracy(y_true, y_pred):.1%}\n"
        f"  good rows: {cm['good predicted good']}/{n_good} correct "
        f"({cm['good predicted good'] / n_good:.1%})\n"
        f"  bad rows:  {cm['bad predicted bad']}/{n_bad} correct "
        f"({cm['bad predicted bad'] / n_bad:.1%})\n"
        f"  confusion: {cm}"
    )


# ===========================================================================
# STEP 5: The random forest
# ===========================================================================
class RandomForest:
    """Many random trees, each trained on a bootstrap sample, combined by vote.

    Bootstrap sample: n rows drawn WITH replacement from the n training rows.
    Each tree sees a slightly different dataset (about 63% of the distinct
    rows, some repeated), and also draws its own random features at every
    node. Every tree then votes, and the majority wins.

    Rows a tree never saw are its "out-of-bag" (OOB) rows. Predicting each
    training row using only the trees that didn't see it gives a free
    estimate of test accuracy without touching the test set.
    """

    def __init__(self, n_trees=N_TREES, m=M, max_depth=MAX_DEPTH,
                 min_samples_split=MIN_SAMPLES_SPLIT, bootstrap=True, seed=SEED):
        self.n_trees = n_trees
        self.m = m
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.in_bag = []      # for each tree: which training rows it saw

    def fit(self, X, y):
        n = len(y)
        self.trees, self.in_bag = [], []
        for _ in range(self.n_trees):
            if self.bootstrap:
                rows = self.rng.integers(0, n, size=n)   # with replacement
            else:
                rows = np.arange(n)
            saw_row = np.zeros(n, dtype=bool)
            saw_row[rows] = True

            tree = build_tree(X[rows], y[rows], self.m, self.rng,
                              self.max_depth, self.min_samples_split)
            self.trees.append(tree)
            self.in_bag.append(saw_row)
        return self

    def good_vote_share(self, X):
        """For each row, the fraction of trees that vote 'good'."""
        votes = np.array([predict(tree, X) for tree in self.trees])  # (trees, rows)
        return votes.mean(axis=0)

    def predict(self, X):
        """Majority vote. A 50/50 tie goes to 'good'."""
        return (self.good_vote_share(X) >= 0.5).astype(int)

    def oob_accuracy(self, X, y):
        """Accuracy on training rows, each predicted only by trees that never
        saw it. Rows that every tree saw (rare) are skipped."""
        correct, counted = 0, 0
        for i in range(len(y)):
            votes = [predict_one(tree, X[i])
                     for tree, saw in zip(self.trees, self.in_bag) if not saw[i]]
            if votes:
                prediction = 1 if np.mean(votes) >= 0.5 else 0
                correct += prediction == y[i]
                counted += 1
        return correct / counted

    def feature_importance(self):
        """How much each feature reduced Gini, summed over every split in
        every tree. A split counts more if more rows reached it. Scaled so
        all features add up to 1."""
        importance = np.zeros(len(FEATURE_NAMES))

        def add(node, n_root):
            if node.is_leaf():
                return
            rows_here = node.n_good + node.n_bad
            importance[node.feature] += node.gain * rows_here / n_root
            add(node.left, n_root)
            add(node.right, n_root)

        for tree in self.trees:
            add(tree, tree.n_good + tree.n_bad)
        return importance / importance.sum()

    def most_representative_tree(self, X):
        """The single tree whose predictions agree most often with the
        forest's vote on X. A readable stand-in for the whole forest."""
        forest_pred = self.predict(X)
        agreement = [accuracy(forest_pred, predict(tree, X)) for tree in self.trees]
        best = int(np.argmax(agreement))
        return best, agreement[best]


# ===========================================================================
# Run Steps 4 and 5
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Random forest on the Ionosphere data.")
    parser.add_argument("--data", default="ionosphere.data")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    report = []

    def log(text=""):
        print(text)
        report.append(text)

    X, y, _ = load_ionosphere(args.data)
    rng = np.random.default_rng(SEED)
    train, test = stratified_split(y, TEST_FRACTION, rng)
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]

    log("RANDOM FOREST ON THE IONOSPHERE DATA")
    log("=" * 50)
    log(f"Settings: m = {M}, max_depth = {MAX_DEPTH}, "
        f"min_samples_split = {MIN_SAMPLES_SPLIT}, seed = {SEED}")
    log(f"Train: {len(train)} rows ({y_train.sum()} good / {len(train) - y_train.sum()} bad)")
    log(f"Test:  {len(test)} rows ({y_test.sum()} good / {len(test) - y_test.sum()} bad)")
    log(f"Baseline (always guess good) on test: {y_test.mean():.1%}")

    # ---------------- STEP 4: the first random tree ----------------
    log("")
    log("STEP 4: SINGLE RANDOM TREE")
    log("-" * 50)
    tree = build_tree(X_train, y_train, M, np.random.default_rng(SEED))
    log(f"Depth {tree_depth(tree)}, {count_leaves(tree)} leaves")
    log(f"Training accuracy: {accuracy(y_train, predict(tree, X_train)):.1%}")
    log("Test results:")
    log(describe_results(y_test, predict(tree, X_test)))
    log("")
    log("The tree:")
    log(tree_to_text(tree))

    # How much does a single tree depend on its random feature draws?
    scores = []
    for seed in range(N_REPEATS):
        t = build_tree(X_train, y_train, M, np.random.default_rng(seed))
        scores.append(accuracy(y_test, predict(t, X_test)))
    log(f"{N_REPEATS} single trees (same data, different random draws):")
    log(f"  test accuracy mean {np.mean(scores):.1%}, std {np.std(scores):.1%}, "
        f"range {min(scores):.1%} – {max(scores):.1%}")

    # ---------------- STEP 5: the forest ----------------
    log("")
    log(f"STEP 5: RANDOM FOREST ({N_TREES} trees, bootstrap, majority vote)")
    log("-" * 50)
    forest = RandomForest().fit(X_train, y_train)
    log("Test results:")
    log(describe_results(y_test, forest.predict(X_test)))
    log(f"Out-of-bag accuracy (estimated from training data only): "
        f"{forest.oob_accuracy(X_train, y_train):.1%}")

    log("")
    log("Top 10 features by importance:")
    importance = forest.feature_importance()
    for f in np.argsort(importance)[::-1][:10]:
        log(f"  {FEATURE_NAMES[f]:>4}  {importance[f]:.3f}")

    best, agreement = forest.most_representative_tree(X_test)
    rep_tree = forest.trees[best]
    log("")
    log(f"Most representative tree: #{best}, agrees with the forest on "
        f"{agreement:.1%} of test rows")
    log(f"  its own test accuracy: {accuracy(y_test, predict(rep_tree, X_test)):.1%}, "
        f"depth {tree_depth(rep_tree)}, {count_leaves(rep_tree)} leaves")

    (RESULTS_DIR / "forest_summary.txt").write_text("\n".join(report) + "\n")
    print(f"\nSummary written to {(RESULTS_DIR / 'forest_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
