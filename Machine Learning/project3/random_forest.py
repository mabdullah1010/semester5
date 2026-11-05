# COM 307
# Muhammad Abdullah
# Project 3
# Date: Nov 1st 2026

#python random_forest.py --data ionosphere.data


import argparse
from pathlib import Path

import numpy as np

from data_utils import FEATURE_NAMES, load_ionosphere

# hyperparameters
M = 6                 
MAX_DEPTH = None       
MIN_SAMPLES_SPLIT = 2    
N_TREES = 100     
TEST_FRACTION = 0.30  
N_REPEATS = 20        
SEED = 307       

RESULTS_DIR = Path("results")


def gini(n_good, n_total):
    if n_total == 0:
        return 0.0
    
    p_good = n_good / n_total
    p_bad = 1 - p_good

    return 1 - p_good ** 2 - p_bad ** 2


def best_split(X, y, features):

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
            left_good += labels[i]

            if values[i] == values[i + 1]:
                continue


            n_left = i + 1
            n_right = n - n_left
            right_good = total_good - left_good

            weighted_gini = (n_left * gini(left_good, n_left)
                             + n_right * gini(right_good, n_right)) / n
            gain = parent_gini - weighted_gini


            if gain > best_gain + 1e-12:
                best_feature = feature
                best_threshold = (values[i] + values[i + 1]) / 2
                best_gain = gain

    return best_feature, best_threshold, best_gain



class Node:

    def __init__(self, n_good, n_bad):
        self.n_good = n_good
        self.n_bad = n_bad
        self.prediction = 1 if n_good >= n_bad else 0


        self.feature = None
        self.threshold = None
        self.gain = 0.0
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.feature is None


def build_tree(X, y, m, rng, max_depth=MAX_DEPTH,
               min_samples_split=MIN_SAMPLES_SPLIT, depth=0):

    n_good = int(y.sum())
    n_bad = len(y) - n_good
    node = Node(n_good, n_bad)

    if n_good == 0 or n_bad == 0:
        return node
    if len(y) < min_samples_split:
        return node
    if max_depth is not None and depth >= max_depth:
        return node


    usable = [f for f in range(X.shape[1]) if X[:, f].min() < X[:, f].max()]
    if not usable:
        return node
    features = rng.choice(usable, size=min(m, len(usable)), replace=False)


    feature, threshold, gain = best_split(X, y, features)
    if feature is None:
        return node

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

    while not node.is_leaf():

        node = node.left if row[node.feature] <= node.threshold else node.right

    return node.prediction


def predict(tree, X):

    return np.array([predict_one(tree, row) for row in X])


def tree_depth(node):

    if node.is_leaf():
        return 0
    
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_leaves(node):
    if node.is_leaf():
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def tree_to_text(node, indent=""):

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

def stratified_split(y, test_fraction, rng):


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

    return {
        "good predicted good": int(np.sum((y_true == 1) & (y_pred == 1))),
        "good predicted bad":  int(np.sum((y_true == 1) & (y_pred == 0))),
        "bad predicted bad":   int(np.sum((y_true == 0) & (y_pred == 0))),
        "bad predicted good":  int(np.sum((y_true == 0) & (y_pred == 1))),
    }


def describe_results(y_true, y_pred):

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



class RandomForest:


    def __init__(self, n_trees=N_TREES, m=M, max_depth=MAX_DEPTH,
                 min_samples_split=MIN_SAMPLES_SPLIT, bootstrap=True, seed=SEED):
        self.n_trees = n_trees
        self.m = m
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.in_bag = []      # for each tree - which training rows it saw

    def fit(self, X, y):
        n = len(y)
        self.trees, self.in_bag = [], []
        for _ in range(self.n_trees):
            if self.bootstrap:
                rows = self.rng.integers(0, n, size=n)   #with replacement
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

        votes = np.array([predict(tree, X) for tree in self.trees])
        return votes.mean(axis=0)

    def predict(self, X):

        return (self.good_vote_share(X) >= 0.5).astype(int)

    def oob_accuracy(self, X, y):

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

        forest_pred = self.predict(X)
        agreement = [accuracy(forest_pred, predict(tree, X)) for tree in self.trees]
        best = int(np.argmax(agreement))

        return best, agreement[best]



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


    scores = []
    for seed in range(N_REPEATS):
        t = build_tree(X_train, y_train, M, np.random.default_rng(seed))
        scores.append(accuracy(y_test, predict(t, X_test)))
    log(f"{N_REPEATS} single trees (same data, different random draws):")
    log(f"  test accuracy mean {np.mean(scores):.1%}, std {np.std(scores):.1%}, "
        f"range {min(scores):.1%} – {max(scores):.1%}")


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
