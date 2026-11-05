# COM 307
# Muhammad Abdullah
# Project 3
# Date: Nov 1st 2026



#python test_random_forest.py


import numpy as np

from random_forest import RandomForest, best_split, build_tree, gini, predict


def slow_best_split(X, y, features):

    parent = gini(y.sum(), len(y))
    best_gain = 0.0



    for f in features:

        values = np.unique(X[:, f])
        for t in (values[:-1] + values[1:]) / 2:
            left, right = y[X[:, f] <= t], y[X[:, f] > t]
            weighted = (len(left) * gini(left.sum(), len(left))
                        + len(right) * gini(right.sum(), len(right))) / len(y)
            best_gain = max(best_gain, parent - weighted)


    return best_gain


def test_gini_values():
    assert gini(4, 4) == 0.0     
    assert gini(0, 4) == 0.0     
    assert gini(2, 4) == 0.5       
    assert gini(0, 0) == 0.0        


def test_perfect_split():

    X = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]])

    y = np.array([0, 0, 0, 1, 1, 1])

    feature, threshold, gain = best_split(X, y, [0])

    assert feature == 0 and np.isclose(threshold, 0.5) and np.isclose(gain, 0.5)


def test_constant_feature_gives_no_split():
    X = np.zeros((6, 1))                             
    y = np.array([0, 0, 0, 1, 1, 1])
    assert best_split(X, y, [0]) == (None, None, 0.0)


def test_equal_values_stay_together():
    X = np.array([[0.5], [0.5], [0.5], [0.9], [0.9]])
    y = np.array([0, 0, 1, 1, 1])
    _, threshold, _ = best_split(X, y, [0])
    assert np.isclose(threshold, 0.7)                   


def test_matches_slow_version_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, p = rng.integers(5, 50), rng.integers(1, 5)
        X = np.round(rng.uniform(-1, 1, (n, p)), 1)   
        y = rng.integers(0, 2, n)
        _, _, gain = best_split(X, y, range(p))
        assert np.isclose(gain, slow_best_split(X, y, range(p)))


def test_full_tree_memorizes_training_data():

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (80, 5))

    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = build_tree(X, y, m=2, rng=rng)

    assert np.all(predict(tree, X) == y)


def test_forest_beats_guessing_on_easy_data():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (200, 5))
    y = (X[:, 0] > 0).astype(int)
    forest = RandomForest(n_trees=15, m=2, seed=0).fit(X[:150], y[:150])
    assert np.mean(forest.predict(X[150:]) == y[150:]) > 0.9


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


