"""Tests for soft-label expansion and class weights."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.soft import expand, class_weights


def toy():
    y_hard = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    S = np.array([[1, 0, 0, 0], [.667, .333, 0, 0], [0, 1, 0, 0], [.333, .667, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, .333, 0, .667], [0, 0, 0, 1.]])
    S = S / S.sum(1, keepdims=True); X = np.arange(8 * 3, dtype=float).reshape(8, 3)
    return X, S, y_hard


def test_expansion_weights_sum_to_class_weight_per_utterance():
    X, S, y = toy(); Xe, ye, w, r = expand(X, S, y); cw = class_weights(y)
    for i in range(len(y)):
        assert abs(w[r == i].sum() - cw[y[i]]) < 1e-9
    assert (ye >= 0).all() and ((S[r, ye]) > 0).all()            # only classes with nonzero soft mass
    assert len(Xe) == int((S > 0).sum()) and np.array_equal(Xe, X[r])


def test_one_hot_soft_labels_reduce_to_hard_training():
    X, S, y = toy(); S1 = np.eye(4)[y]; Xe, ye, w, r = expand(X, S1, y)
    assert np.array_equal(ye, y) and np.array_equal(r, np.arange(len(y))) and np.allclose(w, class_weights(y)[y])


def test_soft_ce_with_one_hot_equals_hard_ce():
    import torch
    logits = torch.randn(5, 4); y = torch.tensor([0, 1, 2, 3, 1]); T = torch.nn.functional.one_hot(y, 4).float()
    soft = -(T * torch.log_softmax(logits, 1)).sum(1).mean(); hard = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(soft, hard)
