"""Soft-label training helpers (M9). Soft label = rater vote distribution over the 4 classes."""
import numpy as np


def class_weights(y_hard, k=4):
    """sklearn-'balanced' weights from HARD training labels: n / (k * count_c)."""
    return len(y_hard) / (k * np.maximum(np.bincount(y_hard, minlength=k), 1))


def expand(X, y_soft, y_hard):
    """One row per (utterance, class with nonzero soft mass). Row label = that class; row weight =
    y_soft[c] * class_weight[y_hard]. The weights of one utterance therefore sum to its class weight."""
    cw = class_weights(y_hard); rows, labels, w = [], [], []
    for c in range(y_soft.shape[1]):
        m = np.where(y_soft[:, c] > 0)[0]
        rows.append(m); labels.append(np.full(len(m), c)); w.append(y_soft[m, c] * cw[y_hard[m]])
    r = np.concatenate(rows)
    return X[r], np.concatenate(labels), np.concatenate(w), r
