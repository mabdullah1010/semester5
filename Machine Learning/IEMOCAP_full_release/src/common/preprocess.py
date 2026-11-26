"""Preprocessing fitted on training rows only: median imputation, per-speaker z-scoring, global scaler."""
import numpy as np


def speaker_zscore(X, spk):
    """Standardize each speaker's rows with that speaker's own mean/std (uses features only, no labels)."""
    out = np.empty_like(X, dtype=np.float64)
    for s in np.unique(spk):
        m = spk == s; mu = X[m].mean(0); sd = X[m].std(0)
        out[m] = (X[m] - mu) / np.where(sd > 1e-12, sd, 1.0)
    return out


class Preprocessor:
    def __init__(self, per_speaker=True):
        self.per_speaker = per_speaker

    def fit(self, X, spk):
        X = np.asarray(X, dtype=np.float64)
        self.median_ = np.nanmedian(X, axis=0)
        Z = self._pre(X, spk)
        self.mean_ = Z.mean(0); sd = Z.std(0); self.std_ = np.where(sd > 1e-12, sd, 1.0)
        return self

    def _pre(self, X, spk):
        X = np.where(np.isfinite(X), X, self.median_)          # impute with TRAIN medians
        return speaker_zscore(X, np.asarray(spk)) if self.per_speaker else X

    def transform(self, X, spk):
        return (self._pre(np.asarray(X, dtype=np.float64), spk) - self.mean_) / self.std_
