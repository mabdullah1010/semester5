"""Metrics and significance tests. Classes are integers 0..K-1."""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest, chi2


def confusion_matrix(y, p, k=4):
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def per_class_recall(y, p, k=4):
    c = confusion_matrix(y, p, k); n = c.sum(1)
    return np.where(n > 0, c.diagonal() / np.maximum(n, 1), np.nan)


def uar(y, p, k=4):
    """Unweighted average recall over the classes present in y."""
    return float(np.nanmean(per_class_recall(y, p, k)))


def wa(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(y, p, k=4):
    c = confusion_matrix(y, p, k); tp = c.diagonal().astype(float)
    prec = tp / np.maximum(c.sum(0), 1); rec = tp / np.maximum(c.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1[c.sum(1) > 0].mean())


def all_metrics(y, p, k=4):
    return dict(UAR=uar(y, p, k), WA=wa(y, p), macroF1=macro_f1(y, p, k),
                per_class_recall=per_class_recall(y, p, k).tolist(), confusion=confusion_matrix(y, p, k).tolist())


def soft_cross_entropy(probs, y_soft, eps=1e-12):
    """Mean cross-entropy of predicted probabilities against soft target vectors (nats)."""
    return float(-(np.asarray(y_soft) * np.log(np.clip(probs, eps, 1))).sum(1).mean())


def within_session_indices(sessions, B, seed):
    """B x N matrix of bootstrap indices; each resample draws within each session so the session mix is fixed."""
    rng = np.random.default_rng(seed); sessions = np.asarray(sessions)
    out = np.empty((B, len(sessions)), dtype=np.int64); pos = 0
    for s in np.unique(sessions):
        idx = np.where(sessions == s)[0]
        out[:, pos:pos + len(idx)] = idx[rng.integers(0, len(idx), size=(B, len(idx)))]
        pos += len(idx)
    return out


def paired_bootstrap(stat_fn, sessions, B=2000, seed=1337, ci=0.95):
    """stat_fn(idx) -> float. Returns dict(point, lo, hi, samples). All statistics inside stat_fn see
    the SAME resample, so differences between models are paired."""
    sessions = np.asarray(sessions)
    point = float(stat_fn(np.arange(len(sessions))))
    idx = within_session_indices(sessions, B, seed)
    s = np.array([stat_fn(i) for i in idx])
    a = (1 - ci) / 2
    return dict(point=point, lo=float(np.quantile(s, a)), hi=float(np.quantile(s, 1 - a)),
                excludes_zero=bool(np.quantile(s, a) > 0 or np.quantile(s, 1 - a) < 0), samples=s)


def uar_diff_ci(y, pa, pb, sessions, B=2000, seed=1337, ci=0.95):
    """CI for UAR(A) - UAR(B) with paired within-session bootstrap."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    r = paired_bootstrap(lambda i: uar(y[i], pa[i]) - uar(y[i], pb[i]), sessions, B, seed, ci)
    r.pop("samples"); return r


def mcnemar(y, pa, pb):
    """McNemar's exact test on accuracy (A vs B)."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return dict(a_right_b_wrong=b, a_wrong_b_right=c, p_value=float(p))
