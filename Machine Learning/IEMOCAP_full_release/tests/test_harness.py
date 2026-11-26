import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.metrics import uar, wa, macro_f1, within_session_indices, paired_bootstrap, mcnemar, soft_cross_entropy
from src.common.preprocess import Preprocessor, speaker_zscore
from src.common.splits import loso_folds


def make_sessions():
    # 5 sessions x 2 speakers
    sess = np.repeat(np.arange(1, 6), 40); spk = np.array([f"Ses0{s}_{'FM'[i % 2]}" for s, i in zip(sess, range(len(sess)))])
    return sess, spk


def test_folds_have_no_speaker_overlap():
    sess, spk = make_sessions()
    for f in loso_folds(sess):
        assert not set(spk[f["train"]]) & set(spk[f["test"]])
        assert not set(spk[f["inner_train"]]) & set(spk[f["inner_val"]])
        assert not set(spk[f["inner_train"]]) & set(spk[f["test"]]) and not set(spk[f["inner_val"]]) & set(spk[f["test"]])
        assert f["val_session"] == (f["fold"] % 5) + 1 and f["val_session"] != f["fold"]
        assert len(f["train"]) == len(f["inner_train"]) + len(f["inner_val"])
    assert sorted(np.concatenate([f["test"] for f in loso_folds(sess)])) == list(range(len(sess)))


def test_preprocessing_never_fit_on_test_rows():
    rng = np.random.default_rng(0); sess, spk = make_sessions()
    X = rng.normal(5, 2, (200, 6)); tr = sess <= 4; te = ~tr
    pp = Preprocessor().fit(X[tr], spk[tr])
    before = (pp.median_.copy(), pp.mean_.copy(), pp.std_.copy()); Ztr = pp.transform(X[tr], spk[tr])
    Xe = X[te].copy(); Xe[:] = 1e9                              # extreme test values, incl. NaN
    Xe[0, 0] = np.nan
    pp.transform(Xe, spk[te])
    assert all(np.array_equal(a, b) for a, b in zip(before, (pp.median_, pp.mean_, pp.std_)))
    assert np.array_equal(Ztr, pp.transform(X[tr], spk[tr]))
    # refitting on the training rows alone reproduces the same statistics
    pp2 = Preprocessor().fit(X[tr], spk[tr]); assert np.allclose(pp2.mean_, before[1]) and np.allclose(pp2.std_, before[2])


def test_speaker_zscore_uses_only_that_speakers_rows():
    rng = np.random.default_rng(1); X = rng.normal(0, 1, (60, 3)); spk = np.array(["a"] * 30 + ["b"] * 30)
    X[30:] += 100
    Z = speaker_zscore(X, spk)
    assert np.allclose(Z[:30].mean(0), 0, atol=1e-9) and np.allclose(Z[30:].mean(0), 0, atol=1e-9)
    assert np.allclose(Z[:30].std(0), 1) and np.allclose(Z[30:].std(0), 1)


def test_uar_hand_computed():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3]); p = np.array([0, 0, 0, 1, 1, 0, 2, 2])
    # recalls: c0 3/4, c1 1/2, c2 1/1, c3 0/1 -> mean 0.5625 ; accuracy 5/8
    assert abs(uar(y, p) - 0.5625) < 1e-12 and abs(wa(y, p) - 0.625) < 1e-12
    assert abs(uar(y, np.zeros(8, int)) - 0.25) < 1e-12      # constant predictor
    assert 0 < macro_f1(y, p) < 1


def test_bootstrap_stays_within_sessions_and_is_paired():
    sess = np.repeat(np.arange(1, 6), 20); idx = within_session_indices(sess, 50, 0)
    for s in range(1, 6):
        block = idx[:, (s - 1) * 20:s * 20]
        assert ((block >= (s - 1) * 20) & (block < s * 20)).all()
    y = np.tile([0, 1, 2, 3], 25); p = y.copy()
    r = paired_bootstrap(lambda i: uar(y[i], p[i]) - uar(y[i], p[i]), sess, B=100)
    assert r["lo"] == r["hi"] == 0.0                              # identical models -> zero difference every resample


def test_mcnemar_and_soft_ce():
    y = np.array([0, 1, 2, 3] * 10); a = y.copy(); b = y.copy(); b[:8] = (b[:8] + 1) % 4
    m = mcnemar(y, a, b); assert m["a_right_b_wrong"] == 8 and m["a_wrong_b_right"] == 0 and m["p_value"] < 0.05
    P = np.full((2, 4), 0.25); Y = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0]])
    assert abs(soft_cross_entropy(P, Y) - np.log(4)) < 1e-9
