"""Leave-one-session-out splits with an inner validation session."""
import numpy as np


def loso_folds(sessions):
    """Fold k tests on session k; validates on session (k mod 5)+1 (inside the training sessions);
    inner_train = remaining 3 sessions. All values are index arrays into `sessions`."""
    s = np.asarray(sessions); folds = []
    for k in range(1, 6):
        val = (k % 5) + 1
        folds.append(dict(fold=k, val_session=val,
                          test=np.where(s == k)[0], train=np.where(s != k)[0],
                          inner_val=np.where(s == val)[0], inner_train=np.where((s != k) & (s != val))[0]))
    return folds
