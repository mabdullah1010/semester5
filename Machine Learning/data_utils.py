"""
data_utils.py
-------------
Loading helpers for the UCI Ionosphere dataset (COM 307, Project 3).

The raw file `ionosphere.data` has no header row. Each line holds 34 continuous
features followed by a class label, 'g' (good: the radar return shows structure
in the ionosphere) or 'b' (bad: the signal passed straight through).

The 34 features come from 17 pulse numbers, two values per pulse (the real and
imaginary parts of a complex autocorrelation value). Attribute 2k-1 and
Attribute 2k belong to pulse k, so A1/A2 are pulse 1, A3/A4 are pulse 2, and so on.

Every later step of the project imports `load_ionosphere` from here, so the data
is read and encoded the same way everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 34
N_PULSES = 17
FEATURE_NAMES = [f"A{i}" for i in range(1, N_FEATURES + 1)]

# Class encoding used throughout the project: good = 1, bad = 0.
LABEL_MAP = {"g": 1, "b": 0}
CLASS_NAMES = {1: "good", 0: "bad"}


def load_ionosphere(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read the raw Ionosphere file.

    Parameters
    ----------
    path : str or Path
        Location of `ionosphere.data`.

    Returns
    -------
    X : np.ndarray, shape (351, 34), float
        Feature matrix, rows in the original file order.
    y : np.ndarray, shape (351,), int
        Labels, 1 = good, 0 = bad.
    df : pd.DataFrame
        The same data with named columns A1..A34 plus 'class' ('g'/'b') and
        'label' (1/0), convenient for exploration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find the data file at {path.resolve()}")

    df = pd.read_csv(path, header=None, names=FEATURE_NAMES + ["class"])

    if df.shape[1] != N_FEATURES + 1:
        raise ValueError(f"Expected {N_FEATURES + 1} columns, found {df.shape[1]}")

    unknown = set(df["class"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected class labels in file: {unknown}")

    df["label"] = df["class"].map(LABEL_MAP).astype(int)

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y, df


def pulse_of(feature_index: int) -> tuple[int, str]:
    """Map a 0-based feature column to (pulse number 1..17, 'real' or 'imag').

    Example: column 0 (A1) -> (1, 'real'), column 5 (A6) -> (3, 'imag').
    """
    pulse = feature_index // 2 + 1
    part = "real" if feature_index % 2 == 0 else "imag"
    return pulse, part
