# COM 307
# Muhammad Abdullah
# Project 3
# Date: Nov 1st 2026


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 34
N_PULSES = 17


FEATURE_NAMES = [f"A{i}" for i in range(1, N_FEATURES + 1)]


LABEL_MAP = {"g": 1, "b": 0}
CLASS_NAMES = {1: "good", 0: "bad"}


def load_ionosphere(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

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


    pulse = feature_index // 2 + 1
    part = "real" if feature_index % 2 == 0 else "imag"
    return pulse, part
