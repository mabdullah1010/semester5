"""M6: shared evaluation harness (data loading, LOSO folds, preprocessing, prediction store, layer choice)
and the sanity baselines. Run `python src/m6_harness.py` for the M6.4 layer choice and M6.8 sanity checks."""
from __future__ import annotations
import sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config, write_json
from src.common.metrics import uar, wa, all_metrics
from src.common.preprocess import Preprocessor, speaker_zscore
from src.common.splits import loso_folds

CLASSES = ["neu", "hap", "sad", "ang"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Data:
    policy: str; feature: str; utt_id: np.ndarray; X: np.ndarray; y: np.ndarray; y_soft: np.ndarray
    high_agreement: np.ndarray; session: np.ndarray; speaker: np.ndarray; dialog_type: np.ndarray
    feature_names: list; overlap_any: np.ndarray

    @property
    def n(self):
        return len(self.y)

    def with_layer(self, layer):
        d = Data(**{**self.__dict__}); d.X = self.X[:, layer, :]; return d


def load_dataset(policy="P2", feature="egemaps") -> Data:
    """feature: 'egemaps' -> (N,88); 'wavlm' -> (N,13,768) (all layers; use .with_layer(l))."""
    P = pd.read_parquet(f"data/processed/{policy}.parquet")
    L = pd.read_parquet("data/processed/labels.parquet").set_index("utt_id").loc[P.utt_id]
    if feature == "egemaps":
        F = pd.read_parquet(f"features/egemaps_{policy}.parquet"); assert (F.index == P.utt_id.values).all()
        X, names = F.values.astype(np.float64), list(F.columns)
    else:
        idx = pd.read_csv(f"features/wavlm_{policy}_index.csv").utt_id.values; assert (idx == P.utt_id.values).all()
        X, names = np.load(f"features/wavlm_{policy}.npy").astype(np.float64), [f"d{i}" for i in range(768)]
    ptag = f"{int(round(load_config()['overlap']['padding_sec'] * 100)):03d}"
    return Data(policy, feature, P.utt_id.values, X, L.y_hard.map(CLS_IDX).values.astype(int),
                L[[f"p_{c}" for c in CLASSES]].values, L.high_agreement.values.astype(bool), P.session.values,
                P.speaker_id.values, P.dialog_type.values, names, (P[f"overlap_sec_p{ptag}"] > 0).values)


class FoldData:
    """One LOSO fold. `outer()` fits preprocessing on the 4 training sessions only; `inner()` fits it on the
    3 inner-training sessions only (for tuning on the validation session)."""

    def __init__(self, data: Data, fold: dict, per_speaker=True, X=None):
        self.d, self.f, self.per_speaker = data, fold, per_speaker
        self.X = data.X if X is None else X

    def _prep(self, tr, te):
        pp = Preprocessor(self.per_speaker).fit(self.X[tr], self.d.speaker[tr])
        return pp.transform(self.X[tr], self.d.speaker[tr]), pp.transform(self.X[te], self.d.speaker[te])

    def outer(self):
        return self._prep(self.f["train"], self.f["test"])

    def inner(self):
        return self._prep(self.f["inner_train"], self.f["inner_val"])


def choose_layer(data3d: Data, fold: dict, C=0.1, seed=1337):
    """M6.4: pick the WavLM layer by logistic-regression probe UAR on the inner validation session."""
    from sklearn.linear_model import LogisticRegression
    scores = {}
    for l in range(13):
        fd = FoldData(data3d, fold, X=data3d.X[:, l, :])
        Xtr, Xva = fd.inner()
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=300, random_state=seed)
        clf.fit(Xtr, data3d.y[fold["inner_train"]])
        scores[l] = uar(data3d.y[fold["inner_val"]], clf.predict(Xva))
    return max(scores, key=scores.get), scores


def save_preds(path, data: Data, idx, fold, seed, probs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES]); df.insert(0, "utt_id", data.utt_id[idx])
    df["fold"] = fold; df["seed"] = seed; df["y_true"] = data.y[idx]; df["session"] = data.session[idx]
    df.to_parquet(path); return df


def pooled(preds: pd.DataFrame):
    """Pooled LOSO arrays from a prediction frame (sorted by utt_id)."""
    d = preds.sort_values("utt_id"); P = d[[f"p_{c}" for c in CLASSES]].values
    return d.utt_id.values, d.y_true.values, P.argmax(1), P, d.session.values


# ----------------------------------------------------------------- sanity baselines (M6.8)
def run_sanity():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    cfg = load_config(); seeds = cfg["seeds"]; S = {"checks": {}}
    d = load_dataset("P2", "egemaps"); folds = loso_folds(d.session)
    # majority: per-fold UAR is exactly 0.25 when the test fold has all 4 classes
    fold_uars, pm_pred = [], np.zeros(d.n, int)
    for f in folds:
        maj = np.bincount(d.y[f["train"]], minlength=4).argmax()
        pm_pred[f["test"]] = maj; fold_uars.append(uar(d.y[f["test"]], np.full(len(f["test"]), maj)))
    S["majority"] = dict(per_fold_uar=fold_uars, pooled_uar=uar(d.y, pm_pred))
    S["checks"]["majority_uar_exactly_0.25_per_fold"] = all(abs(u - 0.25) < 1e-12 for u in fold_uars)
    # stratified random (class prior from training rows)
    rs = []
    for sd in seeds:
        rng = np.random.default_rng(sd); pr = np.zeros(d.n, int)
        for f in folds:
            prior = np.bincount(d.y[f["train"]], minlength=4) / len(f["train"])
            pr[f["test"]] = rng.choice(4, size=len(f["test"]), p=prior)
        rs.append(uar(d.y, pr))
    S["stratified_random_uar"] = rs; S["checks"]["stratified_random_about_0.25"] = all(abs(u - 0.25) < 0.03 for u in rs)
    # shuffled-label test: LR on eGeMAPS, labels shuffled inside the training folds
    sh = []
    for sd in seeds:
        rng = np.random.default_rng(sd); pr = np.zeros(d.n, int)
        for f in folds:
            Xtr, Xte = FoldData(d, f).outer()
            ytr = rng.permutation(d.y[f["train"]])
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(Xtr, ytr)
            pr[f["test"]] = clf.predict(Xte)
        sh.append(uar(d.y, pr))
    S["shuffled_label_uar"] = sh
    S["checks"]["shuffled_label_uar_in_0.22_0.28"] = all(0.22 <= u <= 0.28 for u in sh)
    # reference: real labels, same untuned LR (not a result)
    pr = np.zeros(d.n, int)
    for f in folds:
        Xtr, Xte = FoldData(d, f).outer()
        pr[f["test"]] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(Xtr, d.y[f["train"]]).predict(Xte)
    S["reference_untuned_lr_egemaps_uar"] = uar(d.y, pr)
    # speaker-ID probe: random stratified split, without and with per-speaker normalization
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]); acc = {}
    for name, X in [("raw", d.X), ("per_speaker_zscore", speaker_zscore(d.X, d.speaker))]:
        Xs = StandardScaler().fit_transform(X); a = []
        for tr, te in skf.split(Xs, d.speaker):
            clf = LogisticRegression(C=1.0, max_iter=500).fit(Xs[tr], d.speaker[tr]); a.append(clf.score(Xs[te], d.speaker[te]))
        acc[name] = float(np.mean(a))
    S["speaker_id_probe_accuracy"] = acc; S["speaker_id_chance"] = 1 / len(np.unique(d.speaker))
    S["checks"]["normalization_sharply_reduces_speaker_id"] = acc["per_speaker_zscore"] < acc["raw"] - 0.30
    return S


def run_layer_choice():
    d3 = load_dataset("P2", "wavlm"); out = {}
    for f in loso_folds(d3.session):
        t = time.time(); best, sc = choose_layer(d3, f)
        out[f["fold"]] = dict(best_layer=best, val_session=f["val_session"], val_uar_by_layer={k: round(v, 4) for k, v in sc.items()})
        print(f"fold {f['fold']}: layer {best} (val UAR {sc[best]:.3f}) {time.time() - t:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    S = run_sanity(); print({k: S[k] for k in S if k != "checks"}); print(S["checks"])
    layers = run_layer_choice(); S["wavlm_layer_choice"] = layers
    S["all_checks_pass"] = all(S["checks"].values())
    write_json(S, "results/m6/sanity.json"); write_json(layers, "results/m6/wavlm_layers.json")
