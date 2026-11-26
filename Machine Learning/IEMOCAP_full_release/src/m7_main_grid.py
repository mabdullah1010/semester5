"""M7: 2x2 grid (features x model type), hard labels. One run = (model, feature set) over 5 LOSO folds.
Hyperparameters are tuned on the inner validation session only; test sessions are never used for tuning.
Usage: python src/m7_main_grid.py --policy P2 --models l1lr,tree --features egemaps,wavlm [--force]"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config
from src.common.metrics import uar, wa, macro_f1
from src.common.splits import loso_folds
from src.m6_harness import load_dataset, FoldData, save_preds, CLASSES

CFG = load_config(); SEEDS = CFG["seeds"]; BUDGET_FEATS = CFG["complexity_budget"]["lr_max_nonzero_features"]
BUDGET_LEAVES = CFG["complexity_budget"]["tree_max_leaves"]
GROUP = {"l1lr": "interpretable", "tree": "interpretable", "l2lr": "interpretable_unbudgeted",
         "hgb": "blackbox", "rf": "blackbox", "mlp": "blackbox"}
STOCHASTIC = {"hgb", "rf", "mlp"}
C_L1 = np.logspace(-3.5, -0.5, 10)
C_L2 = np.logspace(-4, 0, 7)


def _lr(C, penalty, seed=1337):
    from sklearn.linear_model import LogisticRegression
    if penalty == "l1":
        return LogisticRegression(l1_ratio=1.0, C=C, solver="saga", class_weight="balanced", max_iter=400, tol=1e-3, random_state=seed)
    return LogisticRegression(C=C, class_weight="balanced", max_iter=500, random_state=seed)


def _nnz(clf):
    return int((np.abs(clf.coef_).max(0) > 1e-8).sum())      # a feature counts if ANY class uses it


# --- each fn: (Xtr_in, ytr_in, Xva, yva, Xtr, ytr, Xte) -> (dict seed->probs, meta)
def run_l1lr(a):
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = None
    for C in C_L1:
        clf = _lr(C, "l1").fit(Xi, yi); n = _nnz(clf)
        if n <= BUDGET_FEATS:
            u = uar(yv, clf.predict(Xv))
            if best is None or u > best[1]:
                best = (C, u, n)
    if best is None:
        best = (C_L1[0], float("nan"), -1)
    ci = list(C_L1).index(best[0])
    while True:                                           # enforce the budget on the final (4-session) fit
        clf = _lr(C_L1[ci], "l1").fit(Xt, yt); n = _nnz(clf)
        if n <= BUDGET_FEATS or ci == 0:
            break
        ci -= 1
    return {1337: clf.predict_proba(Xe)}, dict(C=float(C_L1[ci]), C_inner=float(best[0]), val_uar=best[1], nnz=n,
                                              selected=np.where(np.abs(clf.coef_).max(0) > 1e-8)[0].tolist(),
                                              coef=clf.coef_[:, np.abs(clf.coef_).max(0) > 1e-8].round(4).tolist())


def run_l2lr(a):
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = max(((C, uar(yv, _lr(C, "l2").fit(Xi, yi).predict(Xv))) for C in C_L2), key=lambda t: t[1])
    clf = _lr(best[0], "l2").fit(Xt, yt)
    return {1337: clf.predict_proba(Xe)}, dict(C=float(best[0]), val_uar=best[1])


def run_tree(a):
    from sklearn.tree import DecisionTreeClassifier
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = None
    for leaves in (4, 8, BUDGET_LEAVES):
        for msl in (5, 20, 50):
            t = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=msl, class_weight="balanced", random_state=1337).fit(Xi, yi)
            u = uar(yv, t.predict(Xv))
            if best is None or u > best[2]:
                best = (leaves, msl, u)
    t = DecisionTreeClassifier(max_leaf_nodes=best[0], min_samples_leaf=best[1], class_weight="balanced", random_state=1337).fit(Xt, yt)
    assert t.get_n_leaves() <= BUDGET_LEAVES
    return {1337: t.predict_proba(Xe)}, dict(max_leaf_nodes=best[0], min_samples_leaf=best[1], val_uar=best[2], n_leaves=int(t.get_n_leaves()))


def run_hgb(a):
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = None
    mk = lambda lr, ml, sd: H(learning_rate=lr, max_leaf_nodes=ml, max_iter=300, early_stopping=True, n_iter_no_change=15,
                              validation_fraction=0.1, class_weight="balanced", random_state=sd)
    for lr in (0.03, 0.1):
        for ml in (15, 31):
            m = mk(lr, ml, 1337).fit(Xi, yi); u = uar(yv, m.predict(Xv))
            if best is None or u > best[2]:
                best = (lr, ml, u)
    return {sd: mk(best[0], best[1], sd).fit(Xt, yt).predict_proba(Xe) for sd in SEEDS}, dict(learning_rate=best[0], max_leaf_nodes=best[1], val_uar=best[2])


def run_rf(a):
    from sklearn.ensemble import RandomForestClassifier as R
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = None
    mk = lambda mf, msl, n, sd: R(n_estimators=n, max_features=mf, min_samples_leaf=msl, class_weight="balanced_subsample", n_jobs=-1, random_state=sd)
    for mf in ("sqrt", 0.3):
        for msl in (1, 5):
            m = mk(mf, msl, 200, 1337).fit(Xi, yi); u = uar(yv, m.predict(Xv))   # tuned with 200 trees, final fit 500
            if best is None or u > best[2]:
                best = (mf, msl, u)
    return {sd: mk(best[0], best[1], 500, sd).fit(Xt, yt).predict_proba(Xe) for sd in SEEDS}, dict(max_features=best[0], min_samples_leaf=best[1], val_uar=best[2])


def _mlp_train(Xtr, ytr, n_layers, seed, epochs, Xva=None, yva=None, patience=10):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(4)
    d = Xtr.shape[1]; layers = []; k = d
    for _ in range(n_layers):
        layers += [nn.Linear(k, 256), nn.ReLU(), nn.Dropout(0.3)]; k = 256
    net = nn.Sequential(*layers, nn.Linear(k, 4))
    w = torch.tensor(len(ytr) / (4 * np.maximum(np.bincount(ytr, minlength=4), 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss(weight=w)
    X = torch.tensor(Xtr, dtype=torch.float32); y = torch.tensor(ytr); g = torch.Generator().manual_seed(seed)
    best = (-1, 0); hist = []
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]; opt.zero_grad(); lossf(net(X[b]), y[b]).backward(); opt.step()
        if Xva is not None:
            net.eval()
            with torch.no_grad():
                u = uar(yva, net(torch.tensor(Xva, dtype=torch.float32)).argmax(1).numpy())
            if u > best[0]:
                best = (u, ep)
            if ep - best[1] >= patience:
                break
    return net, best


def run_mlp(a):
    import torch
    Xi, yi, Xv, yv, Xt, yt, Xe = a; best = None
    for nl in (1, 2):
        _, (u, ep) = _mlp_train(Xi, yi, nl, 1337, 100, Xv, yv)
        if best is None or u > best[1]:
            best = (nl, u, ep)
    out = {}
    for sd in SEEDS:
        net, _ = _mlp_train(Xt, yt, best[0], sd, max(best[2], 5)); net.eval()
        with torch.no_grad():
            out[sd] = torch.softmax(net(torch.tensor(Xe, dtype=torch.float32)), 1).numpy()
    return out, dict(hidden_layers=best[0], epochs=best[2], val_uar=best[1])


RUNNERS = dict(l1lr=run_l1lr, l2lr=run_l2lr, tree=run_tree, hgb=run_hgb, rf=run_rf, mlp=run_mlp)


def run(model, feature, policy="P2", outdir=None, force=False, folds_only=None, layers=None, partial=False):
    outdir = Path(outdir or ("results/m7" if policy == "P2" else f"results/m7_{policy}"))
    pf = outdir / "preds" / f"{model}__{feature}.parquet"; hf = outdir / "hparams" / f"{model}__{feature}.json"
    if pf.exists() and not force and folds_only is None:
        print(f"skip {model} {feature} (cached)"); return
    d = load_dataset(policy, feature); folds = loso_folds(d.session); parts, meta = [], {}
    if feature == "wavlm":
        layers = layers or {int(k): v["best_layer"] for k, v in json.load(open(
            "results/m6/wavlm_layers.json" if policy == "P2" else f"results/m6/wavlm_layers_{policy}.json")).items()}
    for f in folds:
        if folds_only and f["fold"] not in folds_only:
            continue
        t0 = time.time(); X = d.X[:, layers[f["fold"]], :] if feature == "wavlm" else d.X
        fd = FoldData(d, f, X=X); Xi, Xv = fd.inner(); Xt, Xe = fd.outer()
        a = (Xi, d.y[f["inner_train"]], Xv, d.y[f["inner_val"]], Xt, d.y[f["train"]], Xe)
        probs, m = RUNNERS[model](a)
        for sd, P in probs.items():
            parts.append(save_preds(outdir / "tmp.parquet", d, f["test"], f["fold"], sd, P))
        m["uar_test_fold_seed0"] = uar(d.y[f["test"]], list(probs.values())[0].argmax(1)); m["seconds"] = round(time.time() - t0, 1)
        if feature == "wavlm":
            m["layer"] = layers[f["fold"]]
        meta[f["fold"]] = m
        print(f"{policy} {model} {feature} fold {f['fold']}: test UAR {m['uar_test_fold_seed0']:.3f} val {m.get('val_uar', float('nan')):.3f} {m['seconds']}s", flush=True)
    if folds_only is None:
        pf.parent.mkdir(parents=True, exist_ok=True); hf.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_parquet(pf); hf.write_text(json.dumps(meta, indent=1, default=str))
    elif partial:                                          # one-fold runs, merged later with --merge
        pdir = outdir / "partial"; pdir.mkdir(parents=True, exist_ok=True)
        for fo in folds_only:
            pd.concat([x for x in parts if (x.fold == fo).all()]).to_parquet(pdir / f"{model}__{feature}__f{fo}.parquet")
            (pdir / f"{model}__{feature}__f{fo}.json").write_text(json.dumps({fo: meta[fo]}, indent=1, default=str))
    return meta


def merge_partial(model, feature, policy="P2", outdir=None):
    outdir = Path(outdir or ("results/m7" if policy == "P2" else f"results/m7_{policy}")); pdir = outdir / "partial"
    files = [pdir / f"{model}__{feature}__f{k}.parquet" for k in range(1, 6)]
    assert all(f.exists() for f in files), "missing folds: " + str([f.name for f in files if not f.exists()])
    (outdir / "preds").mkdir(parents=True, exist_ok=True); (outdir / "hparams").mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_parquet(f) for f in files]).to_parquet(outdir / "preds" / f"{model}__{feature}.parquet")
    meta = {}
    for k in range(1, 6):
        meta.update(json.loads((pdir / f"{model}__{feature}__f{k}.json").read_text()))
    (outdir / "hparams" / f"{model}__{feature}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("merged", model, feature)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="P2"); ap.add_argument("--models", default="l1lr,tree,l2lr,hgb,rf,mlp")
    ap.add_argument("--features", default="egemaps,wavlm"); ap.add_argument("--force", action="store_true"); ap.add_argument("--folds", default=None); ap.add_argument("--partial", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    for feat in a.features.split(","):
        for mdl in a.models.split(","):
            if a.merge:
                merge_partial(mdl, feat, a.policy); continue
            run(mdl, feat, a.policy, force=a.force, folds_only=[int(x) for x in a.folds.split(",")] if a.folds else None, partial=a.partial)
