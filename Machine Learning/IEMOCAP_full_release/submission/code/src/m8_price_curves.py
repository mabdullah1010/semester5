"""M8: price-of-interpretability curves on eGeMAPS (P2, hard labels). Each point is a FIXED setting (C or max_leaf_nodes)
fit on the 4 training sessions of each LOSO fold; nothing is selected on test data. COULD item M8.3 (WavLM L1 sweep) not run here."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config, write_json
from src.common.metrics import uar, within_session_indices
from src.common.splits import loso_folds
from src.m6_harness import load_dataset, FoldData

CFG = load_config(); BUD_F = CFG["complexity_budget"]["lr_max_nonzero_features"]; BUD_L = CFG["complexity_budget"]["tree_max_leaves"]
CS = np.logspace(-3.5, 0.5, 30); LEAVES = [2, 4, 8, 16, 32, 64, 128, 256]; TREE_MSL = 5   # fixed before running


def l1_fold(C, f, d):
    from sklearn.linear_model import LogisticRegression
    Xt, Xe = FoldData(d, f).outer()
    clf = LogisticRegression(l1_ratio=1.0, C=C, solver="saga", class_weight="balanced", max_iter=600, tol=1e-3, random_state=1337).fit(Xt, d.y[f["train"]])
    return f["fold"], C, int((np.abs(clf.coef_).max(0) > 1e-8).sum()), clf.predict(Xe)


def tree_fold(L, f, d):
    from sklearn.tree import DecisionTreeClassifier
    Xt, Xe = FoldData(d, f).outer()
    t = DecisionTreeClassifier(max_leaf_nodes=L, min_samples_leaf=TREE_MSL, class_weight="balanced", random_state=1337).fit(Xt, d.y[f["train"]])
    return f["fold"], L, int(t.get_n_leaves()), t.predict(Xe)


def collect(res, d, folds, key_name):
    rows = []
    by = {}
    for fo, k, n, pred in res:
        by.setdefault(k, {})[fo] = (n, pred)
    for k, v in by.items():
        pred = np.zeros(d.n, int)
        for f in folds:
            pred[f["test"]] = v[f["fold"]][1]
        rows.append(dict(**{key_name: k}, complexity=np.mean([v[fo][0] for fo in v]), complexity_by_fold=[v[fo][0] for fo in sorted(v)],
                         UAR=uar(d.y, pred), pred=pred))
    return rows


def main():
    d = load_dataset("P2", "egemaps"); folds = loso_folds(d.session); B = CFG["bootstrap"]["n_resamples"]
    idx = within_session_indices(d.session, B, CFG["seed"])
    l1 = collect(Parallel(n_jobs=5)(delayed(l1_fold)(C, f, d) for C in CS for f in folds), d, folds, "C")
    tr = collect(Parallel(n_jobs=5)(delayed(tree_fold)(L, f, d) for L in LEAVES for f in folds), d, folds, "max_leaf_nodes")
    out = []
    for name, rows in [("l1lr", l1), ("tree", tr)]:
        for r in sorted(rows, key=lambda r: r["complexity"]):
            s = np.array([uar(d.y[i], r["pred"][i]) for i in idx])
            out.append(dict(model=name, setting=r.get("C", r.get("max_leaf_nodes")), complexity=r["complexity"],
                            complexity_by_fold=r["complexity_by_fold"], UAR=r["UAR"], lo=np.quantile(s, .025), hi=np.quantile(s, .975)))
    cur = pd.DataFrame(out); cur.to_csv("results/m8/curves.csv", index=False)

    # Budget-respecting FIXED-C point: the largest C whose fits have <= 20 nonzero features in EVERY fold.
    # Chosen from nonzero counts only (no UAR), so no test selection. Compared with the M7 L1-LR and the M7 black box.
    from src.m7_costs import load_runs
    ok = [r for r in l1 if max(r["complexity_by_fold"]) <= BUD_F]; strict = max(ok, key=lambda r: r["C"])
    runs, _ = load_runs(Path("results/m7")); y = runs[("mlp", "egemaps")]["y"]
    assert (runs[("mlp", "egemaps")]["utt"] == d.utt_id[np.argsort(d.utt_id)]).all()
    order = np.argsort(d.utt_id)                                   # M7 predictions are sorted by utt_id
    sp = strict["pred"][order]; ys = d.y[order]; sess = d.session[order]; idx2 = within_session_indices(sess, B, CFG["seed"])
    mlp = list(runs[("mlp", "egemaps")]["seeds"].values()); l1m7 = list(runs[("l1lr", "egemaps")]["seeds"].values())
    U = lambda i, p: uar(ys[i], p[i])
    d_mlp = np.array([np.mean([U(i, m["pred"]) for m in mlp]) - U(i, sp) for i in idx2])
    d_m7 = np.array([U(i, sp) - np.mean([U(i, m["pred"]) for m in l1m7]) for i in idx2])
    pt = lambda a, b=None: float(a)
    strict_info = dict(C=float(strict["C"]), nnz_by_fold=strict["complexity_by_fold"], UAR=float(strict["UAR"]),
        delta_model_vs_mlp_egemaps=dict(point=float(np.mean([uar(ys, m["pred"]) for m in mlp]) - strict["UAR"]), lo=float(np.quantile(d_mlp, .025)), hi=float(np.quantile(d_mlp, .975))),
        gain_over_m7_l1lr=dict(point=float(strict["UAR"] - np.mean([uar(ys, m["pred"]) for m in l1m7])), lo=float(np.quantile(d_m7, .025)), hi=float(np.quantile(d_m7, .975))),
        m7_l1lr_mean_nnz=float(np.mean([v["nnz"] for v in json.load(open("results/m7/hparams/l1lr__egemaps.json")).values()])))

    costs = json.load(open("results/m7/costs.json")); tab = {(r["model"], r["features"]): r["UAR"] for r in costs["table"]}
    bb_e, bb_w, ceil = tab[("mlp", "egemaps")], tab[("mlp", "wavlm")], costs["human_ceiling_uar"]; l2_e = tab[("l2lr", "egemaps")]
    S = dict(strict_budget_fixed_C_point=strict_info, black_box_egemaps_mlp=bb_e, black_box_wavlm_mlp=bb_w, human_ceiling=ceil, l2lr_egemaps_unbudgeted=l2_e, tree_min_samples_leaf=TREE_MSL, checks={})
    for name in ["l1lr", "tree"]:
        c = cur[cur.model == name].sort_values("complexity"); S[name] = {}
        for tgt_name, tgt in [("black_box_egemaps", bb_e), ("black_box_wavlm", bb_w)]:
            for pts in (1, 3):
                thr = tgt - pts / 100; ok = c[c.UAR >= thr]
                first = float(ok.complexity.iloc[0]) if len(ok) else None
                stays = None
                for i in range(len(c)):
                    if (c.UAR.iloc[i:] >= thr).all():
                        stays = float(c.complexity.iloc[i]); break
                S[name][f"within_{pts}pt_of_{tgt_name}"] = dict(threshold=round(thr, 4), first_crossing_complexity=first, stays_above_from_complexity=stays)
        S[name]["max_UAR"] = float(c.UAR.max()); S[name]["UAR_at_budget"] = float(np.interp(BUD_F if name == "l1lr" else BUD_L, c.complexity, c.UAR))
        S[name]["spearman_uar_vs_log_complexity"] = float(pd.Series(c.UAR.values).corr(pd.Series(np.log(c.complexity.values)), method="spearman"))
        S["checks"][f"{name}_curve_ends_at_or_below_black_box"] = bool(c.UAR.iloc[-3:].max() <= bb_e + 0.01)
        S["checks"][f"{name}_no_nan"] = bool(c[["UAR", "lo", "hi", "complexity"]].notna().all().all())
    S["l1_nnz_range"] = [float(cur[cur.model == 'l1lr'].complexity.min()), float(cur[cur.model == 'l1lr'].complexity.max())]

    for name, bud, xl, fn in [("l1lr", BUD_F, "mean # nonzero eGeMAPS features (log scale)", "m8_price_curve_lr.png"),
                              ("tree", BUD_L, "mean # leaves (log scale)", "m8_price_curve_tree.png")]:
        c = cur[(cur.model == name) & (cur.complexity > 0)].sort_values("complexity")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(c.complexity, c.UAR, "o-", c="#4C72B0", label=f"{'L1-LR' if name == 'l1lr' else 'Decision tree'} (eGeMAPS)")
        ax.fill_between(c.complexity, c.lo, c.hi, alpha=.2, color="#4C72B0")
        ax.axhline(bb_e, c="#DD8452", ls="-", label=f"MLP on eGeMAPS {bb_e:.3f}"); ax.axhline(bb_w, c="#C44E52", ls="-", label=f"MLP on WavLM {bb_w:.3f}")
        ax.axhline(ceil, c="k", ls="--", lw=1, label=f"approx. human ceiling {ceil:.3f}"); ax.axhline(.25, c="gray", ls=":", label="chance 0.25")
        ax.axhline(l2_e, c="#8172B2", ls="-.", lw=1, label=f"L2-LR eGeMAPS (unbudgeted) {l2_e:.3f}")
        ax.axvline(bud, c="green", ls="--", lw=1, label=f"M7 budget ({bud})"); ax.set_xscale("log")
        if name == "l1lr":
            ax.plot([np.mean(strict["complexity_by_fold"])], [strict["UAR"]], "*", c="green", ms=13, label=f"largest C with <=20 feats in every fold ({strict['UAR']:.3f})")
            ax.plot([S["strict_budget_fixed_C_point"]["m7_l1lr_mean_nnz"]], [tab[("l1lr", "egemaps")]], "s", c="red", label=f"M7 tuned L1-LR ({tab[('l1lr', 'egemaps')]:.3f})")
        ax.set_xlabel(xl); ax.set_ylabel("pooled LOSO UAR (95% CI)"); ax.set_ylim(0.2, 0.75); ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"Price of interpretability: {'L1 logistic regression' if name == 'l1lr' else 'decision tree'}"); plt.tight_layout()
        plt.savefig(f"figures/{fn}", dpi=130); plt.close(); S["checks"][f"{fn}_exists"] = Path(f"figures/{fn}").exists()
    write_json(S, "results/m8/summary.json")
    print(cur[["model", "setting", "complexity", "UAR", "lo", "hi"]].round(3).to_string(index=False)); print(json.dumps({k: S[k] for k in ["strict_budget_fixed_C_point", "checks"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
