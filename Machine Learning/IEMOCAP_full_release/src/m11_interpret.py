"""M11: what does the readable model learn? Budgeted L1-LR coefficients (stability over folds), budgeted tree, plain-language summary.
Uses the pre-registered M7 models (primary policy P2, eGeMAPS, hard labels). Coefficients are in standardized units."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.metrics import uar
from src.common.splits import loso_folds
from src.m6_harness import load_dataset, FoldData, CLASSES

TOK = {"loudness": "loudness", "F0semitoneFrom27.5Hz": "pitch (F0)", "jitterLocal": "pitch jitter", "shimmerLocaldB": "amplitude shimmer",
       "HNRdBACF": "harmonics-to-noise ratio", "logRelF0-H1-H2": "H1-H2 (voice quality)", "logRelF0-H1-A3": "H1-A3 (voice quality)",
       "alphaRatio": "alpha ratio (high/low-band energy)", "hammarbergIndex": "Hammarberg index (spectral peak ratio)", "spectralFlux": "spectral flux",
       "slope0-500": "spectral slope 0-500 Hz", "slope500-1500": "spectral slope 500-1500 Hz", "equivalentSoundLevel": "equivalent sound level",
       "loudnessPeaksPerSec": "loudness peaks per second", "VoicedSegmentsPerSec": "voiced segments per second", "MeanVoicedSegmentLengthSec": "mean voiced segment length",
       "StddevVoicedSegmentLengthSec": "variability of voiced segment length", "MeanUnvoicedSegmentLength": "mean unvoiced segment length",
       "StddevUnvoicedSegmentLength": "variability of unvoiced segment length", "F1frequency": "formant F1 frequency", "F2frequency": "formant F2 frequency",
       "F3frequency": "formant F3 frequency", "F1bandwidth": "formant F1 bandwidth", "F2bandwidth": "formant F2 bandwidth", "F3bandwidth": "formant F3 bandwidth",
       "F1amplitudeLogRelF0": "formant F1 amplitude", "F2amplitudeLogRelF0": "formant F2 amplitude", "F3amplitudeLogRelF0": "formant F3 amplitude"}
STAT = {"amean": "average", "stddevNorm": "variability (normalized)", "stddevFallingSlope": "variability of falling slopes", "stddevRisingSlope": "variability of rising slopes",
        "meanFallingSlope": "mean falling slope", "meanRisingSlope": "mean rising slope", "percentile20.0": "20th percentile", "percentile50.0": "median",
        "percentile80.0": "80th percentile", "pctlrange0-2": "20-80th percentile range", "stddev": "variability"}


def humanize(name):
    """openSMILE feature name -> readable text, e.g. F0semitoneFrom27.5Hz_sma3nz_amean -> 'pitch (F0) - average'."""
    parts = name.split("_"); base = parts[0]; stat = parts[-1] if len(parts) > 1 and parts[-1] in STAT else ""
    seg = ""; m = re.match(r"^(alphaRatio|hammarbergIndex|spectralFlux|slope|mfcc\d+)(UV|V)?(.*)$", base)
    if m:
        root, s, rest = m.groups(); seg = {"V": " [voiced]", "UV": " [unvoiced]", None: ""}[s]; base = root + rest
    label = f"MFCC {base[4:]}" if base.startswith("mfcc") else TOK.get(base, base)
    return label + seg + (f" - {STAT[stat]}" if stat else "")


def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    d = load_dataset("P2", "egemaps"); names = d.feature_names; h = json.load(open("results/m7/hparams/l1lr__egemaps.json"))
    # ---- M11.1 L1-LR coefficients
    coef = np.zeros((5, 88, 4)); sel = np.zeros((5, 88), bool)
    for f, v in h.items():
        ix = v["selected"]; coef[int(f) - 1][ix] = np.array(v["coef"]).T; sel[int(f) - 1][ix] = True
    stab = sel.sum(0); meanc = coef.mean(0)
    rows = []
    for j in np.where(stab > 0)[0]:
        rows.append(dict(feature=names[j], plain_name=humanize(names[j]), folds_selected=int(stab[j]),
                         **{f"coef_{c}": float(meanc[j, i]) for i, c in enumerate(CLASSES)}, max_abs_coef=float(np.abs(meanc[j]).max())))
    tab = pd.DataFrame(rows).sort_values(["folds_selected", "max_abs_coef"], ascending=False); tab.to_csv("results/m11/coefficients.csv", index=False)
    S = dict(n_features_ever_selected=int(len(tab)), selected_in_all_5_folds=int((stab == 5).sum()), selected_in_ge3_folds=int((stab >= 3).sum()),
             selected_in_1_fold_only=int((stab == 1).sum()), per_fold_nnz=[int(x) for x in sel.sum(1)],
             jaccard_between_folds=float(np.mean([(sel[a] & sel[b]).sum() / (sel[a] | sel[b]).sum() for a in range(5) for b in range(a + 1, 5)])))

    # ---- M11.2 tree from the representative fold (median test UAR of the M7 tree)
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    ht = json.load(open("results/m7/hparams/tree__egemaps.json")); ups = {int(k): v["uar_test_fold_seed0"] for k, v in ht.items()}
    rep = sorted(ups, key=ups.get)[2]; fold = loso_folds(d.session)[rep - 1]
    Xt, Xe = FoldData(d, fold).outer(); p = ht[str(rep)]
    t = DecisionTreeClassifier(max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"], class_weight="balanced", random_state=1337).fit(Xt, d.y[fold["train"]])
    assert t.get_n_leaves() <= 16 and abs(uar(d.y[fold["test"]], t.predict(Xe)) - ups[rep]) < 1e-9
    txt = export_text(t, feature_names=names, class_names=CLASSES) if False else export_text(t, feature_names=list(names))
    Path("results/m11/tree_fold%d.txt" % rep).write_text(txt)
    fig, ax = plt.subplots(figsize=(18, 9)); plot_tree(t, feature_names=[humanize(n) for n in names], class_names=CLASSES, filled=True, rounded=True, fontsize=7, ax=ax, impurity=False)
    ax.set_title(f"Budgeted decision tree (fold {rep} test session, {t.get_n_leaves()} leaves, test UAR {ups[rep]:.3f}); features are standardized"); plt.tight_layout(); plt.savefig("figures/m11_tree.png", dpi=110); plt.close()
    imp = pd.Series(t.feature_importances_, index=names).sort_values(ascending=False); S["tree"] = dict(fold=rep, n_leaves=int(t.get_n_leaves()), test_uar=ups[rep],
        top_features={humanize(k): round(float(v), 3) for k, v in imp.head(5).items()})
    tree_feats_all = {}
    for f_, pp in ht.items():
        fd = FoldData(d, loso_folds(d.session)[int(f_) - 1]); a, _ = fd.outer(); tt = DecisionTreeClassifier(max_leaf_nodes=pp["max_leaf_nodes"], min_samples_leaf=pp["min_samples_leaf"], class_weight="balanced", random_state=1337).fit(a, d.y[fd.f["train"]])
        tree_feats_all[f_] = set(np.array(names)[tt.feature_importances_ > 0])
    S["tree_feature_sets_per_fold_overlap_jaccard"] = float(np.mean([len(tree_feats_all[a] & tree_feats_all[b]) / len(tree_feats_all[a] | tree_feats_all[b]) for a in tree_feats_all for b in tree_feats_all if a < b]))

    # ---- raw class means (sanity for direction) for the top features
    raw = pd.DataFrame(d.X, columns=names); raw["y"] = d.y
    top = tab[tab.folds_selected >= 3].head(12).feature.tolist()
    S["class_means_raw_top_features"] = {humanize(f_): {CLASSES[i]: round(float(raw[raw.y == i][f_].mean()), 3) for i in range(4)} for f_ in top}

    # ---- M11.4 plain-language story (stable features only: selected in >= 3 of 5 folds)
    stable = tab[tab.folds_selected >= 3]; lines = ["# What the budgeted L1 logistic regression learned (eGeMAPS, P2, hard labels)", "",
        f"Generated by `src/m11_interpret.py` from `results/m7/hparams/l1lr__egemaps.json`. The model uses at most 20 features per fold (mean {np.mean(S['per_fold_nnz']):.1f}). "
        f"{S['n_features_ever_selected']} different features were chosen across the 5 folds; only {S['selected_in_all_5_folds']} in all 5 and {S['selected_in_ge3_folds']} in at least 3, so the "
        f"*specific* feature list is unstable (mean fold-to-fold Jaccard {S['jaccard_between_folds']:.2f}), as expected when eGeMAPS features are highly correlated. "
        "Coefficients below are averages over the 5 folds in standardized units (per-speaker z-scored); a positive value means a higher feature value pushes the prediction toward that class.", ""]
    for i, c in enumerate(CLASSES):
        s = stable.assign(v=stable[f"coef_{c}"]); s = s[s.v.abs() >= 0.05].sort_values("v", key=np.abs, ascending=False).head(3)
        if len(s):
            lines.append(f"- **{c}**: " + "; ".join(f"{'higher' if r.v > 0 else 'lower'} {r.plain_name} ({r.v:+.2f}, {r.folds_selected}/5 folds)" for r in s.itertuples()))
        else:
            lines.append(f"- **{c}**: no stable feature with a non-negligible coefficient (|coef| < 0.05). The model predicts {c} mostly by default, when the evidence for the other classes is weak.")
    lines += ["", "## Stable features (selected in >= 3 of 5 folds)", "", stable[["plain_name", "folds_selected"] + [f"coef_{c}" for c in CLASSES]].round(2).pipe(md) if len(stable) else "(none)", "",
              f"## Tree (fold {rep}, {t.get_n_leaves()} leaves)", "", "Top features by impurity importance: " + ", ".join(f"{k} ({v})" for k, v in S["tree"]["top_features"].items()) +
              f". Different folds' trees use different features (mean Jaccard {S['tree_feature_sets_per_fold_overlap_jaccard']:.2f}).", "",
              "Caveat: these are associations in acted speech from 10 speakers, in a model with UAR about 0.50, not causal statements about emotion."]
    Path("results/m11/feature_story.md").write_text("\n".join(lines)); Path("results/m11/summary.json").write_text(json.dumps(S, indent=2, default=str))
    print("\n".join(lines[:12])); print(json.dumps({k: S[k] for k in ["n_features_ever_selected", "selected_in_all_5_folds", "selected_in_ge3_folds", "jaccard_between_folds", "tree", "tree_feature_sets_per_fold_overlap_jaccard"]}, indent=1))
    print(json.dumps(S["class_means_raw_top_features"], indent=0)[:1500])


if __name__ == "__main__":
    main()
