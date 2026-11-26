"""M9 analysis: hard vs soft training, H3 (gap shrinkage) and H4 (high-agreement vs all test utterances).
Reads results/m7 (hard-trained) and results/m9 (soft-trained) predictions. Same folds, tuning protocol and budgets."""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config, write_json
from src.common.metrics import uar, wa, macro_f1, within_session_indices
from src.m6_harness import load_dataset
from src.m7_costs import load_runs, GROUP, NAMES

CFG = load_config(); B = CFG["bootstrap"]["n_resamples"]; FEATS = ["egemaps", "wavlm"]; EPS = 1e-4
BB = ["hgb", "rf", "mlp"]; INT = ["l1lr", "tree"]


def main():
    d = load_dataset("P2", "egemaps"); o = np.argsort(d.utt_id)                      # preds are sorted by utt_id
    y, sess, ha, ysoft = d.y[o], d.session[o], d.high_agreement[o], d.y_soft[o]
    R = {"hard": load_runs(Path("results/m7")), "soft": load_runs(Path("results/m9"))}
    ceiling = json.load(open("results/m4/human_ceiling.json"))["P2_primary"]["UAR"]
    keys = {t: sorted(R[t][0]) for t in R}
    for t in R:
        for k in keys[t]:
            assert (R[t][0][k]["utt"] == d.utt_id[o]).all()
    idx = within_session_indices(sess, B, CFG["seed"])
    full = np.arange(len(y))

    def U(t, k, i, view):
        m = ha[i] if view == "ha" else np.ones(len(i), bool); ii = i[m]
        return float(np.mean([uar(y[ii], s["pred"][ii]) for s in R[t][0][k]["seeds"].values()]))
    pt, bt = {}, {}
    for t in R:
        for k in keys[t]:
            for v in ["all", "ha"]:
                pt[(t, k, v)] = U(t, k, full, v); bt[(t, k, v)] = np.array([U(t, k, i, v) for i in idx])
    ci = lambda a: (float(np.quantile(a, .025)), float(np.quantile(a, .975)))
    def comb(terms):                       # terms: [(coef, (t,k,v))] -> point, lo, hi, excludes0
        p = sum(c * pt[x] for c, x in terms); b = sum(c * bt[x] for c, x in terms); lo, hi = ci(b)
        return dict(point=float(p), lo=lo, hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0))

    def val(t, k):
        return float(np.nanmean([R[t][1][k][str(f)].get("val_uar", np.nan) for f in range(1, 6)]))
    sel = {(t, g, f): max([k for k in keys[t] if k[1] == f and GROUP[k[0]] == g], key=lambda k: val(t, k))
           for t in R for g in ["interpretable", "blackbox"] for f in FEATS}
    out = dict(selection="best model per (training, group, features) by mean inner-validation UAR",
               selected={f"{t}|{g}|{f}": "+".join(k) for (t, g, f), k in sel.items()}, human_ceiling=ceiling)

    # --- per-run table (soft_grid.csv)
    rows = []
    for t in R:
        for k in keys[t]:
            for v in ["all", "ha"]:
                mk = np.ones(len(y), bool) if v == "all" else ha
                accs = [wa(y[mk], s["pred"][mk]) for s in R[t][0][k]["seeds"].values()]
                f1s = [macro_f1(y[mk], s["pred"][mk]) for s in R[t][0][k]["seeds"].values()]
                ce = np.mean([-(ysoft[mk] * np.log(np.clip(s["probs"][mk], EPS, 1))).sum(1).mean() for s in R[t][0][k]["seeds"].values()])
                lo, hi = ci(bt[(t, k, v)])
                rows.append(dict(model=k[0], features=k[1], training=t, view=v, UAR=pt[(t, k, v)], UAR_lo=lo, UAR_hi=hi,
                                 WA=float(np.mean(accs)), macroF1=float(np.mean(f1s)), soft_CE=float(ce), mean_inner_val_uar=val(t, k)))
    grid = pd.DataFrame(rows); grid.to_csv("results/m9/soft_grid.csv", index=False)

    # --- soft - hard effect per model
    eff = {}
    for k in keys["hard"]:
        if k in keys["soft"]:
            eff["+".join(k)] = {v: comb([(1, ("soft", k, v)), (-1, ("hard", k, v))]) for v in ["all", "ha"]}
    out["soft_minus_hard_uar_per_model"] = eff

    # --- Delta_model per condition, selected models
    def delta(t, f, v, sl=None):
        a = (sl or sel)[(t, "blackbox", f)] if sl is None else sl[("blackbox", f)]
        b = (sl or sel)[(t, "interpretable", f)] if sl is None else sl[("interpretable", f)]
        return [(1, (t, a, v)), (-1, (t, b, v))]
    out["delta_model"] = {f"{t}|{f}|{v}": comb(delta(t, f, v)) for t in R for f in FEATS for v in ["all", "ha"]}
    # H3: gap shrinkage = Delta(hard) - Delta(soft)
    out["H3_gap_shrinkage"] = {f"{f}|{v}": comb(delta("hard", f, v) + [(-c, x) for c, x in delta("soft", f, v)]) for f in FEATS for v in ["all", "ha"]}
    # H4: Delta on high-agreement minus Delta on all (per training condition)
    out["H4_gap_ha_minus_all"] = {f"{t}|{f}": comb(delta(t, f, "ha") + [(-c, x) for c, x in delta(t, f, "all")]) for t in R for f in FEATS}
    # robustness 1: fixed pair (selected under hard training) used for both
    fixed = {(g, f): sel[("hard", g, f)] for g in ["interpretable", "blackbox"] for f in FEATS}
    out["robust_fixed_pair_selected_under_hard"] = dict(
        pairs={f"{g}|{f}": "+".join(k) for (g, f), k in fixed.items()},
        delta_model={f"{t}|{f}|{v}": comb([(1, (t, (fixed[("blackbox", f)][0], f), v)), (-1, (t, (fixed[("interpretable", f)][0], f), v))]) for t in R for f in FEATS for v in ["all", "ha"]})
    out["robust_fixed_pair_selected_under_hard"]["H3_gap_shrinkage"] = {
        f"{f}|{v}": comb([(1, ("hard", (fixed[("blackbox", f)][0], f), v)), (-1, ("hard", (fixed[("interpretable", f)][0], f), v)),
                          (-1, ("soft", (fixed[("blackbox", f)][0], f), v)), (1, ("soft", (fixed[("interpretable", f)][0], f), v))]) for f in FEATS for v in ["all", "ha"]}
    # robustness 2: selection-free, mean over all black boxes minus mean over both budgeted models
    def meandelta(t, f, v):
        return [(1 / len(BB), (t, (m, f), v)) for m in BB] + [(-1 / len(INT), (t, (m, f), v)) for m in INT]
    out["robust_mean_over_models"] = dict(
        delta_model={f"{t}|{f}|{v}": comb(meandelta(t, f, v)) for t in R for f in FEATS for v in ["all", "ha"]},
        H3_gap_shrinkage={f"{f}|{v}": comb(meandelta("hard", f, v) + [(-c, x) for c, x in meandelta("soft", f, v)]) for f in FEATS for v in ["all", "ha"]},
        H4_gap_ha_minus_all={f"{t}|{f}": comb(meandelta(t, f, "ha") + [(-c, x) for c, x in meandelta(t, f, "all")]) for t in R for f in FEATS})
    # soft cross-entropy (selected models)
    out["soft_cross_entropy_selected"] = {f"{t}|{g}|{f}": float(grid[(grid.model == k[0]) & (grid.features == f) & (grid.training == t) & (grid.view == "all")].soft_CE.iloc[0])
                                          for (t, g, f), k in sel.items()}
    write_json(out, "results/m9/h3_h4.json")

    # --- headline table (plan Section 5)
    ht = []
    for t, v in [("hard", "all"), ("soft", "all"), ("hard", "ha"), ("soft", "ha")]:
        r = dict(setting=f"{t} labels, {'all' if v == 'all' else 'high-agreement'} test")
        for g, f in [("interpretable", "egemaps"), ("blackbox", "egemaps"), ("interpretable", "wavlm"), ("blackbox", "wavlm")]:
            k = sel[(t, g, f)]; lo, hi = ci(bt[(t, k, v)]); r[f"{g}_{f}"] = f"{pt[(t, k, v)]:.3f} [{lo:.3f},{hi:.3f}] ({NAMES[k[0]]})"
        r["human_ceiling"] = f"{ceiling:.3f}"; ht.append(r)
    pd.DataFrame(ht).to_csv("results/m9/headline_table.csv", index=False)

    # --- figures
    c1, c2 = "#4C72B0", "#DD8452"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, f in enumerate(FEATS):
        for i, t in enumerate(["hard", "soft"]):
            e = out["delta_model"][f"{t}|{f}|all"]
            ax[0].bar(j + (i - .5) * .35, e["point"], .35, yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], capsize=4, color=[c1, c2][i], label=f"{t}-label training" if j == 0 else None)
    ax[0].axhline(0, c="k", lw=.8); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["eGeMAPS", "WavLM"]); ax[0].set_ylabel("Delta_model = UAR(black box) - UAR(interpretable)")
    ax[0].set_title("Interpretability gap, all test utterances"); ax[0].legend()
    for j, f in enumerate(FEATS):
        for v_i, v in enumerate(["all", "ha"]):
            e = out["H3_gap_shrinkage"][f"{f}|{v}"]
            ax[1].bar(j + (v_i - .5) * .35, e["point"], .35, yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], capsize=4, color=["#55A868", "#8172B2"][v_i], label={"all": "all test", "ha": "high-agreement test"}[v] if j == 0 else None)
    ax[1].axhline(0, c="k", lw=.8); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["eGeMAPS", "WavLM"]); ax[1].set_ylabel("gap shrinkage = Delta(hard) - Delta(soft)")
    ax[1].set_title("H3: does soft training shrink the gap? (>0 = yes)"); ax[1].legend()
    plt.tight_layout(); plt.savefig("figures/m9_gap_hard_vs_soft.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, f in enumerate(FEATS):
        for i, (t, v) in enumerate([("hard", "all"), ("hard", "ha"), ("soft", "all"), ("soft", "ha")]):
            e = out["delta_model"][f"{t}|{f}|{v}"]
            ax.bar(j * 5 + i, e["point"], .8, yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], capsize=3, color=["#4C72B0", "#99B3D9", "#DD8452", "#F0BE9E"][i],
                   label=f"{t}, {'all' if v == 'all' else 'high-agreement'}" if j == 0 else None)
    ax.set_xticks([1.5, 6.5]); ax.set_xticklabels(["eGeMAPS", "WavLM"]); ax.axhline(0, c="k", lw=.8); ax.set_ylabel("Delta_model (UAR)"); ax.legend(fontsize=8)
    ax.set_title("H4: gap on high-agreement vs all test utterances"); plt.tight_layout(); plt.savefig("figures/m9_gap_by_agreement.png", dpi=130); plt.close()
    f_ = lambda e: f"{e['point']:+.3f} [{e['lo']:+.3f},{e['hi']:+.3f}]"
    print(pd.DataFrame(ht).to_string(index=False)); print(json.dumps(out["selected"], indent=1))
    for kk in ["delta_model", "H3_gap_shrinkage", "H4_gap_ha_minus_all"]:
        print(kk); [print("  ", a, f_(e)) for a, e in out[kk].items()]
    print("robust mean over models: H3", {a: f_(e) for a, e in out["robust_mean_over_models"]["H3_gap_shrinkage"].items()})
    print("robust fixed pair: H3", {a: f_(e) for a, e in out["robust_fixed_pair_selected_under_hard"]["H3_gap_shrinkage"].items()})
    print("soft - hard per model (all view):", {a: f_(e["all"]) for a, e in eff.items()})
    print("soft CE selected:", {a: round(v, 3) for a, v in out["soft_cross_entropy_selected"].items()})


if __name__ == "__main__":
    main()
