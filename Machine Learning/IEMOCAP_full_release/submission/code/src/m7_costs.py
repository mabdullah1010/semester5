"""M7.4: grid table, costs (Delta_model, Delta_feature, Delta_total) with paired bootstrap CIs, figures.
Usage: python src/m7_costs.py [--dir results/m7] [--tag P2]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config, write_json
from src.common.metrics import uar, wa, macro_f1, confusion_matrix, within_session_indices, mcnemar
from src.m6_harness import CLASSES

GROUP = {"l1lr": "interpretable", "tree": "interpretable", "l2lr": "unbudgeted", "hgb": "blackbox", "rf": "blackbox", "mlp": "blackbox"}
NAMES = {"l1lr": "L1-LR (<=20 feats)", "tree": "Tree (<=16 leaves)", "l2lr": "L2-LR (unbudgeted)", "hgb": "HistGB", "rf": "Random forest", "mlp": "MLP"}


def load_runs(d):
    runs, meta = {}, {}
    for f in sorted((d / "preds").glob("*.parquet")):
        m, feat = f.stem.split("__"); df = pd.read_parquet(f).sort_values(["seed", "utt_id"])
        seeds = {}
        for sd, g in df.groupby("seed"):
            seeds[int(sd)] = dict(pred=g[[f"p_{c}" for c in CLASSES]].values.argmax(1), probs=g[[f"p_{c}" for c in CLASSES]].values,
                                  fold=g.fold.values)
        g0 = df[df.seed == df.seed.min()]
        runs[(m, feat)] = dict(seeds=seeds, y=g0.y_true.values, session=g0.session.values, utt=g0.utt_id.values)
        meta[(m, feat)] = json.loads((d / "hparams" / f"{m}__{feat}.json").read_text())
    return runs, meta


def main(d, tag):
    d = Path(d); cfg = load_config(); runs, meta = load_runs(d)
    keys = sorted(runs); ref = runs[keys[0]]
    for k in keys:
        assert (runs[k]["utt"] == ref["utt"]).all() and (runs[k]["y"] == ref["y"]).all()
    y, sess = ref["y"], ref["session"]; B = cfg["bootstrap"]["n_resamples"]
    ceiling = json.loads(Path("results/m4/human_ceiling.json").read_text())["P2_primary" if tag == "P2" else "P0_4class"]["UAR"]

    # ---- grid.csv (per fold, per seed)
    rows = []
    for (m, feat), r in runs.items():
        for sd, s in r["seeds"].items():
            for fo in range(1, 6):
                mk = s["fold"] == fo
                rows.append(dict(model=m, features=feat, fold=fo, seed=sd, UAR=uar(y[mk], s["pred"][mk]), WA=wa(y[mk], s["pred"][mk]),
                                 macroF1=macro_f1(y[mk], s["pred"][mk])))
    grid = pd.DataFrame(rows); grid.to_csv(d / "grid.csv", index=False)

    # ---- bootstrap: all run UARs on the same resamples (paired)
    idx = within_session_indices(sess, B, cfg["seed"])
    def U(k, i):
        return float(np.mean([uar(y[i], s["pred"][i]) for s in runs[k]["seeds"].values()]))
    full = np.arange(len(y)); point = {k: U(k, full) for k in keys}
    boot = {k: np.array([U(k, i) for i in idx]) for k in keys}
    ci = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975)))

    # ---- model selection on inner-validation UAR (never on test)
    def val(k):
        v = [meta[k][str(f)].get("val_uar", np.nan) for f in range(1, 6)]; return float(np.nanmean(v))
    sel = {}
    for feat in ["egemaps", "wavlm"]:
        for grp in ["interpretable", "blackbox"]:
            cands = [k for k in keys if k[1] == feat and GROUP[k[0]] == grp]
            sel[(grp, feat)] = max(cands, key=val)
    allc = [k for k in keys if GROUP[k[0]] != "unbudgeted"]; best_overall = max(allc, key=val)
    oracle = {kk: max([k for k in keys if k[1] == kk[1] and GROUP[k[0]] == kk[0]], key=lambda k: point[k]) for kk in sel}

    def diff(a, b):
        v = boot[a] - boot[b]; lo, hi = ci(v)
        return dict(point=point[a] - point[b], lo=lo, hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0))
    def diff2(pairs):   # sum of differences, e.g. (a-b)-(c-e)
        v = sum(s * boot[k] for k, s in pairs); p = sum(s * point[k] for k, s in pairs); lo, hi = ci(v)
        return dict(point=p, lo=lo, hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0))
    I_e, I_w = sel[("interpretable", "egemaps")], sel[("interpretable", "wavlm")]
    B_e, B_w = sel[("blackbox", "egemaps")], sel[("blackbox", "wavlm")]
    costs = dict(selection_rule="best model per cell by mean inner-validation UAR (not test)",
                 selected={f"{g}|{f}": "+".join(k) for (g, f), k in sel.items()}, best_overall="+".join(best_overall),
                 oracle_test_selected={f"{g}|{f}": "+".join(k) for (g, f), k in oracle.items()},
                 human_ceiling_uar=ceiling)
    costs["delta_model"] = {"egemaps": diff(B_e, I_e), "wavlm": diff(B_w, I_w)}
    costs["delta_feature"] = {m: diff((m, "wavlm"), (m, "egemaps")) for m in GROUP if (m, "wavlm") in runs and (m, "egemaps") in runs}
    costs["delta_total"] = diff(best_overall, I_e)
    costs["H2_feature_minus_model"] = {
        "interpretable_feature_gain_minus_model_gap_on_egemaps": diff2([(I_w, 1), (I_e, -1), (B_e, -1), (I_e, 1)]),
        "note": "Delta_feature(interpretable) - Delta_model(egemaps); >0 means the feature change buys more than the model change"}
    costs["decomposition_total"] = dict(
        feature_first=dict(feature=diff(I_w, I_e), then_model=diff(B_w, I_w)),
        model_first=dict(model=diff(B_e, I_e), then_feature=diff(B_w, B_e)))
    oracle_costs = dict(delta_model={f: diff(oracle[("blackbox", f)], oracle[("interpretable", f)]) for f in ["egemaps", "wavlm"]})
    costs["delta_model_oracle_selection_for_reference"] = oracle_costs["delta_model"]

    # ---- per-run table
    table = []
    for k in keys:
        g = grid[(grid.model == k[0]) & (grid.features == k[1])]
        per_fold = g.groupby("fold").UAR.mean()
        lo, hi = ci(boot[k]); s0 = runs[k]["seeds"][min(runs[k]["seeds"])]
        table.append(dict(model=k[0], features=k[1], group=GROUP[k[0]], UAR=point[k], UAR_lo=lo, UAR_hi=hi,
                          WA=float(g.groupby("seed").WA.mean().mean()), macroF1=float(g.groupby("seed").macroF1.mean().mean()),
                          UAR_by_fold=per_fold.round(4).tolist(), fold_std=float(per_fold.std()), hardest_fold=int(per_fold.idxmin()),
                          seed_std=float(np.std([uar(y, s["pred"]) for s in runs[k]["seeds"].values()])) if len(runs[k]["seeds"]) > 1 else 0.0,
                          mean_inner_val_uar=val(k), gap_to_human_ceiling=ceiling - point[k]))
    tbl = pd.DataFrame(table); tbl.to_csv(d / "table.csv", index=False)
    mc = {}
    for f in ["egemaps", "wavlm"]:
        a, b = sel[("blackbox", f)], sel[("interpretable", f)]; sa = min(runs[a]["seeds"]); sb = min(runs[b]["seeds"])
        mc[f] = mcnemar(y, runs[a]["seeds"][sa]["pred"], runs[b]["seeds"][sb]["pred"])
    costs["mcnemar_blackbox_vs_interpretable"] = mc

    # ---- checks (rough ranges, logged not asserted) and ceiling
    chk = dict(
        egemaps_lr_uar_expected_050_060={m: point[(m, "egemaps")] for m in ["l1lr", "l2lr"] if (m, "egemaps") in point},
        wavlm_lr_uar_expected_060_070={m: point[(m, "wavlm")] for m in ["l1lr", "l2lr"] if (m, "wavlm") in point},
        any_model_above_human_ceiling=[ "+".join(k) for k in keys if point[k] > ceiling],
        hardest_fold_counts=tbl.hardest_fold.value_counts().to_dict())
    costs["checks_logged"] = chk
    costs["table"] = table
    write_json(costs, d / "costs.json")

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); w = 0.35
    for j, grp in enumerate(["interpretable", "blackbox"]):
        vals = [point[sel[(grp, f)]] for f in ["egemaps", "wavlm"]]
        errs = [[point[sel[(grp, f)]] - ci(boot[sel[(grp, f)]])[0] for f in ["egemaps", "wavlm"]],
                [ci(boot[sel[(grp, f)]])[1] - point[sel[(grp, f)]] for f in ["egemaps", "wavlm"]]]
        ax[0].bar(np.arange(2) + (j - .5) * w, vals, w, yerr=errs, capsize=4, label=grp, color=["#4C72B0", "#DD8452"][j])
        for x, f in zip(np.arange(2), ["egemaps", "wavlm"]):
            ax[0].text(x + (j - .5) * w, point[sel[(grp, f)]] + 0.03, NAMES[sel[(grp, f)][0]], ha="center", fontsize=7, rotation=0)
    ax[0].axhline(ceiling, ls="--", c="k", lw=1, label=f"approx. human ceiling {ceiling:.2f}"); ax[0].axhline(.25, ls=":", c="gray", label="chance")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["eGeMAPS (88 named)", "WavLM (learned)"]); ax[0].set_ylabel("pooled LOSO UAR (95% CI)")
    ax[0].set_ylim(0.2, 0.8); ax[0].set_title(f"2x2 grid ({tag}, hard labels)"); ax[0].legend(fontsize=8, loc="upper left")
    order = ["l1lr", "tree", "l2lr", "hgb", "rf", "mlp"]; cols = {"interpretable": "#4C72B0", "unbudgeted": "#8172B2", "blackbox": "#DD8452"}
    for j, f in enumerate(["egemaps", "wavlm"]):
        for i, m in enumerate(order):
            if (m, f) in point:
                lo, hi = ci(boot[(m, f)]); ax[1].bar(i + (j - .5) * 0.4, point[(m, f)], 0.4, color=cols[GROUP[m]], alpha=1 if f == "egemaps" else .55,
                                                    yerr=[[point[(m, f)] - lo], [hi - point[(m, f)]]], capsize=2)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([NAMES[m] for m in order], rotation=25, ha="right", fontsize=8)
    ax[1].axhline(ceiling, ls="--", c="k", lw=1); ax[1].set_ylim(0.2, 0.8); ax[1].set_title("all models (dark = eGeMAPS, light = WavLM)"); ax[1].set_ylabel("UAR")
    plt.tight_layout(); plt.savefig(Path("figures") / f"m7_grid{'' if tag == 'P2' else '_' + tag}.png", dpi=130); plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a_, (grp, f) in zip(ax, [("interpretable", "egemaps"), ("blackbox", "egemaps"), ("interpretable", "wavlm"), ("blackbox", "wavlm")]):
        k = sel[(grp, f)]; s = runs[k]["seeds"][min(runs[k]["seeds"])]; c = confusion_matrix(y, s["pred"]); c = c / c.sum(1, keepdims=True)
        a_.imshow(c, vmin=0, vmax=1, cmap="Blues"); a_.set_xticks(range(4)); a_.set_xticklabels(CLASSES); a_.set_yticks(range(4)); a_.set_yticklabels(CLASSES)
        a_.set_xlabel("predicted"); a_.set_ylabel("true"); a_.set_title(f"{NAMES[k[0]]} / {f}\nUAR {point[k]:.3f}", fontsize=9)
        for i in range(4):
            for j in range(4):
                a_.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", color="white" if c[i, j] > .5 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(Path("figures") / f"m7_confusions{'' if tag == 'P2' else '_' + tag}.png", dpi=130); plt.close()
    print(tbl[["model", "features", "UAR", "UAR_lo", "UAR_hi", "WA", "fold_std", "mean_inner_val_uar"]].round(3).to_string(index=False))
    print(json.dumps({k: costs[k] for k in ["selected", "best_overall", "delta_model", "delta_total", "H2_feature_minus_model", "decomposition_total", "checks_logged"]}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/m7"); ap.add_argument("--tag", default="P2"); a = ap.parse_args()
    main(a.dir, a.tag)
