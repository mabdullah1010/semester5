"""M10 robustness: P0 and overlap (H5), padding sensitivity, impro vs script, per-speaker normalization.
Only models l1lr, tree, mlp are rerun per condition (best interpretable and best black box per feature set)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config, write_json
from src.common.metrics import uar, wa, per_class_recall, within_session_indices
from src.m6_harness import load_dataset, CLASSES
from src.m7_costs import load_runs

CFG = load_config(); B = CFG["bootstrap"]["n_resamples"]; FEATS = ["egemaps", "wavlm"]; MODELS = ["l1lr", "tree", "mlp"]
CONDS = {"P2 (primary)": "results/m7", "P0 (all clips)": "results/m10/P0", "P2 no speaker-norm": "results/m10/nonorm",
         "P2 pad 0.00": "results/m10/P2_pad000", "P2 pad 0.25": "results/m10/P2_pad025"}
ci = lambda a: (float(np.quantile(a, .025)), float(np.quantile(a, .975)))


def load_cond(path):
    p = Path(path)
    avail = [f for f in FEATS if all((p / "preds" / f"{m}__{f}.parquet").exists() for m in MODELS)]
    if not avail:
        return None
    runs, meta = load_runs(p); runs = {k: v for k, v in runs.items() if k[0] in MODELS and k[1] in avail}
    k0 = next(iter(runs)); y, sess, utt = runs[k0]["y"], runs[k0]["session"], runs[k0]["utt"]
    idx = within_session_indices(sess, B, CFG["seed"])
    return dict(runs=runs, meta=meta, y=y, sess=sess, utt=utt, idx=idx, avail=avail)


def U(c, k, i, mask=None):
    ii = i if mask is None else i[mask[i]]
    return float(np.mean([uar(c["y"][ii], s["pred"][ii]) for s in c["runs"][k]["seeds"].values()]))


def analyse(c):
    keys = sorted(c["runs"]); full = np.arange(len(c["y"]))
    pt = {k: U(c, k, full) for k in keys}; bt = {k: np.array([U(c, k, i) for i in c["idx"]]) for k in keys}
    val = lambda k: float(np.nanmean([c["meta"][k][str(f)].get("val_uar", np.nan) for f in range(1, 6)]))
    def comb(terms):
        p = sum(s * pt[k] for k, s in terms); b = sum(s * bt[k] for k, s in terms); lo, hi = ci(b)
        return dict(point=float(p), lo=lo, hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0))
    R = dict(n=len(c["y"]), class_counts=np.bincount(c["y"], minlength=4).tolist(),
             uar={f"{k[0]}|{k[1]}": dict(point=pt[k], lo=ci(bt[k])[0], hi=ci(bt[k])[1]) for k in keys})
    R["delta_model"], R["delta_model_fixed"], R["selected_interpretable"] = {}, {}, {}
    R["features_available"] = c["avail"]
    for f in c["avail"]:
        sel = max([("l1lr", f), ("tree", f)], key=val); R["selected_interpretable"][f] = sel[0]
        R["delta_model"][f] = comb([(("mlp", f), 1), (sel, -1)])
        for m in ["l1lr", "tree"]:
            R["delta_model_fixed"][f"mlp-{m}|{f}"] = comb([(("mlp", f), 1), ((m, f), -1)])
    R["delta_feature"] = {m: comb([((m, "wavlm"), 1), ((m, "egemaps"), -1)]) for m in MODELS} if len(c["avail"]) == 2 else {}
    return R, pt, bt


def main():
    out = {"conditions": {}, "flags": []}; rows = []; C = {}
    for name, path in CONDS.items():
        c = load_cond(path)
        if c is None:
            out["conditions"][name] = "not available (runs missing)"; continue
        R, pt, bt = analyse(c); out["conditions"][name] = R; C[name] = (c, pt, bt)
        for f in R["delta_model"]:
            e = R["delta_model"][f]; rows.append(dict(analysis="delta_model_selected_interp_vs_mlp", condition=name, features=f, value=e["point"], lo=e["lo"], hi=e["hi"]))
        for m, e in R["delta_feature"].items():
            rows.append(dict(analysis="delta_feature", condition=name, features=m, value=e["point"], lo=e["lo"], hi=e["hi"]))
        for k, e in R["uar"].items():
            rows.append(dict(analysis="uar", condition=name, features=k, value=e["point"], lo=e["lo"], hi=e["hi"]))
    base = out["conditions"]["P2 (primary)"]
    # ---- do any main conclusions flip? (H1: Delta_model>0 with CI excluding 0 on both feature sets; interaction: mlp gains from WavLM, budgeted models do not)
    for name, R in out["conditions"].items():
        if isinstance(R, str):
            continue
        for f in R["delta_model"]:
            e = R["delta_model"][f]
            if not (e["lo"] > 0):
                out["flags"].append(f"{name}: Delta_model({f}) CI includes 0 or is negative {e}")
        if R["delta_feature"] and not R["delta_feature"]["mlp"]["lo"] > 0:
            out["flags"].append(f"{name}: MLP gain from WavLM not significant")
        for m in (["l1lr", "tree"] if R["delta_feature"] else []):
            if R["delta_feature"][m]["lo"] > 0:
                out["flags"].append(f"{name}: budgeted {m} now GAINS from WavLM {R['delta_feature'][m]}")

    # ---- M10.1 H5: clean vs overlapped test utterances under P0 (same trained models)
    if "P0 (all clips)" in C:
        c, pt, bt = C["P0 (all clips)"]; d = load_dataset("P0", "egemaps"); o = np.argsort(d.utt_id); assert (d.utt_id[o] == c["utt"]).all()
        ov = d.overlap_any[o]; dur = pd.read_parquet("data/processed/P0.parquet").set_index("utt_id").loc[c["utt"]]
        frac = pd.read_parquet("data/interim/overlap.parquet").set_index("utt_id").loc[c["utt"], "overlap_frac_p010"].values
        H = dict(n_clean=int((~ov).sum()), n_overlapped=int(ov.sum()), clean_class_counts=np.bincount(c["y"][~ov], minlength=4).tolist(),
                 overlapped_class_counts=np.bincount(c["y"][ov], minlength=4).tolist(),
                 mean_duration_clean=float(dur.duration[~ov].mean()), mean_duration_overlapped=float(dur.duration[ov].mean()), models={})
        for k in sorted(c["runs"]):
            def stat(i, which):
                m = (~ov) if which == "clean" else ov
                return U(c, k, i, m)
            diff = np.array([stat(i, "clean") - stat(i, "ov") for i in c["idx"]])
            full = np.arange(len(c["y"])); s0 = c["runs"][k]["seeds"][min(c["runs"][k]["seeds"])]["pred"]
            rc = lambda m: per_class_recall(c["y"][m], s0[m]).round(3).tolist()
            bins = {}
            for lo_, hi_ in [(0, 0), (1e-9, .25), (.25, .5), (.5, 1.01)]:
                m = (frac >= lo_) & (frac <= hi_) if lo_ == 0 else (frac > lo_) & (frac <= hi_)
                bins[f"{lo_:.2f}-{hi_:.2f}"] = dict(n=int(m.sum()), accuracy=float((c["y"][m] == s0[m]).mean()) if m.sum() else None)
            H["models"]["|".join(k)] = dict(uar_clean=stat(full, "clean"), uar_overlapped=stat(full, "ov"), diff_clean_minus_overlapped=dict(
                point=stat(full, "clean") - stat(full, "ov"), lo=ci(diff)[0], hi=ci(diff)[1], ci_excludes_0=bool(ci(diff)[0] > 0 or ci(diff)[1] < 0)),
                per_class_recall_clean=rc(~ov), per_class_recall_overlapped=rc(ov), accuracy_by_overlap_fraction=bins)
            rows.append(dict(analysis="H5_uar_clean", condition="P0", features="|".join(k), value=H["models"]["|".join(k)]["uar_clean"], lo=np.nan, hi=np.nan))
            rows.append(dict(analysis="H5_uar_overlapped", condition="P0", features="|".join(k), value=H["models"]["|".join(k)]["uar_overlapped"], lo=np.nan, hi=np.nan))
            rows.append(dict(analysis="H5_diff", condition="P0", features="|".join(k), value=diff.mean(), lo=ci(diff)[0], hi=ci(diff)[1]))
        out["H5_overlap"] = H

    # ---- M10.3 impro vs script (primary policy predictions)
    if "P2 (primary)" in C:
        c, pt, bt = C["P2 (primary)"]; d = load_dataset("P2", "egemaps"); o = np.argsort(d.utt_id); dt = d.dialog_type[o]
        IS = {}
        for typ in ["impro", "script"]:
            m = dt == typ; IS[typ] = dict(n=int(m.sum()), class_counts=np.bincount(c["y"][m], minlength=4).tolist(),
                                          uar={"|".join(k): dict(point=U(c, k, np.arange(len(m)), m), lo=ci([U(c, k, i, m) for i in c["idx"]])[0],
                                                                 hi=ci([U(c, k, i, m) for i in c["idx"]])[1]) for k in sorted(c["runs"])})
        sel = {f: out["conditions"]["P2 (primary)"]["selected_interpretable"][f] for f in FEATS}; IS["delta_model"] = {}
        for f in FEATS:
            bs = {}
            for typ in ["impro", "script"]:
                m = dt == typ; bs[typ] = np.array([U(c, ("mlp", f), i, m) - U(c, (sel[f], f), i, m) for i in c["idx"]])
                IS["delta_model"][f"{typ}|{f}"] = dict(point=float(np.mean(bs[typ])), lo=ci(bs[typ])[0], hi=ci(bs[typ])[1])
            dd = bs["impro"] - bs["script"]; IS["delta_model"][f"impro_minus_script|{f}"] = dict(point=float(dd.mean()), lo=ci(dd)[0], hi=ci(dd)[1])
        out["impro_vs_script"] = IS
        for typ in ["impro", "script"]:
            for k, e in IS[typ]["uar"].items():
                rows.append(dict(analysis=f"uar_{typ}", condition="P2 (primary)", features=k, value=e["point"], lo=e["lo"], hi=e["hi"]))

    # ---- padding counts
    cnt = json.load(open("results/m3/summary.json"))["padding_sensitivity_counts"]; out["padding_sensitivity_counts_P2"] = {p: v["P2"] for p, v in cnt.items()}
    pd.DataFrame(rows).to_csv("results/m10/robustness.csv", index=False); write_json(out, "results/m10/summary.json")

    # ---- figures
    names = [n for n in CONDS if isinstance(out["conditions"][n], dict)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, f in enumerate(FEATS):
        for i, n in enumerate(names):
            if f not in out["conditions"][n]["delta_model"]:
                continue
            e = out["conditions"][n]["delta_model"][f]
            ax[j].bar(i, e["point"], .7, yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], capsize=3, color=plt.cm.tab10(i))
        ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=25, ha="right", fontsize=8); ax[j].axhline(0, c="k", lw=.8)
        ax[j].set_title(f"Delta_model on {f} (MLP minus best budgeted interpretable)"); ax[j].set_ylabel("UAR difference (95% CI)")
    plt.tight_layout(); plt.savefig("figures/m10_delta_by_condition.png", dpi=130); plt.close()
    if "H5_overlap" in out:
        ks = list(out["H5_overlap"]["models"]); fig, ax = plt.subplots(figsize=(8, 4))
        for i, k in enumerate(ks):
            e = out["H5_overlap"]["models"][k]; ax.bar(i - .2, e["uar_clean"], .4, color="#55A868", label="clean (no partner overlap)" if i == 0 else None)
            ax.bar(i + .2, e["uar_overlapped"], .4, color="#C44E52", label="overlapped" if i == 0 else None)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=8); ax.set_ylabel("UAR on P0 test utterances")
        ax.set_title(f"H5: clean (n={out['H5_overlap']['n_clean']}) vs overlapped (n={out['H5_overlap']['n_overlapped']})"); ax.legend(); ax.set_ylim(0, .9)
        plt.tight_layout(); plt.savefig("figures/m10_h5_overlap.png", dpi=130); plt.close()
    if "impro_vs_script" in out:
        fig, ax = plt.subplots(figsize=(8, 4)); ks = list(out["impro_vs_script"]["impro"]["uar"])
        for i, k in enumerate(ks):
            ax.bar(i - .2, out["impro_vs_script"]["impro"]["uar"][k]["point"], .4, color="#4C72B0", label="improvised" if i == 0 else None)
            ax.bar(i + .2, out["impro_vs_script"]["script"]["uar"][k]["point"], .4, color="#DD8452", label="scripted" if i == 0 else None)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=8); ax.set_ylabel("UAR (P2)"); ax.legend(); ax.set_ylim(0, .9)
        ax.set_title("Improvised vs scripted dialogs"); plt.tight_layout(); plt.savefig("figures/m10_impro_vs_script.png", dpi=130); plt.close()

    f_ = lambda e: f"{e['point']:+.3f} [{e['lo']:+.3f},{e['hi']:+.3f}]"
    for n in names:
        R = out["conditions"][n]; s = f"{n}: n={R['n']} classes={R['class_counts']} | Delta_model eGeMAPS {f_(R['delta_model']['egemaps'])} ({R['selected_interpretable']['egemaps']})"
        if R["delta_feature"]:
            s += (f", WavLM {f_(R['delta_model']['wavlm'])} ({R['selected_interpretable']['wavlm']}) | MLP UAR eg {R['uar']['mlp|egemaps']['point']:.3f} wl {R['uar']['mlp|wavlm']['point']:.3f}"
                  f" | Dfeat mlp {f_(R['delta_feature']['mlp'])} l1lr {f_(R['delta_feature']['l1lr'])} tree {f_(R['delta_feature']['tree'])}")
        else:
            s += f" | (WavLM not available) MLP UAR eg {R['uar']['mlp|egemaps']['point']:.3f}"
        print(s)
    if "H5_overlap" in out:
        H = out["H5_overlap"]; print("H5:", {k: (round(v["uar_clean"], 3), round(v["uar_overlapped"], 3), f_(v["diff_clean_minus_overlapped"])) for k, v in H["models"].items()},
                                     "clean classes", H["clean_class_counts"], "dur clean/ov", round(H["mean_duration_clean"], 2), round(H["mean_duration_overlapped"], 2))
    if "impro_vs_script" in out:
        print("impro vs script Delta:", {k: f_(v) for k, v in out["impro_vs_script"]["delta_model"].items()})
    print("FLAGS:", out["flags"])


if __name__ == "__main__":
    main()
