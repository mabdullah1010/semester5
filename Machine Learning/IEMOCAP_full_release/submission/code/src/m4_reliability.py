"""M4: label reliability, approximate human ceiling, label tiers."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import krippendorff
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config

CLS = ["neu", "hap", "sad", "ang"]
IDX = {"neu": 0, "hap": 1, "exc": 1, "sad": 2, "ang": 3}     # everything else -> 4 ("other" bucket)
NC = 5


def rater_weights(votes):
    """-> {utt_id: array (n_raters, 5)}; each rater's vote spread 1/k over the categories listed."""
    out = {}
    for (u, r), g in votes.groupby(["utt_id", "rater_id"]):
        w = np.zeros(NC)
        for c in g.category_code:
            w[IDX.get(c, 4)] += 1.0 / len(g)
        out.setdefault(u, []).append(w)
    return {u: np.array(v) for u, v in out.items()}


def alpha_fractional(units):
    """Krippendorff's alpha (nominal) where each rater's vote may be a distribution (rows of W).
    Equals the standard alpha when every row is one-hot."""
    k = units[0].shape[1]; o = np.zeros((k, k))
    for W in units:
        m = len(W)
        if m < 2:
            continue
        S = W.sum(0)
        o += (np.outer(S, S) - W.T @ W) / (m - 1)
    n = o.sum(); nc = o.sum(1)
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (nc ** 2).sum()) / (n * (n - 1))
    return float(1 - d_o / d_e)


def fleiss_fractional(units):
    """Fleiss' kappa for items with the same number of raters (here 3); fractional votes allowed."""
    P, tot = [], np.zeros(units[0].shape[1])
    for W in units:
        m = len(W); S = W.sum(0)
        P.append((np.outer(S, S) - W.T @ W).trace() / (m * (m - 1)))
        tot += S / m
    pe = ((tot / len(units)) ** 2).sum()
    return float((np.mean(P) - pe) / (1 - pe))


def stats(units):
    if len(units) < 10:
        return dict(n=len(units), alpha=None, kappa=None)
    return dict(n=len(units), alpha=alpha_fractional(units), kappa=fleiss_fractional(units))


def ovr(units, c):
    return [np.stack([W[:, c], 1 - W[:, c]], 1) for W in units]


def reliability_block(ids, W, meta):
    """alpha/kappa overall, per session, per class (one-vs-rest), per dialog type for utt_ids `ids`."""
    ids = [i for i in ids if i in W]
    R = dict(overall=stats([W[i] for i in ids]))
    m = meta.loc[ids]
    R["per_session"] = {int(s): stats([W[i] for i in g.index]) for s, g in m.groupby("session")}
    R["per_dialog_type"] = {t: stats([W[i] for i in g.index]) for t, g in m.groupby("dialog_type")}
    R["per_class_ovr"] = {CLS[c]: stats(ovr([W[i] for i in ids], c)) for c in range(4)}
    R["per_class_ovr"]["other"] = stats(ovr([W[i] for i in ids], 4))
    return R


def main():
    cfg = load_config()
    utt = pd.read_parquet("data/interim/utterances.parquet").set_index("utt_id")
    votes = pd.read_parquet("data/interim/votes.parquet")
    P2 = pd.read_parquet("data/processed/P2.parquet"); P0 = pd.read_parquet("data/processed/P0.parquet")
    Path("results/m4").mkdir(parents=True, exist_ok=True)
    ext = votes[votes.rater_type == "external"]
    W = rater_weights(ext)
    S = {"checks": {}}
    sets = {"all_10039": list(utt.index), "P0_4class": list(P0.utt_id), "P2_primary": list(P2.utt_id)}

    # ---- M4.1
    S["agreement"] = {k: reliability_block(v, W, utt) for k, v in sets.items()}
    S["n_raters_per_item"] = pd.Series({u: len(w) for u, w in W.items()}).value_counts().to_dict()
    # sanity: standard krippendorff package on single-label (first listed category) data
    first = ext.sort_values(["utt_id", "rater_id"]).drop_duplicates(["utt_id", "rater_id"], keep="first")
    first = first.assign(c=first.category_code.map(lambda c: IDX.get(c, 4)))
    mat = first.assign(slot=first.groupby("utt_id").cumcount()).pivot(index="slot", columns="utt_id", values="c").values.astype(float)
    S["krippendorff_pkg_first_listed_alpha_all"] = float(krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal"))
    hard = [np.eye(NC)[np.argmax(W[u], 1)] for u in utt.index]   # argmax-vote version
    S["alpha_hard_argmax_version_all"] = alpha_fractional(hard)
    print("alpha/kappa (all):", S["agreement"]["all_10039"]["overall"], "| P2:", S["agreement"]["P2_primary"]["overall"])
    print("pkg alpha (first-listed):", S["krippendorff_pkg_first_listed_alpha_all"])

    # ---- M4.2 self vs observers
    selfv = votes[votes.rater_type == "self"]
    rows = []
    for u, g in selfv.groupby("utt_id"):
        cats = {IDX[c] for c in g.category_code if c in IDX}
        rows.append(dict(utt_id=u, self_set=sorted(cats), n_self_cats=g.category_code.nunique()))
    sv = pd.DataFrame(rows).merge(utt.reset_index()[["utt_id", "label_official", "in_4class"]], on="utt_id")
    sv = sv[sv.in_4class]
    sv["y"] = sv.label_official.map(IDX)
    sv["lenient"] = [y in s for y, s in zip(sv.y, sv.self_set)]
    sv["strict"] = [s == [y] for y, s in zip(sv.y, sv.self_set)]
    S["self_vs_observers"] = dict(
        n=len(sv), lenient_match=float(sv.lenient.mean()), strict_match=float(sv.strict.mean()),
        per_class={CLS[c]: dict(n=int((sv.y == c).sum()), lenient=float(sv[sv.y == c].lenient.mean()),
                                strict=float(sv[sv.y == c].strict.mean())) for c in range(4)},
        note="lenient: official 4-class label is among the actor's listed categories; strict: actor listed only that class. Only sessions 1,2,3,5 have self-evaluations.")
    print("self vs observers:", S["self_vs_observers"]["lenient_match"], S["self_vs_observers"]["strict_match"])

    # ---- M4.3 approximate human ceiling
    def ceiling(ids):
        conf = np.zeros((4, NC)); n_items = 0; n_rec = 0; n_cand = 0
        for u in ids:
            Wu = W.get(u)
            if Wu is None or len(Wu) < 3:
                continue
            n_cand += len(Wu)
            for r in range(len(Wu)):
                others = np.delete(Wu, r, 0)
                support = np.where(others.sum(0) > 0)[0]
                if len(support) == 1 and support[0] < 4:
                    conf[support[0]] += Wu[r]; n_rec += 1
        rec = conf.diagonal() / conf.sum(1)
        pair_ag = []
        for u in ids:
            Wu = W[u]
            for i in range(len(Wu)):
                for j in range(i + 1, len(Wu)):
                    pair_ag.append(float(Wu[i] @ Wu[j]))
        return dict(n_records=n_rec, n_candidate_records=n_cand, UAR=float(np.nanmean(rec)), WA=float(np.trace(conf[:, :4]) / conf.sum()),
                    per_class_recall={CLS[c]: float(rec[c]) for c in range(4)},
                    confusion_rows_true_cols_rater=conf.round(2).tolist(), mean_pairwise_agreement=float(np.mean(pair_ag)))
    S_ceiling = {k: ceiling(sets[k]) for k in ["P0_4class", "P2_primary"]}
    for k, v in S_ceiling.items():
        print("human ceiling", k, "UAR %.3f WA %.3f records %d/%d pair-agree %.3f" % (v["UAR"], v["WA"], v["n_records"], v["n_candidate_records"], v["mean_pairwise_agreement"]))
    Path("results/m4/human_ceiling.json").write_text(json.dumps(
        {"name": "approximate human ceiling (optimistic: only items where the other raters agree)", **S_ceiling}, indent=2))

    # ---- M4.4 label tiers
    cnt = {}
    for u, g in ext.groupby("utt_id"):
        c = np.zeros(NC)
        for code in g.category_code:
            c[IDX.get(code, 4)] += 1                      # rule A: one count per listed category
        cnt[u] = c
    d4 = utt[utt.in_4class].copy()
    lab = pd.DataFrame(index=d4.index)
    lab["y_hard"] = d4.label_4class
    C = np.stack([cnt[u] for u in d4.index])
    lab["outside_mass"] = C[:, 4] / C.sum(1)
    soft = C[:, :4] / C[:, :4].sum(1, keepdims=True)
    for i, c in enumerate(CLS):
        lab[f"p_{c}"] = soft[:, i]
    # high agreement: every external vote (category) maps to the same one of the 4 classes
    lab["high_agreement"] = [(np.count_nonzero(W[u].sum(0)) == 1 and W[u].sum(0)[:4].sum() > 0) for u in d4.index]
    lab["session"] = d4.session; lab["speaker_id"] = d4.speaker_id; lab["dialog_type"] = d4.dialog_type
    lab["in_P2"] = lab.index.isin(P2.utt_id)
    ytrue = lab.y_hard.map({c: i for i, c in enumerate(CLS)}).values
    S["checks"]["y_soft_sums_to_1"] = bool(np.allclose(soft.sum(1), 1, atol=1e-6))
    top = soft.max(1)
    is_top = soft[np.arange(len(soft)), ytrue] >= top - 1e-9               # y_hard is among argmax
    unique_top = (np.sum(soft >= top[:, None] - 1e-9, 1) == 1)
    lab["argmax_soft_ne_hard"] = ~is_top
    lab["argmax_soft_tie_with_hard"] = is_top & ~unique_top
    S["argmax_vs_hard"] = dict(n=len(lab), strict_disagree=int((~is_top).sum()), disagree_rate=float((~is_top).mean()),
                               ties_containing_hard=int((is_top & ~unique_top).sum()))
    cases = lab[lab.argmax_soft_ne_hard].copy()
    cases["soft"] = [np.round(r, 3).tolist() for r in cases[[f"p_{c}" for c in CLS]].values]
    cases[["y_hard", "soft", "outside_mass"]].to_csv("results/m4/argmax_disagreements.csv")
    lab.reset_index().to_parquet("data/processed/labels.parquet")
    for name, m in [("P0_4class", lab), ("P2_primary", lab[lab.in_P2])]:
        S.setdefault("tiers", {})[name] = dict(
            n=len(m), high_agreement=int(m.high_agreement.sum()),
            high_agreement_per_class=m[m.high_agreement].y_hard.value_counts().reindex(CLS).fillna(0).astype(int).to_dict(),
            high_agreement_per_session=m[m.high_agreement].groupby("session").size().to_dict(),
            outside_mass_mean=float(m.outside_mass.mean()), outside_mass_frac_gt0=float((m.outside_mass > 0).mean()),
            outside_mass_quantiles=m.outside_mass.quantile([.5, .75, .9, .99]).round(3).to_dict(),
            soft_entropy_mean_bits=float(-(m[[f"p_{c}" for c in CLS]].values * np.log2(np.clip(m[[f"p_{c}" for c in CLS]].values, 1e-12, 1))).sum(1).mean()))
    print("tiers:", json.dumps(S["tiers"], default=str)); print("argmax vs hard:", S["argmax_vs_hard"])

    # ---- checks
    ab = []
    for k in S["agreement"]:
        ab += [S["agreement"][k]["overall"]["alpha"], S["agreement"][k]["overall"]["kappa"]]
    S["checks"]["alpha_kappa_in_0_1"] = all(0 <= x <= 1 for x in ab)
    S["checks"]["all_items_3_raters"] = set(S["n_raters_per_item"]) == {3}
    S["checks"]["high_agreement_ge100_per_class_P2"] = all(v >= 100 for v in S["tiers"]["P2_primary"]["high_agreement_per_class"].values())
    S["all_checks_pass"] = all(S["checks"].values())
    Path("results/m4/reliability.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S["checks"]))

    # ---- figure
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(5); w = 0.27
    names = CLS + ["other"]
    for i, k in enumerate(["all_10039", "P0_4class", "P2_primary"]):
        vals = [S["agreement"][k]["per_class_ovr"][c]["alpha"] for c in names]
        ax.bar(x + (i - 1) * w, vals, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_xlabel("class (one-vs-rest)")
    ax.set_ylabel("Krippendorff's alpha"); ax.set_title("Rater agreement by class"); ax.set_ylim(0, 1); ax.legend()
    plt.tight_layout(); plt.savefig("figures/m4_agreement_by_class.png", dpi=130)


if __name__ == "__main__":
    main()
