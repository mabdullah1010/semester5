"""M12: build report/report.md and report/tables/*.csv from results/*.json (no hand-typed numbers), then optionally verify.
Verdicts are computed from the confidence intervals and ASSERTED, so the script fails if a claim is not supported by the data.
Usage: python src/m12_report.py            (build)        python src/m12_report.py --verify   (final verification pass)"""
import json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config

J = lambda p: json.load(open(p))
CL = ["neu", "hap", "sad", "ang"]
NICE = {"l1lr": "L1-LR (<=20 features)", "tree": "decision tree (<=16 leaves)", "l2lr": "L2-LR (no budget)", "hgb": "HistGradientBoosting", "rf": "random forest", "mlp": "MLP"}


def E(e, d=3):          # estimate with CI
    return f"{e['point']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]"
def U3(x): return f"{x:.3f}"
def P(x, d=1): return f"{100 * x:.{d}f}%"
def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    return "\n".join(out + ["| " + " | ".join(str(v).replace("|", " / ") for v in r) + " |" for r in df.itertuples(index=False)])
def sel(c, name): return c["selected"][name]


def load_all():
    return dict(m1=J("results/m1/summary.json"), m2=J("results/m2/summary.json"), m3=J("results/m3/summary.json"), ov=J("results/m3/overlap_summary.json"), oh=J("results/m3/overlap_hist.json"),
                ct=J("results/m3/crosstalk_calibration.json"), m4=J("results/m4/reliability.json"), hc=J("results/m4/human_ceiling.json"), m5=J("results/m5/summary.json"),
                m6=J("results/m6/sanity.json"), m7=J("results/m7/costs.json"), m8=J("results/m8/summary.json"), m9=J("results/m9/h3_h4.json"),
                m10=J("results/m10/summary.json"), m11=J("results/m11/summary.json"), m9t=pd.read_csv("results/m9/headline_table.csv"),
                curves=pd.read_csv("results/m8/curves.csv"), co=pd.read_csv("results/m11/coefficients.csv"))


def verdicts(R):
    v = {}; m7, m9, m10 = R["m7"], R["m9"], R["m10"]
    dm = m7["delta_model"]
    assert dm["egemaps"]["lo"] > 0 and dm["wavlm"]["lo"] > 0, "H1 not supported by M7 CIs"
    v["H1"] = ("Supported", f"Delta_model = {E(dm['egemaps'])} on eGeMAPS and {E(dm['wavlm'])} on WavLM (CI excludes 0 on both).")
    df = m7["delta_feature"]; dec = m7["decomposition_total"]
    assert df["l1lr"]["lo"] <= 0 and df["mlp"]["lo"] > 0, "H2 interaction pattern changed"
    v["H2"] = ("Not supported as stated; the two factors interact", f"Switching to WavLM does not help the budgeted L1-LR ({E(df['l1lr'])}) or the tree ({E(df['tree'])}) but helps the dense MLP "
               f"({E(df['mlp'])}). Feature share of the total gap is {P(dec['model_first']['then_feature']['point'] / m7['delta_total']['point'], 0)} if the model changes first and negative if the features change first.")
    sh = m9["H3_gap_shrinkage"]; sm = m9["robust_mean_over_models"]["H3_gap_shrinkage"]
    assert not (sh["egemaps|all"]["lo"] > 0 and sh["wavlm|all"]["lo"] > 0), "H3 now looks supported; rewrite the text"
    v["H3"] = ("Not supported", f"Gap shrinkage (hard minus soft) on all test utterances: eGeMAPS {E(sh['egemaps|all'])} (selection-free {E(sm['egemaps|all'])}), WavLM {E(sh['wavlm|all'])} (selection-free {E(sm['wavlm|all'])}); "
               "the sign depends on the feature set.")
    h4 = m9["H4_gap_ha_minus_all"]; h4s = m9["robust_mean_over_models"]["H4_gap_ha_minus_all"]
    assert all(h4s[k]["point"] > 0 for k in h4s), "H4 direction changed"
    n_sig = sum(h4s[k]["lo"] > 0 for k in h4s); n_sig_sel = sum(h4[k]["lo"] > 0 for k in h4)
    v["H4"] = ("Supported in direction: the gap is larger, not smaller, on high-agreement utterances", f"Delta(high-agreement) minus Delta(all) is positive in all 4 selection-free comparisons "
               f"(CI excludes 0 in {n_sig} of 4) and in {n_sig_sel} of 4 selected-model comparisons.")
    return v


def tables(R):
    m7, m3, m4, hc, m9, m10 = R["m7"], R["m3"], R["m4"], R["hc"], R["m9"], R["m10"]; T = {}
    T["t1_label_reconstruction"] = pd.DataFrame([dict(rule=k, match_rate=U3(v)) for k, v in R["m2"]["rule_match_rates"].items()])
    pol = []
    for p in ["P0", "P1", "P2"]:
        c = m3["counts"][p]; pol.append(dict(policy=p, total=c["total"], **{k: c["per_class"][k] for k in CL}))
    T["t2_policies"] = pd.DataFrame(pol)
    ag = []
    for k, n in [("all_10039", "all utterances"), ("P0_4class", "P0 4-class"), ("P2_primary", "P2 primary")]:
        a = m4["agreement"][k]; ag.append(dict(set=n, n=a["overall"]["n"], alpha=U3(a["overall"]["alpha"]), kappa=U3(a["overall"]["kappa"]),
                                               **{f"alpha_{c}": U3(a["per_class_ovr"][c]["alpha"]) for c in CL}))
    T["t3_agreement"] = pd.DataFrame(ag)
    t = pd.DataFrame(m7["table"]); t["UAR (95% CI)"] = [f"{r.UAR:.3f} [{r.UAR_lo:.3f}, {r.UAR_hi:.3f}]" for r in t.itertuples()]
    T["t4_grid"] = t[["model", "features", "group", "UAR (95% CI)", "WA", "macroF1", "fold_std"]].round(3)
    cost = [dict(quantity="Delta_model eGeMAPS", value=E(m7["delta_model"]["egemaps"])), dict(quantity="Delta_model WavLM", value=E(m7["delta_model"]["wavlm"])),
            dict(quantity="Delta_total", value=E(m7["delta_total"]))] + [dict(quantity=f"Delta_feature {k}", value=E(v)) for k, v in m7["delta_feature"].items()]
    T["t5_costs"] = pd.DataFrame(cost)
    T["t6_headline_soft_hard"] = R["m9t"]
    h = [dict(quantity=f"{k}", value=E(v)) for k, v in m9["H3_gap_shrinkage"].items()]
    T["t7_h3_shrinkage"] = pd.DataFrame(h)
    T["t8_h4"] = pd.DataFrame([dict(condition=k, selected_models=E(m9["H4_gap_ha_minus_all"][k]), selection_free=E(m9["robust_mean_over_models"]["H4_gap_ha_minus_all"][k])) for k in m9["H4_gap_ha_minus_all"]])
    rb = []
    for n, c in m10["conditions"].items():
        if isinstance(c, dict):
            rb.append(dict(condition=n, n=c["n"], delta_model_egemaps=E(c["delta_model"]["egemaps"]) if "egemaps" in c["delta_model"] else "-",
                           delta_model_wavlm=E(c["delta_model"]["wavlm"]) if "wavlm" in c["delta_model"] else "(not run)",
                           mlp_uar_egemaps=U3(c["uar"]["mlp|egemaps"]["point"]), mlp_uar_wavlm=U3(c["uar"]["mlp|wavlm"]["point"]) if "mlp|wavlm" in c["uar"] else "-"))
    T["t9_robustness"] = pd.DataFrame(rb)
    H5 = m10["H5_overlap"]
    T["t10_h5"] = pd.DataFrame([dict(model=k, uar_clean=U3(v["uar_clean"]), uar_overlapped=U3(v["uar_overlapped"]), clean_minus_overlapped=E(v["diff_clean_minus_overlapped"])) for k, v in H5["models"].items()])
    IS = m10["impro_vs_script"]
    T["t11_impro_script"] = pd.DataFrame([dict(quantity=k, value=E(v)) for k, v in IS["delta_model"].items()])
    for k, v in T.items():
        v.to_csv(f"report/tables/{k}.csv", index=False)
    return T


def build():
    R = load_all(); V = verdicts(R); T = tables(R)
    m1, m2, m3, ov, ct, m4, hc, m5, m6, m7, m8, m9, m10, m11 = (R[k] for k in ["m1", "m2", "m3", "ov", "ct", "m4", "hc", "m5", "m6", "m7", "m8", "m9", "m10", "m11"])
    ceil = hc["P2_primary"]["UAR"]; sel_ = m7["selected"]; tb = {(r["model"], r["features"]): r for r in m7["table"]}
    cell = lambda g, f: tb[tuple(sel_[f"{g}|{f}"].split("+"))]
    cs = ct["calibration"]["chosen"]; strict = m8["strict_budget_fixed_C_point"]; tiers = m4["tiers"]["P2_primary"]
    H5 = m10["H5_overlap"]; mw = H5["models"]["mlp|wavlm"]; flags = m10["flags"]
    # ---- values previously typed by hand, now computed (and qualitative claims asserted)
    smh = m9["soft_minus_hard_uar_per_model"]; small = [k for k, v in smh.items() if abs(v["all"]["point"]) <= 0.006]; big = [k for k in smh if k not in small]
    hardest = max(m7["checks_logged"]["hardest_fold_counts"].items(), key=lambda kv: kv[1])[0]
    drop_frac = m3["P2_stats"]["n_dropped"] / m3["counts"]["P0"]["total"]
    nn7 = {k: v["nnz"] for k, v in J("results/m7/hparams/l1lr__egemaps.json").items()}; nn9 = {k: v["nnz"] for k, v in J("results/m9/hparams/l1lr__egemaps.json").items()}
    jump_fold = max(nn9, key=lambda k: abs(nn9[k] - nn7[k]))
    nd = m5["near_dup"]["audio_check"]; assert m5["reextract_max_abs_diff"]["egemaps"] == 0 and m5["reextract_max_abs_diff"]["wavlm"] == 0
    assert all(m2["checks"].values()) and not m1["unreadable"] and m1["n_duplicate_hash_groups"] == 0 and nd["pairs_ncc_gt_0p9"] == 0
    iss = m10["impro_vs_script"]; assert all(iss["script"]["uar"][k]["point"] < iss["impro"]["uar"][k]["point"] for k in iss["impro"]["uar"]), "script no longer lower than impro"
    ag = m4["agreement"]["P2_primary"]["per_class_ovr"]; assert min(CL, key=lambda c: ag[c]["alpha"]) == "neu"
    assert max(r["UAR"] for r in m7["table"] if r["features"] == "egemaps") < m8["black_box_wavlm_mlp"] - 0.05
    cur = R["curves"]; tr = cur[cur.model == "tree"].sort_values("UAR"); tree_peak = int(tr.iloc[-1].setting)
    n_h34 = sum(len(m9[k]) for k in ["H3_gap_shrinkage", "H4_gap_ha_minus_all"]) + sum(len(m9["robust_mean_over_models"][k]) for k in ["H3_gap_shrinkage", "H4_gap_ha_minus_all"]) + len(m9["robust_fixed_pair_selected_under_hard"]["H3_gap_shrinkage"])
    conds_ok = [k for k, v in m10["conditions"].items() if isinstance(v, dict)]; oth_eg = [m10["conditions"][k]["delta_model"]["egemaps"]["point"] for k in conds_ok if k != "P2 (primary)"]
    assert all(k in conds_ok for k in ["P2 pad 0.00", "P2 pad 0.25"]) and m10["conditions"]["P2 pad 0.25"]["delta_feature"], "padding conditions incomplete"
    assert m10["conditions"]["P2 (primary)"]["delta_model"]["egemaps"]["point"] > max(oth_eg), "primary gap no longer the largest"
    P2c, P0c, NNc = m10["conditions"]["P2 (primary)"], m10["conditions"]["P0 (all clips)"], m10["conditions"]["P2 no speaker-norm"]
    nonorm_shift = {k: NNc["uar"][k]["point"] - P2c["uar"][k]["point"] for k in P2c["uar"] if k.startswith("mlp")}
    co = R["co"].set_index("feature"); assert co.loc["loudness_sma3_meanFallingSlope", "coef_ang"] > 0.05 and co.loc["equivalentSoundLevel_dBp", "coef_sad"] < -0.05
    h1j = J("results/m7/hparams/l1lr__egemaps.json"); zero_folds = [k for k, v in h1j.items() if (np.abs(np.array(v["coef"])[:2]).max() if len(v["coef"]) else 0) < 1e-8]
    mw0 = m10["H5_overlap"]["models"]["mlp|wavlm"]; rc, ro = mw0["per_class_recall_clean"], mw0["per_class_recall_overlapped"]
    assert rc[2] > ro[2] and rc[0] < ro[0] and rc[1] < ro[1] and rc[3] < ro[3], "H5 per-class pattern changed"
    sel_wl = sel_["interpretable|wavlm"].split("+")[0]; best_int_test_wl = max([r for r in m7["table"] if r["features"] == "wavlm" and r["group"] == "interpretable"], key=lambda r: r["UAR"])["model"]
    L = []; A = L.append
    A("# The Cost of Interpretability in Speech Emotion Recognition on IEMOCAP\n")
    A(f"*Generated {time.strftime('%Y-%m-%d')} by `src/m12_report.py` from `results/`. Every number below is read from a results file; verdicts are asserted against confidence intervals in the script.*\n")
    A("## Summary\n")
    A(f"We ask how much UAR (unweighted average recall) is lost when an interpretable model replaces a black box for 4-class emotion recognition (neutral, happy+excited, sad, angry) on IEMOCAP, "
      f"using leave-one-session-out evaluation on {m3['counts']['P2']['total']:,} utterances after removing partner-overlap. With named eGeMAPS features the best black box reaches UAR {U3(cell('blackbox', 'egemaps')['UAR'])} "
      f"and the best budgeted interpretable model {U3(cell('interpretable', 'egemaps')['UAR'])} (gap {E(m7['delta_model']['egemaps'])}). With WavLM features the black box reaches {U3(cell('blackbox', 'wavlm')['UAR'])}, "
      f"but a simple model (at most 20 features) on those unreadable features does not benefit (gap {E(m7['delta_model']['wavlm'])}). A dense linear model on WavLM (no budget) gets {U3(tb[('l2lr', 'wavlm')]['UAR'])}, so the cost is set mostly by the complexity budget, "
      f"not by linearity. Training on rater-vote distributions (soft labels) does not shrink the gap consistently.\n")
    A("| Hypothesis | Verdict | Evidence |\n|---|---|---|")
    for k in ["H1", "H2", "H3", "H4"]:
        A(f"| {k} | {V[k][0]} | {V[k][1]} |")
    A("")
    A("## 1. Question and hypotheses\n")
    A("**Question.** How much accuracy do we lose by using an interpretable model instead of a black box for speech emotion recognition on IEMOCAP, and does that gap shrink when models are trained on the raters' disagreement (soft labels) instead of a single majority label?\n")
    A("- **H1** Black-box models beat interpretable models on the same features.\n- **H2** Most of the total gap comes from the features, not the model.\n- **H3** The gap shrinks when training on soft labels.\n"
      "- **H4** The gap differs on high-agreement utterances compared with all utterances.\n\nA fifth planned hypothesis, that overlapping speech hurts accuracy, was dropped: each utterance wav contains only the speaker's own microphone and the clean subset is too small and too different in class mix to test it (section 4.4). Overlap is treated as a robustness question instead: does the choice of overlap handling change the conclusions?\n")
    A("Any outcome is reportable; nothing was tuned to make a hypothesis work.\n")
    A("## 2. Data audit\n")
    A(f"**Inventory.** {m1['n_dialogs']} dialogs and {m1['n_sentence_wavs']:,} utterance wavs were found, all 16 kHz mono 16-bit (dialog wavs: 16 kHz stereo). There are {len(m1['unreadable'])} unreadable files and {m1['n_duplicate_hash_groups']} exact duplicates; "
      f"{m1['n_junk_skipped']:,} junk files (`._*`, `.pk`, `.DS_Store`) were skipped.\n")
    A(f"**Labels.** The official label is reproduced exactly by rule A (plurality of the three external raters, each rater counting once per listed category, ties to `xxx`): match rate {U3(m2['rule_match_rates']['A'])} "
      f"(rule B {U3(m2['rule_match_rates']['B'])}, rule C including self-evaluation {U3(m2['rule_match_rates']['C'])}). All 10,039 utterances, per-session counts and the 4-class set of 5,531 utterances match the dataset guide.\n")
    A(f"**Rater agreement.** Krippendorff's alpha (nominal; multi-category votes kept as distributions) is {U3(m4['agreement']['all_10039']['overall']['alpha'])} on all utterances and {U3(m4['agreement']['P2_primary']['overall']['alpha'])} on the primary set. "
      f"By class (one-vs-rest, primary set): " + ", ".join(f"{c} {U3(m4['agreement']['P2_primary']['per_class_ovr'][c]['alpha'])}" for c in CL) + f". Neutral is the contested class. "
      f"The actor's own label includes the observers' label in {P(m4['self_vs_observers']['lenient_match'], 0)} of {m4['self_vs_observers']['n']:,} self-evaluated utterances.\n")
    A(f"**Approximate human ceiling.** Predicting each rater's vote from the other two raters (only where those two agree) gives UAR {U3(ceil)} and WA {U3(hc['P2_primary']['WA'])}. This is called *approximate* because it is optimistic in one way "
      f"(items without agreement are skipped) and not like-for-like in another (a single rater is compared with a consensus); it is not a true upper bound, and {len(m7['checks_logged']['any_model_above_human_ceiling'])} of our models exceed it ({', '.join(m7['checks_logged']['any_model_above_human_ceiling'])}).\n")
    A(f"**Overlap.** Partner speech overlaps {P(ov['overlap_rate_4class'], 0)} of 4-class utterances in the transcript timings (padding {load_config()['overlap']['padding_sec']} s), more for angry ({P(ov['overlap_rate_by_class_4class']['ang'], 0)}) and neutral "
      f"({P(ov['overlap_rate_by_class_4class']['neu'], 0)}) than sad ({P(ov['overlap_rate_by_class_4class']['sad'], 0)}); class and overlap are strongly dependent (chi-square p = {m3['chi2_class_vs_overlap_P0']['p']:.0e}). "
      f"Every utterance wav is a sample-exact copy of one channel (the speaker's own microphone) of the stereo dialog, so the partner is present only as microphone bleed. A stereo crosstalk detector was calibrated on the transcript overlaps "
      f"but failed its own gate (recall {U3(cs['recall'])} at tau = {cs['tau']:.0f} dB, required 0.5), so it was dropped. The strictly clean policy P1 keeps only {m3['counts']['P1']['total']} utterances "
      f"({m3['counts']['P1']['per_class']['ang']} angry) and fails the retention rule. The primary policy is **P2**: overlapping spans are cut and the longest clean piece (>= 1 s) kept, giving {m3['counts']['P2']['total']:,} utterances "
      f"({', '.join(f'{c} {m3['counts']['P2']['per_class'][c]}' for c in CL)}); every class keeps {P(min(v['retention'] for v in m3['P2_retention'].values()), 0)} or more of its P0 count. P0 (all clips) is carried through as a secondary run.\n")
    oh = R["oh"]
    A(f"**How large are the overlaps?** Overlap is common but mostly sits at the clip edges. Among the {oh['n_overlapped']:,} overlapped utterances the median overlap is {oh['overlap_sec_quantiles']['0.5']:.1f} s, which is {P(oh['overlap_frac_quantiles']['0.5'], 0)} of the clip; "
      f"only {P(oh['share_overlap_lt_0p5s'], 0)} overlap for less than 0.5 s and {P(oh['share_overlap_ge_2s'], 0)} for 2 s or more. In {P(oh['share_edge_only'], 0)} of overlapped utterances every overlapped stretch touches the start or the end of the clip, so P2 mostly trims clip edges (turn boundaries) rather than cutting out the middle. "
      f"The edges are not small, however: P2 trimmed {P(oh['p2_trimmed_share'], 0)} of its clips and kept a median of {P(oh['p2_retained_frac_among_trimmed_median'], 0)} of the duration of the trimmed ones. The overlap histogram is in the figure below.\n")
    A("![overlap histogram](../figures/m3_overlap_hist.png)\n")
    A(f"**Leakage checks.** Leave-one-session-out means no speaker appears in both train and test (asserted in tests and in M3). A shuffled-label logistic regression gives UAR {', '.join(U3(x) for x in m6['shuffled_label_uar'])} "
      f"(must lie in 0.22-0.28), a majority predictor gives exactly 0.25 in every fold, and per-speaker z-scoring cuts a speaker-ID probe from {U3(m6['speaker_id_probe_accuracy']['raw'])} to "
      f"{U3(m6['speaker_id_probe_accuracy']['per_speaker_zscore'])} (chance {U3(m6['speaker_id_chance'])}). Re-extracting 20 random clips reproduced features exactly (max abs difference {m5['reextract_max_abs_diff']['egemaps']:.0f} for eGeMAPS and {m5['reextract_max_abs_diff']['wavlm']:.0f} for WavLM). "
      f"{nd['pairs_checked']} near-duplicate candidates were checked by waveform correlation (max {nd['max_waveform_ncc']:.2f}): none are duplicates. Scripted lines are reused across sessions, but scripted-dialog UAR is lower than improvised for every model, and the models never see text, so there is no sign that the reused scripts inflate scores.\n")
    A("![overlap and policies](../figures/m3_overlap_by_class.png)\n")
    A("Table 1: policy counts.\n\n" + md(T["t2_policies"]) + "\n")
    A("Table 2: rater agreement.\n\n" + md(T["t3_agreement"]) + "\n")
    A("## 3. Method\n")
    A("**Features.** *Interpretable*: the 88 eGeMAPSv02 functionals (openSMILE, named acoustic measurements). *Learned*: mean-pooled hidden states of frozen `microsoft/wavlm-base-plus` (13 layers); one layer per fold is chosen with a "
      "logistic-regression probe on an inner validation session (training sessions only). Layers chosen: " + ", ".join(f"fold {k}: {v['best_layer']}" for k, v in m6["wavlm_layer_choice"].items()) + ".\n")
    A("**Evaluation.** Five folds, each testing on one session and training on the other four; hyperparameters are tuned on a held-out training session and the model is refit on all four training sessions. "
      "Preprocessing (median imputation, per-speaker z-scoring using each speaker's own unlabeled features, global scaler) is fit inside each fold only. Metric: UAR; secondary: accuracy and macro-F1. "
      "Significance: paired bootstrap over test utterances resampled within each session (2,000 resamples, 95% intervals); McNemar's test as a second opinion. Stochastic models use 3 seeds.\n")
    A(f"**Complexity budget (fixed before any M7 result).** Interpretable = L1 logistic regression with at most 20 nonzero features (a feature counts if any class uses it) or a decision tree with at most 16 leaves. "
      "Black boxes: HistGradientBoosting, random forest, MLP. An unbudgeted L2 logistic regression is reported as a reference. The best model per cell is chosen by mean inner-validation UAR, never by test UAR.\n")
    A("**Soft labels.** Each training utterance is expanded into one row per class with nonzero rater-vote mass; the row weight is that mass times the balanced class weight of the utterance's hard label (the weights of one utterance sum to its class weight). "
      "The MLP minimizes weighted cross-entropy against the soft target. Tuning and test labels stay hard, so hard- and soft-trained models are scored on identical test sets.\n")
    A("## 4. Results\n")
    A("### 4.1 The 2x2 grid (hard labels, primary policy) - H1 and H2\n")
    A("| | Interpretable (budgeted) | Black box |\n|---|---|---|")
    for f, nm in [("egemaps", "eGeMAPS"), ("wavlm", "WavLM")]:
        i, b = cell("interpretable", f), cell("blackbox", f)
        A(f"| {nm} | {'*simple model, unreadable features*: ' if f == 'wavlm' else ''}{NICE[i['model']]}: **{U3(i['UAR'])}** [{i['UAR_lo']:.3f}, {i['UAR_hi']:.3f}] | {NICE[b['model']]}: **{U3(b['UAR'])}** [{b['UAR_lo']:.3f}, {b['UAR_hi']:.3f}] |")
    A(f"\nApproximate human ceiling {U3(ceil)}; chance 0.25.\n")
    A("**Reading the WavLM row.** The WavLM / interpretable cell is a *simple model on unreadable features*: a 20-feature L1 model or a 16-leaf tree over WavLM dimensions is small, but nobody can read it, because the dimensions have no names. It shows what a small model can do on learned features; only the eGeMAPS cell is interpretable in the sense of a human-readable model.\n")
    A("![grid](../figures/m7_grid.png)\n")
    A("Table 3: all models, pooled leave-one-session-out UAR.\n\n" + md(T["t4_grid"]) + "\n")
    A("Table 4: costs.\n\n" + md(T["t5_costs"]) + "\n")
    A(f"**H1: {V['H1'][0]}.** {V['H1'][1]} McNemar's test agrees (p = {m7['mcnemar_blackbox_vs_interpretable']['egemaps']['p_value']:.0e} and {m7['mcnemar_blackbox_vs_interpretable']['wavlm']['p_value']:.0e}).\n")
    A(f"**H2: {V['H2'][0]}.** {V['H2'][1]} Order matters because of the interaction: model-first gives {E(m7['decomposition_total']['model_first']['model'])} then {E(m7['decomposition_total']['model_first']['then_feature'])}; "
      f"feature-first gives {E(m7['decomposition_total']['feature_first']['feature'])} then {E(m7['decomposition_total']['feature_first']['then_model'])}. WavLM helps only models that can use many dimensions.\n")
    A(f"**Caution on selection.** On WavLM the validation rule picks {sel_wl} as the best budgeted model although {best_int_test_wl} scored higher on test; with test-oracle selection Delta_model(WavLM) would be {E(m7['delta_model_oracle_selection_for_reference']['wavlm'])}. "
      f"Models above the approximate human ceiling: {', '.join(m7['checks_logged']['any_model_above_human_ceiling'])}. We found no sign of leakage (checks in section 2), and the ceiling is not a like-for-like bound.\n")
    A("### 4.2 Price of interpretability\n")
    l1 = m8["l1lr"]; w3 = l1["within_3pt_of_black_box_egemaps"]; w1 = l1["within_1pt_of_black_box_egemaps"]
    A(f"Sweeping the L1 penalty on eGeMAPS, UAR rises with the number of features and flattens at about {m8['l1lr']['max_UAR']:.3f}; it is within 3 UAR points of the best eGeMAPS black box ({U3(m8['black_box_egemaps_mlp'])}) from about "
      f"{w3['first_crossing_complexity']:.0f} features and within 1 point first at about {w1['first_crossing_complexity']:.0f} features. No eGeMAPS model gets near the WavLM black box ({U3(m8['black_box_wavlm_mlp'])}). "
      f"Decision trees do not follow this pattern: UAR peaks at {tree_peak} leaves ({U3(m8['tree']['max_UAR'])}) and then declines (overfitting at minimum leaf size 5), so trees never come within 3 points of the black box.\n")
    A(f"**Sensitivity of the headline.** The tuned L1 model used only {strict['m7_l1lr_mean_nnz']:.1f} features on average, under-using the budget. A fixed penalty chosen from feature counts only (largest C with at most 20 features in every fold, "
      f"{min(strict['nnz_by_fold'])}-{max(strict['nnz_by_fold'])} features) gives UAR {U3(strict['UAR'])} and Delta_model(eGeMAPS) = {E(strict['delta_model_vs_mlp_egemaps'])} instead of {E(m7['delta_model']['egemaps'])}. "
      "H1 still holds on eGeMAPS; the pre-registered number remains the headline.\n")
    A("![lr curve](../figures/m8_price_curve_lr.png)\n\n![tree curve](../figures/m8_price_curve_tree.png)\n")
    A("### 4.3 Soft labels - H3 and H4\n")
    A("Table 5: hard- versus soft-trained (best model per cell, UAR with 95% CI).\n\n" + md(T["t6_headline_soft_hard"]) + "\n")
    A(f"**H3: {V['H3'][0]}.** {V['H3'][1]} Per-model changes from soft training are within +-0.006 UAR for {len(small)} of {len(smh)} runs; the exceptions are " + ", ".join(f"{k} ({E(smh[k]['all'])})" for k in big) + f". The L1-LR on eGeMAPS change is driven by tuning variance: in fold {jump_fold} the inner tuning chose {nn9[jump_fold]} features under soft training versus {nn7[jump_fold]} under hard training. "
      f"Soft training does improve how well predicted distributions match rater distributions: cross-entropy of the best black box on WavLM falls from {m9['soft_cross_entropy_selected']['hard|blackbox|wavlm']:.3f} to {m9['soft_cross_entropy_selected']['soft|blackbox|wavlm']:.3f} nats.\n")
    A("Table 6: gap shrinkage Delta(hard) - Delta(soft); positive means the gap shrank.\n\n" + md(T["t7_h3_shrinkage"]) + "\n")
    A(f"**H4: {V['H4'][0]}.** {V['H4'][1]} If the black box's advantage came from fitting label noise, the gap would shrink on high-agreement utterances; it grows. "
      f"The high-agreement subset has {tiers['high_agreement']:,} utterances ({', '.join(f'{c} {tiers['high_agreement_per_class'][c]}' for c in CL)}), mean soft-label entropy is only {tiers['soft_entropy_mean_bits']:.2f} bits, so soft targets are close to one-hot for most clips.\n")
    A("Table 7: Delta(high-agreement) - Delta(all).\n\n" + md(T["t8_h4"]) + "\n")
    A("![h3](../figures/m9_gap_hard_vs_soft.png)\n\n![h4](../figures/m9_gap_by_agreement.png)\n")
    A("### 4.4 Robustness: does overlap handling change the conclusions?\n")
    A("Table 8: Delta_model and MLP UAR under alternative settings (interpretable model chosen by inner-validation UAR in each condition).\n\n" + md(T["t9_robustness"]) + "\n")
    A("**Overlap handling does not change the conclusions.** Under P2, P0 (no trimming), padding 0.00 s and 0.25 s, and without per-speaker normalization, black boxes beat interpretable models and the WavLM interaction pattern persists (Table 8).\n")
    A(f"*Exploratory, not a hypothesis test.* On P0, UAR on clean utterances is lower than on overlapped ones for all {len(H5['models'])} models, but the clean subset is only {H5['n_clean']} clips with a different class mix. Recall on clean clips is higher for sad ({rc[2]:.2f} vs {ro[2]:.2f} for MLP/WavLM) and lower for neutral ({rc[0]:.2f} vs {ro[0]:.2f}), happy ({rc[1]:.2f} vs {ro[1]:.2f}) and angry ({rc[3]:.2f} vs {ro[3]:.2f}); "
      f"the clean subset is {P(H5['clean_class_counts'][2] / H5['n_clean'], 0)} sad with only {H5['clean_class_counts'][3]} angry clips and its clips are longer ({H5['mean_duration_clean']:.1f} s vs {H5['mean_duration_overlapped']:.1f} s), "
      "so class mix and turn structure are confounded with overlap. This data cannot isolate an effect of overlap, so we make no claim about it; because each clip contains only the speaker's own microphone, large effects were not expected.\n")
    A("Table 9: UAR on clean versus overlapped utterances (P0), exploratory.\n\n" + md(T["t10_h5"]) + "\n")
    A(f"**Improvised versus scripted.** The gap is similar: Delta_model(eGeMAPS) is {E(m10['impro_vs_script']['delta_model']['impro|egemaps'])} improvised and {E(m10['impro_vs_script']['delta_model']['script|egemaps'])} scripted "
      f"(difference {E(m10['impro_vs_script']['delta_model']['impro_minus_script|egemaps'])}); on WavLM {E(m10['impro_vs_script']['delta_model']['impro|wavlm'])} and {E(m10['impro_vs_script']['delta_model']['script|wavlm'])}.\n")
    A("Table 10: improvised versus scripted.\n\n" + md(T["t11_impro_script"]) + "\n")
    A(("**Conclusions that flipped under robustness checks: none** (every condition keeps Delta_model > 0 with a CI excluding 0 and the same interaction pattern). " if not flags else
       "**Robustness flags (conclusions that changed):** " + "; ".join(flags) + ". ") +
      f"Magnitudes vary: the budgeted L1-LR on WavLM is unstable (UAR {U3(m10['conditions']['P2 (primary)']['uar']['l1lr|wavlm']['point'])} on P2 but {U3(m10['conditions']['P0 (all clips)']['uar']['l1lr|wavlm']['point'])} on P0), "
      "so Delta_model on WavLM ranges widely depending on which budgeted model is selected. "
      f"The primary-policy eGeMAPS gap ({P2c['delta_model']['egemaps']['point']:+.3f}) is the largest of the {len(conds_ok)} conditions; the others range from {min(oth_eg):+.3f} to {max(oth_eg):+.3f}, consistent with the tuned L1 model under-using its budget (section 4.2). "
      f"Moving the overlap padding to 0.00 s or 0.25 s changes the MLP's UAR by at most {max(abs(m10['conditions'][k]['uar'][u]['point'] - P2c['uar'][u]['point']) for k in ['P2 pad 0.00', 'P2 pad 0.25'] for u in ['mlp|egemaps', 'mlp|wavlm']):.3f}. Removing per-speaker normalization changes the MLP's UAR by " + ", ".join(f"{k.split('|')[1]} {v:+.3f}" for k, v in nonorm_shift.items()) + " and does not alter the conclusions.\n")
    A("![robustness](../figures/m10_delta_by_condition.png)\n\n![clean versus overlapped, exploratory](../figures/m10_h5_overlap.png)\n")
    A("## 5. What the interpretable model learned\n")
    A(f"The budgeted L1-LR chose {m11['n_features_ever_selected']} different eGeMAPS features across the five folds; only {m11['selected_in_all_5_folds']} were chosen in every fold (mean fold-to-fold Jaccard {m11['jaccard_between_folds']:.2f}), "
      "so the specific list is unstable when features are correlated. The stable part is a loudness story: for **angry**, steeper loudness rises and falls and higher voiced spectral flux; for **sad**, lower overall sound level, "
      "more variable unvoiced-segment length and a lower low-frequency spectral slope. "
      f"In folds {', '.join(sorted(zero_folds))} the model has **no coefficients at all for neutral and happy**: it separates sad and angry from the rest and "
      f"decides between neutral and happy through its intercepts, which helps explain its UAR of {U3(tb[('l1lr', 'egemaps')]['UAR'])}. Trees built in different folds also use different features "
      f"(mean Jaccard {m11['tree_feature_sets_per_fold_overlap_jaccard']:.2f}). These are associations in acted speech, not causal statements. "
      "Details: `results/m11/feature_story.md`, `figures/m11_tree.png`.\n")
    A("## 6. Limitations\n")
    for s in [f"Acted emotion from 10 actors in 5 sessions; five folds and one dataset, so differences between sessions (fold {hardest} is the hardest in most runs) matter and generalization to natural speech is untested.",
              "The approximate human ceiling is optimistic in how it filters items and is not like-for-like with consensus-label evaluation; it should not be read as an upper bound.",
              f"Overlap handling: utterance wavs are single-microphone, so partner bleed rather than partner speech is what contaminates a clip. The crosstalk detector failed its recall gate ({U3(cs['recall'])} < 0.5). "
              f"The primary policy P2 cuts whole overlapped spans, which also removes the speaker's own speech during them, drops {P(drop_frac, 0)} of clips non-randomly, and shifts the class mix slightly; P0 is reported alongside.",
              f"Happy and excited are merged; frustration and other classes are excluded, which removes the hardest, noisiest categories. Soft labels are 4-class renormalizations ({P(tiers['outside_mass_frac_gt0'])} of primary-policy utterances had votes outside the 4 classes).",
              f"The 'best' budgeted model is chosen by inner-validation UAR; this is leak-free but noisy (the budgeted L1-LR on WavLM is unstable; validation chose {sel_wl} while {best_int_test_wl} scored higher on test), and the tuned L1-LR under-used its budget, so Delta_model(eGeMAPS) is {strict['delta_model_vs_mlp_egemaps']['point']:+.3f} with a fixed-penalty sensitivity point versus {m7['delta_model']['egemaps']['point']:+.3f} pre-registered.",
              f"About {n_h34} H3/H4-related intervals are reported at 95% without multiple-comparison correction; bounds close to zero (for example H3 on eGeMAPS) are fragile.",
              "Gradient boosting's built-in early-stopping split can place duplicated soft-label rows of one utterance on both sides. The L1 sweep on WavLM (M8.3), a cleaned-training variant (M9.5), a 5-class variant (M10.5) and permutation importance for the black box (M11.3) were not run.",
              "Text was not used; a text or multimodal comparison is out of scope."]:
        A(f"- {s}")
    A("\n## 7. Conclusion\n")
    A(f"On IEMOCAP, with leave-one-session-out evaluation, black-box models beat interpretable ones by {E(m7['delta_model']['egemaps'], 2)} UAR on named acoustic features and by far more on learned features. The cost is not a fixed number: it depends on the complexity budget "
      f"(a dense linear model on WavLM is within {abs(tb[('mlp', 'wavlm')]['UAR'] - tb[('l2lr', 'wavlm')]['UAR']):.3f} UAR of the MLP, but a 20-feature one is not), and it is not explained by label noise or by soft labels. "
      f"In practice: an L1 model on eGeMAPS comes within 3 UAR points of the best eGeMAPS black box at about {w3['first_crossing_complexity']:.0f} features, although the tuned model constrained to at most 20 features was {abs(m7['delta_model']['egemaps']['point']):.3f} behind; "
      "if accuracy matters most, the learned representation with a dense model is clearly better.\n")
    Path("report/report.md").write_text("\n".join(L)); print("wrote report/report.md,", len(L), "lines;", len(T), "tables")
    for k, v in V.items():
        print(k, "->", v[0])


def verify():
    out = {"checks": {}}; R = load_all(); txt = Path("report/report.md").read_text()
    # 1. tests and sanity
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], capture_output=True, text=True); out["pytest"] = r.stdout.strip().splitlines()[-1]; out["checks"]["pytest_pass"] = r.returncode == 0
    from src.m6_harness import run_sanity
    s = run_sanity(); old = R["m6"]
    out["m6_sanity_rerun_matches"] = dict(shuffled=bool(np.allclose(s["shuffled_label_uar"], old["shuffled_label_uar"])), speaker_probe=bool(np.isclose(s["speaker_id_probe_accuracy"]["raw"], old["speaker_id_probe_accuracy"]["raw"])),
                                          all_checks=all(s["checks"].values()))
    out["checks"]["sanity_rerun_pass"] = all(out["m6_sanity_rerun_matches"].values())
    # 2. recompute 3 headline numbers with FRESH code (sklearn balanced accuracy = UAR) from saved predictions
    from sklearn.metrics import balanced_accuracy_score
    heads = {}
    for name, d, m, f in [("MLP/WavLM (hard)", "results/m7", "mlp", "wavlm"), ("L1-LR/eGeMAPS (hard)", "results/m7", "l1lr", "egemaps"), ("MLP/WavLM (soft)", "results/m9", "mlp", "wavlm")]:
        df = pd.read_parquet(f"{d}/preds/{m}__{f}.parquet"); u = []
        for sd, g in df.groupby("seed"):
            g = g.sort_values("utt_id"); u.append(balanced_accuracy_score(g.y_true.values, g[[f"p_{c}" for c in CL]].values.argmax(1)))
        val = float(np.mean(u)); heads[name] = dict(recomputed=val, in_report=f"{val:.3f}" in txt)
    out["headline_recomputed"] = heads; out["checks"]["headline_numbers_match_report"] = all(v["in_report"] for v in heads.values())
    # 3. figures exist and are newer than their source results
    figs = sorted(set(re.findall(r"\(\.\./(figures/[^)]+)\)", txt))); bad = []
    src = {"m3_overlap_by_class": "results/m3/summary.json", "m7_grid": "results/m7/costs.json", "m8_price_curve_lr": "results/m8/summary.json", "m8_price_curve_tree": "results/m8/summary.json",
           "m9_gap_hard_vs_soft": "results/m9/h3_h4.json", "m9_gap_by_agreement": "results/m9/h3_h4.json", "m10_delta_by_condition": "results/m10/summary.json", "m10_h5_overlap": "results/m10/summary.json", "m3_overlap_hist": "results/m3/overlap_hist.json"}
    for f in figs:
        p = Path(f); k = p.stem
        if not p.exists() or (k in src and p.stat().st_mtime + 1 < Path(src[k]).stat().st_mtime):
            bad.append(f)
    out["figures_checked"] = figs; out["figures_missing_or_stale"] = bad; out["checks"]["figures_ok"] = not bad
    # 4. every 'better/supported' claim vs CI (recomputed from raw json, independent of verdict code)
    c = []
    dm = R["m7"]["delta_model"]; c.append(("H1 eGeMAPS CI>0", dm["egemaps"]["lo"] > 0)); c.append(("H1 WavLM CI>0", dm["wavlm"]["lo"] > 0))
    h4s = R["m9"]["robust_mean_over_models"]["H4_gap_ha_minus_all"]; c.append(("H4 direction positive in all selection-free", all(v["point"] > 0 for v in h4s.values())))
    c.append(("H3 not claimed (not both shrinkage CIs > 0)", not (R["m9"]["H3_gap_shrinkage"]["egemaps|all"]["lo"] > 0 and R["m9"]["H3_gap_shrinkage"]["wavlm|all"]["lo"] > 0)))
    c.append(("overlap-handling conditions all keep Delta_model CI > 0", all(v["delta_model"][f]["lo"] > 0 for k, v in R["m10"]["conditions"].items() if "delta_model" in v for f in ("egemaps", "wavlm"))))
    out["claim_checks"] = {k: bool(v) for k, v in c}; out["checks"]["claims_match_cis"] = all(v for _, v in c)
    out["all_checks_pass"] = all(out["checks"].values()); Path("results/m12").mkdir(exist_ok=True); Path("results/m12/verification.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else build()
