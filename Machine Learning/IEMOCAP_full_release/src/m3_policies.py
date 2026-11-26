"""M3.5 audio policies (P0/P1/P2) and M3.6 leakage checks."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config
from src.m3_overlap import merge

SR = 16000
CLASSES = ["neu", "hap", "sad", "ang"]


def complement(iv, dur):
    out, cur = [], 0.0
    for a, b in merge(iv):
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < dur:
        out.append((cur, dur))
    return out


def longest_piece(iv, dur, min_len):
    pieces = complement(iv, dur)
    if not pieces:
        return None
    a, b = max(pieces, key=lambda p: p[1] - p[0])
    return (a, b) if b - a >= min_len else None


def counts(df):
    return df.groupby("label_4class").size().reindex(CLASSES).fillna(0).astype(int).to_dict()


def main(write_audio=True):
    cfg = load_config(); root = Path(cfg["iemocap_root"])
    pad = cfg["overlap"]["padding_sec"]; ptag = f"{int(round(pad * 100)):03d}"
    min_keep = cfg["overlap"]["trim_min_remaining_sec"]; ct_max = cfg["overlap"]["crosstalk_max_detected_sec"]
    rule = cfg["primary_policy_rule"]
    utt = pd.read_parquet("data/interim/utterances.parquet"); ov = pd.read_parquet("data/interim/overlap.parquet")
    ct = pd.read_parquet("data/interim/crosstalk.parquet")
    kept = bool(ct.detector_kept.iloc[0])
    S = {"detector_kept": kept, "checks": {}}
    d = utt[utt.in_4class].merge(ov, on="utt_id").merge(ct[["utt_id", "crosstalk_sec", "crosstalk_intervals", "detector_ran"]], on="utt_id")
    d = d.rename(columns={"wav_path": "audio_path_orig"})

    def intervals(r, ptag_):
        iv = json.loads(r[f"overlap_intervals_p{ptag_}"])
        if kept and ptag_ == ptag:
            iv += json.loads(r.crosstalk_intervals)
        return iv

    def p1_mask(ptag_):
        m = d[f"overlap_sec_p{ptag_}"] == 0
        if kept and ptag_ == ptag:
            m &= d.crosstalk_sec <= ct_max
        return m

    def p2_table(ptag_):
        rows = []
        for r in d.itertuples(index=False):
            r = r._asdict()
            iv = intervals(pd.Series(r), ptag_)
            lp = longest_piece(iv, r["duration"], min_keep) if iv else (0.0, r["duration"])
            rows.append(dict(utt_id=r["utt_id"], trim_start=lp[0] if lp else np.nan, trim_end=lp[1] if lp else np.nan,
                             keep=lp is not None, trimmed=bool(iv)))
        return pd.DataFrame(rows)

    # ---- policies
    P0 = d.copy()
    P1 = d[p1_mask(ptag)].copy()
    t2 = p2_table(ptag)
    P2 = d.merge(t2, on="utt_id"); P2 = P2[P2.keep].copy()
    P2["new_duration"] = P2.trim_end - P2.trim_start
    out_dir = Path("data/processed/P2_audio"); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in P2.itertuples():
        if r.trimmed and write_audio:
            x, sr = sf.read(root / r.audio_path_orig, dtype="int16"); assert sr == SR
            a, b = int(round(r.trim_start * SR)), int(round(r.trim_end * SR))
            p = out_dir / f"{r.utt_id}.wav"
            sf.write(p, x[a:b], SR, subtype="PCM_16")
            paths.append(str(p))
        else:
            paths.append(str(root / r.audio_path_orig))
    P2["audio_path"] = paths
    P0["audio_path"] = [str(root / p) for p in P0.audio_path_orig]
    P1["audio_path"] = [str(root / p) for p in P1.audio_path_orig]
    for name, df in [("P0", P0), ("P1", P1), ("P2", P2)]:
        df.drop(columns=[c for c in df.columns if c.startswith("overlap_intervals")]).to_parquet(f"data/processed/{name}.parquet")

    # ---- counts and decision
    S["counts"] = {n: dict(per_class=counts(df), per_session=df.groupby("session").size().to_dict(), total=len(df))
                   for n, df in [("P0", P0), ("P1", P1), ("P2", P2)]}
    c0 = S["counts"]["P0"]["per_class"]

    def meets(df):
        c = counts(df)
        return all(c[k] >= rule["min_class_retention"] * c0[k] and c[k] >= rule["min_class_count"] for k in CLASSES), \
            {k: dict(count=c[k], retention=round(c[k] / c0[k], 3)) for k in CLASSES}
    S["P1_meets_rule"], S["P1_retention"] = meets(P1)
    S["P2_meets_rule"], S["P2_retention"] = meets(P2)
    S["primary_policy"] = "P1" if S["P1_meets_rule"] else ("P2" if S["P2_meets_rule"] else "NONE_STOP_AND_REPORT")
    S["P2_stats"] = dict(n_trimmed=int(P2.trimmed.sum()), n_untouched=int((~P2.trimmed).sum()),
                         n_dropped=int((~t2.keep).sum()), mean_new_duration=float(P2.new_duration.mean()),
                         mean_orig_duration=float(P0.duration.mean()))
    # padding sensitivity (counts only)
    S["padding_sensitivity_counts"] = {}
    for p in cfg["overlap"]["padding_sensitivity"]:
        tg = f"{int(round(p * 100)):03d}"
        S["padding_sensitivity_counts"][str(p)] = dict(
            P1=counts(d[d[f"overlap_sec_p{tg}"] == 0]),
            P2=counts(d.merge(p2_table(tg), on="utt_id").query("keep")))
    print("counts:", json.dumps({k: v["per_class"] for k, v in S["counts"].items()}))
    print("P1 meets rule:", S["P1_meets_rule"], S["P1_retention"]); print("P2 meets rule:", S["P2_meets_rule"], S["P2_retention"])
    print("PRIMARY:", S["primary_policy"])

    # ---- class mix and chi-square
    P1mix = P1.label_4class.value_counts(normalize=True).reindex(CLASSES).to_dict()
    P0mix = P0.label_4class.value_counts(normalize=True).reindex(CLASSES).to_dict()
    S["class_mix"] = dict(P0=P0mix, P1=P1mix, P2=P2.label_4class.value_counts(normalize=True).reindex(CLASSES).to_dict())
    ct_tab = pd.crosstab(P0.label_4class, P0[f"overlap_sec_p{ptag}"] > 0)
    chi2, p, dof, _ = chi2_contingency(ct_tab)
    S["chi2_class_vs_overlap_P0"] = dict(chi2=float(chi2), p=float(p), dof=int(dof))

    # ---- asserts (M3 checks)
    c = S["checks"]
    c["P1_all_overlap_zero"] = bool((P1[f"overlap_sec_p{ptag}"] == 0).all())
    c["P2_all_ge_min_duration"] = bool((P2.new_duration >= min_keep - 1e-9).all())
    ok_dur = True
    for r in P2[P2.trimmed].itertuples():
        if write_audio:
            ok_dur &= abs(sf.info(r.audio_path).duration - r.new_duration) <= 0.02
    c["P2_trimmed_duration_matches_interval"] = bool(ok_dur)

    # ---- M3.6 leakage
    hashes = pd.read_parquet("results/m1/wav_md5.parquet")
    h4 = hashes[hashes.utt_id.isin(P0.utt_id)]
    c["zero_exact_duplicates_in_4class"] = not h4.duplicated("md5").any()
    spk_sessions = utt.groupby("speaker_id").session.nunique()
    c["each_speaker_in_exactly_one_session"] = bool((spk_sessions == 1).all())
    leak = []
    for k in range(1, 6):
        te = set(utt[utt.session == k].speaker_id); tr = set(utt[utt.session != k].speaker_id)
        leak.append(len(te & tr))
    c["loso_no_speaker_overlap"] = sum(leak) == 0
    # overlapping utterances of the two speakers share audio only within one dialog -> one session -> one fold
    pairs_cross_session = 0
    for dlg, g in d.groupby("dialog"):
        pairs_cross_session += int(g.session.nunique() > 1)
    c["overlapping_utterances_same_fold"] = pairs_cross_session == 0
    S["all_checks_pass"] = all(c.values())

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rate = P0.groupby("label_4class")[f"overlap_sec_p{ptag}"].apply(lambda s: (s > 0).mean()).reindex(CLASSES)
    ax[0].bar(CLASSES, rate.values * 100, color="#4C72B0"); ax[0].set_ylabel("% utterances with any partner overlap")
    ax[0].set_xlabel("class"); ax[0].set_title(f"Transcript overlap by class (padding {pad}s)"); ax[0].set_ylim(0, 100)
    x = np.arange(4); w = 0.27
    for i, (n, df) in enumerate([("P0", P0), ("P1", P1), ("P2", P2)]):
        ax[1].bar(x + (i - 1) * w, [counts(df)[k] for k in CLASSES], w, label=n)
    ax[1].axhline(rule["min_class_count"], ls="--", c="k", lw=1, label=f"min class count {rule['min_class_count']}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(CLASSES); ax[1].set_xlabel("class"); ax[1].set_ylabel("utterances")
    ax[1].set_title("Utterances kept per class by policy"); ax[1].legend()
    plt.tight_layout(); plt.savefig("figures/m3_overlap_by_class.png", dpi=130)

    Path("results/m3/summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S["checks"], indent=1)); print("class mix:", S["class_mix"])


if __name__ == "__main__":
    main()
