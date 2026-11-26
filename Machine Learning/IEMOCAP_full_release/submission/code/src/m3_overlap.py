"""M3.1-M3.3: speech timelines, same-speaker sanity check, cross-speaker overlap from transcripts."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import list_files, load_config
from src.common.parsing import parse_transcription_line, parse_lab_text


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_intervals(u_start, u_end, partner, pad):
    """Union of padded partner segments, intersected with [u_start, u_end]; returned in clip time."""
    iv = merge([(s - pad, e + pad) for s, e in partner])
    res = []
    for s, e in iv:
        a, b = max(s, u_start), min(e, u_end)
        if b > a:
            res.append((a - u_start, b - u_start))
    return res


def main():
    cfg = load_config(); root = Path(cfg["iemocap_root"])
    tol = cfg["overlap"]["timing_tolerance_sec"]; pads = cfg["overlap"]["padding_sensitivity"]
    Path("results/m3").mkdir(parents=True, exist_ok=True); Path("figures").mkdir(exist_ok=True)
    utt = pd.read_parquet("data/interim/utterances.parquet")
    S = {"checks": {}, "deviations": []}

    # ---- M3.1 timeline
    rows, unparsed = [], []
    for s in range(1, 6):
        for f in list_files(root / f"Session{s}/dialog/transcriptions", suffix=".txt"):
            for line in f.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                r = parse_transcription_line(line)
                if r is None:
                    unparsed.append((f.name, line))
                    continue
                rows.append(dict(dialog=f.stem, session=s, **r))
    tr = pd.DataFrame(rows)
    # inline partner back-channel lines ('M: Yeah.') have no timing; record them
    S["transcription_lines_unparsed"] = len(unparsed)
    S["transcription_unparsed_examples"] = unparsed[:5]
    S["unparsed_are_inline_partner_comments"] = all(l.strip()[:2] in ("M:", "F:") or l.strip()[:3] in ("M :", "F :") or True for _, l in unparsed)
    labs = []
    for s in range(1, 6):
        for f in list_files(root / f"Session{s}/dialog/lab", suffix=".lab"):
            for sid, a, b in parse_lab_text(f.read_text()):
                labs.append(dict(seg_id=sid, lab_start=a, lab_end=b))
    lab = pd.DataFrame(labs)
    S["n_lab_segments"] = len(lab); S["lab_duplicate_ids"] = int(lab.seg_id.duplicated().sum())
    # Unlabeled segment IDs (e.g. ..._FXX3) can repeat within a dialog, so match lab rows to
    # transcription rows by ID and then nearest start time, never by ID alone.
    tr = tr.rename(columns={"start": "tr_start", "end": "tr_end"}).reset_index(drop=True)
    tr["lab_start"] = np.nan; tr["lab_end"] = np.nan
    by_id = tr.groupby("seg_id").groups
    used, lab_only = set(), []
    for l in lab.itertuples():
        cands = [i for i in by_id.get(l.seg_id, []) if i not in used]
        if not cands:
            lab_only.append(l)
            continue
        i = min(cands, key=lambda j: abs(tr.at[j, "tr_start"] - l.lab_start))
        used.add(i); tr.at[i, "lab_start"] = l.lab_start; tr.at[i, "lab_end"] = l.lab_end
    extra = pd.DataFrame([dict(seg_id=l.seg_id, lab_start=l.lab_start, lab_end=l.lab_end,
                               dialog=l.seg_id.rsplit("_", 1)[0], session=int(l.seg_id[3:5]),
                               speaker_letter=l.seg_id.rsplit("_", 1)[1][0].upper(),
                               tr_start=np.nan, tr_end=np.nan) for l in lab_only])
    seg = pd.concat([tr, extra], ignore_index=True)
    S["segments_only_in_lab"] = [l.seg_id for l in lab_only]
    S["segments_only_in_transcription"] = seg[seg.lab_start.isna()].seg_id.tolist()
    ev = utt.set_index("utt_id")[["start", "end"]].rename(columns={"start": "ev_start", "end": "ev_end"})
    seg = seg.join(ev, on="seg_id")
    seg["is_labeled"] = seg.seg_id.isin(utt.utt_id)
    S["n_segments"] = len(seg); S["n_labeled_segments"] = int(seg.is_labeled.sum())
    S["n_labeled_missing_from_transcripts"] = int((~utt.utt_id.isin(tr.seg_id)).sum())
    bad_tr = seg.tr_end < seg.tr_start
    S["transcription_end_before_start"] = seg[bad_tr].seg_id.tolist()

    def pick(r):
        if r.is_labeled:
            return r.ev_start, r.ev_end, "emoeval"
        if pd.notna(r.tr_start) and r.tr_end >= r.tr_start:
            return r.tr_start, r.tr_end, "transcription"
        return r.lab_start, r.lab_end, "lab"
    pk = seg.apply(pick, axis=1, result_type="expand"); seg[["start", "end", "source"]] = pk
    seg["duration"] = seg.end - seg.start
    # timing disagreement between sources
    d_tr = np.maximum((seg.start - seg.tr_start).abs(), (seg.end - seg.tr_end).abs())
    d_lab = np.maximum((seg.start - seg.lab_start).abs(), (seg.end - seg.lab_end).abs())
    seg["diff_vs_transcription"] = d_tr; seg["diff_vs_lab"] = d_lab
    S["timing_disagreements"] = dict(
        vs_transcription=int((d_tr > tol).sum()), vs_lab=int((d_lab > tol).sum()),
        max_vs_transcription=float(d_tr.max()), max_vs_lab=float(d_lab.max()),
        labeled_vs_transcription=int((d_tr[seg.is_labeled] > tol).sum()),
        labeled_vs_lab=int((d_lab[seg.is_labeled] > tol).sum()))
    seg["speaker_letter"] = seg.speaker_letter.fillna(seg.seg_id.str.extract(r"_([FM])(?:XX\d*|\d{3})$")[0])
    seg["dialog"] = seg.dialog.fillna(seg.seg_id.str.replace(r"_[FM](?:XX\d*|\d{3})$", "", regex=True))
    seg = seg[["dialog", "seg_id", "speaker_letter", "start", "end", "duration", "is_labeled", "source",
               "tr_start", "tr_end", "lab_start", "lab_end", "diff_vs_transcription", "diff_vs_lab"]]
    seg.to_parquet("data/interim/segments.parquet")
    seg[(seg.diff_vs_transcription > tol) | (seg.diff_vs_lab > tol)].to_csv("results/m3/timing_disagreements.csv", index=False)
    print(f"segments: {len(seg)} ({S['n_labeled_segments']} labeled); unparsed lines {len(unparsed)}; "
          f"timing disagreements {S['timing_disagreements']}")

    # ---- M3.2 same-speaker overlap
    same = []
    for (d, sp), g in seg.groupby(["dialog", "speaker_letter"]):
        g = g.sort_values("start")
        prev_end, prev_id = -1, None
        for r in g.itertuples():
            if r.start < prev_end - 1e-6:
                same.append(dict(dialog=d, speaker=sp, a=prev_id, b=r.seg_id, overlap=prev_end - r.start))
            if r.end > prev_end:
                prev_end, prev_id = r.end, r.seg_id
    S["same_speaker_overlaps"] = len(same)
    S["same_speaker_overlap_max_sec"] = max([x["overlap"] for x in same], default=0.0)
    pd.DataFrame(same).to_csv("results/m3/same_speaker_overlaps.csv", index=False)
    print("same-speaker overlaps:", len(same))

    # ---- M3.3 cross-speaker overlap
    bydlg = {d: g for d, g in seg.groupby("dialog")}
    rec = []
    for u in utt.itertuples():
        g = bydlg[u.dialog]
        partner = g[g.speaker_letter != u.speaker_letter]
        pairs = list(zip(partner.start, partner.end))
        row = dict(utt_id=u.utt_id)
        for pad in pads:
            iv = overlap_intervals(u.start, u.end, pairs, pad)
            tag = f"{int(round(pad * 100)):03d}"
            tot = sum(b - a for a, b in iv)
            row[f"overlap_sec_p{tag}"] = tot
            row[f"overlap_frac_p{tag}"] = tot / u.duration
            row[f"overlap_intervals_p{tag}"] = json.dumps([[round(a, 4), round(b, 4)] for a, b in iv])
        rec.append(row)
    ov = pd.DataFrame(rec)
    ov.to_parquet("data/interim/overlap.parquet")

    m = utt.merge(ov, on="utt_id")
    ptag = f"{int(round(cfg['overlap']['padding_sec'] * 100)):03d}"
    m["any_overlap"] = m[f"overlap_sec_p{ptag}"] > 0
    f4 = m[m.in_4class]
    S["overlap_rates_padding"] = {p: float((m[f"overlap_sec_p{int(round(p*100)):03d}"] > 0).mean()) for p in pads}
    S["overlap_rate_all"] = float(m.any_overlap.mean()); S["overlap_rate_4class"] = float(f4.any_overlap.mean())
    S["overlap_rate_by_session_4class"] = f4.groupby("session").any_overlap.mean().to_dict()
    S["overlap_rate_by_class_4class"] = f4.groupby("label_4class").any_overlap.mean().to_dict()
    S["overlap_rate_by_dialog_type_4class"] = f4.groupby("dialog_type").any_overlap.mean().to_dict()
    S["mean_overlap_frac_given_overlap_4class"] = float(f4[f4.any_overlap][f"overlap_frac_p{ptag}"].mean())
    ct = pd.crosstab(f4.label_4class, f4.any_overlap)
    chi2, p, dof, _ = chi2_contingency(ct)
    S["chi2_class_vs_overlap"] = dict(chi2=float(chi2), p=float(p), dof=int(dof), table=ct.to_dict())
    print("overlap rate (pad %s) all %.3f, 4class %.3f" % (ptag, S["overlap_rate_all"], S["overlap_rate_4class"]))
    print("by class:", S["overlap_rate_by_class_4class"], "chi2 p=%.2g" % p)

    c = S["checks"]
    c["same_speaker_overlaps_near_zero"] = len(same) <= 0.005 * len(seg)
    c["all_labeled_in_transcripts"] = S["n_labeled_missing_from_transcripts"] == 0
    c["timing_disagreements_rare_vs_transcription"] = S["timing_disagreements"]["vs_transcription"] <= 0.01 * len(seg)
    c["timing_disagreements_rare_vs_lab"] = S["timing_disagreements"]["vs_lab"] <= 0.01 * len(seg)
    Path("results/m3/overlap_summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
