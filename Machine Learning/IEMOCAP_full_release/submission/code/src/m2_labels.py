"""M2: label parsing and ground-truth reconstruction."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import list_files, load_config
from src.common.parsing import (parse_summary_text, parse_cat_text, parse_utt_id, map_category, rater_type)

EXP_SESSION = {1: 1819, 2: 1811, 3: 2136, 4: 2103, 5: 2170}
EXP_LABELS = dict(xxx=2507, fru=1849, neu=1708, ang=1103, sad=1084, exc=1041, hap=595, sur=107, fea=40, oth=3, dis=2)
FOUR = {"neu", "hap", "exc", "sad", "ang"}



def rule_label(votes: pd.DataFrame, rule: str):
    """votes: rows for one utterance. Returns 3-letter code or 'xxx' on tie / no votes."""
    v = votes[votes.category_code.notna()]
    if rule in ("A", "B"):
        v = v[v.rater_type == "external"]
    if v.empty:
        return "xxx"
    w = v.weight if rule == "B" else pd.Series(1.0, index=v.index)
    tot = w.groupby(v.category_code).sum().sort_values(ascending=False)
    if len(tot) > 1 and abs(tot.iloc[0] - tot.iloc[1]) < 1e-9:
        return "xxx"
    return tot.index[0]


def main():
    cfg = load_config(); root = Path(cfg["iemocap_root"]); tol = cfg["overlap"]["timing_tolerance_sec"]
    Path("results/m2").mkdir(parents=True, exist_ok=True)
    S = {"checks": {}, "deviations": []}

    # M2.1 / M2.2
    urows, vrows, unknown = [], [], Counter()
    for s in range(1, 6):
        for f in list_files(root / f"Session{s}/dialog/EmoEvaluation"):
            if f.parent.name != "EmoEvaluation" or f.suffix != ".txt":
                continue
            for u in parse_summary_text(f.read_text(errors="replace")):
                meta = parse_utt_id(u["utt_id"])
                urows.append({**{k: u[k] for k in ["utt_id"]}, **{k: meta[k] for k in
                              ["session", "dialog", "dialog_type", "mocap_letter", "speaker_letter", "speaker_id"]},
                              **{k: u[k] for k in ["start", "end"]}, "duration": u["end"] - u["start"],
                              **{k: u[k] for k in ["label_official", "val", "act", "dom"]}})
                for rid, cats in u["votes"]:
                    k = len(cats)
                    for c in cats:
                        code = map_category(c)
                        if code is None:
                            unknown[c] += 1
                        vrows.append(dict(utt_id=u["utt_id"], rater_id=rid, rater_type=rater_type(rid),
                                          category_raw=c, category_code=code, weight=1.0 / k))
    utt = pd.DataFrame(urows); votes = pd.DataFrame(vrows)
    S["unknown_category_strings"] = dict(unknown)
    S["raw_category_strings"] = votes.category_raw.value_counts().to_dict()
    S["external_rater_ids_per_session"] = {
        int(s): sorted(votes[(votes.rater_type == "external") & votes.utt_id.isin(g.utt_id)].rater_id.unique())
        for s, g in utt.groupby("session")}
    S["self_rater_ids"] = sorted(votes[votes.rater_type == "self"].rater_id.unique())
    S["n_external_raters_per_utt"] = (votes[votes.rater_type == "external"].groupby("utt_id").rater_id.nunique()
                                      .value_counts().sort_index().to_dict())
    n_no_ext = int((~utt.utt_id.isin(votes[votes.rater_type == "external"].utt_id)).sum())
    S["utts_without_external_votes"] = n_no_ext
    print(f"parsed {len(utt)} utterances, {len(votes)} vote rows; unknown categories: {dict(unknown)}")

    # M2.3 cross-check with per-rater files
    mism, ncmp = [], 0
    vk = defaultdict(set)
    for r in votes.itertuples():
        vk[(r.utt_id, r.rater_id)].add(r.category_code if r.category_code else r.category_raw)
    cat_unknown = Counter()
    seen_pairs = set()
    for s in range(1, 6):
        d = root / f"Session{s}/dialog/EmoEvaluation"
        for sub in ["Categorical", "Self-evaluation"]:
            for f in list_files(d / sub, suffix="_cat.txt"):
                rid = re.search(r"_([efm]\d+)_cat\.txt$", f.name)
                rid = rid.group(1).upper()
                for u, cats in parse_cat_text(f.read_text(errors="replace")).items():
                    codes = set()
                    for c in cats:
                        code = map_category(c)
                        if code is None:
                            cat_unknown[c] += 1   # e.g. 'Overlap' in self-eval files
                        else:
                            codes.add(code)
                    seen_pairs.add((u, rid))
                    ncmp += 1
                    other = {x for x in vk.get((u, rid), set()) if x in set(map_category(k) for k in ["Neutral","Happiness","Excited","Sadness","Anger","Frustration","Surprise","Fear","Disgust","Other"])}
                    if codes != other:
                        mism.append(dict(utt_id=u, rater_id=rid, cat_file=sorted(codes), summary=sorted(other)))
    # (utt, rater) pairs present in summary but absent from per-rater files
    only_summary = [k for k in vk if k not in seen_pairs]
    S["m2_3"] = dict(compared=ncmp, mismatches=len(mism), mismatch_rate=len(mism) / max(ncmp, 1),
                     pairs_only_in_summary=len(only_summary), cat_file_nonCategory_tokens=dict(cat_unknown))
    pd.DataFrame(mism).to_csv("results/m2/cat_file_mismatches.csv", index=False)
    print("M2.3:", S["m2_3"])

    # M2.4 reconstruct official label
    g = {k: d for k, d in votes.groupby("utt_id")}
    empty = votes.iloc[0:0]
    rules = {}
    for rule in "ABC":
        rules[rule] = pd.Series({u: rule_label(g.get(u, empty), rule) for u in utt.utt_id})
    match = {r: float((rules[r].reindex(utt.utt_id).values == utt.label_official.values).mean()) for r in rules}
    S["rule_match_rates"] = match
    best = max(match, key=match.get)
    S["best_rule"] = best
    mm = utt.assign(reconstructed=rules[best].reindex(utt.utt_id).values)
    mm = mm[mm.reconstructed != mm.label_official]
    mm["votes_shown"] = [";".join(f"{r.rater_id}:{r.category_code}" for r in g[u].itertuples()) if u in g else ""
                         for u in mm.utt_id]
    mm[["utt_id", "label_official", "reconstructed", "votes_shown"]].to_csv("results/m2/label_mismatches.csv", index=False)
    S["n_mismatches_best_rule"] = len(mm)
    S["mismatch_confusion_top"] = (mm.groupby(["label_official", "reconstructed"]).size()
                                   .sort_values(ascending=False).head(15).reset_index().values.tolist())
    print("rule match rates:", match, "best:", best, "mismatches:", len(mm))

    # M2.5 4-class
    utt["label_4class"] = utt.label_official.where(utt.label_official != "exc", "hap")
    utt["in_4class"] = utt.label_official.isin(FOUR)
    utt.loc[~utt.in_4class, "label_4class"] = None

    # M2.6 join wav, M2.7 duration check
    hdr = pd.read_parquet("results/m1/wav_headers.parquet")
    hdr = hdr[hdr.folder_type == "sentences/wav"][["utt_id", "path", "duration"]].rename(
        columns={"path": "wav_path", "duration": "wav_duration"})
    utt = utt.merge(hdr, on="utt_id", how="left")
    utt["duration_diff"] = (utt.wav_duration - utt.duration).abs()
    utt["duration_flag"] = utt.duration_diff > tol
    S["n_missing_wav"] = int(utt.wav_path.isna().sum())
    S["n_missing_wav_4class"] = int(utt[utt.in_4class].wav_path.isna().sum())
    S["duration_flags"] = dict(n=int(utt.duration_flag.sum()), n_4class=int(utt[utt.in_4class].duration_flag.sum()),
                               max_diff=float(utt.duration_diff.max()), median_diff=float(utt.duration_diff.median()))
    S["wavs_without_utterance_row"] = int((~hdr.utt_id.isin(utt.utt_id)).sum())
    print("M2.7:", S["duration_flags"])

    # checks
    c = S["checks"]
    c["n_utts_10039"] = len(utt) == 10039
    c["utt_ids_unique"] = bool(utt.utt_id.is_unique)
    c["per_session_counts"] = utt.groupby("session").size().to_dict() == EXP_SESSION
    c["official_label_counts"] = utt.label_official.value_counts().to_dict() == EXP_LABELS
    c["n_4class_5531"] = int(utt.in_4class.sum()) == 5531
    c["10_speakers"] = utt.speaker_id.nunique() == 10
    S["utts_per_speaker"] = utt.groupby("speaker_id").size().to_dict()
    c["every_4class_has_wav"] = S["n_missing_wav_4class"] == 0
    c["best_rule_ge_98pct"] = match[best] >= 0.98
    c["no_unknown_categories"] = not unknown
    S["all_checks_pass"] = all(c.values())

    utt.to_parquet("data/interim/utterances.parquet")
    votes.to_parquet("data/interim/votes.parquet")
    Path("results/m2/summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(c, indent=1)); print("per speaker:", S["utts_per_speaker"])
    print("M2 gate:", "PASS" if S["all_checks_pass"] else "REVIEW")
    if match[best] < 0.95:
        print("🛑 best rule match < 95% — stop and report")


if __name__ == "__main__":
    main()
