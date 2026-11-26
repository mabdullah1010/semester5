"""M3.4: untranscribed partner speech (crosstalk) detector. Usage: python src/m3_crosstalk.py [--force]"""
import json, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config
from src.m3_overlap import merge

SR = 16000; FRAME = 320; DT = 0.02; VAD_CHUNK = 512
_VAD = None


def _vad():
    global _VAD
    if _VAD is None:
        import torch
        torch.set_num_threads(1)
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    return _VAD


def ncc_best(clip, ch):
    c = clip - clip.mean(); n = len(c)
    if len(ch) < n:
        return 0.0, 0
    num = fftconvolve(ch, c[::-1], mode="valid")
    e = np.cumsum(np.r_[0, ch.astype(np.float64) ** 2]); win = e[n:] - e[:-n]
    r = num / (np.sqrt(win * (c ** 2).sum()) + 1e-9)
    i = int(np.argmax(r))
    return float(r[i]), i


def vad_probs(x):
    """Silero speech probability per 512-sample chunk (sequential, stateful)."""
    import torch
    m = _vad(); m.reset_states()
    n = int(np.ceil(len(x) / VAD_CHUNK)); xp = np.zeros(n * VAD_CHUNK, np.float32); xp[:len(x)] = x
    t = torch.from_numpy(xp)
    with torch.no_grad():
        return np.array([m(t[i * VAD_CHUNK:(i + 1) * VAD_CHUNK], SR).item() for i in range(n)], np.float32)


def process_dialog(args):
    dialog, rows, root = args          # rows: list of (utt_id, speaker_letter, start, end, wav_path)
    session = int(dialog[3:5])
    d, sr = sf.read(f"{root}/Session{session}/dialog/wav/{dialog}.wav", dtype="float32")
    assert sr == SR and d.ndim == 2
    # --- M3.4a channel mapping
    votes = {"F": [], "M": []}
    for sp in "FM":
        cand = [r for r in rows if r[1] == sp and r[3] - r[2] >= 1.0]
        pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(5, len(cand))).round().astype(int))] if cand else []
        for utt_id, _, st, en, wp in pick:
            clip, _ = sf.read(f"{root}/{wp}", dtype="float32")
            a = max(int((st - 0.5) * SR), 0); b = int((en + 0.5) * SR)
            res = [ncc_best(clip, d[a:b, ci]) for ci in (0, 1)]
            best = int(res[1][0] > res[0][0])
            votes[sp].append(dict(channel=best, ncc=res[best][0], ncc_other=res[1 - best][0],
                                  lag_sec=((a + res[best][1]) / SR) - st))
    cmap, mrows = {}, []
    ok = True
    for sp in "FM":
        v = votes[sp]
        ch = [x["channel"] for x in v]
        agree = len(set(ch)) == 1 and len(ch) > 0
        mc = int(np.bincount(ch).argmax()) if ch else -1
        cmap[sp] = mc
        sp_ok = agree and np.mean([x["ncc"] for x in v]) >= 0.9
        ok &= bool(sp_ok)
        mrows.append(dict(dialog=dialog, speaker=sp, channel=mc, n_votes=len(v), agree=bool(agree),
                          ncc_mean=float(np.mean([x["ncc"] for x in v])) if v else np.nan,
                          ncc_other_mean=float(np.mean([x["ncc_other"] for x in v])) if v else np.nan,
                          lag_median_ms=float(np.median([x["lag_sec"] for x in v]) * 1000) if v else np.nan))
    ok &= cmap["F"] != cmap["M"]
    for m in mrows:
        m["channel_map_ok"] = bool(ok)
    frames = []
    if ok:
        for utt_id, sp, st, en, wp in rows:
            S_ch, P_ch = d[:, cmap[sp]], d[:, cmap["M" if sp == "F" else "F"]]
            a = int(round(st * SR)); n = int((en - st) / DT)
            if n < 1:
                continue
            seg_s = S_ch[a:a + n * FRAME]; seg_p = P_ch[a:a + n * FRAME]
            if len(seg_s) < n * FRAME:
                n = len(seg_s) // FRAME; seg_s, seg_p = seg_s[:n * FRAME], seg_p[:n * FRAME]
                if n < 1:
                    continue
            es = 10 * np.log10((seg_s.reshape(n, FRAME) ** 2).mean(1) + 1e-10)
            ep = 10 * np.log10((seg_p.reshape(n, FRAME) ** 2).mean(1) + 1e-10)
            pv = vad_probs(seg_p)
            centers = ((np.arange(n) + 0.5) * FRAME).astype(int) // VAD_CHUNK
            frames.append(pd.DataFrame(dict(utt_id=utt_id, frame=np.arange(n, dtype=np.int32),
                                            t_dialog=(st + (np.arange(n) + 0.5) * DT).astype(np.float32),
                                            e_s=es.astype(np.float32), e_p=ep.astype(np.float32),
                                            vad_p=pv[np.minimum(centers, len(pv) - 1)])))
    return mrows, (pd.concat(frames) if frames else None)


def covered(t, iv):
    """t: array of times; iv: merged sorted [[s,e],...]. Boolean mask of t inside any interval."""
    if not iv:
        return np.zeros(len(t), bool)
    s = np.array([x[0] for x in iv]); e = np.array([x[1] for x in iv])
    i = np.searchsorted(s, t, "right") - 1
    return (i >= 0) & (t <= e[np.maximum(i, 0)])


def main(force=False):
    cfg = load_config(); root = cfg["iemocap_root"]; pad = cfg["overlap"]["padding_sec"]
    out_frames = Path("results/m3/crosstalk_frames.parquet"); out_map = Path("results/m3/channel_map.csv")
    utt = pd.read_parquet("data/interim/utterances.parquet"); seg = pd.read_parquet("data/interim/segments.parquet")
    t0 = time.time()
    if force or not out_frames.exists() or not out_map.exists():
        jobs = [(d, list(zip(g.utt_id, g.speaker_letter, g.start, g.end, g.wav_path)), root) for d, g in utt.groupby("dialog")]
        maps, frames = [], []
        with ProcessPoolExecutor(8) as ex:
            for i, (m, f) in enumerate(ex.map(process_dialog, jobs)):
                maps += m
                if f is not None:
                    frames.append(f)
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(jobs)} dialogs, {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(maps).to_csv(out_map, index=False)
        pd.concat(frames).reset_index(drop=True).to_parquet(out_frames)
    cm = pd.read_csv(out_map); fr = pd.read_parquet(out_frames)
    S = {"channel_map": dict(
        n_dialogs=int(cm.dialog.nunique()), n_dialogs_ok=int(cm.groupby("dialog").channel_map_ok.first().sum()),
        dialogs_not_ok=sorted(cm[~cm.channel_map_ok].dialog.unique()),
        ncc_mean_min=float(cm.ncc_mean.min()), ncc_other_mean_max=float(cm.ncc_other_mean.max()),
        lag_median_ms_abs_max=float(cm.lag_median_ms.abs().max()),
        dialogs_with_lag_gt_20ms=sorted(cm[cm.lag_median_ms.abs() > 20].dialog.unique()))}
    print("channel map:", S["channel_map"])

    # ground truth frames from transcripts
    fr = fr.merge(utt[["utt_id", "dialog", "speaker_letter"]], on="utt_id")
    fr["gt_overlap0"] = False; fr["gt_far"] = False; fr["cov_pad"] = False
    bydlg = {d: g for d, g in seg.groupby("dialog")}
    for (d, sp), idx in fr.groupby(["dialog", "speaker_letter"]).groups.items():
        g = bydlg[d]; p = g[g.speaker_letter != sp]
        t = fr.loc[idx, "t_dialog"].values.astype(np.float64)
        pairs = list(zip(p.start, p.end))
        fr.loc[idx, "gt_overlap0"] = covered(t, merge(pairs))
        fr.loc[idx, "gt_far"] = ~covered(t, merge([(s - 1.0, e + 1.0) for s, e in pairs]))
        fr.loc[idx, "cov_pad"] = covered(t, merge([(s - pad, e + pad) for s, e in pairs]))
    fr["de"] = fr.e_p - fr.e_s
    pos = fr[fr.gt_overlap0]; neg = fr[fr.gt_far]
    # M3.4d calibrate tau
    sweep = []
    for tau in np.arange(0, 15.01, 0.5):
        rule = lambda x: (x.vad_p > 0.5) & (x.de > tau)
        tp = rule(pos).sum(); fp = rule(neg).sum()
        recall = tp / max(len(pos), 1); fpr = fp / max(len(neg), 1); prec = tp / max(tp + fp, 1)
        sweep.append(dict(tau=float(tau), recall=float(recall), fpr=float(fpr), precision=float(prec)))
    ok = [s for s in sweep if s["fpr"] <= 0.02]
    chosen = ok[0] if ok else sweep[-1]
    gate_pass = chosen["recall"] >= 0.5 and bool(ok)
    S["calibration"] = dict(n_pos_frames=int(len(pos)), n_neg_frames=int(len(neg)), chosen=chosen,
                            fpr_target=0.02, recall_gate=0.5, detector_kept=bool(gate_pass), sweep=sweep)
    print("calibration: pos %d neg %d chosen %s -> %s" % (len(pos), len(neg), chosen, "KEPT" if gate_pass else "DROPPED"))
    # M3.4e apply (excluding frames already covered by transcript overlap at padding_sec)
    tau = chosen["tau"]
    fr["active"] = (fr.vad_p > 0.5) & (fr.de > tau) & ~fr.cov_pad
    rec = []
    for u, g in fr.groupby("utt_id"):
        act = g.sort_values("frame")
        fidx = act.frame.values[act.active.values]
        iv = merge([(f * DT, (f + 1) * DT) for f in fidx])
        rec.append(dict(utt_id=u, crosstalk_sec=float(len(fidx) * DT),
                        crosstalk_intervals=json.dumps([[round(a, 4), round(b, 4)] for a, b in iv])))
    ct = pd.DataFrame(rec)
    ct = utt[["utt_id"]].merge(ct, on="utt_id", how="left")
    ct["detector_ran"] = ct.crosstalk_sec.notna()
    ct["crosstalk_sec"] = ct.crosstalk_sec.fillna(0.0); ct["crosstalk_intervals"] = ct.crosstalk_intervals.fillna("[]")
    ct["detector_kept"] = bool(gate_pass)
    ct.to_parquet("data/interim/crosstalk.parquet")
    u4 = utt[["utt_id", "in_4class", "label_4class"]].merge(ct, on="utt_id")
    S["crosstalk_summary"] = dict(
        n_utts_detector_ran=int(ct.detector_ran.sum()),
        frac_utts_crosstalk_gt_max=float((u4.crosstalk_sec > cfg["overlap"]["crosstalk_max_detected_sec"]).mean()),
        frac_4class_crosstalk_gt_max=float((u4[u4.in_4class].crosstalk_sec > cfg["overlap"]["crosstalk_max_detected_sec"]).mean()),
        by_class_gt_max=(u4[u4.in_4class].assign(x=lambda d: d.crosstalk_sec > cfg["overlap"]["crosstalk_max_detected_sec"])
                         .groupby("label_4class").x.mean().to_dict()),
        mean_crosstalk_sec=float(u4.crosstalk_sec.mean()))
    S["seconds"] = round(time.time() - t0, 1)
    Path("results/m3/crosstalk_calibration.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S["crosstalk_summary"], indent=1))


if __name__ == "__main__":
    main("--force" in sys.argv)
