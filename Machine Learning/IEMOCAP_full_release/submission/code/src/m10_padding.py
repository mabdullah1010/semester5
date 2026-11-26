"""M10.2: rebuild the primary policy (P2 trimmed) at other overlap paddings and extract features for the new trimmed clips.
Untouched clips reuse their P0 features. Usage: python src/m10_padding.py 0.0 0.25"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import load_config
from src.m3_policies import longest_piece
from src.m5_features_egemaps import extract as eg_extract

SR = 16000


def build(pad):
    cfg = load_config(); root = Path(cfg["iemocap_root"]); tag = f"P2_pad{int(round(pad * 100)):03d}"; ptag = f"{int(round(pad * 100)):03d}"
    P0 = pd.read_parquet("data/processed/P0.parquet"); ov = pd.read_parquet("data/interim/overlap.parquet")[["utt_id", f"overlap_intervals_p{ptag}"]]
    d = P0.merge(ov, on="utt_id"); rows = []
    for r in d.itertuples(index=False):
        iv = [tuple(x) for x in json.loads(getattr(r, f"overlap_intervals_p{ptag}"))]
        lp = longest_piece(iv, r.duration, cfg["overlap"]["trim_min_remaining_sec"]) if iv else (0.0, r.duration)
        rows.append(dict(trim_start=lp[0] if lp else np.nan, trim_end=lp[1] if lp else np.nan, keep=lp is not None, trimmed=bool(iv)))
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1); d = d[d.keep].copy()
    d["new_duration"] = d.trim_end - d.trim_start
    out = Path(f"data/processed/{tag}_audio"); out.mkdir(parents=True, exist_ok=True); paths = []
    for r in d.itertuples():
        if r.trimmed:
            x, sr = sf.read(root / r.audio_path.replace(str(root) + "/", ""), dtype="int16") if False else sf.read(r.audio_path, dtype="int16")
            p = out / f"{r.utt_id}.wav"; sf.write(p, x[int(round(r.trim_start * SR)):int(round(r.trim_end * SR))], SR, subtype="PCM_16"); paths.append(str(p))
        else:
            paths.append(r.audio_path)
    d["audio_path"] = paths
    d.drop(columns=[c for c in d.columns if c.startswith("overlap_intervals")]).to_parquet(f"data/processed/{tag}.parquet")
    print(tag, "kept", len(d), "trimmed", int(d.trimmed.sum()), d.label_4class.value_counts().to_dict(), flush=True)
    return tag, d


def features(tag, d):
    t = d[d.trimmed]; P0 = pd.read_parquet("data/processed/P0.parquet")
    # eGeMAPS
    f_eg = Path(f"features/egemaps_{tag}.parquet")
    if not f_eg.exists():
        with ProcessPoolExecutor(8) as ex:
            rows = list(ex.map(eg_extract, t.audio_path, chunksize=32))
        new = pd.DataFrame(rows); new.index = t.utt_id.values
        base = pd.read_parquet("features/egemaps_P0.parquet")
        m = pd.concat([base.loc[d[~d.trimmed].utt_id], new]).loc[d.utt_id]; m.index.name = "utt_id"; m.to_parquet(f_eg)
        print(tag, "egemaps", m.shape, flush=True)
    # WavLM (chunk-cached so an interrupted run resumes)
    f_wl = Path(f"features/wavlm_{tag}.npy")
    if not f_wl.exists() and not __import__("os").environ.get("M10_SKIP_WAVLM"):
        import torch
        from src.m5_features_wavlm import embed, load
        load(); torch.set_num_threads(int(__import__("os").environ.get("M10_THREADS", "4")))
        cdir = Path(f"features/_m10_{tag}"); cdir.mkdir(exist_ok=True); CH = 250; parts = []
        for ci, s in enumerate(range(0, len(t), CH)):
            f = cdir / f"chunk_{ci:04d}.npy"
            if not f.exists():
                np.save(f, np.stack([embed(p) for p in t.audio_path.iloc[s:s + CH]])); print(tag, "wavlm chunk", ci, f"{s + CH}/{len(t)}", flush=True)
            parts.append(np.load(f))
        new = np.concatenate(parts); base = np.load("features/wavlm_P0.npy"); pos = {u: i for i, u in enumerate(P0.utt_id)}
        npos = {u: i for i, u in enumerate(t.utt_id)}
        m = np.stack([new[npos[u]] if u in npos else base[pos[u]] for u in d.utt_id])
        np.save(f_wl, m); pd.DataFrame({"utt_id": d.utt_id.values}).to_csv(f"features/wavlm_{tag}_index.csv", index=False)
        print(tag, "wavlm", m.shape, flush=True)


if __name__ == "__main__":
    for pad in [float(x) for x in sys.argv[1:]]:
        tag, d = build(pad); features(tag, d)
