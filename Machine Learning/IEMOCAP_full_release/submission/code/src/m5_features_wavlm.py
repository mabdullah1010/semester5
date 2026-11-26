"""M5.2: WavLM-base-plus mean-pooled hidden states for all 13 layers -> (N, 13, 768).
Resumable: results are cached in features/wavlm_chunks/ every CHUNK clips."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.clips import clip_table, policy_index

MODEL = "microsoft/wavlm-base-plus"; CHUNK = 250; MAX_SAMPLES = 20 * 16000
_M = None


def load():
    global _M
    if _M is None:
        from transformers import AutoFeatureExtractor, WavLMModel
        torch.set_num_threads(8)
        _M = (AutoFeatureExtractor.from_pretrained(MODEL), WavLMModel.from_pretrained(MODEL).eval())
    return _M


@torch.no_grad()
def embed(path):
    """Mean-pooled hidden states per layer. Clips > 20 s are split into equal chunks of <= 20 s
    and averaged weighted by chunk length."""
    fe, model = load()
    x, sr = sf.read(path, dtype="float32"); assert sr == 16000 and x.ndim == 1
    n = int(np.ceil(len(x) / MAX_SAMPLES)); parts = np.array_split(x, n)
    acc = np.zeros((13, 768), np.float64)
    for p in parts:
        inp = fe(p, sampling_rate=16000, return_tensors="pt")
        hs = model(inp.input_values, output_hidden_states=True).hidden_states
        acc += np.stack([h[0].mean(0).numpy() for h in hs]).astype(np.float64) * len(p)
    return (acc / len(x)).astype(np.float32)


def main(force=False):
    ct = clip_table(); cdir = Path("features/wavlm_chunks"); cdir.mkdir(parents=True, exist_ok=True)
    out = Path("features/wavlm_clips.npy"); t0 = time.time()
    if force or not out.exists():
        for ci, s in enumerate(range(0, len(ct), CHUNK)):
            f = cdir / f"chunk_{ci:04d}.npy"
            if f.exists() and not force:
                continue
            sub = ct.iloc[s:s + CHUNK]
            arr = np.stack([embed(p) for p in sub.path]); np.save(f, arr)
            print(f"chunk {ci} done ({s + len(sub)}/{len(ct)}) {time.time() - t0:.0f}s", flush=True)
        np.save(out, np.concatenate([np.load(cdir / f"chunk_{i:04d}.npy") for i in range(int(np.ceil(len(ct) / CHUNK)))]))
    clips = np.load(out); assert len(clips) == len(ct)
    pos = {k: i for i, k in enumerate(ct.key)}
    info = {"seconds_last_run": round(time.time() - t0, 1)}
    for pol in ["P0", "P2"]:
        idx = policy_index(pol)
        m = clips[[pos[k] for k in idx]]
        np.save(f"features/wavlm_{pol}.npy", m)
        pd.DataFrame({"utt_id": [k.split(":")[1] for k in idx]}).to_csv(f"features/wavlm_{pol}_index.csv", index=False)
        info[pol] = dict(shape=list(m.shape), nan=int(np.isnan(m).sum()), inf=int(np.isinf(m).sum()))
        print(pol, m.shape, info[pol])
    Path("results/m5").mkdir(exist_ok=True)
    Path("results/m5/wavlm_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main("--force" in sys.argv)
