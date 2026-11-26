"""M5.1: eGeMAPSv02 functionals (88-d) per clip."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.clips import clip_table, policy_index

_S = None


def extract(path):
    global _S
    import opensmile
    if _S is None:
        _S = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    x, sr = sf.read(path, dtype="float32")
    return _S.process_signal(x, sr).iloc[0]


def main(force=False):
    out = Path("features/egemaps_clips.parquet")
    ct = clip_table()
    t0 = time.time()
    if force or not out.exists():
        with ProcessPoolExecutor(8) as ex:
            rows = list(ex.map(extract, ct.path, chunksize=32))
        df = pd.DataFrame(rows); df.index = ct.key.values
        df.to_parquet(out)
        print(f"extracted {len(df)} clips in {time.time() - t0:.0f}s")
    clips = pd.read_parquet(out)
    for pol in ["P0", "P2"]:
        idx = policy_index(pol)
        m = clips.loc[idx].copy(); m.index = [k.split(":")[1] for k in idx]; m.index.name = "utt_id"
        m.to_parquet(f"features/egemaps_{pol}.parquet")
        print(pol, m.shape, "NaN:", int(m.isna().sum().sum()), "inf:", int(np.isinf(m.values).sum()),
              "constant cols:", int((m.std() == 0).sum()))


if __name__ == "__main__":
    main("--force" in sys.argv)
