"""M5.4/M5.5 checks: shapes, NaNs, reproducibility, class sanity, near-duplicate search."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.clips import clip_table
from src.m5_features_egemaps import extract
from src.m5_features_wavlm import embed


def cos_pairs(X, thr, ids, same_session_only=False):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = []
    for s in range(0, len(X), 1000):
        S = X[s:s + 1000] @ X.T
        for i in range(S.shape[0]):
            S[i, :s + i + 1] = -1                      # upper triangle only
        r, c = np.where(S > thr)
        out += [(ids[s + a], ids[b], float(S[a, b])) for a, b in zip(r, c)]
    return out


def main():
    S = {"checks": {}}
    lab = pd.read_parquet("data/processed/labels.parquet").set_index("utt_id")
    P = {p: pd.read_parquet(f"data/processed/{p}.parquet") for p in ["P0", "P2"]}
    eg = {p: pd.read_parquet(f"features/egemaps_{p}.parquet") for p in P}
    wl = {p: np.load(f"features/wavlm_{p}.npy") for p in P}
    for p in P:
        S[p] = dict(n=len(P[p]), egemaps_shape=list(eg[p].shape), wavlm_shape=list(wl[p].shape),
                    egemaps_nan=int(eg[p].isna().sum().sum()), wavlm_nan=int(np.isnan(wl[p]).sum()),
                    egemaps_constant_cols=int((eg[p].std() == 0).sum()),
                    index_match=bool((eg[p].index == P[p].utt_id.values).all()))
        S["checks"][f"{p}_shapes"] = eg[p].shape == (len(P[p]), 88) and wl[p].shape == (len(P[p]), 13, 768)
        S["checks"][f"{p}_no_nan"] = S[p]["egemaps_nan"] == 0 and S[p]["wavlm_nan"] == 0
        S["checks"][f"{p}_index_match"] = S[p]["index_match"]

    # reproducibility on 20 random clips
    ct = clip_table(); rng = np.random.default_rng(1337); pick = rng.choice(len(ct), 20, replace=False)
    clips_eg = pd.read_parquet("features/egemaps_clips.parquet"); clips_wl = np.load("features/wavlm_clips.npy")
    d_eg = max(float(np.abs(extract(ct.path[i]).values - clips_eg.iloc[i].values).max()) for i in pick)
    d_wl = max(float(np.abs(embed(ct.path[i]) - clips_wl[i]).max()) for i in pick)
    S["reextract_max_abs_diff"] = dict(egemaps=d_eg, wavlm=d_wl)
    S["checks"]["egemaps_reproducible"] = d_eg == 0.0
    S["checks"]["wavlm_reproducible_1e-4"] = d_wl < 1e-4

    # light class sanity (log, don't assert)
    m = eg["P2"].join(lab.y_hard)
    cols = ["loudness_sma3_amean", "F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"]
    S["class_means_P2_raw"] = {c: m.groupby("y_hard")[c].mean().round(3).to_dict() for c in cols if c in m}
    cm = S["class_means_P2_raw"]["loudness_sma3_amean"]
    S["loudness_ang_gt_sad"] = bool(cm["ang"] > cm["sad"])

    # near duplicates (P0 = superset of all 4-class clips)
    ids = list(P["P0"].utt_id)
    Z = (eg["P0"] - eg["P0"].mean()) / eg["P0"].std()
    W = wl["P0"][:, 12, :]                                # last layer
    raw = cos_pairs(W, 0.99, ids); cen = cos_pairs(W - W.mean(0), 0.99, ids); ez = cos_pairs(Z.values, 0.99, ids)
    S["near_dup"] = dict(wavlm_last_layer_raw_gt_0p99=len(raw), wavlm_last_layer_centered_gt_0p99=len(cen), egemaps_z_gt_0p99=len(ez),
                         examples_centered=cen[:10], examples_egemaps=ez[:10])
    special = lambda a, b: any(k in a or k in b for k in ["impro05a", "impro05b", "impro08a", "impro08b", "script01_1b"])
    S["near_dup"]["special_dialog_pairs_flagged"] = [x for x in cen + ez if special(x[0], x[1])][:20]
    # direct audio check for special dialogs: sentence-id pairs with identical timing across impro05a/b etc. is covered by M1 md5 (0 dups)
    pd.DataFrame(cen, columns=["a", "b", "cos_wavlm_centered"]).to_csv("results/m5/near_duplicates_wavlm.csv", index=False)
    S["all_checks_pass"] = all(S["checks"].values())
    Path("results/m5/summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps({k: S[k] for k in ["checks", "reextract_max_abs_diff", "class_means_P2_raw", "near_dup"]}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
