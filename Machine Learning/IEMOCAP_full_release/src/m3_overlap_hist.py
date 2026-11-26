"""M3.4: how much of each 4-class utterance is overlapped? Histogram + summary (results/m3/overlap_hist.json)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    ov = pd.read_parquet("data/interim/overlap.parquet")
    p0 = pd.read_parquet("data/processed/P0.parquet")[["utt_id", "duration", "label_4class"]]
    p2 = pd.read_parquet("data/processed/P2.parquet")[["utt_id", "duration", "new_duration", "trimmed"]]
    d = p0.merge(ov[["utt_id", "overlap_sec_p010", "overlap_frac_p010", "overlap_intervals_p010"]], on="utt_id")
    tol = 0.05
    def edge_only(iv, dur):   # every overlapped interval touches the clip start or end
        return len(iv) > 0 and all(a <= tol or b >= dur - tol for a, b in iv)
    d["edge_only"] = [edge_only(iv, du) for iv, du in zip(d.overlap_intervals_p010.map(json.loads), d.duration)]
    o = d[d.overlap_sec_p010 > 0]
    q = lambda x: {k: float(np.quantile(x, k)) for k in (.25, .5, .75, .9)}
    S = dict(n=len(d), n_overlapped=len(o), share_overlapped=len(o) / len(d),
             overlap_sec_quantiles=q(o.overlap_sec_p010), overlap_frac_quantiles=q(o.overlap_frac_p010),
             share_overlap_lt_0p5s=float((o.overlap_sec_p010 < 0.5).mean()), share_overlap_lt_1s=float((o.overlap_sec_p010 < 1).mean()),
             share_overlap_ge_2s=float((o.overlap_sec_p010 >= 2).mean()),
             share_edge_only=float(o.edge_only.mean()), share_overlap_frac_ge_half=float((o.overlap_frac_p010 >= .5).mean()),
             p2_n=len(p2), p2_trimmed_share=float(p2.trimmed.mean()),
             p2_retained_frac_median=float((p2.new_duration / p2.duration).median()),
             p2_retained_frac_among_trimmed_median=float((p2.new_duration / p2.duration)[p2.trimmed].median()))
    Path("results/m3").mkdir(exist_ok=True)
    json.dump(S, open("results/m3/overlap_hist.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(o.overlap_sec_p010.clip(upper=8), bins=40, color="#4477aa"); ax[0].set_xlabel("seconds overlapped per utterance (clipped at 8)"); ax[0].set_ylabel("utterances")
    ax[0].axvline(o.overlap_sec_p010.median(), color="k", ls="--", lw=1)
    ax[1].hist(o.overlap_frac_p010, bins=40, color="#cc6677"); ax[1].set_xlabel("fraction of utterance overlapped")
    ax[0].set_title(f"{len(o):,} of {len(d):,} utterances overlapped (padding 0.1 s)", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/m3_overlap_hist.png", dpi=150)
    print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
