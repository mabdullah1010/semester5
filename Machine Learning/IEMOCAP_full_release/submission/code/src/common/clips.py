"""Which audio clips need features. P2 clips that were not trimmed are identical to their P0 clip,
so only trimmed P2 clips are extracted separately; the rest reuse the P0 row."""
import pandas as pd


def clip_table():
    p0 = pd.read_parquet("data/processed/P0.parquet")
    p2 = pd.read_parquet("data/processed/P2.parquet")
    a = pd.DataFrame(dict(key="P0:" + p0.utt_id, utt_id=p0.utt_id, policy="P0", path=p0.audio_path))
    t = p2[p2.trimmed]
    b = pd.DataFrame(dict(key="P2:" + t.utt_id, utt_id=t.utt_id, policy="P2", path=t.audio_path))
    return pd.concat([a, b], ignore_index=True)


def policy_index(policy):
    """Row keys (in P0/P2 order) to assemble a policy's matrix from the extracted clips."""
    if policy == "P0":
        return ["P0:" + u for u in pd.read_parquet("data/processed/P0.parquet").utt_id]
    p2 = pd.read_parquet("data/processed/P2.parquet")
    return [("P2:" if t else "P0:") + u for u, t in zip(p2.utt_id, p2.trimmed)]
