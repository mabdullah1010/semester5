"""M1: file inventory and integrity. Usage: python src/m1_inventory.py [--force]"""
import hashlib, json, re, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.io import list_files, load_config

DIALOG_RE = re.compile(r"^(Ses\d\d[FM]_(?:impro\d+[ab]?|script\d+_\d+b?))")
UTT_RE = re.compile(r"^(Ses\d\d[FM]_.+)_([FM]\d{3})$")
EXPECTED_DIALOGS = {1: 28, 2: 30, 3: 32, 4: 30, 5: 31}
EXPECTED_UTTS = 10039


def classify(root: Path, p: Path):
    parts = p.relative_to(root).parts
    session = int(re.match(r"Session(\d)", parts[0]).group(1))
    m = DIALOG_RE.match(p.stem)
    dialog = m.group(1) if m else None
    utt = None
    if parts[1] == "sentences":
        ftype = "/".join(parts[1:3])               # e.g. sentences/wav
        if len(parts) > 3 and DIALOG_RE.match(parts[3]):
            dialog = parts[3]
        if UTT_RE.match(p.stem):
            utt = p.stem
    else:
        ftype = "/".join(parts[1:-1])              # e.g. dialog/wav, dialog/lab/Ses01_F
    return dict(path=str(p.relative_to(root)), session=session, folder_type=ftype,
                dialog=dialog, utt_id=utt, extension=p.suffix.lower(), size_bytes=p.stat().st_size)


def header(path):
    try:
        i = sf.info(path)
        return dict(sample_rate=i.samplerate, channels=i.channels, frames=i.frames,
                    duration=i.duration, subtype=i.subtype, error=None)
    except Exception as e:
        return dict(sample_rate=None, channels=None, frames=None, duration=None, subtype=None, error=repr(e))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(force=False):
    cfg = load_config(); root = Path(cfg["iemocap_root"])
    out_inv = Path("data/interim/inventory.parquet")
    out_hdr = Path("results/m1/wav_headers.parquet")
    Path("results/m1").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {"checks": {}, "deviations": []}

    # M1.1 inventory
    junk = []
    files = []
    for s in range(1, 6):
        files += list_files(root / f"Session{s}", count_junk=junk)
    if force or not out_inv.exists():
        inv = pd.DataFrame([classify(root, p) for p in files])
        inv.to_parquet(out_inv)
    inv = pd.read_parquet(out_inv)
    summary["n_files"] = len(inv); summary["n_junk_skipped"] = len(junk)
    summary["junk_by_kind"] = {"._": sum(Path(j).name.startswith("._") for j in junk),
                               ".pk": sum(j.endswith(".pk") for j in junk),
                               "other": sum(not (Path(j).name.startswith("._") or j.endswith(".pk")) for j in junk)}
    summary["files_by_type"] = inv.groupby("folder_type").size().to_dict()
    print(f"inventory: {len(inv)} files, {len(junk)} junk skipped {summary['junk_by_kind']}")

    # M1.2 dialogs per session
    dw = inv[inv.folder_type == "dialog/wav"]
    dpc = dw.groupby("session").dialog.nunique().to_dict()
    summary["dialogs_per_session"] = dpc; summary["n_dialogs"] = int(dw.dialog.nunique())
    ok = dpc == EXPECTED_DIALOGS and summary["n_dialogs"] == 151
    summary["checks"]["dialog_counts"] = ok
    if not ok: summary["deviations"].append(f"dialog counts {dpc} != {EXPECTED_DIALOGS}")

    # M1.3 per-dialog file completeness
    def have(ft, ext):
        return set(inv[(inv.folder_type == ft) & (inv.extension == ext)].dialog.dropna())
    dialogs = set(dw.dialog)
    req = {"dialog_wav": have("dialog/wav", ".wav"),
           "emoeval": have("dialog/EmoEvaluation", ".txt"),
           "transcription": have("dialog/transcriptions", ".txt")}
    for sp in "FM":
        sub = inv[inv.folder_type.str.match(rf"dialog/lab/Ses0\d_{sp}$") & (inv.extension == ".lab")]
        req[f"lab_{sp}"] = set(sub.dialog.dropna())
    missing = {k: sorted(dialogs - v) for k, v in req.items()}
    missing = {k: v for k, v in missing.items() if v}
    summary["missing_per_dialog_files"] = missing
    summary["checks"]["per_dialog_files_complete"] = not missing
    if missing: summary["deviations"].append(f"missing dialog files: {missing}")

    # M1.4 wav headers
    if force or not out_hdr.exists():
        wavs = inv[(inv.extension == ".wav") & inv.folder_type.isin(["sentences/wav", "dialog/wav"])]
        paths = [str(root / p) for p in wavs.path]
        with ProcessPoolExecutor() as ex:
            hdrs = list(ex.map(header, paths, chunksize=64))
        h = pd.DataFrame(hdrs); h.insert(0, "path", wavs.path.values)
        h["folder_type"] = wavs.folder_type.values; h["session"] = wavs.session.values
        h["dialog"] = wavs.dialog.values; h["utt_id"] = wavs.utt_id.values
        h.to_parquet(out_hdr)
    h = pd.read_parquet(out_hdr)
    sw = h[h.folder_type == "sentences/wav"]; dwh = h[h.folder_type == "dialog/wav"]
    summary["n_sentence_wavs"] = len(sw); summary["n_dialog_wavs"] = len(dwh)
    summary["unreadable"] = h[h.error.notna()].path.tolist()
    summary["sentence_wav_formats"] = sw.groupby(["sample_rate", "channels", "subtype"]).size().reset_index().astype(str).values.tolist()
    summary["dialog_wav_formats"] = dwh.groupby(["sample_rate", "channels", "subtype"]).size().reset_index().astype(str).values.tolist()
    summary["checks"]["sentence_wavs_16k_mono_16bit"] = bool(((sw.sample_rate == 16000) & (sw.channels == 1) & (sw.subtype == "PCM_16")).all())
    summary["checks"]["dialog_wavs_16k_stereo"] = bool(((dwh.sample_rate == 16000) & (dwh.channels == 2)).all())
    summary["checks"]["no_unreadable_files"] = not summary["unreadable"]
    summary["sentence_wavs_per_session"] = sw.groupby("session").size().to_dict()
    summary["total_sentence_hours"] = float(sw.duration.sum() / 3600)
    summary["checks"]["n_sentence_wavs_close_to_10039"] = abs(len(sw) - EXPECTED_UTTS) <= 50
    print(f"headers: {len(sw)} sentence wavs, {len(dwh)} dialog wavs, unreadable={len(summary['unreadable'])}")

    # M1.5 MD5 of sentence wavs
    out_hash = Path("results/m1/wav_md5.parquet")
    if force or not out_hash.exists():
        paths = [str(root / p) for p in sw.path]
        with ProcessPoolExecutor() as ex:
            hs = list(ex.map(md5, paths, chunksize=32))
        pd.DataFrame({"path": sw.path.values, "utt_id": sw.utt_id.values, "md5": hs}).to_parquet(out_hash)
    hh = pd.read_parquet(out_hash)
    dup = hh[hh.duplicated("md5", keep=False)].sort_values("md5")
    summary["n_duplicate_hash_groups"] = int(dup.md5.nunique())
    summary["duplicate_pairs"] = dup.groupby("md5").utt_id.apply(list).tolist()
    summary["checks"]["zero_exact_duplicates"] = dup.empty
    if not dup.empty: summary["deviations"].append(f"exact duplicate groups: {summary['duplicate_pairs']}")

    summary["quirks"] = {
        "ses1_impro_dialogs": sorted(d for d in dialogs if d.startswith("Ses01") and "impro" in d),
        "ses03M_split_dialogs": sorted(d for d in dialogs if re.search(r"impro0[58][ab]$", d)),
        "ses05M_script01_1b": "Ses05M_script01_1b" in dialogs}
    summary["all_checks_pass"] = all(summary["checks"].values())
    summary["seconds"] = round(time.time() - t0, 1)
    Path("results/m1/summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["checks", "deviations", "quirks"]}, indent=1, default=str))
    print("M1 gate:", "PASS" if summary["all_checks_pass"] else "REVIEW DEVIATIONS")


if __name__ == "__main__":
    main("--force" in sys.argv)
