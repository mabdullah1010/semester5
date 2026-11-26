"""M14: assemble submission/ (code, data notes, results, predictions, write-ups).
The IEMOCAP audio is licensed and not copied; large feature caches are regenerable."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"

def cp(src, dst, ignore=None):
    s, d = ROOT / src, OUT / dst
    if not s.exists():
        return
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=True, ignore=ignore)
    else:
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, d)

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "log_*.txt", "*_log.txt", "tmp.parquet",
                                 "partial", "*stdout.txt", "crosstalk_frames.parquet")
    cp("src", "code/src", ign); cp("tests", "code/tests", ign)
    for f in ("config.yaml", "requirements.txt", "PROGRESS.md", "IEMOCAP_PROJECT_PLAN.md"):
        cp(f, f"code/{f}")
    for f in (ROOT / "data/processed").glob("*.parquet"):   # tables only, no audio
        cp(f"data/processed/{f.name}", f"data/processed/{f.name}")
    cp("data/interim/utterances.parquet", "data/interim/utterances.parquet")
    cp("data/interim/votes.parquet", "data/interim/votes.parquet")
    # results: summaries, tables, validation outputs, test-set predictions
    cp("results", "results", ign)
    cp("figures", "figures")
    cp("latex", "writeup/full_report_latex"); cp("latex_short", "writeup/short_report_latex")
    cp("report/report.md", "writeup/report.md"); cp("report/report.html", "writeup/report.html")
    cp("report/tables", "writeup/tables")
    shutil.copy(ROOT / "src/submission_README.md", OUT / "README.md")
    print("submission size MB:", sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) // 2**20)

if __name__ == "__main__":
    main()
