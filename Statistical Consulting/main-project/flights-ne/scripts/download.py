"""Download BTS Reporting Carrier On-Time Performance monthly zips.

Skips months already downloaded, retries the (flaky) BTS server with
backoff, and stops cleanly on months not yet published.

    python scripts/download.py --year 2025
    python scripts/download.py --year 2025 --months 1 2 3
"""
import argparse
import time
import zipfile
from pathlib import Path

import requests
import urllib3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def fetch(url: str, dest: Path, verify: bool) -> None:
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True,
                      timeout=120, verify=verify) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise IOError("downloaded file is not a valid zip")
    tmp.rename(dest)


def download_month(year: int, month: int, tries: int = 5) -> bool:
    dest = RAW / f"ontime_{year}_{month:02d}.zip"
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"  {year}-{month:02d}: already have it")
        return True
    url = URL.format(y=year, m=month)
    verify = True
    for attempt in range(1, tries + 1):
        try:
            fetch(url, dest, verify)
            mb = dest.stat().st_size / 1e6
            print(f"  {year}-{month:02d}: ok ({mb:.0f} MB)")
            return True
        except requests.exceptions.SSLError:
            # BTS has had certificate-chain problems; fall back once.
            if verify:
                print("  SSL verification failed; retrying without it")
                urllib3.disable_warnings()
                verify = False
                continue
            raise
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"  {year}-{month:02d}: not published yet")
                return False
            err = e
        except (requests.RequestException, IOError) as e:
            err = e
        wait = 10 * attempt
        print(f"  {year}-{month:02d}: attempt {attempt} failed ({err}); "
              f"retrying in {wait}s")
        time.sleep(wait)
    print(f"  {year}-{month:02d}: FAILED after {tries} tries, rerun later")
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.year} to {RAW}")
    ok = [m for m in a.months if download_month(a.year, m)]
    missing = sorted(set(a.months) - set(ok))
    print(f"Have {len(ok)} of {len(a.months)} months."
          + (f" Missing: {missing}. Rerun to resume." if missing else ""))


if __name__ == "__main__":
    main()
