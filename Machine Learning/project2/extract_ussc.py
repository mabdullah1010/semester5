"""
One-time step: stream the full USSC FY2018 CSV line by line and save only the
columns the project needs to ussc2018_slim.csv. main.py then reads the slim file.

Usage:
    python3 extract_ussc.py /path/to/opafy18nid.csv
"""

import csv
import sys

KEEP = ["USSCIDN", "OFFGUIDE", "DISTRICT", "MONRACE", "HISPORIG",
        "NEWRACE", "MONSEX", "SENTMON"]
OUT = "ussc2018_slim.csv"

csv.field_size_limit(sys.maxsize)
src = sys.argv[1]

with open(src, newline="", encoding="latin-1") as fin, \
     open(OUT, "w", newline="") as fout:
    reader = csv.reader(fin)
    header = [h.strip().upper() for h in next(reader)]
    missing = [c for c in KEEP if c not in header]
    if missing:
        sys.exit(f"Columns not found in header: {missing}")
    idx = [header.index(c) for c in KEEP]

    writer = csv.writer(fout)
    writer.writerow(KEEP)
    n = 0
    for row in reader:
        writer.writerow([row[i].strip() if i < len(row) else "" for i in idx])
        n += 1
        if n % 10000 == 0:
            print(f"  {n} rows...", flush=True)

print(f"Done: {n} rows written to {OUT}")