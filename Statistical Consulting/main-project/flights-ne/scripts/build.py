"""Filter 2025 flights to the Northeast and save one clean parquet file.

Northeast = U.S. Census Northeast region (CT, ME, MA, NH, RI, VT, NJ, NY, PA).
A flight is kept if it departs from OR arrives at a Northeast airport;
flags ne_origin / ne_dest say which.

Output: data/processed/flights_2025_ne.parquet
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
NORTHEAST = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}

KEEP = [
    # when / who
    "FlightDate", "Month", "DayOfWeek",
    "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    # where
    "Origin", "OriginCityName", "OriginState",
    "Dest", "DestCityName", "DestState", "Distance",
    # schedule vs actual
    "CRSDepTime", "DepTime", "DepDelay", "DepDel15", "DepTimeBlk",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15", "ArrTimeBlk",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime",
    # disruptions
    "Cancelled", "CancellationCode", "Diverted",
    # delay causes (only filled when ArrDelay >= 15)
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]
CATEGORICAL = ["Reporting_Airline", "Origin", "OriginCityName", "OriginState",
               "Dest", "DestCityName", "DestState", "DepTimeBlk", "ArrTimeBlk",
               "CancellationCode"]


def read_month(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        csv = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv) as f:
            header = pd.read_csv(f, nrows=0).columns
        # match column names case-insensitively, in case BTS changes casing
        lookup = {c.lower(): c for c in header}
        present = {k: lookup[k.lower()] for k in KEEP if k.lower() in lookup}
        missing = [k for k in KEEP if k not in present]
        if missing:
            print(f"  warning: {zpath.name} lacks columns {missing}")
        with z.open(csv) as f:
            df = pd.read_csv(f, usecols=list(present.values()),
                             dtype={present.get("Tail_Number", "_"): "string",
                                    present.get("CancellationCode", "_"): "string"},
                             low_memory=False)
    return df.rename(columns={v: k for k, v in present.items()})


def hhmm_to_datetime(date: pd.Series, hhmm: pd.Series) -> pd.Series:
    """Local clock time 'hhmm' (e.g. 1435, 2400 = midnight) -> timestamp."""
    t = pd.to_numeric(hhmm, errors="coerce")
    minutes = (t // 100) * 60 + (t % 100)      # 2400 -> 1440 = next midnight
    return date + pd.to_timedelta(minutes, unit="m")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["ne_origin"] = df["OriginState"].isin(NORTHEAST)
    df["ne_dest"] = df["DestState"].isin(NORTHEAST)
    df = df[df["ne_origin"] | df["ne_dest"]].copy()

    # Scheduled times as timestamps. BTS times are LOCAL: sched_dep is in the
    # origin's time zone, sched_arr in the destination's. Don't subtract them
    # across time zones; use CRSElapsedTime for scheduled duration instead.
    # Scheduled arrival rolls to the next day when earlier than departure.
    df["sched_dep"] = hhmm_to_datetime(df["FlightDate"], df["CRSDepTime"])
    arr = hhmm_to_datetime(df["FlightDate"], df["CRSArrTime"])
    overnight = arr < df["sched_dep"]
    df["sched_arr"] = arr + pd.to_timedelta(overnight.astype(int), unit="D")
    df["sched_dep_hour"] = df["sched_dep"].dt.hour.astype("int8")

    for c in ["Cancelled", "Diverted"]:
        df[c] = df[c].fillna(0).astype(bool)
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    a = p.parse_args()
    zips = sorted(RAW.glob(f"ontime_{a.year}_*.zip"))
    if not zips:
        raise SystemExit("No zips found. Run scripts/download.py first.")

    parts = []
    for z in zips:
        df = clean(read_month(z))
        print(f"  {z.name}: {len(df):,} Northeast flights")
        parts.append(df)
    flights = pd.concat(parts, ignore_index=True)
    for c in CATEGORICAL:                       # re-unify categories after concat
        if c in flights:
            flights[c] = flights[c].astype("category")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"flights_{a.year}_ne.parquet"
    flights.to_parquet(out, index=False)

    # Quick sanity summary
    ops = flights[~flights["Cancelled"] & ~flights["Diverted"]]
    print(f"\nSaved {out} — {len(flights):,} flights, "
          f"{flights['FlightDate'].dt.month.nunique()} months")
    print(f"cancelled: {flights['Cancelled'].mean():.1%} | "
          f"arrived 15+ min late (completed flights): {ops['ArrDel15'].mean():.1%}")
    bdl = flights[(flights["Origin"] == "BDL") | (flights["Dest"] == "BDL")]
    print(f"Bradley (BDL) flights: {len(bdl):,}")
    print("\nTop Northeast departure airports:")
    print(flights.loc[flights["ne_origin"], "Origin"].astype(str)
          .value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
