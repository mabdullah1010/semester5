# Northeast flight delays, 2025 (BTS on-time data)

## Setup (Ubuntu, once)
    bash setup.sh

That creates `.venv`, downloads the 12 monthly 2025 files from BTS
(~30–60 MB each), and builds `data/processed/flights_2025_ne.parquet`:
every flight departing from or arriving at an airport in CT, ME, MA, NH,
RI, VT, NJ, NY or PA.

The BTS server is slow and sometimes drops connections. If a month fails,
rerun `python scripts/download.py --year 2025`; it skips what you already
have. Then rerun `python scripts/build.py --year 2025`.

## Daily use
    source .venv/bin/activate
    jupyter lab

    import pandas as pd
    f = pd.read_parquet("data/processed/flights_2025_ne.parquet")

## Notes on the data
- Delay columns are empty for cancelled and diverted flights.
- Delay-cause minutes are filled only when ArrDelay >= 15.
- Times are local: departure in origin time, arrival in destination time.
- Reporting_Airline is the operating carrier (e.g. Endeavor = 9E flies for
  Delta), not the brand on the ticket.
- Field definitions: the "Reporting Carrier On-Time Performance" download
  page on transtats.bts.gov links each column to its description.
