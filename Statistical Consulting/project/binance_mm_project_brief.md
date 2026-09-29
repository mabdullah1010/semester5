# Project Brief: Testing Avellaneda-Stoikov Assumptions on Binance Perpetual Futures Data

This document describes a semester-long statistical consulting project in full. It is written so that someone (or an AI agent) with no prior context can understand the goals, get the data, set up the environment, run the exploratory analysis, and help with the modeling and writeup. Read it top to bottom once before doing anything.

---

## 1. Context

**Who:** Abdullah, a junior at Connecticut College double-majoring in Computer Science and Statistics & Data Science.

**Course:** Statistical consulting class, fall 2026. The semester project runs the whole term and ends in a written consulting report plus a presentation. Consulting courses grade on:

- a clear client question
- careful, defensible inference
- honest limitations
- communication a non-technical client can act on

Technical impressiveness alone does not carry the grade.

**Why this topic:** Abdullah is second author on EvoMM, a paper accepted at IEEE CIFEr 2026. It applies quality-diversity search to automated market making inside an **Avellaneda-Stoikov (A-S) simulator**. He wants to keep working in market making. This project takes the assumptions baked into that simulator and tests them against real market data.

**Earlier plan:** He had been planning a project on 2025 BTS flight-delay data for the Northeast. That remains the fallback if the Binance data turns out unusable (see Section 11, "Go/no-go check").

**Related lab work:** Jim O'Connor's lab, where EvoMM was written, plans to acquire real limit order book (LOB) data. This class project is positioned as a **pilot**: it builds the methods and pipeline on free public data, so the code and findings can carry over to the lab's richer data later. It must stand on its own as a class project and must not depend on the lab's data arriving.

---

## 2. The client

Consulting projects need a client. Be honest about which clients are real:

- **Primary (real, pending):** Jim O'Connor's lab. The lab faces a real decision: when real LOB data arrives, which A-S assumptions should the simulator relax first, and what parameter ranges are realistic?
  - **Status:** Abdullah has not yet asked Jim to act as client. If Jim agrees, plan one scoping meeting early in the semester and one results meeting in December.
  - **Overlap check:** Also confirm with Jim that this public-data project does not duplicate something the lab is already doing.
- **Hypothetical personas** (label them as hypothetical in the report):
  - A small crypto market-making desk deciding how far from the midprice to quote and when to widen.
  - An execution trader at a fund who needs to know when trading crypto is cheapest.

---

## 3. Background: the Avellaneda-Stoikov model in brief

A-S (2008) models a market maker who quotes a bid and an ask around a reference price and manages inventory risk. Its key simplifying assumptions are the ones this project tests.

1. **Midprice dynamics.** The midprice follows arithmetic Brownian motion with **constant volatility σ**. Returns are normal and independent.
2. **Fill intensity.** Market orders hit a quote placed at distance δ from the midprice as a Poisson process with intensity **λ(δ) = A·e^(−kδ)**. Fill probability therefore decays **exponentially** with distance.
3. **No adverse selection.** Fills carry no information about future prices. Being filled does not predict the midprice moving against you.

The model also has a risk-aversion parameter γ, which cannot be observed from market data. It can only be explored by sensitivity analysis.

The project's overall message to the client takes this form: "Here is where the A-S world matches a real market, here is where it breaks, and here is what that implies for strategies tuned in simulation."

---

## 4. Research questions

Questions are tiered. If time runs short, cut from the bottom and **never cut Q3 or Q4**.

### Core (must answer)

| # | Question | Client | Tests A-S assumption |
|---|---|---|---|
| Q1 | Is midprice volatility constant, or does it cluster and follow intraday/weekly patterns? How large and persistent are the effects? | Jim's lab | 1 |
| Q2 | Are midprice returns close to normal at market-making horizons (1s, 10s, 1min)? *Can be a subsection of Q1.* | Jim's lab | 1 |
| Q3 | Does fill probability decay exponentially with distance from the mid? What are A and k? | Jim's lab, hypothetical desk | 2 |
| Q4 | How large is adverse selection: after a buyer- or seller-initiated trade, how far does the mid move in that direction over the next 1–60s? | Hypothetical desk, Jim's lab | 3 |

### Secondary (if core finishes smoothly)

| # | Question | Client |
|---|---|---|
| Q5 | Are k and adverse selection stable across hours, volatility regimes, and pairs? | Jim's lab |
| Q6 | Does adverse selection grow with trade size? Does it differ between liquid and illiquid pairs? | Hypothetical desk |
| Q7 | When is liquidity cheapest and most expensive (time-weighted spread and top-of-book size, by hour and weekday)? | Hypothetical execution trader |

### Stretch

| # | Question | Client |
|---|---|---|
| Q8 | Do the 8-hourly funding timestamps (00:00, 08:00, 16:00 UTC) disrupt liquidity and volatility? Use an event study with matched control windows. | Hypothetical desk |
| Q9 | Does the order-flow-imbalance signal from Q4 hold on a held-out month? | Hypothetical desk |
| Q10 | Plugging estimated σ, A, k into the A-S optimal quote formulas (with a sensitivity sweep over γ), do implied spreads resemble observed spreads? *Interpretive capstone, not a formal test.* | Jim's lab |

### Out of scope (do not attempt)

- **Backtesting A-S strategy profitability.** This needs queue position, which top-of-book data cannot give. Any P&L number would be fiction.
- **Price prediction as a goal.** It is not a consulting question and invites overfitting.
- **Causal claims** such as "volatility causes wider spreads." Spreads and volatility are jointly determined. Describe associations and name the simultaneity as a limitation.

---

## 5. Data

### 5.1 Source

Binance publishes free historical market data at **https://data.binance.vision**, with the source repo at github.com/binance/binance-public-data. The data is organized as:

```
https://data.binance.vision/data/<market>/<cadence>/<datatype>/<SYMBOL>/<SYMBOL>-<datatype>-<DATE>.zip
```

The fields are:

- **`<market>`**: `spot`, `futures/um` (USD-M, the one we use), `futures/cm`, `option`
- **`<cadence>`**: `daily` (DATE = `YYYY-MM-DD`) or `monthly` (DATE = `YYYY-MM`)
- **Checksums**: each zip normally has a sibling `.zip.CHECKSUM` file containing a SHA-256 hash. Verify downloads against it.

Examples:

```
https://data.binance.vision/data/futures/um/daily/bookTicker/ETHUSDT/ETHUSDT-bookTicker-2024-01-15.zip
https://data.binance.vision/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-2024-01-15.zip
https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-01-15.zip
```

(Klines add an interval level to the path.)

### 5.2 Datasets used and their columns

**bookTicker** (USD-M futures): best bid and ask. There is one row every time the top of book changes, which means millions of rows per day for liquid pairs.

| column | meaning |
|---|---|
| `update_id` | order book update ID |
| `best_bid_price`, `best_bid_qty` | best bid price and size |
| `best_ask_price`, `best_ask_qty` | best ask price and size |
| `transaction_time` | matching-engine time (ms since epoch, UTC) |
| `event_time` | publish time (ms since epoch, UTC). **Use this as the timestamp.** |

**trades** (USD-M futures): one row per trade.

| column | meaning |
|---|---|
| `id` | sequential trade ID. Gaps mean missing trades. |
| `price`, `qty`, `quote_qty` | price, base size, notional in USDT |
| `time` | ms since epoch, UTC |
| `is_buyer_maker` | `True` means the buyer was the resting (maker) order, so the **seller was the aggressor**. `False` means the buyer was the aggressor. |

**klines** (optional, 1-minute candles): OHLC, volume, quote volume, trade count, taker buy volume. Use them for sanity checks, not as the primary data.

### 5.3 Critical data limitations (verify, do not assume)

These determine what the project can and cannot do.

- **bookTicker was discontinued.** Users on the binance-public-data GitHub reported that futures bookTicker began around May 2023, that the last BTCUSDT file is from **March 2024**, and that **other symbols ended earlier**. A 2025 developer-forum post confirmed it had not been updated since 2024.
  - **Consequence:** quote data exists only for a fixed historical window, roughly May 2023 to March 2024 at most, and the window varies by symbol.
  - **Action:** before choosing symbols or dates, browse `https://data.binance.vision/?prefix=data/futures/um/daily/bookTicker/` and record the actual first and last available date per candidate symbol.
- **bookTicker is futures-only.** Spot has no equivalent archive.
- **No full order book depth.** The old L2 download page is gone. We have top of book only. (`bookDepth` files exist for futures, but they give coarse aggregated depth at percentage bands, not a true order book. Investigate only as an optional extra.)
- **Out-of-order rows.** Users reported BTC/ETH bookTicker files from January 2024 onward with unordered rows. **Always sort by `(event_time, update_id)`** before any time-based computation.
- **Header rows are inconsistent.** Some files have a header line and some do not. The loader must detect this.
- **Timestamp units.** Futures files use milliseconds. (Spot files from 2025 onward switched to microseconds. That's irrelevant here unless spot data is added.)
- **Live alternative:** recording the websocket bookTicker stream going forward is possible but yields only weeks of data and needs a process kept running. It is not the plan unless the archive proves unusable.

### 5.4 Symbol choice

**Plan:**

- **ETHUSDT perp:** very liquid.
- **One or two mid-cap USD-M perps:** chosen for wider, more variable spreads and enough trades. Candidates must have bookTicker coverage in the chosen window (check per 5.3).
- **BTCUSDT (optional):** extreme-liquidity reference only.

**Warning:** on BTCUSDT and likely ETHUSDT, the spread sits at **one tick** almost all the time. Spread-based analyses on those pairs have almost no variance to explain. The EDA plot "01_spread_ticks" settles this empirically. Liquidity variation on liquid pairs should be measured through top-of-book size instead.

### 5.5 Sample window

- **EDA:** 3–5 ordinary days per symbol (no holidays, no crash days).
- **Final analysis:** about 6 months within the available window (for example, Oct 2023 to Mar 2024 if coverage allows). Reserve the final month as a **held-out test set** for Q9 and never look at it during model building.

---

## 6. Environment setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install pandas numpy scipy statsmodels matplotlib pyarrow duckdb arch
```

What the packages are for:

| package | use |
|---|---|
| `statsmodels` | ACF, OLS with Newey-West (HAC) errors, quantile regression, GLMs |
| `arch` | GARCH models (Q1) |
| `pyarrow` | Parquet files |
| `duckdb` | querying months of Parquet without loading all of it into memory (`polars` is an acceptable alternative) |

Suggested layout:

```
binance-mm/
├── raw/                 # downloaded .zip files (never edited)
├── parquet/             # cleaned, sorted, typed data; one file per symbol per day per type
├── eda_out/<SYMBOL>/    # EDA plots and reports
├── notebooks/
├── src/
│   ├── download.py
│   ├── to_parquet.py
│   └── binance_eda.py
└── report/
```

---

## 7. Downloading

### 7.1 Bulk download with checksum verification

`src/download.py`:

```python
import hashlib, os, sys, urllib.request
from datetime import date, timedelta

BASE = "https://data.binance.vision/data/futures/um/daily"

def daterange(start, end):
    d = start
    while d <= end:
        yield d.isoformat()
        d += timedelta(days=1)

def get(url, timeout=300):
    return urllib.request.urlopen(url, timeout=timeout).read()

def download(kind, symbol, day, outdir="raw"):
    fname = f"{symbol}-{kind}-{day}.zip"
    path = os.path.join(outdir, fname)
    if os.path.exists(path):
        return "exists"
    url = f"{BASE}/{kind}/{symbol}/{fname}"
    try:
        data = get(url)
    except Exception as e:
        return f"MISSING ({e})"
    try:
        expected = get(url + ".CHECKSUM").decode().split()[0]
        if hashlib.sha256(data).hexdigest() != expected:
            return "CHECKSUM MISMATCH"
    except Exception:
        pass  # checksum file absent; keep data but note it
    os.makedirs(outdir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return "ok"

if __name__ == "__main__":
    # usage: python download.py ETHUSDT 2024-01-15 2024-01-17
    sym, s, e = sys.argv[1], date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
    for day in daterange(s, e):
        for kind in ("bookTicker", "trades"):
            print(day, kind, download(kind, sym, day))
```

Log every `MISSING` day. Missing days are data-quality findings for the report, not errors to hide.

### 7.2 Convert to Parquet once

Reading zipped CSVs repeatedly is slow. After downloading, convert each file once:

1. Detect whether a header is present.
2. Assign the column names from Section 5.2.
3. Sort (bookTicker by `event_time, update_id`; trades by `time, id`).
4. Convert timestamps to `datetime64[ns, UTC]`.
5. Cast `is_buyer_maker` to bool.
6. Write to `parquet/<type>/<SYMBOL>/<DATE>.parquet`.

The loader functions in `binance_eda.py` (`_read_zip_csv`, `load`) already do steps 1–5 and can be reused.

**Implementation note (pandas ≥ 2):** `pd.to_datetime(..., unit="ms")` can produce `datetime64[ms]`, while adding a `Timedelta` produces a different resolution. `merge_asof` then fails with "incompatible merge keys." Cast all timestamps explicitly to `datetime64[ns, UTC]`.

---

## 8. Conventions (use these everywhere)

- **Timezone:** UTC throughout. Label plot axes "UTC."
- **Timestamp:** bookTicker `event_time`, trades `time`.
- **Midprice:** `(best_bid_price + best_ask_price) / 2`.
- **Spread:**
  - in ticks: `round((ask − bid) / tick)`
  - in basis points: `1e4 · (ask − bid) / mid`
- **Tick size:** infer from data as the smallest positive difference between observed prices. Confirm against Binance's exchange info (for reference, ETHUSDT perp is expected to be 0.01).
- **Quote imbalance:** `(bid_qty − ask_qty) / (bid_qty + ask_qty)`, in [−1, 1].
- **Trade sign:**
  - `+1` (buyer-initiated) if `is_buyer_maker == False`
  - `−1` (seller-initiated) if `is_buyer_maker == True`
- **Order-flow imbalance (per interval):** `(buy-initiated volume − sell-initiated volume) / total volume`.
- **Resampling mids:** take the last quote in each interval, then forward-fill. A quote stays live until it changes.
- **Averaging quote-level quantities:** **time-weight** by how long each quote was live. Do not weight by update count. Cap individual durations (for example, at 60s) so outage gaps don't dominate.
- **Returns:** log midprice returns, reported in basis points.
- **Markout at horizon h:** `sign · (mid(t+h) − mid(t)) / mid(t) · 1e4`, where mid(t) is the last mid at or before the trade. Positive means the price moved in the aggressor's direction, which is bad for the maker.
- **Outages:** any gap longer than 60s between bookTicker updates on a liquid pair is treated as missing data. Log it and exclude it from time-weighted averages and return calculations.

---

## 9. Exploratory data analysis

The script `binance_eda.py` implements everything in this section.

**Usage:**

```bash
python binance_eda.py --symbol ETHUSDT --dates 2024-01-15 2024-01-16 2024-01-17
python binance_eda.py --symbol ETHUSDT --dates ... --local-dir raw      # use downloaded zips
python binance_eda.py --symbol ETHUSDT --dates ... --fill-window 1s     # fill-curve sensitivity
```

**Outputs** go to `eda_out/<SYMBOL>/`:

- `integrity_report.txt`
- plots `01`–`07`
- `minute_table.parquet`
- `fill_curve.csv`

**Status:** the script was tested end to end only on **synthetic** files that mimic Binance's formats (headerless files, out-of-order rows). It has not yet run on real downloads, so expect minor fixes on the first real run. Some integrity numbers were artifacts of the synthetic data and are meaningless. Re-check everything on real data.

### 9.1 Stage 1: integrity checks (run before any plot)

| check | healthy result | if unhealthy |
|---|---|---|
| time span per day | about 00:00 to 23:59 UTC | truncated file; re-download or exclude the day |
| duplicate `update_id` / trade `id` | about 0 | deduplicate; note in report |
| crossed quotes (ask < bid) | rare | drop them; many means a file problem |
| locked quotes (ask = bid) | rare | drop them |
| `event_time − transaction_time` | small, a few ms | large lags go in limitations |
| gaps between book updates | median well under 1s for ETH | gaps > 60s mean outage or missing data: log and exclude |
| trade ID jumps > 1 | 0 | missing trades; log them |

Record all findings. They become the report's data-quality section.

### 9.2 Stage 2: derived variables

Compute the following as defined in Section 8:

- midprice
- spread (ticks and bps)
- quote imbalance
- trade sign
- a **one-minute table** per symbol with:
  - time-weighted spread
  - share of time at a 1-tick spread
  - realized volatility (from 1s mid returns)
  - number of quote updates
  - volume, trade count
  - order-flow imbalance
  - hour, weekday, funding-window flag
  - next-minute return

This table is the backbone of Q1, Q5, Q7, and Q9.

### 9.3 Stage 3: EDA plots

| file | plot | question it previews | what to look for |
|---|---|---|---|
| `01_spread_ticks.png` | time share at spread = 1, 2, … ticks | Q7, symbol choice | If ETH is at 1 tick ~95% of the time, spread analysis on ETH is dead; use the mid-cap pair and top-of-book size instead. |
| `02_intraday.png` | median spread, 1-min realized vol, quote updates by UTC hour, funding times marked | Q1, Q7, Q8 | US-session volatility hump; spikes near 00/08/16 UTC |
| `02b_heatmap_vol.png` | hour × weekday realized vol (only with ≥14 days) | Q1, Q7 | Weekend and night effects |
| `03_qq_returns.png` + kurtosis table | QQ plots of mid returns at 1s, 10s, 1min | Q2 | Heavy tails that shrink with horizon. At 1s most returns are exactly zero; the script drops zeros there, and that choice must be stated. |
| `04_acf.png` | ACF of 1-min returns vs absolute returns | Q1 | Returns ≈ uncorrelated; absolute returns show slowly decaying positive ACF (volatility clustering). Likely the most persuasive early figure. |
| `05_markouts.png` | mean signed mid move 1–60s after trades | Q4 | Curve above zero means adverse selection. CIs are naive (ignore overlap); replace with block bootstrap later. |
| `06_fill_curve.png` | P(trade reaches a quote δ ticks from mid within window), log y-axis, with exponential fit | Q3 | Straight line means exponential, consistent with A-S. Curvature means the assumption breaks. The slope gives a rough k. |
| `07_trade_sizes.png` | log-log survival function of trade notional | Q6 | Roughly straight tail indicates power-law-like sizes, which motivates size buckets. |

**Fill-curve definition** (crude on purpose, for EDA only):

1. At the start of each window (default 10s), take the mid.
2. A hypothetical ask at `mid + δ·tick` counts as "touched" if any trade in the window prints at or above it (symmetric for the bid).

Known weaknesses, which refining is the real work of Q3:

- It ignores queue position (touched ≠ filled).
- It ignores the mid moving within the window.
- It is sensitive to the window length.

### 9.4 EDA deliverable

A one-page memo for the client (and the first meeting with Jim, if he agrees) containing:

- a data-quality summary
- symbols and date coverage
- time share at a 1-tick spread per symbol
- the kurtosis table
- three figures: ACF, markouts, fill curve
- a short list of surprises and next steps

---

## 10. Modeling plan (after EDA)

### Q1/Q2: Volatility

- **Descriptive:** realized volatility by hour and weekday; the ACF of absolute and squared returns; Ljung-Box tests.
- **Model:** GARCH(1,1) on 1-min returns using `arch`. Report persistence (α+β).
- **Seasonality:** consider deseasonalizing by the intraday volatility profile first.
- **Deliverable:** effect sizes. Include peak-to-trough ratio of hourly volatility, a half-life of volatility shocks, and tail-exponent or kurtosis by horizon.

### Q3: Fill intensity

1. Refine the fill-curve definition. Candidates:
   - condition on the mid at quote time
   - multiple window lengths
   - separate bid and ask
2. Fit `P(δ) = A·e^(−kδ)` by nonlinear least squares or a binomial/Poisson GLM with log link.
3. Compare against at least one alternative shape (for example, a power law) using out-of-sample fit or likelihood.
4. Get confidence intervals by **block bootstrap** over time (blocks of an hour or a day), because windows are serially dependent.

### Q4: Adverse selection

- **Markout curves:** by horizon, with block-bootstrap bands.
- **Regression:** next-interval return on order-flow imbalance and quote imbalance, with **Newey-West (HAC)** standard errors.
- **Expectations:** a tiny R² is expected. The consulting point is that a small, stable effect matters to a high-frequency quoter. Explain this explicitly.

### Q5–Q7

- **Q5:** interactions with hour, a volatility regime (for example, terciles of trailing realized vol), and pair. Alternatively, separate fits compared across groups.
- **Q6:** pooled regression with pair fixed effects and trade-size buckets.
- **Q7:** time-weighted spread and top-of-book size heatmaps. Use quantile regression (50th, 90th, 99th percentiles) because the client cares about bad minutes, not averages.

### Q8–Q10

- **Q8:** event study around funding timestamps against matched non-funding windows.
- **Q9:** fit on training months; evaluate out-of-sample R² and coefficient stability on the held-out month.
- **Q10:** plug σ, A, k into A-S closed-form quotes; sweep γ; compare implied spreads to observed ones.

### Statistical rules for every question

- **Never** use iid standard errors on time-series data. Use HAC or block bootstrap.
- Report effect sizes in client units (bps, seconds, ticks), not just p-values.
- Check that results survive **1-min vs 5-min** aggregation and **ETH vs mid-cap**.
- Treat everything as association, not causation.

---

## 11. Go/no-go check (first week)

Download about 3 days for ETHUSDT and one mid-cap pair. Run the integrity report and `06_fill_curve.png`.

- **Go:** files are mostly complete, integrity issues are manageable, and the fill curve looks like a noisy decaying function. Commit to the project.
- **No-go:** coverage is too spotty in the needed window, or the mid-cap pair has too few trades to estimate anything. Consider another symbol first, then fall back to the BTS flight-delay project.

---

## 12. Timeline (about 14 weeks)

| weeks | work |
|---|---|
| 1 | Verify coverage, go/no-go check, ask Jim to be the client |
| 1–3 | Bulk download, Parquet conversion, integrity checks, one-minute tables |
| 4–6 | Full EDA; EDA memo to client |
| 7–11 | Modeling: Q1–Q4 first, then Q5–Q7 if on schedule |
| 12–14 | Robustness, writing, presentation. Q8–Q10 only if everything else is done. |

---

## 13. Final report structure (15–25 pages plus appendix)

1. **Executive summary** (1 page, no equations): three or four findings and what the client should do.
2. **Client question and scope.**
3. **Data:** source, symbols, window, bookTicker discontinuation, sorting issue, aggregation, exclusions with justification.
4. **Exploratory analysis.**
5. **Volatility** (Q1/Q2).
6. **Fill intensity** (Q3).
7. **Adverse selection** (Q4).
8. **Secondary questions**, if done.
9. **Robustness and diagnostics.**
10. **Recommendations** for the client: which A-S assumptions to relax first, realistic parameter ranges, quoting implications.
11. **Limitations:**
    - 2023–24 window only
    - top of book only
    - single venue
    - touch ≠ fill
    - simultaneity between spreads and volatility
12. **Appendix:** pipeline code, variable dictionary, extra tables.

---

## 14. Open items and decisions

- [ ] Verify bookTicker date coverage per candidate symbol (Section 5.3).
- [ ] Pick the mid-cap symbol(s).
- [ ] Ask Jim O'Connor to act as client, and confirm no overlap with the lab's LOB plans.
- [ ] Run the EDA script on real data and fix any issues.
- [ ] Decide the final 6-month window and the held-out month.

---

## 15. Notes for an AI agent helping on this project

- Abdullah wants **direct, honest critique**, not reassurance. Flag weak statistical choices plainly.
- He prefers **surgical, verified edits**. When changing code, change only what is needed, and run it where possible.
- He wants writing that **sounds human** and is free of AI tells, especially in the report.
- For LaTeX, give **paste-ready Overleaf source** rather than compiled PDFs.
- Do not invent results. Any numbers in the report must come from actually running the analysis.
- Do not claim a Binance data fact (coverage dates, columns, tick sizes) without checking it against the actual files or data.binance.vision. Coverage in particular has changed over time.
- Keep the scope discipline in Section 4. Resist expanding into strategy backtests or price prediction.
