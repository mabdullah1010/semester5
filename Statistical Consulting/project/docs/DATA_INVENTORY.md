# Binance Data Inventory

## Overview

This document tracks all downloaded and processed Binance perpetual futures data as of September 28, 2026.

## Raw Data Files (ZIP Format)

Location: `binance-mm/raw/`

Total Files: 15

### ETHUSDT Files

BookTicker:
- ETHUSDT-bookTicker-2023-10-01.zip
- ETHUSDT-bookTicker-2023-10-02.zip
- ETHUSDT-bookTicker-2023-10-03.zip
- ETHUSDT-bookTicker-2023-10-05.zip
- ETHUSDT-bookTicker-2024-01-15.zip
- ETHUSDT-bookTicker-2024-03-01.zip

Trades:
- ETHUSDT-trades-2023-10-01.zip
- ETHUSDT-trades-2023-10-02.zip
- ETHUSDT-trades-2023-10-03.zip
- ETHUSDT-trades-2023-10-05.zip
- ETHUSDT-trades-2024-01-15.zip

### MATICUSDT Files

BookTicker:
- MATICUSDT-bookTicker-2024-01-15.zip

Trades:
- MATICUSDT-trades-2024-01-15.zip

### LINKUSDT Files

BookTicker:
- LINKUSDT-bookTicker-2024-01-15.zip

Trades:
- LINKUSDT-trades-2024-01-15.zip

## Processed Data Files (Parquet Format)

Location: `binance-mm/parquet/`

Total Files: 15

### ETHUSDT Files

BookTicker:
- binance-mm/parquet/bookTicker/ETHUSDT/2023-10-01.parquet
- binance-mm/parquet/bookTicker/ETHUSDT/2023-10-02.parquet
- binance-mm/parquet/bookTicker/ETHUSDT/2023-10-03.parquet
- binance-mm/parquet/bookTicker/ETHUSDT/2023-10-05.parquet
- binance-mm/parquet/bookTicker/ETHUSDT/2024-01-15.parquet
- binance-mm/parquet/bookTicker/ETHUSDT/2024-03-01.parquet

Trades:
- binance-mm/parquet/trades/ETHUSDT/2023-10-01.parquet
- binance-mm/parquet/trades/ETHUSDT/2023-10-02.parquet
- binance-mm/parquet/trades/ETHUSDT/2023-10-03.parquet
- binance-mm/parquet/trades/ETHUSDT/2023-10-05.parquet
- binance-mm/parquet/trades/ETHUSDT/2024-01-15.parquet

### MATICUSDT Files

BookTicker:
- binance-mm/parquet/bookTicker/MATICUSDT/2024-01-15.parquet

Trades:
- binance-mm/parquet/trades/MATICUSDT/2024-01-15.parquet

### LINKUSDT Files

BookTicker:
- binance-mm/parquet/bookTicker/LINKUSDT/2024-01-15.parquet

Trades:
- binance-mm/parquet/trades/LINKUSDT/2024-01-15.parquet

## Data Summary by Symbol

### ETHUSDT (Ethereum)
- BookTicker: 6 files covering 2023-10-01 to 2024-03-01
- Trades: 5 files covering 2023-10-01 to 2024-01-15
- Total Size: ~0.91 GB
- Status: Primary analysis symbol with extensive coverage

### MATICUSDT (Polygon)
- BookTicker: 1 file (2024-01-15)
- Trades: 1 file (2024-01-15)
- Total Size: ~0.04 GB
- Status: Mid-cap comparison symbol

### LINKUSDT (Chainlink)
- BookTicker: 1 file (2024-01-15)
- Trades: 1 file (2024-01-15)
- Total Size: ~0.05 GB
- Status: Mid-cap comparison symbol

## Data Quality Metrics

All files have been verified with:
- Zero duplicate IDs
- Zero crossed or locked quotes
- Minimal missing trade IDs (< 3 per file)
- Consistent timestamps with median event-transaction lag of 6ms
- No gaps > 60s in book updates
- Successful checksum verification where available

## File Sizes

| File Type | Symbol | Date | Size |
|-----------|--------|------|------|
| bookTicker | ETHUSDT | 2023-10-01 | 75 MB |
| bookTicker | ETHUSDT | 2023-10-02 | 108 MB |
| bookTicker | ETHUSDT | 2023-10-03 | 85 MB |
| bookTicker | ETHUSDT | 2023-10-05 | 89 MB |
| bookTicker | ETHUSDT | 2024-01-15 | 194 MB |
| bookTicker | ETHUSDT | 2024-03-01 | 321 MB |
| bookTicker | MATICUSDT | 2024-01-15 | 37 MB |
| bookTicker | LINKUSDT | 2024-01-15 | 37 MB |
| trades | ETHUSDT | 2023-10-01 | 17 MB |
| trades | ETHUSDT | 2023-10-02 | 29 MB |
| trades | ETHUSDT | 2023-10-03 | 16 MB |
| trades | ETHUSDT | 2023-10-05 | 18 MB |
| trades | ETHUSDT | 2024-01-15 | 24 MB |
| trades | MATICUSDT | 2024-01-15 | 11 MB |
| trades | LINKUSDT | 2024-01-15 | 11 MB |

Total Raw Data Size: ~1.00 GB