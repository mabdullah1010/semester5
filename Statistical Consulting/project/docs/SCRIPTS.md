# Project Scripts Documentation

## Overview

This document describes all custom scripts created for the Binance Market Making project to facilitate data acquisition, processing, and analysis.

## Data Download Scripts

### download.py

Location: `src/download.py`

Core download script that handles individual file downloads with checksum verification.

Features:
- Downloads bookTicker and trades data from Binance data repository
- Performs SHA-256 checksum verification when available
- Handles HTTP timeouts and missing files gracefully
- Prevents re-downloading existing files

Usage:
```bash
python src/download.py ETHUSDT 2024-01-15 2024-01-17
```

### bulk_download.py

Location: `src/bulk_download.py`

Script for downloading multiple dates in parallel to speed up bulk acquisition.

Features:
- Concurrent downloads using ThreadPoolExecutor
- Progress tracking and error handling
- Configurable date ranges
- Multi-symbol support

Usage:
```bash
python src/bulk_download.py
```

### parallel_download.py

Location: `src/parallel_download.py`

Alternative parallel download implementation focused on downloading multiple symbols simultaneously.

Features:
- Parallel symbol downloading
- Configurable date ranges per symbol
- Detailed progress reporting

Usage:
```bash
python src/parallel_download.py
```

### monitor_downloads.py

Location: `src/monitor_downloads.py`

Monitoring utility to track download progress and report statistics.

Features:
- Real-time file count and size reporting
- Per-symbol breakdown of downloaded data
- Date range coverage analysis
- Total storage usage calculation

Usage:
```bash
python src/monitor_downloads.py
```

## Data Processing Scripts

### to_parquet_fixed.py

Location: `src/to_parquet_fixed.py`

Main conversion script that transforms ZIP files to efficient Parquet format.

Features:
- Reads Binance CSV files with header detection
- Proper timestamp conversion to datetime64[ns, UTC]
- Sorting by appropriate columns (event_time/update_id for bookTicker, time/id for trades)
- Boolean conversion for is_buyer_maker field
- Directory organization by data type, symbol, and date
- Skip existing files to prevent redundant processing

Usage:
```bash
python src/to_parquet_fixed.py
```

## Analysis Scripts

### binance_eda.py

Location: `binance_eda.py` (provided in project)

Comprehensive EDA script implementing all analysis functions described in project brief.

Features:
- Data loading with header detection
- Integrity checking with detailed reporting
- Derived variable calculation (midprice, spread, imbalance, etc.)
- Visualization generation (10+ plot types)
- Statistical analysis functions (ACF, markouts, fill curves)
- Export to Parquet and CSV formats

Usage:
```bash
python binance_eda.py --symbol ETHUSDT --dates 2024-01-15 --local-dir raw
```

## Script Development History

### Version 1: Basic Download Script
Initial implementation based on project brief specifications.

### Version 2: Bulk Download Enhancements
Added parallel processing and progress tracking.

### Version 3: Parquet Conversion Fixes
Resolved date parsing issues in file naming.

## Error Handling

All scripts include robust error handling for:
- Network timeouts and connectivity issues
- Missing or corrupted files
- Checksum mismatches
- Disk space limitations
- Invalid date ranges
- Unsupported symbols

## Performance Considerations

### Download Optimization
- Parallel downloads using threading
- Existing file detection to avoid re-downloads
- Configurable timeouts for unreliable connections

### Processing Optimization
- Parquet format for efficient storage and retrieval
- Sorted data for optimal time-series operations
- Memory-efficient chunked processing for large files

## Future Enhancements

Planned improvements to existing scripts:
1. Resume interrupted downloads
2. Bandwidth throttling options
3. Enhanced progress bars with ETA
4. Automatic retry mechanisms
5. Integration with download managers for large files

## Dependencies

All scripts require:
- Python 3.10+
- Standard library modules (urllib, hashlib, concurrent.futures, etc.)
- Third-party packages (pandas, numpy)

Installation:
```bash
pip install pandas numpy scipy statsmodels matplotlib pyarrow duckdb arch
```

## Configuration

Scripts use sensible defaults but can be customized through:
- Command-line arguments
- Environment variables
- Configuration files (planned)

## Maintenance

Scripts follow Python best practices:
- Clear function and variable naming
- Comprehensive docstrings
- Error handling with descriptive messages
- Modular design for easy modification
- Compatibility with Python 3.10+