# Data Files (Not Committed)

These CSVs are too large for Git (~91MB total). The app loads them from Databricks SQL in production.

## Local Development Setup

Copy from the shared download location:
```bash
cp /path/to/shared/data/facilities_complete.csv .
cp /path/to/shared/data/nfhs_health.csv .
cp /path/to/shared/data/pincode_directory.csv .
```

Or re-download using the script in the project root's `requirements.txt` setup instructions.
