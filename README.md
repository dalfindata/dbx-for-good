# 🏥 Medical Desert Planner

**Track 2: Where are the real, highest-risk gaps in care?**

> 61 million people live in India's 40 worst healthcare deserts. This app tells you which ones,
> why, and what to do about it — with every number traceable to its source.

## The Problem

India's 1.4 billion people face uneven healthcare access. A naive facility count is misleading —
a district of 3 million with 2 facilities is far worse off than a small district with 2. Planners
need **per-capita**, **confidence-rated**, **evidence-cited** tools to prioritize.

## How It Works

Each district gets a **care-gap score** = *health need* (NFHS-5 z-score) − *facility supply per 100k people*.

**What makes this trustworthy (not just another dashboard):**

| # | Problem with naive approach | Our fix |
|---|---|---|
| 1 | Raw counts ignore population | Census-2011 → **facilities per 100k** |
| 2 | Self-reported pincodes create false zeros | **Coordinate geocoding** (nearest post office) |
| 3 | 88% private = coverage bias | Labelled as upper bounds, not absolute |
| 4 | Need not matched to supply type | Maternal gaps vs **obstetric-capable** supply only |
| 5 | Duplicates inflate counts | Deduped on entity key (cluster_id) |
| 6 | NFHS small-sample values look solid | Reliability flags ⚠ carried through |
| 7 | Post-2011 splits have no population | Flagged separately, never given fake numbers |

## App Tabs

| Tab | What it does |
|-----|-------------|
| 🗺️ **Map & Ranking** | Bubble map (size=population, color=gap score) + top-25 table |
| 🔎 **District Detail** | NFHS indicators, per-facility cited evidence text, planner actions |
| ⭐ **Shortlist** | Persisted decisions — status, notes, priority overrides, export CSV |
| 📋 **Data Readiness** | Audit of the raw dataset: field coverage, column bleed, what was fixed |
| 🤖 **AI Analyst** | Natural language Q&A powered by Databricks Foundation Models |

## Datasets

| Dataset | Records | Role |
|---------|---------|------|
| Healthcare Facilities (FDR) | 10,088 | Supply — what facilities exist (noisy, treat as claims) |
| NFHS-5 Health Indicators | 706 districts | Demand — where health needs are highest |
| India Post Pincode Directory | 160,721 | Geocoding bridge (coordinate → district assignment) |
| Census 2011 | 640 districts | Population denominator for per-capita scoring |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│         Streamlit UI (5 tabs)                           │
├────────────────────────────────────────────────────────┤
│  Gap Scoring Engine  │  AI Agent  │  Persistence       │
├────────────────────────────────────────────────────────┤
│  Pre-computed Parquet (district_gaps + citations)       │
│  + Databricks Foundation Models (meta-llama-3.3-70b)   │
│  + Lakebase / SQLite (planner actions)                 │
└────────────────────────────────────────────────────────┘
```

## Databricks Tools Used

1. **Databricks Apps** — Managed Streamlit deployment (Free Edition)
2. **Foundation Models** — `meta-llama-3.3-70b-instruct` for AI Analyst
3. **Lakebase** — Persists planner actions (Postgres, with SQLite fallback)
4. **Unity Catalog + Delta Sharing** — Source dataset access

## Run Locally

```bash
pip install -r requirements.txt
cd src && streamlit run app.py
```

## Deploy to Databricks Apps

1. Sync this repo to your Databricks workspace
2. Create an App pointing at `src/`; `app.yaml` sets the entrypoint
3. (Optional) Attach Lakebase and set `PGHOST`/`PGUSER`/`PGPASSWORD`/`PGDATABASE`

## Rebuild Data (Optional)

```bash
# Requires source CSVs + census2011_district_population.csv in parent folder
pip install scipy
python src/prepare_app_data.py
```

## Team

**DalFin Data** — DAIS Apps & Agents Hackathon for Good 2026

## License

MIT
