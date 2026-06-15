# Medical Desert Planner — DAIS 2026 (Track 2)

**Core question:** *Where are the real, highest-risk gaps in care?*

A Databricks App that ranks India's districts by a **per-capita care-gap score** — health need
(NFHS-5) against facility supply (FDR dataset) per 100k people — with every weak number **flagged**
and every score **traced to the underlying facility text**.

## Why this is trustworthy (the data-readiness story)
The naive answer ("districts with the fewest facilities") is wrong, and we can prove it. This app fixes
the four things that make a raw ranking misleading:

1. **Per-capita denominator** — Census-2011 district population, so a 3-million-person district with 2
   facilities outranks a tiny one. *(Naive count missed this entirely.)*
2. **Coordinate geocoding** — facilities placed by nearest post office (lat/long), not self-reported
   pincode strings. This collapsed one state's false-zeros from 392 orphaned facilities to 11.
3. **Specialty matching** — maternal gaps compare need against *obstetric-capable* facilities.
4. **Honest uncertainty** — dedupe by entity key; confidence flags (🟢 measured / 🟡 real-sparse /
   🔴 low-confidence); 67 post-2011-split districts shown separately as *population-unresolved* rather
   than given fake per-capita numbers; NFHS small-sample values marked ⚠.

## Headline finding
Per-capita reframes the answer: the worst gaps are huge-population **Bihar/UP** districts (Araria: 2.8 M
people, 1 facility) **and** Northeast/tribal absolute deserts (Nagaland, Bastar). **~61 million people**
live in the 40 highest-gap districts.

## Layout
```
medical_desert_planner/
  app.py                 Streamlit UI (map · ranking · district detail w/ citations · shortlist)
  persistence.py         user actions — Lakebase/Postgres in prod, SQLite locally
  prepare_app_data.py    builds the two parquet tables from the source CSVs
  app_data/
    district_gaps.parquet        one row per district (score, drivers, supply, confidence, centroid)
    facility_citations.parquet   one row per geocoded facility (cited evidence text)
  requirements.txt
  app.yaml               Databricks Apps entrypoint
```

## Run locally
```bash
# from the parent folder that holds the source CSVs:
python medical_desert_planner/prepare_app_data.py        # (re)build the data tables
pip install -r medical_desert_planner/requirements.txt
streamlit run medical_desert_planner/app.py
```

## Deploy to Databricks Apps (Free Edition)
1. Sync this folder to your workspace (`databricks sync` or the workspace UI), or include it in a bundle.
2. Create an App pointing at this directory; `app.yaml` runs Streamlit on port 8000.
3. **Persistence:** attach a Lakebase (Postgres) instance and set `PGHOST`/`PGUSER`/`PGPASSWORD`/`PGDATABASE`
   (or `DATABASE_URL`) in the app env, then uncomment `psycopg[binary]` in `requirements.txt`.
   With no PG vars set, the app falls back to local SQLite so it always runs.

## Data sources
- **Facilities** — provided FDR dataset (`...virtue_foundation_dataset.facilities`). *Noisy fields are
  treated as claims to verify.*
- **Health need** — `nfhs_5_district_health_indicators` (NFHS-5, 2019-21).
- **Geocoding bridge** — `india_post_pincode_directory`.
- **Population denominator** — Census 2011 district populations (`census2011_district_population.csv`).

## Known limitations (shown in-app)
- Supply = *findable / largely-private* facilities, not the full public PHC/CHC network → gaps are upper bounds.
- Census-2011 vintage; 67 post-2011-split districts lack a matched population (flagged, ranked by need only).
- Need is z-scored (relative); pair with absolute coverage thresholds when committing resources.
