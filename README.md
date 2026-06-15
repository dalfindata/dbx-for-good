# 🏥 Medical Desert Planner

**Track 2: Where are the real, highest-risk gaps in care?**

> A Databricks App that helps non-technical health planners identify and prioritize healthcare access gaps across India using data from 10,088 facilities, 706 district-level health indicators, and 160,721 geographic reference points.

## The Problem

India's 1.4 billion people face uneven healthcare access. Some districts have abundant hospitals while others — "medical deserts" — have critically few facilities relative to health needs. Planners need tools to identify the highest-risk gaps, understand the evidence, and coordinate interventions.

## Our Solution

The **Medical Desert Planner** combines three datasets to compute a **Medical Desert Risk Score** for each district:

| Dataset | Records | Purpose |
|---------|---------|---------|
| Healthcare Facilities (FDR) | 10,088 | Supply — what facilities exist |
| NFHS-5 Health Indicators | 706 districts | Demand — where health needs are highest |
| India Post Pincode Directory | 160,721 | Geographic reference (planned for district-level joins) |

### Key Features

- **Desert Score** (0-100): Composite risk index from institutional birth rates, child stunting, insurance coverage, anaemia prevalence, and underweight children
- **Interactive Map**: Visualize facility distribution and identify sparse regions
- **District Deep Dive**: Click any district to see health indicators vs national averages
- **AI Facility Analyst**: LLM-powered natural language Q&A about healthcare gaps with cited evidence
- **Uncertainty Communication**: Data confidence levels (High/Medium/Low) shown alongside scores
- **Persist Actions**: Planners can save notes, shortlist districts, export reports

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Streamlit UI (Medical Desert Planner)   │
├─────────────────────────────────────────────────┤
│  Desert Score Engine │ Map Renderer │ Persistence │
├─────────────────────────────────────────────────┤
│      Databricks SQL Warehouse (Serverless)       │
├─────────────────────────────────────────────────┤
│  Facilities  │  NFHS-5   │  Pincode Directory    │
│  (Delta Sharing — Virtue Foundation FDR)         │
└─────────────────────────────────────────────────┘
```

## Databricks Tools Used

1. **Lakebase** — Persists planner notes, shortlists, and review decisions
2. **SQL Warehouse (Serverless)** — Queries 10K+ facility records on-demand
3. **Unity Catalog + Delta Sharing** — Accesses shared FDR dataset securely
4. **Databricks Apps** — Deploys as a managed Streamlit application

## Local Development

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Deploy to Databricks

This project uses a Databricks Asset Bundle (DAB). `databricks.yml` is the
declarative source of truth for the Databricks App, deploy targets, variables,
and workspace resources. The Databricks CLI uses that file to validate, deploy,
bind, and run the app repeatably from a laptop or CI/CD.

Targets:

- `dev` deploys `dev-medical-desert-planner`
- `prod` deploys `medical-desert-planner`

The separate dev app name avoids colliding with the existing production app.

Authenticate first:

```bash
databricks auth login \
  --host https://dbc-32a89911-6688.cloud.databricks.com \
  --profile dbc-32a89911-6688
```

Deploy locally:

```bash
PROFILE=dbc-32a89911-6688 \
TARGET=dev \
WAREHOUSE_ID=<sql-warehouse-id> \
scripts/deploy_databricks_app.sh
```

The local deploy script always runs Python compilation and `mypy`. Install the
typechecker before deploying:

```bash
python3 -m pip install mypy
```

Only bypass `mypy` for emergency deploys:

```bash
RUN_MYPY=0 scripts/deploy_databricks_app.sh
```

For production, set `TARGET=prod`. If the target app already exists, the script
attempts to bind it to the bundle before deploying. After binding, the bundle
manages that app and future deployments update it instead of creating a
duplicate.

### GitHub Actions Deployment

The `Deploy Databricks App` workflow can be run manually from GitHub Actions.
Configure these environment secrets for both `dev` and `prod` environments:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_WAREHOUSE_ID`

Then run the workflow and choose the target environment.

### Data Access

The app service principal needs `USE CATALOG`, `USE SCHEMA`, and `SELECT` on
the Unity Catalog data used by the app:

- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities`
- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators`
- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory`

The bundle does not declare these UC tables as app resources because doing so
requires the deployer to have `MANAGE` on each table. If deployment succeeds but
the app cannot load data, ask a catalog/table owner or workspace admin to grant
the required Unity Catalog privileges to the Databricks App service principal.

To print the app details and a grant template:

```bash
PROFILE=dbc-32a89911-6688 scripts/print_app_data_grants.sh
```

For the dev app, the grant principal is the service principal client ID:
`fb4754d9-7dd1-4e21-a5d9-9bcf63068a05`.

## Team

- **Team: DalFin Data**
- DAIS Apps & Agents Hackathon for Good 2026

## License

MIT
