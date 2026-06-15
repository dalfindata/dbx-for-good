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
| India Post Pincode Directory | 160,721 | Geographic bridge between datasets |

### Key Features

- **Desert Score** (0-100): Composite risk index from institutional birth rates, child stunting, insurance coverage, anaemia prevalence, and underweight children
- **Interactive Map**: Visualize facility distribution and identify sparse regions
- **District Deep Dive**: Click any district to see health indicators vs national averages
- **Evidence Citations**: Every score links back to source data — no black-box claims
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

```bash
databricks bundle deploy --target dev
```

## Team

- **Team: DalFin Data**
- DAIS Apps & Agents Hackathon for Good 2026

## License

MIT
