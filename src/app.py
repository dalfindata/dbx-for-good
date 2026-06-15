"""
Medical Desert Planner  —  DAIS 2026 Hackathon for Good, Track 2
"Where are the real, highest-risk gaps in care?"

A planning tool that ranks India's districts by a PER-CAPITA care-gap score
(health need from NFHS-5  vs.  facility supply from the FDR dataset), with every
weak number flagged and every score traceable to the underlying facility text.

Run locally:   streamlit run app.py
Deploy:        Databricks App (see README.md)
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import persistence
from agent import run_agent, get_quick_stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "app_data")

st.set_page_config(page_title="Medical Desert Planner", page_icon="🏥", layout="wide")

# friendly labels for NFHS driver indicators: (label, bad_direction, unit)
LABELS = {
    "anc4": ("4+ antenatal-care visits", "low"), "inst_birth": ("Institutional births", "low"),
    "skilled_birth": ("Births w/ skilled attendant", "low"),
    "pnc": ("Postnatal care (skilled provider)", "low"),
    "full_immun": ("Children fully immunised", "low"),
    "insurance": ("Households w/ health insurance", "low"),
    "ari_careseek": ("Child fever/ARI taken to facility", "low"),
    "cerv_screen": ("Women ever had cervical screen", "low"),
    "unmet_fp": ("Unmet need, family planning", "high"), "stunting": ("Children stunted (u5)", "high"),
    "wasting": ("Children wasted (u5)", "high"), "underweight": ("Children underweight (u5)", "high"),
    "child_anaemia": ("Children anaemic (6-59m)", "high"), "women_anaemia": ("Women anaemic (15-49)", "high"),
}
CONF_BADGE = {
    "measured": ("🟢 Measured", "Facilities were geocoded into this district — supply count is observed."),
    "real_sparse": ("🟡 Real but sparse", "0 facilities found, and this state has almost none in the dataset — "
                                          "the gap is real, not a data error."),
    "low_join_conf": ("🔴 Low confidence", "0 facilities found, but this state has many facilities that didn't "
                                           "match a district — treat this zero with caution."),
}


@st.cache_data
def load():
    g = pd.read_parquet(f"{DATA}/district_gaps.parquet")
    c = pd.read_parquet(f"{DATA}/facility_citations.parquet")
    # NFHS names carry stray whitespace — strip so selectbox/lookup/citation joins stay consistent
    for df in (g, c):
        df["district_name"] = df["district_name"].astype(str).str.strip()
        df["state_ut"] = df["state_ut"].astype(str).str.strip()
    return g, c


@st.cache_data
def load_readiness():
    p = os.path.join(DATA, "data_readiness.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


gaps, cites = load()
persistence.init_db()


# --------------------------------------------------------------------------- header
st.title("🏥 Medical Desert Planner")
st.caption("DAIS 2026 — Track 2 · *Where are the real, highest-risk gaps in care?*")

with st.expander("How to read this — method & uncertainty (please read once)", expanded=False):
    st.markdown(
        """
**The score.** Each district gets a **care-gap score** = *health need* − *facility supply per 100k people*.
- **Need** is a z-scored composite of NFHS-5 indicators: maternal & newborn care access, child immunisation,
  treatment-seeking, insurance, plus disease burden (child malnutrition, anaemia). Higher = worse.
- **Supply** is facilities from the FDR dataset, **geocoded by coordinates** and divided by
  **Census-2011 district population** — so a district of 3 million with 2 facilities ranks above a small one.

**We do not hide weak evidence.** Every district carries a **confidence flag**:
🟢 *measured* · 🟡 *real but sparse* · 🔴 *low confidence (possible data-join gap)*.
67 districts are post-2011 splits with **unresolved population** — they are shown separately and ranked by need only,
never given a fake per-capita number.

**Every claim is cited.** District scores link to the NFHS-5 indicators behind them; supply links to the
**underlying facility records** (name, description, capability, source URLs) in the detail view.

*Supply reflects **findable / largely-private** facilities, not the full public PHC/CHC network — so gaps are
upper bounds, directionally reliable for targeting.*
        """
    )


# --------------------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
view = st.sidebar.radio("Score to rank by", ["Overall care gap", "Maternal-care gap"], index=0)
GAPCOL = "gap" if view.startswith("Overall") else "maternal_gap"

states = ["(All states)"] + sorted(gaps["state_ut"].dropna().unique().tolist())
state_sel = st.sidebar.selectbox("State / UT", states)

conf_sel = st.sidebar.multiselect(
    "Supply confidence", ["measured", "real_sparse", "low_join_conf"],
    default=["measured", "real_sparse"],
    format_func=lambda x: CONF_BADGE[x][0],
)
only_resolved = st.sidebar.checkbox("Only population-resolved districts", value=True)
min_pop = st.sidebar.slider("Minimum population", 0, 3_000_000, 0, step=100_000)

f = gaps.copy()
if state_sel != "(All states)":
    f = f[f["state_ut"] == state_sel]
if conf_sel:
    f = f[f["supply_confidence"].isin(conf_sel)]
if only_resolved:
    f = f[f["pop_resolved"]]
f = f[(f["pop"].fillna(0) >= min_pop)]
f = f.sort_values(GAPCOL, ascending=False)

st.sidebar.markdown("---")
st.sidebar.metric("Districts shown", len(f))
if len(f):
    st.sidebar.metric("People in view", f"{f['pop'].sum()/1e6:,.0f} M")


# --------------------------------------------------------------------------- tabs
tab_map, tab_detail, tab_short, tab_ready, tab_agent = st.tabs(
    ["🗺️ Map & ranking", "🔎 District detail", "⭐ Shortlist", "📋 Data readiness", "🤖 AI Analyst"])

# ===== TAB 1: MAP + RANKING ================================================
with tab_map:
    c1, c2 = st.columns([3, 2])
    with c1:
        m = f.dropna(subset=["lat", "lon"]).copy()
        if len(m):
            m["People"] = m["pop"].fillna(0)
            fig = px.scatter_map(
                m, lat="lat", lon="lon", color=GAPCOL, size=np.sqrt(m["People"].clip(lower=1)),
                color_continuous_scale="OrRd", size_max=22, zoom=3.3,
                hover_name="district_name",
                hover_data={"state_ut": True, GAPCOL: ":.2f", "fac_per_100k": ":.2f",
                            "pop": ":,.0f", "lat": False, "lon": False},
                map_style="carto-positron", height=560,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar=dict(title="gap"))
            st.plotly_chart(fig, width='stretch')
            st.caption("Bubble size = population · colour = care-gap score (darker = worse). "
                       "Centroids approximated from facility & post-office coordinates.")
        else:
            st.info("No districts match the current filters.")
    with c2:
        st.subheader("Highest-risk districts")
        show = f.head(25)[["district_name", "state_ut", GAPCOL, "fac_per_100k",
                           "pop", "supply_confidence"]].copy()
        show["pop"] = show["pop"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        show["fac_per_100k"] = show["fac_per_100k"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        show[GAPCOL] = show[GAPCOL].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        show["conf"] = show["supply_confidence"].map(lambda c: CONF_BADGE[c][0])
        show = show.drop(columns=["supply_confidence"]).rename(
            columns={"district_name": "District", "state_ut": "State",
                     GAPCOL: "Gap", "fac_per_100k": "Fac/100k", "pop": "Population"})
        st.dataframe(show, hide_index=True, width='stretch', height=520)

    # population-unresolved but high-need (honest separate bucket)
    unr = gaps[(~gaps["pop_resolved"]) & (gaps["need_z"] > 0.3)].sort_values("need_z", ascending=False)
    if state_sel != "(All states)":
        unr = unr[unr["state_ut"] == state_sel]
    if len(unr):
        with st.expander(f"⚠️ {len(unr)} high-need districts with UNRESOLVED population "
                         f"(post-2011 splits — ranked by need only, no per-capita number)"):
            u = unr[["district_name", "state_ut", "need_z", "n_fac"]].copy()
            u["need_z"] = u["need_z"].map(lambda x: f"{x:+.2f}")
            st.dataframe(u.rename(columns={"district_name": "District", "state_ut": "State",
                                           "need_z": "Need (z)", "n_fac": "Facilities"}),
                         hide_index=True, width='stretch')


# ===== TAB 2: DISTRICT DETAIL ==============================================
with tab_detail:
    opts = (f["district_name"] + "  —  " + f["state_ut"]).tolist()
    if not opts:
        opts = (gaps["district_name"] + "  —  " + gaps["state_ut"]).tolist()
    pick = st.selectbox("Choose a district", opts)
    d_name, d_state = [s.strip() for s in pick.split("—")]
    row = gaps[(gaps["district_name"] == d_name) & (gaps["state_ut"] == d_state)].iloc[0]

    badge, badge_help = CONF_BADGE[row["supply_confidence"]]
    st.subheader(f"{d_name}, {d_state}")
    st.markdown(f"**Supply confidence:** {badge} — {badge_help}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Priority rank", f"#{int(row['priority_rank'])}", help="Lower = higher risk")
    k2.metric("Population (2011)", f"{row['pop']:,.0f}" if pd.notna(row["pop"]) else "unresolved")
    k3.metric("Facilities found", int(row["n_fac"]),
              help=f"{int(row['n_hosp'])} hospitals · {int(row['n_obgyn'])} list obstetric/maternal care")
    k4.metric("Facilities / 100k",
              f"{row['fac_per_100k']:.2f}" if pd.notna(row["fac_per_100k"]) else "—")

    st.markdown("##### What's driving the need — NFHS-5 indicators (worst first)")
    st.caption("Source: NFHS-5 (National Family Health Survey, 2019-21) district fact sheets. "
               "⚠ = small unweighted sample (NFHS reliability caveat).")
    drivers = []
    for k, (lbl, bad) in LABELS.items():
        val = row.get(f"ind_{k}")
        if pd.isna(val):
            continue
        flagged = bool(row.get(f"indflag_{k}"))
        # severity for sorting: deficit indicators worse when low; burden worse when high
        sev = (100 - val) if bad == "low" else val
        drivers.append((sev, lbl, val, bad, flagged))
    drivers.sort(reverse=True)
    dd = pd.DataFrame(
        [{"Indicator": f"{l}{' ⚠' if fl else ''}",
          "Value": f"{v:.1f}%",
          "Concern": "low coverage" if bad == "low" else "high burden"} for _, l, v, bad, fl in drivers[:10]]
    )
    st.dataframe(dd, hide_index=True, width='stretch')

    # ----- cited facility evidence -----
    st.markdown("##### Facilities in this district — underlying evidence (cited)")
    fc = cites[(cites["district_name"] == d_name) & (cites["state_ut"] == d_state)]
    if not len(fc):
        st.warning("No facilities from the dataset were geocoded into this district. "
                   "This is the gap itself — but confirm against the confidence flag above before acting.")
    else:
        st.caption(f"{len(fc)} facility record(s). Each card shows the **text the dataset provides** "
                   f"(treat as claims to verify) and its source links.")
        mat_only = st.checkbox("Show only maternal/obstetric-capable facilities", value=False)
        view_fc = fc[fc["is_obgyn"]] if mat_only else fc
        for _, r in view_fc.head(40).iterrows():
            title = r["name"] if pd.notna(r["name"]) else "(unnamed facility)"
            tags = " · ".join([t for t in [r.get("facilityTypeId"), r.get("operatorTypeId"),
                                            "🤰 maternal" if r.get("is_obgyn") else None] if pd.notna(t) and t])
            with st.expander(f"{title}  —  {tags}"):
                if pd.notna(r.get("description")):
                    st.markdown(f"**Description (claim):** {str(r['description'])[:800]}")
                if pd.notna(r.get("capability")):
                    st.markdown(f"**Capability (claim):** {str(r['capability'])[:500]}")
                if pd.notna(r.get("specialties")):
                    st.markdown(f"**Specialties:** {str(r['specialties'])[:300]}")
                meta = []
                for c, lab in [("capacity", "beds"), ("numberDoctors", "doctors"),
                               ("yearEstablished", "est.")]:
                    if pd.notna(r.get(c)):
                        meta.append(f"{lab}: {r[c]}")
                if meta:
                    st.caption(" · ".join(meta))
                src = r.get("source_urls")
                if pd.notna(src):
                    st.markdown(f"**Sources:** {str(src)[:600]}")

    # ----- persist user actions -----
    st.markdown("##### Planner actions")
    saved = persistence.get_one(d_name, d_state)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        status = st.selectbox("Status", ["(none)", "Shortlisted", "Under review", "Dismissed"],
                              index=["(none)", "Shortlisted", "Under review", "Dismissed"].index(
                                  saved["status"]) if saved and saved.get("status") in
                                  ["Shortlisted", "Under review", "Dismissed"] else 0)
        reviewed = st.checkbox("Mark reviewed", value=bool(saved and saved.get("reviewed")))
    with cc2:
        note = st.text_area("Note / override rationale",
                            value=(saved or {}).get("note") or "", height=110,
                            placeholder="e.g. 'Confirmed 1 CHC exists via state portal — keep flagged, "
                                        "send mobile maternal unit'.")
    if st.button("💾 Save action", type="primary"):
        persistence.upsert(d_name, d_state,
                           status=None if status == "(none)" else status,
                           note=note, reviewed=1 if reviewed else 0)
        st.success("Saved.")
        st.cache_data.clear()


# ===== TAB 3: SHORTLIST ====================================================
with tab_short:
    st.subheader("Saved planner actions")
    rows = persistence.get_all()
    if not rows:
        st.info("Nothing saved yet. Use the **District detail** tab to shortlist districts and add notes.")
    else:
        sl = pd.DataFrame(rows)
        merged = sl.merge(gaps[["district_name", "state_ut", "gap", "pop", "fac_per_100k",
                                "supply_confidence", "priority_rank"]],
                          on=["district_name", "state_ut"], how="left")
        merged["conf"] = merged["supply_confidence"].map(lambda c: CONF_BADGE.get(c, ("?",))[0])
        st.dataframe(
            merged[["district_name", "state_ut", "status", "priority_rank", "gap",
                    "pop", "conf", "reviewed", "note", "updated_at"]]
            .rename(columns={"district_name": "District", "state_ut": "State", "status": "Status",
                             "priority_rank": "Rank", "gap": "Gap", "pop": "Population",
                             "reviewed": "Reviewed", "note": "Note", "updated_at": "Updated"}),
            hide_index=True, width='stretch')
        st.download_button("⬇️ Export shortlist (CSV)", merged.to_csv(index=False),
                           "medical_desert_shortlist.csv", "text/csv")
        st.caption(f"Persistence backend: **{persistence.BACKEND}** "
                   f"({'Lakebase/Postgres' if persistence.BACKEND == 'postgres' else 'local SQLite — set PG* env vars for Lakebase'})")


# ===== TAB 4: DATA READINESS (Track 4 lens) ================================
with tab_ready:
    rd = load_readiness()
    st.subheader("Is this data ready to plan on?")
    st.caption("*What must be fixed before planning can trust it?* — audited on the facility "
               "dataset **as delivered**, before any of our fixes.")
    if rd is None:
        st.info("Readiness scorecard not found. Run `python prepare_app_data.py` to generate it.")
    else:
        n = rd["n_rows"]
        g, ow, cb = rd["geocoding"], rd["ownership"], rd["column_bleed"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Facility records", f"{n:,}")
        m2.metric("Geocodable", f"{g['has_coords_pct']:.0f}%",
                  help=f"{g['no_coords']} rows have no coordinates; {g['out_of_range']} fall outside India.")
        m3.metric("Private-owned", f"{ow['private_pct']:.0f}%",
                  help="Coverage bias: the public PHC/CHC network is largely absent, so poor districts "
                       "look emptier than they are. Treat gaps as upper bounds.")
        m4.metric("Duplicate records",
                  f"{rd['duplicates']['cluster_id'] + rd['duplicates']['unique_id']}",
                  help="Rows sharing the dataset's own entity keys (cluster_id / unique_id) — "
                       "double-counts removed before scoring.")

        st.markdown("##### Field coverage — by the role each field plays for a planner")
        st.caption("The dataset is **rich in evidence text** (cite-able) but **thin in the structured "
                   "numbers planning needs** (capacity, doctor counts, clean ownership).")
        cov = pd.DataFrame(rd["coverage"]).rename(
            columns={"field": "Field", "role": "Planner role", "pct": "Coverage %"})
        st.dataframe(
            cov[["Field", "Planner role", "Coverage %"]], hide_index=True, width='stretch',
            column_config={"Coverage %": st.column_config.ProgressColumn(
                "Coverage %", min_value=0, max_value=100, format="%.1f%%")},
        )

        st.markdown("##### 🔴 The trust-breaker: column bleed")
        st.markdown(
            f"`operatorTypeId` should only ever say **private / public / government** — but "
            f"**{cb['n_bad']} rows** instead hold coordinates, URLs, UUIDs or specialty text. The source "
            "FDR text fields contain unescaped commas, quotes and newlines, so values shift into the "
            "wrong columns. **Until this is caught, every per-column number is suspect.** "
            "Examples (cited raw text):")
        if cb["examples"]:
            ex = pd.DataFrame(cb["examples"]).rename(columns={
                "unique_id": "Record ID", "name": "Facility",
                "leaked_value": "Value leaked into operatorTypeId"})
            st.dataframe(ex, hide_index=True, width='stretch')
        st.caption("Fix: strict quoted/multiline-aware parsing + a schema assertion that fails the "
                   "pipeline whenever operatorTypeId is not one of the three valid values.")

        st.markdown("##### What this app fixed before ranking anything")
        st.markdown(
            f"""
| # | Readiness problem (evidence) | Fix applied |
|---|---|---|
| 1 | No population denominator — a raw count can't define a "desert" | Joined Census-2011 population → **facilities per 100k** |
| 2 | Pincode-name geocoding manufactured false zeros | Re-geocoded **{g['has_coords_pct']:.0f}%** from lat/long (nearest post office) |
| 3 | **{ow['private_pct']:.0f}% private** coverage bias | Labelled supply *findable / largely-private* → gaps are **upper bounds** |
| 4 | Supply not matched to need ({rd['specialty']['maternal_pct']:.0f}% mention maternal care) | Score maternal need vs **obstetric-capable** supply only |
| 5 | {rd['duplicates']['cluster_id']} duplicate `cluster_id` rows | Deduped on the dataset's own entity key |
| 6 | NFHS small-sample `(x)` values look as solid as the rest | Carried reliability flags through → ⚠ in the district detail |
| 7 | 67 post-2011 split districts have no matched population | Flagged *population-unresolved*, ranked by need only (never a fake per-capita) |
            """)
        st.caption("Each fix is visible elsewhere in this app: confidence badges on the map, ⚠ on NFHS "
                   "values, the separate unresolved-population bucket, and cited facility text per district.")


# ===== TAB 5: AI ANALYST ==================================================
with tab_agent:
    st.subheader("🤖 AI Facility Analyst")
    st.caption("Ask natural-language questions about healthcare gaps. The agent gathers facility & NFHS "
               "evidence, then uses a Foundation Model to synthesize findings with citations.")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        agent_states = sorted(gaps["state_ut"].dropna().unique().tolist())
        agent_state = st.selectbox("State to analyse", agent_states, key="agent_state")
    with col_a2:
        agent_districts = ["(All districts)"] + sorted(
            gaps[gaps["state_ut"] == agent_state]["district_name"].dropna().unique().tolist()
        )
        agent_district = st.selectbox("District (optional)", agent_districts, key="agent_district")

    # Quick stats (instant, no LLM)
    if agent_state:
        with st.expander("📊 Quick data summary (instant)", expanded=True):
            st.markdown(get_quick_stats(cites, agent_state))

    # Preset question buttons
    st.markdown("**Common questions:**")
    presets = [
        "What are the biggest healthcare gaps in this area?",
        "Which specialties are underrepresented relative to health needs?",
        "What facilities exist for maternal care and are they sufficient?",
        "Recommend where to prioritize a new facility placement.",
        "What are the data quality issues I should be aware of?",
    ]
    preset_cols = st.columns(len(presets))
    icons = ["🏥", "🔬", "👶", "📍", "⚠️"]
    for i, (col, q) in enumerate(zip(preset_cols, presets)):
        with col:
            if st.button(icons[i], help=q, key=f"preset_{i}"):
                st.session_state["agent_q"] = q

    query = st.text_area(
        "Your question",
        value=st.session_state.get("agent_q", ""),
        placeholder="e.g., Are there enough pediatric specialists in this region?",
        key="agent_query_input"
    )

    if st.button("🔍 Analyse", type="primary"):
        if query.strip():
            dist_val = None if agent_district == "(All districts)" else agent_district
            with st.spinner("🤖 Analysing facility data and health indicators..."):
                result = run_agent(
                    query=query,
                    facilities_df=cites,
                    nfhs_df=gaps,
                    state=agent_state,
                    district=dist_val
                )
            st.markdown("---")
            st.markdown(result)

            if "agent_history" not in st.session_state:
                st.session_state.agent_history = []
            st.session_state.agent_history.append({
                "query": query, "state": agent_state,
                "district": dist_val, "result": result
            })
        else:
            st.warning("Please enter a question.")

    # History
    if st.session_state.get("agent_history"):
        st.markdown("---")
        st.markdown("#### Previous analyses")
        for entry in reversed(st.session_state.agent_history[-5:]):
            with st.expander(f"Q: {entry['query'][:80]}... ({entry['state']})"):
                st.markdown(entry["result"])
