"""
Medical Desert Planner — DAIS Hackathon for Good 2026
Track 2: Where are the real, highest-risk gaps in care?

Combines:
- 10,088 healthcare facilities (FDR dataset)
- 706 district-level health indicators (NFHS-5)
- 160,721 pincode entries for geo-mapping
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from agent import run_agent, get_quick_stats

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Desert Planner",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_facilities():
    """Load facilities from Databricks or local CSV."""
    # Try Databricks SQL first (when deployed)
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import Disposition
        w = WorkspaceClient()
        warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "7701ddcec8332cb1")
        catalog = os.getenv("DATABRICKS_CATALOG", "databricks_virtue_foundation_dataset_dais_2026")
        schema = os.getenv("DATABRICKS_SCHEMA", "virtue_foundation_dataset")
        
        resp = w.statement_execution.execute_statement(
            statement=f"SELECT * FROM {catalog}.{schema}.facilities",
            warehouse_id=warehouse_id,
            wait_timeout="50s"
        )
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            return pd.DataFrame(resp.result.data_array, columns=cols)
    except Exception:
        pass
    
    # Fallback to local CSV
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "facilities_complete.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        st.error("Could not load facilities data. Place facilities_complete.csv in data/ or deploy to Databricks.")
        return pd.DataFrame()


@st.cache_data
def load_nfhs():
    """Load NFHS-5 district health indicators."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "7701ddcec8332cb1")
        catalog = os.getenv("DATABRICKS_CATALOG", "databricks_virtue_foundation_dataset_dais_2026")
        schema = os.getenv("DATABRICKS_SCHEMA", "virtue_foundation_dataset")
        
        resp = w.statement_execution.execute_statement(
            statement=f"SELECT * FROM {catalog}.{schema}.nfhs_5_district_health_indicators",
            warehouse_id=warehouse_id,
            wait_timeout="50s"
        )
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            return pd.DataFrame(resp.result.data_array, columns=cols)
    except Exception:
        pass
    
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "nfhs_health.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return pd.DataFrame()


@st.cache_data
def load_pincode():
    """Load India Post pincode directory."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import Disposition
        w = WorkspaceClient()
        warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID", "7701ddcec8332cb1")
        catalog = os.getenv("DATABRICKS_CATALOG", "databricks_virtue_foundation_dataset_dais_2026")
        schema = os.getenv("DATABRICKS_SCHEMA", "virtue_foundation_dataset")
        
        resp = w.statement_execution.execute_statement(
            statement=f"SELECT * FROM {catalog}.{schema}.india_post_pincode_directory",
            warehouse_id=warehouse_id,
            wait_timeout="50s"
        )
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            return pd.DataFrame(resp.result.data_array, columns=cols)
    except Exception:
        pass
    
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "pincode_directory.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return pd.DataFrame()


# ─── Data Processing ─────────────────────────────────────────────────
def normalize_state(state_name):
    """Normalize state names to match between datasets."""
    if pd.isna(state_name):
        return None
    state_map = {
        'nct of delhi': 'Delhi', 'nct delhi': 'Delhi', 'new delhi': 'Delhi',
        'west delhi': 'Delhi', 'chattisgarh': 'Chhattisgarh',
        'orissa': 'Odisha', 'tamilnadu': 'Tamil Nadu',
        'telengana': 'Telangana', 'u.p.': 'Uttar Pradesh',
        'up': 'Uttar Pradesh', 'jammu and kashmir': 'Jammu & Kashmir',
        'jammu & kashmir': 'Jammu & Kashmir',
        'andaman & nicobar islands': 'Andaman & Nicobar Islands',
        'dadra and nagar haveli and daman and diu': 'Dadra & Nagar Haveli and Daman & Diu',
    }
    normalized = state_map.get(str(state_name).lower().strip(), state_name)
    return normalized


def compute_desert_score(row):
    """
    Compute a Medical Desert Risk Score (0-100) for a district.
    Higher = more likely a healthcare desert.
    
    Components:
    - Low institutional birth rate (proxy for facility access)
    - High child stunting (proxy for chronic health neglect)
    - Low health insurance coverage
    - Low facility density (facilities per population proxy)
    - High anaemia prevalence
    """
    score = 0
    weights = []
    
    # Institutional births (lower = worse access)
    inst_birth = pd.to_numeric(row.get('institutional_birth_5y_pct'), errors='coerce')
    if pd.notna(inst_birth):
        score += (100 - inst_birth) * 0.25
        weights.append(0.25)
    
    # Child stunting (higher = worse health)
    stunting = pd.to_numeric(row.get('child_u5_who_are_stunted_height_for_age_18_pct'), errors='coerce')
    if pd.notna(stunting):
        score += stunting * 0.20
        weights.append(0.20)
    
    # Health insurance (lower = less access)
    insurance = pd.to_numeric(row.get('hh_member_covered_health_insurance_pct'), errors='coerce')
    if pd.notna(insurance):
        score += (100 - insurance) * 0.20
        weights.append(0.20)
    
    # Anaemia (higher = worse health)
    anaemia = pd.to_numeric(row.get('all_w15_49_who_are_anaemic_pct'), errors='coerce')
    if pd.notna(anaemia):
        score += anaemia * 0.20
        weights.append(0.20)
    
    # Child underweight (higher = worse)
    underweight = pd.to_numeric(row.get('child_u5_who_are_underweight_weight_for_age_18_pct'), errors='coerce')
    if pd.notna(underweight):
        score += underweight * 0.15
        weights.append(0.15)
    
    if not weights:
        return None
    
    # Normalize by actual weights used
    return round(score / sum(weights), 1)


def get_confidence_level(row):
    """Determine data confidence for a district assessment."""
    available = 0
    total = 5
    for col in ['institutional_birth_5y_pct', 'child_u5_who_are_stunted_height_for_age_18_pct',
                'hh_member_covered_health_insurance_pct', 'all_w15_49_who_are_anaemic_pct',
                'child_u5_who_are_underweight_weight_for_age_18_pct']:
        if pd.notna(row.get(col)) and str(row.get(col)).strip() != '':
            available += 1
    pct = available / total * 100
    if pct >= 80:
        return "High"
    elif pct >= 60:
        return "Medium"
    else:
        return "Low"


# ─── Persistence (Lakebase) ──────────────────────────────────────────
_DB_TYPE = None  # 'postgres' or 'sqlite'


def get_db_connection():
    """Get connection to Lakebase (or local SQLite fallback)."""
    global _DB_TYPE
    try:
        import psycopg
        host = os.getenv("LAKEBASE_DB_HOST")
        if host:
            conn = psycopg.connect(
                host=host,
                port=os.getenv("LAKEBASE_DB_PORT", "5432"),
                dbname=os.getenv("LAKEBASE_DB_NAME", "hackathon"),
                user=os.getenv("LAKEBASE_DB_USER", "admin"),
                password=os.getenv("LAKEBASE_DB_PASSWORD", "")
            )
            # Initialize schema on Lakebase too
            conn.execute("""CREATE TABLE IF NOT EXISTS planner_notes (
                id SERIAL PRIMARY KEY,
                district TEXT NOT NULL,
                state TEXT,
                note TEXT,
                priority TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS shortlist (
                id SERIAL PRIMARY KEY,
                district TEXT NOT NULL,
                state TEXT,
                desert_score REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.commit()
            _DB_TYPE = 'postgres'
            return conn
    except Exception:
        pass
    
    # Fallback to SQLite for local dev
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "planner.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS planner_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT NOT NULL,
        state TEXT,
        note TEXT,
        priority TEXT DEFAULT 'medium',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS shortlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT NOT NULL,
        state TEXT,
        desert_score REAL,
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    _DB_TYPE = 'sqlite'
    return conn


def _placeholder():
    """Return SQL placeholder based on active DB type."""
    return "%s" if _DB_TYPE == 'postgres' else "?"


def save_note(conn, district, state, note, priority):
    """Save planner note for a district."""
    try:
        ph = _placeholder()
        conn.execute(
            f"INSERT INTO planner_notes (district, state, note, priority) VALUES ({ph}, {ph}, {ph}, {ph})",
            (district, state, note, priority)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to save note: {e}")
        return False


def save_to_shortlist(conn, district, state, score, reason):
    """Add district to intervention shortlist."""
    try:
        ph = _placeholder()
        conn.execute(
            f"INSERT INTO shortlist (district, state, desert_score, reason) VALUES ({ph}, {ph}, {ph}, {ph})",
            (district, state, score, reason)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False


def get_notes(conn, district=None):
    """Retrieve planner notes."""
    if district:
        ph = _placeholder()
        return pd.read_sql(f"SELECT * FROM planner_notes WHERE district = {ph} ORDER BY created_at DESC", conn, params=(district,))
    return pd.read_sql("SELECT * FROM planner_notes ORDER BY created_at DESC", conn)


def get_shortlist(conn):
    """Retrieve intervention shortlist."""
    return pd.read_sql("SELECT * FROM shortlist ORDER BY desert_score DESC", conn)


# ─── Main App ────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🏥 Medical Desert Planner")
    st.markdown("**Track 2: Where are the real, highest-risk gaps in care?**")
    st.markdown("*Helping non-technical planners identify healthcare access gaps across India*")
    
    # Load data
    with st.spinner("Loading datasets..."):
        facilities = load_facilities()
        nfhs = load_nfhs()
    
    if facilities.empty or nfhs.empty:
        st.error("Failed to load data. Please check your connection.")
        return
    
    # Normalize states in facilities
    facilities['state_normalized'] = facilities['address_stateOrRegion'].apply(normalize_state)
    facilities['latitude'] = pd.to_numeric(facilities['latitude'], errors='coerce')
    facilities['longitude'] = pd.to_numeric(facilities['longitude'], errors='coerce')
    
    # Compute desert scores for NFHS districts
    nfhs['desert_score'] = nfhs.apply(compute_desert_score, axis=1)
    nfhs['confidence'] = nfhs.apply(get_confidence_level, axis=1)
    
    # Count facilities per state
    fac_per_state = facilities.groupby('state_normalized').size().reset_index(name='facility_count')
    
    # Merge with NFHS
    nfhs_with_fac = nfhs.merge(fac_per_state, left_on='state_ut', right_on='state_normalized', how='left')
    nfhs_with_fac['facility_count'] = nfhs_with_fac['facility_count'].fillna(0)
    
    # Database connection for persistence
    conn = get_db_connection()
    
    # ─── Sidebar ─────────────────────────────────────────────────────
    st.sidebar.header("🔍 Filters")
    
    states = sorted(nfhs['state_ut'].dropna().unique())
    selected_states = st.sidebar.multiselect("Filter by State", states, default=[])
    
    min_score = st.sidebar.slider("Minimum Desert Score", 0, 100, 40)
    
    confidence_filter = st.sidebar.multiselect(
        "Data Confidence",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    
    # Filter data
    filtered = nfhs_with_fac[
        (nfhs_with_fac['desert_score'] >= min_score) &
        (nfhs_with_fac['confidence'].isin(confidence_filter))
    ]
    if selected_states:
        filtered = filtered[filtered['state_ut'].isin(selected_states)]
    
    # ─── Overview Tab ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🗺️ Desert Map", "🔍 District Detail",
        "🤖 AI Analyst", "📋 Shortlist", "📝 Notes"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Facilities", f"{len(facilities):,}")
        col2.metric("Districts Analyzed", f"{len(nfhs):,}")
        col3.metric("High-Risk Districts", f"{len(filtered[filtered['desert_score'] >= 60]):,}")
        col4.metric("States Covered", f"{len(states)}")
        
        st.markdown("---")
        st.subheader("Desert Score Distribution")
        
        fig_hist = px.histogram(
            nfhs[nfhs['desert_score'].notna()],
            x='desert_score',
            nbins=30,
            color_discrete_sequence=['#C8102E'],
            labels={'desert_score': 'Medical Desert Risk Score'}
        )
        fig_hist.add_vline(x=60, line_dash="dash", line_color="red",
                          annotation_text="High Risk Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top 10 worst districts
        st.subheader("⚠️ Top 10 Highest-Risk Districts")
        top_10 = nfhs.nlargest(10, 'desert_score')[
            ['district_name', 'state_ut', 'desert_score', 'confidence',
             'institutional_birth_5y_pct', 'child_u5_who_are_stunted_height_for_age_18_pct',
             'hh_member_covered_health_insurance_pct']
        ].rename(columns={
            'district_name': 'District',
            'state_ut': 'State',
            'desert_score': 'Desert Score',
            'confidence': 'Data Confidence',
            'institutional_birth_5y_pct': 'Inst. Birth %',
            'child_u5_who_are_stunted_height_for_age_18_pct': 'Stunting %',
            'hh_member_covered_health_insurance_pct': 'Insurance %'
        })
        st.dataframe(top_10, use_container_width=True, hide_index=True)
        
        st.caption("⚠️ **Uncertainty Note:** Desert scores are computed from available NFHS-5 indicators. "
                   "Districts with 'Low' confidence have incomplete data — scores may not reflect true conditions.")
    
    with tab2:
        st.subheader("Healthcare Desert Risk Map")
        
        # Map facilities
        valid_fac = facilities[
            (facilities['latitude'] >= 6) & (facilities['latitude'] <= 37) &
            (facilities['longitude'] >= 68) & (facilities['longitude'] <= 98)
        ]
        
        fig_map = px.scatter_mapbox(
            valid_fac,
            lat='latitude',
            lon='longitude',
            color='facilityTypeId',
            hover_name='name',
            hover_data=['address_city', 'address_stateOrRegion', 'capacity'],
            zoom=4,
            center={"lat": 22.5, "lon": 82},
            mapbox_style="carto-positron",
            opacity=0.6,
            title="Healthcare Facilities Across India"
        )
        fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.caption("Each dot represents a healthcare facility. Sparse regions indicate potential medical deserts.")
        
        # State-level facility count
        st.subheader("Facility Count by State (Bottom 20 — potential deserts)")
        state_summary = fac_per_state.sort_values('facility_count', ascending=True).head(20)
        fig_bar = px.bar(
            state_summary,
            x='facility_count',
            y='state_normalized',
            orientation='h',
            color='facility_count',
            color_continuous_scale='RdYlGn',
            labels={'facility_count': 'Number of Facilities', 'state_normalized': 'State'}
        )
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.subheader("District Deep Dive")
        
        # Select district
        district_options = sorted(nfhs['district_name'].dropna().unique())
        selected_district = st.selectbox("Select a District", district_options)
        
        if selected_district:
            district_data = nfhs[nfhs['district_name'] == selected_district].iloc[0]
            state_name = district_data.get('state_ut', 'Unknown')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Desert Score", f"{district_data.get('desert_score', 'N/A')}/100")
            col2.metric("State", state_name)
            col3.metric("Data Confidence", district_data.get('confidence', 'Unknown'))
            
            st.markdown("---")
            st.markdown("#### Health Indicators")
            
            indicators = {
                'Institutional Births %': 'institutional_birth_5y_pct',
                'Child Stunting %': 'child_u5_who_are_stunted_height_for_age_18_pct',
                'Child Underweight %': 'child_u5_who_are_underweight_weight_for_age_18_pct',
                'Health Insurance %': 'hh_member_covered_health_insurance_pct',
                'Women Anaemic %': 'all_w15_49_who_are_anaemic_pct',
                'Vaccination Coverage %': 'child_12_23m_fully_vaccinated_based_on_information_from_eit_pct',
            }
            
            ind_data = []
            for label, col in indicators.items():
                val = pd.to_numeric(district_data.get(col), errors='coerce')
                nat_avg = pd.to_numeric(nfhs[col], errors='coerce').mean()
                if pd.notna(val):
                    ind_data.append({
                        'Indicator': label,
                        'District Value': f"{val:.1f}",
                        'National Avg': f"{nat_avg:.1f}",
                        'Status': '🔴 Worse' if (('Stunting' in label or 'Underweight' in label or 'Anaemic' in label) and val > nat_avg) or
                                  (('Institutional' in label or 'Insurance' in label or 'Vaccination' in label) and val < nat_avg) else '🟢 Better'
                    })
                else:
                    ind_data.append({
                        'Indicator': label,
                        'District Value': '⚠️ No Data',
                        'National Avg': f"{nat_avg:.1f}" if pd.notna(nat_avg) else 'N/A',
                        'Status': '⚪ Unknown'
                    })
            
            st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)
            
            # Facilities in this state
            st.markdown("---")
            st.markdown(f"#### Facilities in {state_name}")
            state_facs = facilities[facilities['state_normalized'] == state_name]
            
            if len(state_facs) > 0:
                st.info(f"Found **{len(state_facs)}** facilities in {state_name}")
                
                # Show facility types breakdown
                type_counts = state_facs['facilityTypeId'].value_counts()
                fig_pie = px.pie(values=type_counts.values, names=type_counts.index,
                                title=f"Facility Types in {state_name}")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # List top facilities
                top_facs = state_facs[['name', 'address_city', 'facilityTypeId', 'capacity', 'specialties']].head(10)
                st.dataframe(top_facs, use_container_width=True, hide_index=True)
            else:
                st.warning(f"⚠️ No facilities found in dataset for {state_name}. "
                          "This could indicate a data gap or state name mismatch.")
            
            # Evidence citation
            st.markdown("---")
            st.markdown("#### 📄 Evidence & Sources")
            st.caption(f"Data source: NFHS-5 (National Family Health Survey, 2019-21) for {selected_district}, {state_name}. "
                      f"Facility data from Virtue Foundation FDR dataset (web-scraped, GenAI-extracted). "
                      f"Desert score is a composite index — not a definitive ranking. "
                      f"Planners should verify with ground-truth surveys before intervention decisions.")
            
            # Add to shortlist
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                reason = st.text_input("Reason for shortlisting", key="shortlist_reason")
                if st.button("➕ Add to Shortlist"):
                    score = district_data.get('desert_score', 0)
                    if save_to_shortlist(conn, selected_district, state_name, score, reason):
                        st.success(f"Added {selected_district} to shortlist!")
            
            with col_b:
                note = st.text_area("Planner Note", key="planner_note")
                priority = st.selectbox("Priority", ["high", "medium", "low"])
                if st.button("💾 Save Note"):
                    if save_note(conn, selected_district, state_name, note, priority):
                        st.success("Note saved!")
    
    with tab4:
        st.subheader("🤖 AI Facility Analyst Agent")
        st.markdown("Ask questions about healthcare gaps, facility capabilities, or get recommendations for a specific area.")
        
        # Agent configuration
        col_agent1, col_agent2 = st.columns(2)
        with col_agent1:
            agent_state = st.selectbox("State to analyze", states, key="agent_state")
        with col_agent2:
            agent_district_options = ["(All districts in state)"] + sorted(
                nfhs[nfhs['state_ut'] == agent_state]['district_name'].dropna().unique()
            ) if agent_state else ["(Select a state first)"]
            agent_district = st.selectbox("District (optional)", agent_district_options, key="agent_district")
        
        # Quick stats (no LLM call)
        if agent_state:
            with st.expander("📊 Quick Data Summary (instant)", expanded=True):
                quick = get_quick_stats(facilities, agent_state)
                st.markdown(quick)
        
        # Single input: preset buttons OR custom text
        st.markdown("**Quick questions:**")
        preset_questions = [
            "What are the biggest healthcare gaps in this area?",
            "Which specialties are underrepresented relative to health needs?",
            "What facilities exist for maternal care and are they sufficient?",
            "Recommend where to prioritize a new facility placement.",
            "What are the data quality issues I should be aware of?",
        ]
        
        # Use buttons for presets
        preset_cols = st.columns(len(preset_questions))
        for i, (col, q) in enumerate(zip(preset_cols, preset_questions)):
            with col:
                if st.button(f"{'🏥❓👶📍⚠️'[i]}", help=q, key=f"preset_{i}"):
                    st.session_state['agent_query_value'] = q
        
        custom_query = st.text_area(
            "Ask the AI Analyst",
            value=st.session_state.get('agent_query_value', ''),
            placeholder="e.g., Are there enough pediatric specialists in this region?",
            key="agent_query"
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if custom_query.strip():
                district_val = None if agent_district == "(All districts in state)" else agent_district
                
                with st.spinner("🤖 Agent analyzing facility data and health indicators..."):
                    result = run_agent(
                        query=custom_query,
                        facilities_df=facilities,
                        nfhs_df=nfhs,
                        state=agent_state,
                        district=district_val
                    )
                
                st.markdown("---")
                st.markdown(result)
                
                # Store in session for reference
                if 'agent_history' not in st.session_state:
                    st.session_state.agent_history = []
                st.session_state.agent_history.append({
                    "query": custom_query,
                    "state": agent_state,
                    "district": district_val,
                    "result": result
                })
            else:
                st.warning("Please enter a question for the agent.")
        
        # Show conversation history
        if 'agent_history' in st.session_state and st.session_state.agent_history:
            st.markdown("---")
            st.markdown("#### Previous Analyses")
            for i, entry in enumerate(reversed(st.session_state.agent_history[-5:])):
                with st.expander(f"Q: {entry['query'][:80]}... ({entry['state']})"):
                    st.markdown(entry['result'])
    
    with tab5:
        st.subheader("📋 Intervention Shortlist")
        shortlist = get_shortlist(conn)
        if len(shortlist) > 0:
            st.dataframe(shortlist, use_container_width=True, hide_index=True)
            
            # Export
            csv = shortlist.to_csv(index=False)
            st.download_button("📥 Export Shortlist (CSV)", csv, "shortlist.csv", "text/csv")
        else:
            st.info("No districts shortlisted yet. Use the District Detail tab to add districts.")
    
    with tab6:
        st.subheader("📝 Planner Notes")
        notes = get_notes(conn)
        if len(notes) > 0:
            st.dataframe(notes, use_container_width=True, hide_index=True)
        else:
            st.info("No notes yet. Add notes from the District Detail tab.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Medical Desert Planner** | DAIS Hackathon for Good 2026 | "
        "Data: Virtue Foundation FDR + NFHS-5 + India Post | "
        "Built with Databricks + Streamlit"
    )


if __name__ == "__main__":
    main()
