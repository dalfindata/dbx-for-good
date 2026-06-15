"""
Facility Analyst Agent — AI-powered healthcare gap analysis.

Uses Databricks Foundation Model APIs (or OpenAI-compatible endpoint) to:
1. Analyze facility capabilities for a given district/state
2. Identify specific care gaps based on facility text data
3. Generate actionable recommendations for planners
4. Cite specific facilities as evidence for claims

All claims include uncertainty markers and source citations.
"""
from __future__ import annotations

import os
import json
import pandas as pd
from typing import Any, Literal, TypeAlias, cast


JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]
LLMClient: TypeAlias = (
    tuple[Literal["databricks"], Any]
    | tuple[Literal["openai"], str]
    | tuple[Literal["mock"], None]
)
CAPABILITY_SAMPLE_LIMIT = 5
SPECIALTY_SAMPLE_LIMIT = 3
SAMPLE_FACILITY_LIMIT = 5
TOP_EVIDENCE_LIMIT = 15
PROMPT_EVIDENCE_LIMIT = 10

# ─── LLM Client ──────────────────────────────────────────────────────

def get_llm_client() -> LLMClient:
    """Get LLM client - tries Databricks Foundation Models first, then OpenAI fallback."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return ("databricks", w)
    except Exception:
        pass

    # Fallback: OpenAI-compatible endpoint (for local dev)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DATABRICKS_TOKEN")
    if api_key:
        return ("openai", api_key)

    return ("mock", None)


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    """Call LLM via Databricks serving or fallback."""
    client_type, client = get_llm_client()

    if client_type == "databricks":
        try:
            # Use Databricks Foundation Model serving endpoint
            endpoint = os.getenv("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
            db_client = cast(Any, client)
            response = db_client.serving_endpoints.query(
                name=endpoint,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return cast(str, response.choices[0].message.content)
        except Exception as e:
            return f"⚠️ LLM call failed ({e}). Showing data-only analysis below."

    elif client_type == "openai":
        try:
            import openai
            oai_client = openai.OpenAI(
                api_key=cast(str, client),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            response = oai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return cast(str, response.choices[0].message.content)
        except Exception as e:
            return f"⚠️ LLM call failed ({e}). Showing data-only analysis below."

    else:
        # Mock mode for local testing without API keys
        return _mock_analysis(user_prompt)


def _mock_analysis(prompt: str) -> str:
    """Provide structured data-driven response when no LLM is available."""
    return (
        "🤖 **AI Agent (Offline Mode)**\n\n"
        "The AI agent requires a Databricks Foundation Model endpoint or OpenAI API key.\n\n"
        "**To enable:**\n"
        "- On Databricks: Automatically uses `databricks-meta-llama-3-3-70b-instruct`\n"
        "- Locally: Set `OPENAI_API_KEY` environment variable\n\n"
        "The data-driven analysis below is still available without AI."
    )


# ─── Agent Tools (Data Analysis Functions) ────────────────────────────

def _numeric_value(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors='coerce')
    if pd.notna(numeric):
        return float(numeric)
    return None


def analyze_district_facilities(
    facilities_df: pd.DataFrame, state: str, district: str | None = None
) -> JSONObject:
    """Gather facility evidence for a district/state for LLM context."""
    # Prefer state_normalized column when available (aligns with NFHS state names)
    if 'state_normalized' in facilities_df.columns:
        state_facs = facilities_df[
            facilities_df['state_normalized'].str.lower().str.strip() == state.lower().strip()
        ]
    else:
        state_facs = facilities_df[
            facilities_df['address_stateOrRegion'].str.lower().str.strip() == state.lower().strip()
        ]

    if district:
        # Try city-level match as proxy for district
        district_facs = state_facs[
            state_facs['address_city'].str.lower().str.strip() == district.lower().strip()
        ]
        if len(district_facs) == 0:
            district_facs = state_facs  # Fall back to state level

    else:
        district_facs = state_facs

    # Extract key evidence
    total = len(district_facs)
    if total == 0:
        return {"total": 0, "message": f"No facilities found in dataset for {state}"}

    # Capability summary
    capabilities: list[str] = []
    for _, row in district_facs.head(20).iterrows():
        cap = row.get('capabilities', '')
        if pd.notna(cap) and str(cap).strip():
            try:
                cap_list: object = json.loads(str(cap))
                if isinstance(cap_list, list):
                    capabilities.extend(str(item) for item in cap_list[:CAPABILITY_SAMPLE_LIMIT])
            except (json.JSONDecodeError, TypeError):
                capabilities.append(str(cap)[:100])

    # Specialty summary
    specialties: list[str] = []
    for _, row in district_facs.head(20).iterrows():
        spec = row.get('specialties', '')
        if pd.notna(spec) and str(spec).strip():
            try:
                spec_list: object = json.loads(str(spec))
                if isinstance(spec_list, list):
                    specialties.extend(str(item) for item in spec_list[:SPECIALTY_SAMPLE_LIMIT])
            except (json.JSONDecodeError, TypeError):
                specialties.append(str(spec)[:50])

    # Type breakdown
    type_counts: JSONObject = {
        str(facility_type): int(count)
        for facility_type, count in district_facs['facilityTypeId'].value_counts().items()
    }

    # Capacity
    caps = pd.to_numeric(district_facs['capacity'], errors='coerce').dropna()
    total_capacity = int(caps.sum()) if len(caps) > 0 else None

    # Sample facility names for citations
    sample_facilities: list[JSONValue] = [
        {
            "name": str(row.get('name', '')),
            "address_city": str(row.get('address_city', '')),
            "facilityTypeId": str(row.get('facilityTypeId', '')),
        }
        for _, row in district_facs[
            ['name', 'address_city', 'facilityTypeId']
        ].head(SAMPLE_FACILITY_LIMIT).iterrows()
    ]
    top_specialties: list[JSONValue] = [item for item in list(set(specialties))[:TOP_EVIDENCE_LIMIT]]
    top_capabilities: list[JSONValue] = [
        item for item in list(set(capabilities))[:TOP_EVIDENCE_LIMIT]
    ]

    return {
        "total": total,
        "type_breakdown": type_counts,
        "total_capacity": total_capacity,
        "top_specialties": top_specialties,
        "top_capabilities": top_capabilities,
        "sample_facilities": sample_facilities,
        "has_capacity_data": f"{len(caps)}/{total} facilities report capacity"
    }


def identify_care_gaps(
    facilities_df: pd.DataFrame,
    nfhs_df: pd.DataFrame,
    state: str,
    district: str | None = None,
) -> JSONObject:
    """Identify gaps between health needs (NFHS) and available facilities."""
    # Get NFHS data
    if district:
        nfhs_row = nfhs_df[nfhs_df['district_name'].str.lower().str.strip() == district.lower().strip()]
    else:
        nfhs_row = nfhs_df[nfhs_df['state_ut'].str.lower().str.strip() == state.lower().strip()]

    # Get facility evidence
    fac_evidence = analyze_district_facilities(facilities_df, state, district)

    nfhs_indicators: JSONObject = {}
    if len(nfhs_row) > 0:
        # Aggregate across all matching rows (e.g., multiple districts per state)
        row = nfhs_row.mean(numeric_only=True) if len(nfhs_row) > 1 else nfhs_row.iloc[0]
        possible_indicators = {
            "institutional_birth_pct": _numeric_value(row.get('institutional_birth_5y_pct')),
            "child_stunting_pct": _numeric_value(row.get('child_u5_who_are_stunted_height_for_age_18_pct')),
            "insurance_pct": _numeric_value(row.get('hh_member_covered_health_insurance_pct')),
            "anaemia_pct": _numeric_value(row.get('all_w15_49_who_are_anaemic_pct')),
            "vaccination_pct": _numeric_value(
                row.get('child_12_23m_fully_vaccinated_based_on_information_from_eit_pct')
            ),
            "c_section_pct": _numeric_value(row.get('births_delivered_by_c_section_5y_pct')),
        }
        nfhs_indicators = {
            key: value for key, value in possible_indicators.items() if value is not None
        }

    return {
        "facility_evidence": fac_evidence,
        "health_indicators": nfhs_indicators,
        "data_completeness": f"{len(nfhs_indicators)}/6 health indicators available"
    }


# ─── Agent Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Medical Desert Planner AI Agent — an expert healthcare access analyst for India.

Your role:
- Analyze healthcare facility data and district health indicators
- Identify gaps between available healthcare supply and community health needs
- Provide actionable, evidence-backed recommendations for planners
- Always cite specific data points and facilities as evidence

CRITICAL RULES:
1. NEVER present weak evidence as fact. Use qualifiers: "suggests", "may indicate", "based on available data"
2. ALWAYS communicate uncertainty when data is incomplete
3. ALWAYS cite specific facilities or indicators as evidence (e.g., "Based on [Facility X] in [City]...")
4. Keep recommendations actionable for non-technical planners
5. Flag data quality issues (e.g., "Only X of Y facilities report capacity")
6. Format output with clear sections and bullet points
7. If data is insufficient to draw conclusions, say so explicitly

Output format:
## Key Findings
- [Finding with evidence citation]

## Care Gaps Identified
- [Gap with supporting data]

## Confidence Assessment
- [What we know well vs. what's uncertain]

## Recommended Actions
- [Concrete next steps for planners]
"""


def build_analysis_prompt(query: str, gap_data: JSONObject) -> str:
    """Build the user prompt with data context for the LLM."""
    fac = cast(JSONObject, gap_data["facility_evidence"])
    indicators = cast(JSONObject, gap_data["health_indicators"])
    top_specialties = cast(list[str], fac.get('top_specialties', []))
    top_capabilities = cast(list[str], fac.get('top_capabilities', []))

    context = f"""## Data Context for Analysis

### Facility Supply
- Total facilities found: {fac.get('total', 0)}
- Type breakdown: {json.dumps(fac.get('type_breakdown', {}), indent=2)}
- Total reported capacity: {fac.get('total_capacity', 'Not reported for most facilities')}
- Capacity data availability: {fac.get('has_capacity_data', 'Unknown')}
- Top specialties: {', '.join(top_specialties[:PROMPT_EVIDENCE_LIMIT]) or 'None extracted'}
- Top capabilities: {', '.join(top_capabilities[:PROMPT_EVIDENCE_LIMIT]) or 'None extracted'}
- Sample facilities (for citations): {json.dumps(fac.get('sample_facilities', []), indent=2)}

### Health Indicators (NFHS-5)
"""
    for key, val in indicators.items():
        context += f"- {key}: {val}%\n"

    if not indicators:
        context += "- ⚠️ No NFHS indicators available for this area\n"

    context += f"\n### Data Completeness\n- {gap_data.get('data_completeness', 'Unknown')}\n"

    context += f"\n## Planner Question\n{query}\n"
    context += "\nProvide analysis following the format in your instructions. Cite specific facilities and data points."

    return context


# ─── Main Agent Interface ─────────────────────────────────────────────

def run_agent(query: str, facilities_df: pd.DataFrame, nfhs_df: pd.DataFrame,
              state: str, district: str | None = None) -> str:
    """
    Run the Facility Analyst Agent.

    Args:
        query: The planner's question
        facilities_df: Full facilities DataFrame
        nfhs_df: NFHS district indicators DataFrame
        state: State to analyze
        district: Optional district for focused analysis

    Returns:
        Formatted analysis string with citations and uncertainty markers
    """
    # Step 1: Gather evidence (tool calls)
    gap_data = identify_care_gaps(facilities_df, nfhs_df, state, district)

    # Step 2: Build prompt with evidence context
    user_prompt = build_analysis_prompt(query, gap_data)

    # Step 3: Call LLM for analysis
    analysis = call_llm(SYSTEM_PROMPT, user_prompt)

    # Step 4: Append data summary footer
    fac = cast(JSONObject, gap_data["facility_evidence"])
    footer = f"""

---
📊 **Data Sources Used:**
- Facilities: {fac.get('total', 0)} records from Virtue Foundation FDR
- Health Indicators: {gap_data.get('data_completeness', 'Unknown')} from NFHS-5 (2019-21)
- ⚠️ Facility data is web-scraped and GenAI-extracted — verify critical claims with ground surveys
"""
    return analysis + footer


def get_quick_stats(facilities_df: pd.DataFrame, state: str) -> str:
    """Get quick data summary without LLM call (for instant feedback)."""
    # Prefer state_normalized column when available (aligns with NFHS state names)
    if 'state_normalized' in facilities_df.columns:
        state_facs = facilities_df[
            facilities_df['state_normalized'].str.lower().str.strip() == state.lower().strip()
        ]
    else:
        state_facs = facilities_df[
            facilities_df['address_stateOrRegion'].str.lower().str.strip() == state.lower().strip()
        ]
    total = len(state_facs)
    if total == 0:
        return f"⚠️ No facilities found for state: {state}"

    types = state_facs['facilityTypeId'].value_counts()
    cities = state_facs['address_city'].nunique()
    has_cap = state_facs['capacity'].notna().sum()

    summary = f"""**Quick Stats for {state}:**
- 🏥 {total} facilities across {cities} cities
- Types: {', '.join(f'{k}: {v}' for k, v in types.head(3).items())}
- Capacity reported: {has_cap}/{total} ({has_cap/total*100:.0f}%)
"""
    return summary
