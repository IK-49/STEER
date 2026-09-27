"""STEER: Statistical Twins for Educational Equity & Resources."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    COMPOSITE,
    DOMAINS,
    IDENTIFIER,
    RECORD_KEY,
    configured_data_path,
    identifier_quality,
    load_dataset,
)
from similarity import (
    SearchError,
    find_positive_deviants,
    find_similar_schools,
    get_systemic_masking_leaderboard,
    synthesize_llm_grant_narrative,
)

st.set_page_config(
    page_title="STEER | School Benchmarking",
    layout="wide",
    page_icon="🏫",
)


@st.cache_data(show_spinner="Loading institutional records…")
def get_data(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    return load_dataset(path)


def school_label(row: pd.Series) -> str:
    return f"{row['Name']} ({row['City']}, {row['State']}) · Composite: {row[COMPOSITE]}"


st.title("STEER: Statistical Twins for Educational Equity & Resources")
st.caption("Decomposing Civic Stress to Target Federal School Funding & Multi-Domain Peer Parity")

try:
    data_path = configured_data_path()
    data_stat = data_path.stat()
    schools = get_data(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

# Required by automated test suite to verify data hygiene
id_quality = identifier_quality(schools)
if id_quality["duplicate_groups"] or id_quality["scientific_notation_rows"]:
    st.warning(
        f"Source NCESSCH needs correction: {id_quality['scientific_notation_rows']:,} values use scientific notation; "
        f"{id_quality['duplicate_groups']:,} repeated-ID groups affect {id_quality['affected_rows']:,} rows. "
        "The app preserves source IDs as supplied and uses generated `rq` row identifiers to distinguish records. "
        "`rq` values follow CSV row order and are temporary until corrected NCESSCH values are available."
    )

default_query = "Biotech" if schools["Name"].str.contains("Biotech", case=False, na=False).any() else ""
query = st.text_input("Search school by name, city, state, or identifier:", value=default_query, max_chars=120)

if not query.strip():
    st.caption("Enter a search term to see matching schools.")
    st.stop()

needle = query.strip().casefold()
search_columns = (IDENTIFIER, "Name", "City", "State")
mask = pd.Series(False, index=schools.index)
for column in search_columns:
    mask |= schools[column].astype("string").str.casefold().str.contains(needle, regex=False, na=False)
found = schools.loc[mask].head(100)

if found.empty:
    st.warning("No matching institutions found. Try another query.")
    st.stop()

selected_id = st.selectbox(
    "Select Target School:",
    found[RECORD_KEY].tolist(),
    format_func=lambda value: school_label(schools.loc[schools[RECORD_KEY].eq(value)].iloc[0]),
)
target = schools.loc[schools[RECORD_KEY].eq(selected_id)].iloc[0]

with st.sidebar:
    st.header("Search & Subspace Controls")
    active_domains = st.multiselect("Active Subspace Dimensions:", DOMAINS, default=list(DOMAINS))
    states = sorted(schools["State"].dropna().astype(str).str.strip().loc[lambda v: v.ne("")].unique())
    state_filter = st.selectbox("Geographic Scope:", ["Nationwide", *states])
    k = st.slider("Number of Statistical Twins (k):", min_value=3, max_value=10, value=5)

state_value = None if state_filter == "Nationwide" else state_filter

try:
    result = find_similar_schools(
        schools, selected_id, active_domains, k=k, state=state_value,
    )
except SearchError as exc:
    st.warning(str(exc))
    st.stop()

# Top KPI Metric Cards
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Composite Hardship", f"{target[COMPOSITE]} / 100", "State Aggregate")
m2.metric("Dominant Civic Driver", result.dominant_domain.upper(), f"+{result.dominant_z}σ Outlier")
m3.metric("Composite Masking Index", f"{result.cmi} σ", "High Distortion" if result.cmi > 0.8 else "Uniform")
m4.metric("Federal Grant Target", result.grant_program.split("&")[0][:26], "Statutory Priority")
st.markdown("---")

# What-If Policy Intervention Simulator & Tipping Point Solver
with st.expander("🧪 What-If Policy Intervention Simulator & Tipping Point Solver", expanded=False):
    st.caption("Simulate targeted grant interventions to observe real-time peer cohort migration and CMI reduction.")
    sim_c1, sim_c2 = st.columns([1.5, 2.5])
    with sim_c1:
        default_sim_idx = active_domains.index(result.dominant_domain) if result.dominant_domain in active_domains else 0
        sim_domain = st.selectbox("Intervention Domain:", active_domains, index=default_sim_idx)
        st.markdown(
            f"🎯 **Tipping Point:** `{result.tipping_point_reduction} pts` reduction in **{result.dominant_domain}** "
            f"stabilizes CMI below **1.0σ**."
        )
    with sim_c2:
        relief_pts = st.slider(
            f"Simulate Relief for {sim_domain} (-pts):",
            min_value=0.0, max_value=30.0, value=0.0, step=0.5,
        )

    if relief_pts > 0:
        simulated_inputs = dict(result.target_values)
        simulated_inputs[sim_domain] = max(0.0, simulated_inputs[sim_domain] - relief_pts)
        sim_result = find_similar_schools(
            schools, selected_id, active_domains, k=k, state=state_value, target_values=simulated_inputs,
        )
        st.info(
            f"Simulation Active: {sim_domain} reduced from {result.target_values[sim_domain]:.1f} to "
            f"{sim_result.target_values[sim_domain]:.1f} | New CMI: {sim_result.cmi}σ (Δ {sim_result.cmi - result.cmi:+.2f}σ)"
        )
        result = sim_result

tab1, tab2, tab3 = st.tabs([
    "🕸️ Stress Topology & Twin Cohort",
    "🚀 Positive Deviance Mentors",
    "🚨 Systemic Masking Leaderboard",
])

with tab1:
    st.subheader("Multi-Domain Stress Topology: Target vs. Twin vs. Cohort Median")
    if len(result.matches) > 0:
        closest_peer = result.matches.iloc[0]
        cats = list(result.domains) + [result.domains[0]]
        t_vals = [result.target_values[d] for d in result.domains] + [result.target_values[result.domains[0]]]
        p_vals = [float(closest_peer[d]) for d in result.domains] + [float(closest_peer[result.domains[0]])]
        b_vals = [float(result.means.get(d, 0.0)) for d in result.domains] + [float(result.means.get(result.domains[0], 0.0))]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=t_vals, theta=cats, fill="toself",
            name=f"Target: {target['Name'][:22]}", line=dict(color="#EA580C", width=2.5),
            fillcolor="rgba(234, 88, 12, 0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=p_vals, theta=cats, fill="toself",
            name=f"Twin: {closest_peer['Name'][:22]}", line=dict(color="#0284C7", width=2.5),
            fillcolor="rgba(2, 132, 199, 0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=b_vals, theta=cats,
            name="Cohort Benchmark", line=dict(color="#94A3B8", width=1.5, dash="dash"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=440, margin=dict(l=40, r=40, t=30, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Retrieved Statistical Twin Cohort")
    table_cols = [
        "Rank", RECORD_KEY, "Name", "City", "State", COMPOSITE,
        "Distance", "Similarity %", "Top contributing domain", *result.domains,
    ]
    st.dataframe(result.matches[table_cols], hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Positive Deviance: Peer Institutions with Superior Education Attainment")
    st.caption("Identifies schools facing matching community distress that achieve higher educational outcomes.")
    deviants = find_positive_deviants(schools, selected_id, k=3)
    if not deviants.empty:
        st.dataframe(
            deviants[["Name", "City", "State", "Composite Score", "Education Score", "Education Outperformance (+pts)", "Headwind Match Distance"]],
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No positive deviant schools with higher educational scores found for this specific profile.")

with tab3:
    st.subheader(f"Systemic Masking Leaderboard ({state_filter})")
    st.caption("Institutions with highest Composite Masking Index (CMI) where scalar scores obscure acute crisis.")
    leaderboard = get_systemic_masking_leaderboard(schools, domains=result.domains, state=state_value, n=10)
    st.dataframe(leaderboard, hide_index=True, use_container_width=True)

# Downstream Policy Actionable Artifact
st.markdown("---")
st.subheader("📄 Automated Statutory Grant Evidence Brief & Grounded LLM Copilot")

col_brief_btn, col_brief_ai = st.columns([1, 1])
with col_brief_btn:
    st.download_button(
        label="📥 Download Legal Grant Application Brief (.txt)",
        data=result.brief_text,
        file_name=f"STEER_Grant_Brief_{target['Name'].replace(' ', '_')}.txt",
        mime="text/plain",
    )

brief_display = result.brief_text
if st.button("✨ Synthesize Grounded Executive Narrative (LLM Copilot)"):
    with st.spinner("Synthesizing grounded narrative from empirical parameters…"):
        brief_display = synthesize_llm_grant_narrative(result, target)

st.text_area("Live Brief Preview:", brief_display, height=220)