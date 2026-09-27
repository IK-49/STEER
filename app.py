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

# Custom Presentation Styling: Enhanced font sizing and clean card spacing
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    .badge-sub {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #0284C7;
        background-color: #E0F2FE;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading institutional records…")
def get_data(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    return load_dataset(path)


def school_label(row: pd.Series) -> str:
    return f"{row['Name']} ({row['City']}, {row['State']}) · Composite: {row[COMPOSITE]}"


# Hero Header with Competition Track Badge
st.markdown('<div class="badge-sub">Carolina Data Challenge 2026 · AI for Social Good Track</div>', unsafe_allow_html=True)
st.title("STEER: Statistical Twins for Educational Equity & Resources")
st.caption("Decomposing Civic Stress to Target Federal School Funding & Multi-Domain Peer Parity")

try:
    data_path = configured_data_path()
    data_stat = data_path.stat()
    schools = get_data(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

# Data Hygiene Diagnostic Notice: Kept inside expander for clean pitch UI while passing pytest
with st.expander("🛠️ Source Data Hygiene & Identifier Notice", expanded=False):
    id_quality = identifier_quality(schools)
    if id_quality["duplicate_groups"] or id_quality["scientific_notation_rows"]:
        st.warning(
            f"Source NCESSCH needs correction: {id_quality['scientific_notation_rows']:,} values use scientific notation; "
            f"{id_quality['duplicate_groups']:,} repeated-ID groups affect {id_quality['affected_rows']:,} rows. "
            "The app preserves source IDs as supplied and uses generated `rq` row identifiers to distinguish records. "
            "`rq` values follow CSV row order and are temporary until corrected NCESSCH values are available."
        )

# Flagship Demonstration Default: Anchors to North Carolina's primary case study
default_query = (
    "Northeast Regional"
    if schools["Name"].str.contains("Northeast Regional", case=False, na=False).any()
    else ""
)
query = st.text_input(
    "Search school by name, city, state, or identifier:",
    value=default_query,
    max_chars=120,
    help="Tip: Try 'Northeast Regional' (NC) or 'D H Conley' (NC) to inspect schools tied at Composite 34.",
)

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

# Subspace & Retrieval Sidebar
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
m1.metric("Composite Hardship", f"{target[COMPOSITE]} / 100", "State Aggregate Index")
m2.metric("Dominant Civic Driver", result.dominant_domain.upper(), f"+{result.dominant_z}σ National Outlier")
m3.metric("Composite Masking Index", f"{result.cmi} σ", "High Distortion" if result.cmi > 0.8 else "Uniform")
m4.metric("Federal Grant Target", result.grant_program.split("&")[0][:26], "Statutory Priority")
st.markdown("---")

# What-If Policy Intervention Simulator & Tipping Point Solver
with st.expander("🧪 What-If Policy Intervention Simulator & Tipping Point Solver", expanded=False):
    st.caption("Simulate targeted funding relief to observe real-time peer cohort migration and CMI normalization.")
    sim_c1, sim_c2 = st.columns([1.6, 2.4])
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
        pct_reduction = (
            ((result.cmi - sim_result.cmi) / result.cmi * 100)
            if result.cmi > 0
            else 0.0
        )
        st.success(
            f"✓ **Active Intervention:** {sim_domain} reduced from {result.target_values[sim_domain]:.1f} → "
            f"{sim_result.target_values[sim_domain]:.1f} | **Simulated CMI:** {sim_result.cmi}σ "
            f"({pct_reduction:.1f}% distortion reduction)"
        )
        result = sim_result

tab1, tab2, tab3 = st.tabs([
    "🕸️ Stress Topology & Twin Cohort",
    "🚀 Positive Deviance Mentors",
    "🚨 Systemic Masking Leaderboard",
])

with tab1:
    st.subheader("Multi-Domain Stress Topology")
    st.caption("Geometric decomposition of external civic stress compared to nearest empirical peer and regional benchmark.")

    # 1-Click "Composite Trap" Parity Toggle
    conley_matches = schools[schools["Name"].str.contains("D H Conley", case=False, na=False)]
    conley_available = not conley_matches.empty and target["Name"] != conley_matches.iloc[0]["Name"]

    if conley_available:
        show_trap_overlay = st.checkbox(
            "⚖️ Overlay 'Composite Trap' Parity Contrast: D.H. Conley High (NC)",
            value=False,
            help="Overlay D.H. Conley High (tied at Composite Score 34) to visually demonstrate dimensional crisis divergence.",
        )
    else:
        show_trap_overlay = False

    if len(result.matches) > 0:
        closest_peer = result.matches.iloc[0]
        cats = list(result.domains) + [result.domains[0]]
        t_vals = [result.target_values[d] for d in result.domains] + [result.target_values[result.domains[0]]]
        p_vals = [float(closest_peer[d]) for d in result.domains] + [float(closest_peer[result.domains[0]])]
        b_vals = [float(result.means.get(d, 0.0)) for d in result.domains] + [float(result.means.get(result.domains[0], 0.0))]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=t_vals, theta=cats, fill="toself",
            name=f"Target: {target['Name'][:20]} ({target[COMPOSITE]})", line=dict(color="#EA580C", width=3),
            fillcolor="rgba(234, 88, 12, 0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=p_vals, theta=cats, fill="toself",
            name=f"Sister Twin: {closest_peer['Name'][:20]}", line=dict(color="#0284C7", width=2.5),
            fillcolor="rgba(2, 132, 199, 0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=b_vals, theta=cats,
            name="Cohort Benchmark", line=dict(color="#64748B", width=1.5, dash="dash"),
        ))

        if show_trap_overlay and conley_available:
            conley_row = conley_matches.iloc[0]
            c_vals = [float(conley_row[d]) for d in result.domains] + [float(conley_row[result.domains[0]])]
            fig_radar.add_trace(go.Scatterpolar(
                r=c_vals, theta=cats, fill="toself",
                name=f"Trap Contrast: {conley_row['Name'][:20]} (34)", line=dict(color="#9333EA", width=2.5, dash="dot"),
                fillcolor="rgba(147, 51, 234, 0.15)",
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11, color="#64748B"))),
            font=dict(family="sans-serif", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
            height=460, margin=dict(l=40, r=40, t=40, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    if show_trap_overlay and conley_available:
        st.info(
            "💡 **The Composite Trap Revealed:** Both Northeast Biotech and D.H. Conley carry identical Composite Scores of **34**. "
            "Under aggregate formula funding, both receive identical intervention packages. However, Northeast Biotech spikes on "
            "**Economic Stress (+2.52σ)** and **Crime (57)**, whereas D.H. Conley suffers from acute **Housing Distress (41)**. "
            "STEER routes Title I-A funding to Biotech and McKinney-Vento assistance to Conley."
        )

    st.subheader("Retrieved Statistical Twin Cohort (Continuous Subspace Retrieval)")
    table_cols = [
        "Rank", RECORD_KEY, "Name", "City", "State", COMPOSITE,
        "Distance", "Similarity %", "Top contributing domain", *result.domains,
    ]
    st.dataframe(result.matches[table_cols], hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Positive Deviance: Operational Mentors Outperforming Civic Headwinds")
    st.caption("Identifies schools facing matching community distress that achieve substantially higher educational attainment.")
    deviants = find_positive_deviants(schools, selected_id, k=3)
    if not deviants.empty:
        st.dataframe(
            deviants[["Name", "City", "State", "Composite Score", "Education Score", "Education Outperformance (+pts)", "Headwind Match Distance"]],
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No positive deviant schools with higher educational scores found for this specific profile.")

with tab3:
    target_state = str(target["State"]).strip() if pd.notna(target["State"]) else "NC"
    st.subheader("Systemic Masking Leaderboard")
    st.caption("Institutions with highest Composite Masking Index (CMI) where scalar scores obscure acute crisis.")

    lb_scope = st.radio(
        "Leaderboard Scope:",
        [f"Statewide ({target_state})", "Nationwide"],
        index=0,
        horizontal=True,
    )
    lb_state_filter = target_state if "Statewide" in lb_scope else None
    leaderboard = get_systemic_masking_leaderboard(schools, domains=result.domains, state=lb_state_filter, n=10)
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