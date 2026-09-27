"""
STEER: Institutional Community Profile Explorer.

Authors: Anish Velagapudi, Izad Khokhar, Aurick Smart
AI Attribution: Initial application scaffolding, layout templates, and widget bindings
were generated with assistance from AI tools (GitHub Copilot / LLM). Record filtering,
out-of-sample scaling coordination, simulation workflows, chart geometry fallbacks,
and presentation layouts were refactored and customized by the project team.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    COMPOSITE,
    DOMAINS,
    RECORD_KEY,
    configured_data_path,
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
    page_title="STEER | School Profile Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading institutional records...")
def get_data(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    return load_dataset(path)


def school_label(row: pd.Series) -> str:
    return str(row.get("Display name") or f"{row['Name']} ({row['City']}, {row['State']})")


st.title("STEER: Explore School Community Profiles")
st.caption("Descriptive school community profile comparison prototype")

try:
    data_path = configured_data_path()
    data_stat = data_path.stat()
    schools = get_data(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

with st.expander("Source data and record identity", expanded=False):
    st.write(
        "Federal school identifiers (NCESSCH) are not loaded due to source formatting issues. "
        "Records are identified and excluded from their own comparison pool using "
        "school name, city, state, and district (county is used when district is missing)."
    )

default_query = (
    "Northeast Regional"
    if schools["Name"].str.contains("Northeast Regional", case=False, na=False).any()
    else ""
)
query = st.text_input(
    "Search school by name, city, or state:",
    value=default_query,
    max_chars=120,
    help="Tip: Try 'Northeast Regional' (NC) or 'D H Conley' (NC) to inspect schools tied at Composite 34.",
)

if not query.strip():
    st.caption("Enter a search term to see matching schools.")
    st.stop()

needle = query.strip().casefold()
search_columns = ("Name", "City", "State")
mask = pd.Series(False, index=schools.index)
for column in search_columns:
    mask |= schools[column].astype("string").str.casefold().str.contains(needle, regex=False, na=False)

total_matches = int(mask.sum())
found = schools.loc[mask].head(250)

if found.empty:
    st.warning("No matching institutions found. Try another query.")
    st.stop()

suggestion_keys = found[RECORD_KEY].head(5).tolist()
st.caption(f"{total_matches:,} matching records · choose a suggestion")
selected_id = st.pills(
    "Matching schools",
    suggestion_keys,
    selection_mode="single",
    default=suggestion_keys[0],
    format_func=lambda value: school_label(schools.loc[schools[RECORD_KEY].eq(value)].iloc[0]),
    key=f"school_suggestion_{query.casefold()}",
    label_visibility="collapsed",
    width="stretch",
)

if selected_id is None:
    st.info("Choose a matching school to show its profile comparison.")
    st.stop()

target = schools.loc[schools[RECORD_KEY].eq(selected_id)].iloc[0]

with st.sidebar:
    st.header("Search & Subspace Controls")
    active_domains = st.multiselect(
        "Active Subspace Dimensions:",
        DOMAINS,
        default=list(DOMAINS),
    )
    states = sorted(
        schools["State"].dropna().astype(str).str.strip().loc[lambda v: v.ne("")].unique()
    )
    state_filter = st.selectbox("Geographic Scope:", ["Nationwide", *states])
    k = st.slider("Number of comparison schools (k):", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.subheader("Model Configuration")
    user_api_key = st.text_input(
        "Gemini API Key (Google AI Studio):",
        type="password",
        help="Optional API key from aistudio.google.com. Can also be set in .streamlit/secrets.toml.",
    )
    if Path("pages/1_Story.py").exists():
        st.page_link("pages/1_Story.py", label="View Project Story")

state_value = None if state_filter == "Nationwide" else state_filter

try:
    result = find_similar_schools(
        schools, selected_id, active_domains, k=k, state=state_value,
    )
except SearchError as exc:
    st.warning(str(exc))
    st.stop()

# What-If Policy Simulation
simulated_result = None
with st.expander("What-If Policy Intervention Simulator and Tipping Point Solver", expanded=False):
    st.caption("Explore how a hypothetical score change affects the descriptive domain-profile spread.")
    sim_c1, sim_c2 = st.columns([1.6, 2.4])
    with sim_c1:
        default_sim_idx = active_domains.index(result.dominant_domain) if result.dominant_domain in active_domains else 0
        sim_domain = st.selectbox("Intervention Domain:", active_domains, index=default_sim_idx)
        if result.tipping_point_reduction is None:
            st.markdown(f"No reduction from current **{result.dominant_domain}** score brings spread to 1.0σ or less.")
        elif result.tipping_point_reduction == 0:
            st.markdown("The current profile already has a spread of 1.0σ or less.")
        else:
            st.markdown(
                f"A hypothetical `{result.tipping_point_reduction} point` reduction in **{result.dominant_domain}** "
                "would bring the spread metric to 1.0σ or less."
            )
    with sim_c2:
        relief_pts = st.slider(
            f"Simulate Relief for {sim_domain} (-pts):",
            min_value=0.0, max_value=30.0, value=0.0, step=0.5,
        )

    if relief_pts > 0:
        if sim_domain not in result.target_values:
            st.info(f"The selected school has no {sim_domain} score to adjust.")
        else:
            simulated_inputs = dict(result.target_values)
            simulated_inputs[sim_domain] = max(0.0, simulated_inputs[sim_domain] - relief_pts)
            simulated_result = find_similar_schools(
                schools, selected_id, active_domains, k=k, state=state_value, target_values=simulated_inputs,
            )
            st.success(
                f"Hypothetical score adjustment: {sim_domain} changed from {result.target_values[sim_domain]:.1f} → "
                f"{simulated_result.target_values[sim_domain]:.1f} | Domain-profile spread: {simulated_result.cmi}σ "
                f"(was {result.cmi}σ before adjustment)"
            )

active_display_result = simulated_result if simulated_result is not None else result

# Metric Cards
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Composite score", f"{target[COMPOSITE]:.1f} / 100", "Source value")
m2.metric("Highest relative domain", active_display_result.dominant_domain.upper(), f"{active_display_result.dominant_z:+.2f}σ in selected cohort")
m3.metric("Domain-profile spread", f"{active_display_result.cmi} σ", "Standard deviation of domain z-scores")
m4.metric(
    "Program area to research",
    textwrap.shorten(active_display_result.grant_program.split("&")[0].strip(), width=26, placeholder="..."),
    "Not an eligibility finding",
)
st.markdown("---")
st.caption(
    f"Scope: {active_display_result.scope_description} · {len(active_display_result.domains)} selected domains · "
    f"{active_display_result.excluded_missing_count:,} in-scope candidate records omitted for missing selected scores."
)
if active_display_result.excluded_target_domains:
    st.info("The selected school has missing scores for: " + ", ".join(active_display_result.excluded_target_domains) + ". These domains were omitted.")

tab1, tab2, tab3 = st.tabs([
    "Profile & Nearest Schools",
    "Profile Comparisons",
    "Domain Spread Leaderboard",
])

with tab1:
    st.subheader("Multi-domain profile")
    st.caption("Compare selected source scores with nearby points in the standardized feature space.")

    conley_matches = schools[
        schools["Name"].str.contains("D H Conley", case=False, na=False)
        & schools["State"].astype(str).str.strip().str.casefold().eq("nc")
    ]
    conley_available = (
        not conley_matches.empty
        and target[RECORD_KEY] != conley_matches.iloc[0][RECORD_KEY]
        and str(target["State"]).strip().upper() == "NC"
    )

    if conley_available:
        show_trap_overlay = st.checkbox(
            "Compare with (D.H. Conley High, NC) (Example)",
            value=False,
            help="Show a second named school profile for descriptive comparison.",
        )
    else:
        show_trap_overlay = False

    if len(active_display_result.matches) > 0:
        closest_peer = active_display_result.matches.iloc[0]

        if len(active_display_result.domains) < 3:
            bar_df = pd.DataFrame({
                "Domain": list(active_display_result.domains),
                "Target": [active_display_result.target_values[d] for d in active_display_result.domains],
                "Nearest Peer": [float(closest_peer[d]) for d in active_display_result.domains],
                "Cohort Mean": [float(active_display_result.means.get(d, 0.0)) for d in active_display_result.domains],
            })
            st.bar_chart(bar_df.set_index("Domain"))
        else:
            cats = list(active_display_result.domains) + [active_display_result.domains[0]]
            t_vals = [active_display_result.target_values[d] for d in active_display_result.domains] + [active_display_result.target_values[active_display_result.domains[0]]]
            p_vals = [float(closest_peer[d]) for d in active_display_result.domains] + [float(closest_peer[active_display_result.domains[0]])]
            b_vals = [float(active_display_result.means.get(d, 0.0)) for d in active_display_result.domains] + [float(active_display_result.means.get(active_display_result.domains[0], 0.0))]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=t_vals, theta=cats, fill="toself",
                name=f"Target: {target['Name'][:20]} ({target[COMPOSITE]})", line=dict(color="#EA580C", width=3),
                fillcolor="rgba(234, 88, 12, 0.2)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=p_vals, theta=cats, fill="toself",
                name=f"Nearest profile: {closest_peer['Name'][:20]}", line=dict(color="#0284C7", width=2.5),
                fillcolor="rgba(2, 132, 199, 0.2)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=b_vals, theta=cats,
                name="Candidate mean", line=dict(color="#64748B", width=1.5, dash="dash"),
            ))

            if show_trap_overlay and conley_available:
                conley_row = conley_matches.iloc[0]
                c_vals = [float(conley_row[d]) for d in active_display_result.domains] + [float(conley_row[active_display_result.domains[0]])]
                fig_radar.add_trace(go.Scatterpolar(
                    r=c_vals, theta=cats, fill="toself",
                    name=f"Comparison: {conley_row['Name'][:20]}", line=dict(color="#9333EA", width=2.5, dash="dot"),
                    fillcolor="rgba(147, 51, 234, 0.15)",
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11, color="#64748B"))),
                font=dict(family="sans-serif", size=13),
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
                height=460, margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_radar, width="stretch")

    if show_trap_overlay and conley_available:
        st.info("These profiles illustrate differences across source dimensions; they do not establish causal intervention requirements.")

    st.subheader("Nearest schools in the selected profile space")
    table_cols = [
        "Rank", "Name", "City", "State", COMPOSITE,
        "Distance", "Top contributing domain", *active_display_result.domains,
    ]
    st.dataframe(active_display_result.matches[table_cols], hide_index=True, width="stretch")

with tab2:
    st.subheader("Profile comparisons")
    st.caption("Lists nearby records across four community stress domains with higher Education scores.")
    deviants = find_positive_deviants(schools, selected_id, k=3, state=state_value)
    if not deviants.empty:
        st.dataframe(
            deviants[["Name", "City", "State", "Composite Score", "Education Score", "Education Outperformance (+pts)", "Headwind Match Distance"]],
            hide_index=True, width="stretch",
        )
    else:
        st.info("No comparison records with complete scores were found for this profile and scope.")

with tab3:
    target_state = str(target["State"]).strip() if pd.notna(target["State"]) else "NC"
    st.subheader("Largest domain-profile spread")
    st.caption("Schools with the largest dispersion across selected standardized domain scores.")

    lb_scope = st.radio(
        "Leaderboard Scope:",
        [f"Statewide ({target_state})", "Nationwide"],
        index=0,
        horizontal=True,
    )
    lb_state_filter = target_state if "Statewide" in lb_scope else None
    leaderboard = get_systemic_masking_leaderboard(schools, domains=active_display_result.domains, state=lb_state_filter, n=10)
    st.dataframe(leaderboard, hide_index=True, width="stretch")

with st.expander("Method and interpretation"):
    st.markdown(
        """
        - School identity uses name, city, state, and district, with county as a fallback. Federal identifiers are omitted.
        - Incomplete cases in selected domains are handled by complete-case filtering without imputation.
        - StandardScaler is fit on the candidate pool for the selected domains; Euclidean distances are computed in that space.
        - Domain-profile spread is the standard deviation of selected domain z-scores.
        - ODIS scores describe synthesized community conditions within School Attendance Boundaries, not student or school performance.
        - Program names serve as references for exploratory research only and do not establish funding eligibility.
        """
    )

st.markdown("---")
st.subheader("Download exploratory profile summary")

col_brief_btn, col_brief_ai = st.columns([1, 1])
with col_brief_btn:
    st.download_button(
        label="Download profile summary (.txt)",
        data=active_display_result.brief_text,
        file_name=f"STEER_Profile_Summary_{target['Name'].replace(' ', '_')}.txt",
        mime="text/plain",
    )

with col_brief_ai:
    if st.button("Draft descriptive summary with AI"):
        with st.spinner("Drafting summary with AI model..."):
            ai_summary = synthesize_llm_grant_narrative(
                active_display_result, target, user_api_key=user_api_key
            )
        st.text_area("Profile summary preview:", ai_summary, height=220)