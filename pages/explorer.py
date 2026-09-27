"""STEER: descriptive school community profile explorer."""

from __future__ import annotations

import textwrap

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
    return str(row.get("Display name") or f"{row['Name']} ({row['City']}, {row['State']})")


# Hero Header with Competition Track Badge
st.markdown('<div class="badge-sub">Carolina Data Challenge 2026 · AI for Social Good Track</div>', unsafe_allow_html=True)
st.title("STEER: Explore School Community Profiles")
st.caption("A descriptive profile comparison prototype for the Carolina Data Challenge")

try:
    data_path = configured_data_path()
    data_stat = data_path.stat()
    schools = get_data(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

# Name and location provide record identity; federal identifiers are not loaded or used.
with st.expander(" Source data and record identity", expanded=False):
    st.write(
        "Federal school identifiers are not loaded. Records are selected and excluded from their own peer set "
        "using school name, city, state, and district (county is used when district is missing)."
    )

# Flagship Demonstration Default: Anchors to North Carolina's primary case study
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
# BUGFIX: count the true number of matches BEFORE truncating to 250,
# and report that count (not len() on it -- it's already an int).
total_matches = int(mask.sum())
found = schools.loc[mask].head(250)

if found.empty:
    st.warning("No matching institutions found. Try another query.")
    st.stop()

suggestion_keys = found[RECORD_KEY].head(5).tolist()
st.caption(f"{total_matches:,} matching records · choose a suggestion")
selected_id = st.pills(
    "Matching schools", suggestion_keys,
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

# Subspace & Retrieval Sidebar
with st.sidebar:
    st.page_link("pages/1_Story.py", label="Story")
    st.header("Search & Subspace Controls")
    active_domains = st.multiselect("Active Subspace Dimensions:", DOMAINS, default=list(DOMAINS))
    states = sorted(schools["State"].dropna().astype(str).str.strip().loc[lambda v: v.ne("")].unique())
    state_filter = st.selectbox("Geographic Scope:", ["Nationwide", *states])
    k = st.slider("Number of comparison schools (k):", min_value=3, max_value=10, value=5)

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
m1.metric("Composite score", f"{target[COMPOSITE]:.1f} / 100", "Source value")
m2.metric("Highest relative domain", result.dominant_domain.upper(), f"{result.dominant_z:+.2f}σ in selected cohort")
m3.metric("Domain-profile spread", f"{result.cmi} σ", "Spread of selected domain z-scores")
m4.metric(
    "Program area to research",
    # BUGFIX: word-boundary-aware truncation instead of a raw character slice,
    # which could previously cut a program name mid-word (e.g. "...Health C").
    textwrap.shorten(result.grant_program.split("&")[0].strip(), width=26, placeholder="…"),
    "Not an eligibility finding",
)
st.markdown("---")
st.caption(
    f"Scope: {result.scope_description} · {len(result.domains)} selected domains · "
    f"{result.excluded_missing_count:,} in-scope candidate records omitted for missing selected scores."
)
if result.excluded_target_domains:
    st.info("The selected school has missing scores for: " + ", ".join(result.excluded_target_domains) + ". These domains were left out of this comparison.")

# What-If Policy Intervention Simulator & Tipping Point Solver
with st.expander(" What-If Policy Intervention Simulator & Tipping Point Solver", expanded=False):
    st.caption("Explore how a hypothetical score change affects the descriptive domain-profile spread. This does not estimate an intervention effect.")
    sim_c1, sim_c2 = st.columns([1.6, 2.4])
    with sim_c1:
        default_sim_idx = active_domains.index(result.dominant_domain) if result.dominant_domain in active_domains else 0
        sim_domain = st.selectbox("Intervention Domain:", active_domains, index=default_sim_idx)
        if result.tipping_point_reduction is None:
            st.markdown(f"No reduction from the current **{result.dominant_domain}** score down to zero brings the spread to 1.0σ or less under this calculation.")
        elif result.tipping_point_reduction == 0:
            st.markdown("The current profile already has a spread of 1.0σ or less.")
        else:
            st.markdown(
                f"A hypothetical `{result.tipping_point_reduction} point` reduction in **{result.dominant_domain}** "
                "would bring the spread metric to 1.0σ or less under this calculation."
            )
    with sim_c2:
        relief_pts = st.slider(
            f"Simulate Relief for {sim_domain} (-pts):",
            min_value=0.0, max_value=30.0, value=0.0, step=0.5,
        )

    if relief_pts > 0:
        if sim_domain not in result.target_values:
            st.info(f"The selected school has no {sim_domain} score to adjust. Remove it from the selected domains or choose a different score.")
        else:
            simulated_inputs = dict(result.target_values)
            simulated_inputs[sim_domain] = max(0.0, simulated_inputs[sim_domain] - relief_pts)
            sim_result = find_similar_schools(
                schools, selected_id, active_domains, k=k, state=state_value, target_values=simulated_inputs,
            )
            st.success(
                f"**Hypothetical score adjustment:** {sim_domain} changed from {result.target_values[sim_domain]:.1f} → "
                f"{sim_result.target_values[sim_domain]:.1f} | **Domain-profile spread:** {sim_result.cmi}σ "
                f"(was {result.cmi}σ before the adjustment)"
            )
            result = sim_result

tab1, tab2, tab3 = st.tabs([
    "Profile & Nearest Schools",
    "Profile Comparisons",
    "Domain Spread Leaderboard",
])

with tab1:
    st.subheader("Multi-domain profile")
    st.caption("Compare selected source scores with nearby points in the standardized feature space.")

    # 1-Click "Composite Trap" Parity Toggle
    # BUGFIX: match on the record identity key (name + city + state + district/county),
    # not just Name, so a same-named school in a different state can't be picked up
    # as the "D.H. Conley (NC)" comparison record.
    conley_matches = schools[
        schools["Name"].str.contains("D H Conley", case=False, na=False)
        & schools["State"].astype(str).str.strip().str.casefold().eq("nc")
    ]
    conley_available = not conley_matches.empty and target[RECORD_KEY] != conley_matches.iloc[0][RECORD_KEY]

    if conley_available:
        show_trap_overlay = st.checkbox(
            "Compare with D.H. Conley High (NC)",
            value=False,
            help="Show a second named school profile for descriptive comparison.",
        )
    else:
        show_trap_overlay = False

    if len(result.matches) > 0:
        closest_peer = result.matches.iloc[0]

        if len(result.domains) < 2:
            # BUGFIX: a radar/polar chart with a single repeated category is a
            # degenerate shape (a sliver/point) and isn't a useful visual.
            # Fall back to a plain comparison for the single selected domain.
            single_domain = result.domains[0]
            st.info(
                "The profile radar needs at least two active domains to draw a shape. "
                "Select more than one domain in the sidebar to see it. Showing the "
                f"single **{single_domain}** value instead:"
            )
            st.metric(
                f"{single_domain} — target vs. nearest peer vs. candidate mean",
                f"{result.target_values[single_domain]:.1f}",
                f"peer {float(closest_peer[single_domain]):.1f} · mean {result.means.get(single_domain, 0.0):.1f}",
            )
        else:
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
                name=f"Nearest profile: {closest_peer['Name'][:20]}", line=dict(color="#0284C7", width=2.5),
                fillcolor="rgba(2, 132, 199, 0.2)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=b_vals, theta=cats,
                name="Candidate mean", line=dict(color="#64748B", width=1.5, dash="dash"),
            ))

            if show_trap_overlay and conley_available:
                conley_row = conley_matches.iloc[0]
                c_vals = [float(conley_row[d]) for d in result.domains] + [float(conley_row[result.domains[0]])]
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
        st.info("These profiles illustrate differences among source dimensions. They do not establish school need, funding formulas, or suitable interventions.")

    st.subheader("Nearest schools in the selected profile space")
    table_cols = [
        "Rank", "Name", "City", "State", COMPOSITE,
        "Distance", "Top contributing domain", *result.domains,
    ]
    st.dataframe(result.matches[table_cols], hide_index=True, width="stretch")

with tab2:
    st.subheader("Profile comparisons")
    st.caption("Lists nearby records on four selected community domains with a higher Education score. This association is descriptive, not causal or an operational best-practice finding.")
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
    st.caption("Schools with the largest spread across selected standardized domain scores. This is not a ranking of need or evidence that a composite score masks outcomes.")

    lb_scope = st.radio(
        "Leaderboard Scope:",
        [f"Statewide ({target_state})", "Nationwide"],
        index=0,
        horizontal=True,
    )
    lb_state_filter = target_state if "Statewide" in lb_scope else None
    leaderboard = get_systemic_masking_leaderboard(schools, domains=result.domains, state=lb_state_filter, n=10)
    st.dataframe(leaderboard, hide_index=True, width="stretch")

with st.expander("Method and interpretation"):
    st.markdown(
        """
        - School identity uses name, city, state, and district, with county as a fallback. Federal identifiers are not loaded. Conflicting profiles on a name/location key stop the load.
        - Missing values in selected domains are handled by complete-case filtering. No scores are imputed.
        - `StandardScaler` is fit on the complete candidate pool for the selected domains. The target is transformed with that scaler; results are Euclidean distances in this feature space.
        - Domain-profile spread is the standard deviation of selected domain z-scores. It is descriptive and depends on the selected domains and search scope.
        - ODIS scores describe community conditions synthesized to School Attendance Boundaries. They are not school performance measures or causal estimates.
        - Program names are references for independent research only. This app does not determine grant eligibility or recommend funding.
        """
    )

# Downstream Policy Actionable Artifact
st.markdown("---")
st.subheader("Download exploratory profile summary")

col_brief_btn, col_brief_ai = st.columns([1, 1])
with col_brief_btn:
    st.download_button(
        label="Download profile summary (.txt)",
        data=result.brief_text,
        file_name=f"STEER_Profile_Summary_{target['Name'].replace(' ', '_')}.txt",
        mime="text/plain",
    )
    
with col_brief_ai:
    if st.button("Draft descriptive summary with AI"):
        with st.spinner("Drafting a descriptive summary…"):
            brief_display = synthesize_llm_grant_narrative(result, target)

        st.text_area("Profile summary preview:", brief_display, height=220)
