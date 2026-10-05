"""
Presentation narrative and evidence for the STEER project.

Authors: Anish Velagapudi, Izad Khokhar, Aurick Smart
Attribution: Narrative synthesis, interactive chart generation, and methodology
breakdowns were co-developed by the project team with AI scaffolding assistance.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import COMPOSITE, DOMAINS, configured_data_path, load_dataset

st.set_page_config(page_title="STEER | Project Story", layout="wide")

# Safe return navigation back to main app
if st.button("Back to Explorer"):
    try:
        st.switch_page("app.py")
    except Exception:
        st.info("Use the sidebar navigation to return to the Explorer.")

st.title("STEER: Project Background and Analytical Story")
st.caption("Statistical Twins for Educational Equity & Resources · Authors: Anish Velagapudi, Izad Khokhar, Aurick Smart")

st.markdown(
    """
    ### Our Starting Point
    We grew up in the Cary, Morrisville, and Apex area of the Triangle. Even though our families and friends 
    lived relatively close to one another, we experienced distinct neighborhood environments and school communities. 
    Many of us had friends and older siblings who attended different high schools across Wake, Durham, and Orange counties. 
    When we were younger, we often wondered: **how different can communities this close together really be?**

    We turned to the Open Data Index for Schools (ODIS) dataset to examine that question. Initially, we looked at 
    single Composite Scores. That revealed geographic variation, but it also exposed a fundamental problem: 
    **can similar composite scores hide completely different community headwinds?**
    """
)

try:
    path = configured_data_path()
    schools = load_dataset(path)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

st.info(
    "Data Notice: ODIS measures describe community stress indicators synthesized to School Attendance Boundaries (SABs). "
    "These are not school performance scores, and all patterns presented here are descriptive rather than causal."
)

st.markdown("---")

st.header("1. Local Exploration: The Limits of One-Number Summaries")
st.write(
    "A composite score summarizes multiple dimensions into a single value. Looking at its distribution across "
    "Triangle communities illustrates how variation exists locally, yet obscures which specific dimensions drive that variation."
)

local_cities = ["Cary", "Morrisville", "Apex", "Raleigh", "Durham", "Chapel Hill"]
local = schools.loc[schools["State"].eq("NC") & schools["City"].isin(local_cities)].copy()

if local.empty:
    st.warning("No exact city-name matches found for the selected Triangle subset.")
else:
    local["City"] = pd.Categorical(local["City"], categories=local_cities, ordered=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_box = px.box(
            local, x="City", y=COMPOSITE, points="all", color="City",
            title="Composite Score Distribution Across Triangle Cities"
        )
        fig_box.update_layout(showlegend=False, xaxis_title="City", yaxis_title="Composite Score")
        st.plotly_chart(fig_box, width="stretch")
    with c2:
        means = (
            local.groupby("City", observed=True)[list(DOMAINS)]
            .mean()
            .reset_index()
            .melt("City", var_name="Domain", value_name="Mean Score")
        )
        fig_line = px.line(
            means, x="City", y="Mean Score", color="Domain", markers=True,
            title="Average Five-Domain Stress Profiles Across Triangle Cities"
        )
        st.plotly_chart(fig_line, width="stretch")

    st.caption(
        f"Inspecting {len(local):,} records across {local['City'].nunique()} Triangle city labels. "
        "City groupings represent an illustrative lens; municipal borders and attendance boundaries do not perfectly align."
    )

    # Standardize locally to show within-profile variance vs composite
    scaled = (local[list(DOMAINS)] - local[list(DOMAINS)].mean()) / local[list(DOMAINS)].std(ddof=0).replace(0, 1)
    local["Profile spread (σ)"] = scaled.std(axis=1, ddof=0)
    fig_scatter = px.scatter(
        local, x=COMPOSITE, y="Profile spread (σ)", color="City", hover_name="Name",
        title="Composite Score vs. Within-Profile Spread (Domain Disparity)"
    )
    fig_scatter.update_layout(xaxis_title="Composite Score", yaxis_title="Domain-Profile Spread (σ)")
    st.plotly_chart(fig_scatter, width="stretch")

st.markdown("---")

st.header("2. The Composite Trap: Similar Scores, Polarized Realities")
st.write(
    "To understand why single scores fail resource allocation, examine two schools from the same district in Arizona: "
    "**Gila Ridge High School** and **San Luis High School** in Yuma Union High School District."
)

col_left, col_right = st.columns([1, 1.2])
with col_left:
    st.markdown(
        """
        **Case Study: Yuma Union High School District**
        - **Gila Ridge High School** (Yuma, AZ)
        - **San Luis High School** (San Luis, AZ)
        - **Composite Scores:** 36 vs. 42 (separated by only 6 points)
        
        Despite sharing a district and near-identical composite scores, their underlying environments are completely polarized:
        - **Education Stress:** 29 vs. 90
        - **Housing Stress:** 55 vs. 6
        
        A formula allocating funds strictly on composite hardship treats these schools almost identically, 
        missing the fact that one requires housing assistance while the other confronts severe language and educational headwinds.
        """
    )

with col_right:
    pair_names = ["Gila Ridge High School", "San Luis High School"]
    pair = schools.loc[schools["Name"].isin(pair_names)].copy()
    if len(pair) == 2:
        pair["Name"] = pd.Categorical(pair["Name"], categories=pair_names, ordered=True)
        long = pair.melt(
            id_vars=["Name", "City", COMPOSITE],
            value_vars=list(DOMAINS),
            var_name="Domain",
            value_name="Score"
        )
        long["School"] = long["Name"].astype(str) + " (" + long["City"] + ")"
        fig_bar = px.bar(
            long, x="Domain", y="Score", color="School", barmode="group",
            title="Contrasting Domain Profiles at Composite 36 vs. 42"
        )
        st.plotly_chart(fig_bar, width="stretch")
        st.dataframe(pair[["Name", "City", "School District", COMPOSITE, *DOMAINS]], hide_index=True, width="stretch")
    else:
        st.warning("The pair case study records could not be matched in the dataset.")

st.markdown("---")

st.header("3. How STEER Was Built")
st.markdown(
    """
    1. **Data Cleaning & Identity Schema:** Federal identifiers (NCESSCH) were corrupted upstream into scientific notation. 
       We bypassed reading them entirely, engineering an unambiguous composite key using `(Name, City, State, District/County)`.
    2. **Out-of-Sample Standardization:** To eliminate data leakage, we fit `StandardScaler` strictly on the candidate pool 
       (excluding the target school). This places disparate indicators on a comparable unit-variance scale.
    3. **Subspace Matching & Distance Attribution:** We compute Euclidean distance across user-selected dimensions and decompose 
       the distance vector to explicitly surface the "Top Contributing Domain" driving divergence.
    4. **Positive Deviance Framework:** By holding community headwinds constant (matching on Economic, Health, Housing, and Crime), 
       we isolate institutions achieving stronger Education marks under matching neighborhood challenges.
    """
)

st.code(
    "Domain Standardization: z = (score − candidate_mean) ÷ candidate_std\n"
    "Subspace Distance: d(u, v) = √( Σ (z_target - z_candidate)² )",
    language="text",
)

st.markdown("---")

st.header("4. Responsible AI Principles")
st.write(
    "STEER is designed as an exploratory, descriptive tool rather than an automated decision maker. "
    "It does not evaluate school quality, predict student achievement, or automate grant distribution. "
    "Instead, it provides multidimensional clarity to support human decision-making, helping educational leaders "
    "look past one-number rankings to find meaningful peer collaboration."
)

with st.expander("Data and Methodology Limitations"):
    st.write(
        "Source: Bundled ODIS v3 dataset (Hawken et al., 2026). Composite scores can conceal domain differences. "
        "Standardization is local to the chosen search cohort; distance is an exploratory metric, not a probability or causal finding. "
        "Local city groupings are presentation summaries and may span multiple school attendance boundaries."
    )