"""Presentation narrative and evidence for the STEER project."""

import pandas as pd
import plotly.express as px
import streamlit as st

from data import COMPOSITE, DOMAINS, configured_data_path, load_dataset

if st.button("Back to Explorer"):
    st.switch_page("pages/explorer.py")

st.title("Nearby places, different school communities")
st.caption("STEER · Carolina Data Challenge · A descriptive exploration of ODIS School Attendance Boundary profiles")

st.markdown("""
### Our starting point

We’re from the Cary, Morrisville, and Apex area. Our friends grew up across nearby
parts of the Triangle, and many of us have older siblings who went to different high
schools. When we were younger, we often wondered: **how different can places this
close together really be?** We used school attendance boundary (SAB) data to explore
that question.

First, we compared composite scores. That showed us variation, but it also raised a
second question: **can similar composites hide very different underlying profiles?**
The Yuma example below shows why we looked beyond one number.
""")

try:
    path = configured_data_path()
    schools = load_dataset(path)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

st.info("ODIS describes community stress indicators associated with School Attendance Boundaries. These are not school performance scores, and patterns here are descriptive rather than causal.")

st.header("1 · A composite score is a useful starting point")
st.write("A composite can summarize multiple dimensions. Looking at its distribution across places helps us ask where the summary varies; it does not tell us which domains account for a difference.")

local_cities = ["Cary", "Morrisville", "Apex", "Raleigh", "Durham", "Chapel Hill"]
local = schools.loc[schools["State"].eq("NC") & schools["City"].isin(local_cities)].copy()
if local.empty:
    st.warning("The current dataset has no exact city-name matches for the Triangle comparison. Use the notebook scaffold to choose and document the intended local records.")
else:
    local["City"] = pd.Categorical(local["City"], categories=local_cities, ordered=True)
    a, b = st.columns(2)
    with a:
        fig = px.box(local, x="City", y=COMPOSITE, points="all", color="City", title="Composite scores across selected Triangle cities")
        fig.update_layout(showlegend=False, xaxis_title="City", yaxis_title="Composite score")
        st.plotly_chart(fig, width="stretch")
    with b:
        means = local.groupby("City", observed=True)[list(DOMAINS)].mean().reset_index().melt("City", var_name="Domain", value_name="Mean score")
        fig = px.line(means, x="City", y="Mean score", color="Domain", markers=True, title="Average domain profiles across the same cities")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Selected {len(local):,} records across {local['City'].nunique()} exact city labels. City groupings are a presentation lens, not a claim that city and SAB boundaries align.")
    scaled = (local[list(DOMAINS)] - local[list(DOMAINS)].mean()) / local[list(DOMAINS)].std(ddof=0).replace(0, 1)
    local["Profile spread (σ)"] = scaled.std(axis=1, ddof=0)
    fig = px.scatter(local, x=COMPOSITE, y="Profile spread (σ)", color="City", hover_name="Name",
                     title="Composite score versus standardized five-domain spread")
    fig.update_layout(xaxis_title="Composite score", yaxis_title="Within-profile spread (σ)")
    st.plotly_chart(fig, width="stretch")

st.header("2 · Similar composites, very different profiles")
left, right = st.columns([1, 1.2])
with left:
    st.markdown("""
    **Two schools in Yuma Union High School District**

    - Gila Ridge High School · Yuma
    - San Luis High School · San Luis
    - Composite scores: **36** and **42** (six points apart)
    - Same district; San Luis is about 20 miles from Yuma, according to the [district history](https://www.yumaunion.org/welcome/welcome/our-history)

    The composite scores look close, but the underlying community profiles do not:
    Education is **29 vs. 90**, and Housing is **55 vs. 6**. The district and the
    similar composite do not tell the whole story. These scores describe community
    conditions associated with school attendance boundaries, not school performance.
    """)
with right:
    pair_names = ["Gila Ridge High School", "San Luis High School"]
    pair = schools.loc[schools["Name"].isin(pair_names)].copy()
    if len(pair) == 2:
        pair["Name"] = pd.Categorical(pair["Name"], categories=pair_names, ordered=True)
        long = pair.melt(id_vars=["Name", "City", COMPOSITE], value_vars=list(DOMAINS), var_name="Domain", value_name="Score")
        long["School"] = long["Name"].astype(str) + " · " + long["City"]
        fig = px.bar(long, x="Domain", y="Score", color="School", barmode="group", title="Composite scores 6 points apart, contrasting five-score profiles")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pair[["Name", "City", "School District", COMPOSITE, *DOMAINS]], hide_index=True, width="stretch")
    else:
        st.warning("The Yuma example could not be matched in the loaded data. Confirm school names before presenting.")

st.header("3 · From one-number ranking to profile comparison")
st.markdown("""
We then compared the five selected scores together. In plain language: put each
score on a comparable scale for the chosen group, measure the gap between two
schools across all selected scores, and find the smallest gaps. A large gap on one
score contributes more to the overall distance. This is a descriptive way to find
nearby profiles; it does not establish shared needs, causes, or transferable policy.
""")
st.code("For each domain: (score − group average) ÷ group spread\nProfile distance: √(gap₁² + gap₂² + … + gap₅²)", language="text")
st.markdown("**Our question for policy:** if two places are treated as similar because their composites match, what could be missed when their underlying domain profiles differ? STEER makes those profiles visible so people can ask better questions before drawing conclusions.")

st.subheader("More ways to show the pattern")
st.markdown("The two Triangle comparisons and Arizona profile plot above are ready-to-use views. The notebook scaffold also lays out a composite distribution, a domain profile heatmap, composite-versus-profile-spread scatter, and the Arizona pair for presentation edits.")

with st.expander("Data and interpretation notes"):
    st.write("Source: bundled ODIS v1 school community profile dataset; see Data/ODIS README v3.pdf and Data/ODIS Technical Report March 2026.pdf. Composite scores can conceal differences among domains. Standardization depends on the selected candidate group; distance is not a probability. The displayed city-level summaries are exploratory and may combine different boundary contexts.")
