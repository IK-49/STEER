"""Streamlit exploration UI for descriptive school similarity."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data import COMPOSITE, DOMAINS, IDENTIFIER, RECORD_KEY, configured_data_path, identifier_quality, load_dataset, missingness_summary
from similarity import SearchError, find_similar_schools

st.set_page_config(page_title="STEER School Similarity", layout="wide")


@st.cache_data(show_spinner="Loading school records…")
def get_data(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    return load_dataset(path)


def school_label(row: pd.Series) -> str:
    return f"{row['Name']} — {row['City']}, {row['State']} · NCESSCH {row[IDENTIFIER]} · {RECORD_KEY} {row[RECORD_KEY]}"


st.title("STEER School Similarity")
st.caption("Find schools with similar measured domain scores for peer learning.")
st.info(
    "Similarity is descriptive. It is not a causal finding, school quality rating, "
    "or recommendation about resource allocation. The source does not document "
    "score definitions, direction, or reference year; score levels should not be "
    "interpreted as better or worse outcomes."
)

try:
    data_path = configured_data_path()
    data_stat = data_path.stat()
    schools = get_data(str(data_path), data_stat.st_mtime_ns, data_stat.st_size)
except Exception as exc:
    st.error(f"The school dataset could not be loaded or validated: {exc}")
    st.stop()

with st.expander("Dataset and missingness"):
    st.write(f"Loaded **{len(schools):,}** schools from `{configured_data_path().name}`.")
    st.dataframe(missingness_summary(schools), hide_index=True, use_container_width=True)
    st.caption("The source CSV is read only. Search preparation uses a new in-memory working copy.")

id_quality = identifier_quality(schools)
if id_quality["duplicate_groups"] or id_quality["scientific_notation_rows"]:
    st.warning(
        f"Source NCESSCH needs correction: {id_quality['scientific_notation_rows']:,} values use scientific notation; "
        f"{id_quality['duplicate_groups']:,} repeated-ID groups affect {id_quality['affected_rows']:,} rows. "
        "The app preserves source IDs as supplied and uses generated `rq` row identifiers to distinguish records. "
        "`rq` values follow CSV row order and are temporary until corrected NCESSCH values are available."
    )

st.header("Find a school")
query = st.text_input("Search by school name, city, state, or NCESSCH", max_chars=120)
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
    st.warning("No matching schools found. Try a different name, city, state, or identifier.")
    st.stop()
if len(found) == 100:
    st.caption("Showing the first 100 matches. Refine your search if needed.")
selected_id = st.selectbox(
    "Choose the school", found[RECORD_KEY].tolist(),
    format_func=lambda value: school_label(schools.loc[schools[RECORD_KEY].eq(value)].iloc[0]),
)
target = schools.loc[schools[RECORD_KEY].eq(selected_id)].iloc[0]

left, right = st.columns([1, 1])
with left:
    st.subheader("School profile")
    st.write(f"**{target['Name']}**")
    st.write(" · ".join(str(target[field]) for field in ("City", "County", "State") if pd.notna(target[field]) and str(target[field]).strip()))
    profile = pd.DataFrame({"Domain": DOMAINS, "Score": [target[domain] for domain in DOMAINS]})
    profile["Score"] = pd.to_numeric(profile["Score"], errors="coerce")
    st.dataframe(profile, hide_index=True, use_container_width=True)
    st.write(f"**Composite Score:** {target[COMPOSITE]}")
    st.caption("Displayed as supplied; its scale and interpretation are undocumented.")
    st.subheader("Domain values used for this search")
    st.caption("Values are prefilled from the dataset when available. You can enter or adjust values for this search only. A blank missing target value omits that domain.")
    target_inputs = {}
    for domain in DOMAINS:
        original = pd.to_numeric(pd.Series([target[domain]]), errors="coerce").iloc[0]
        target_inputs[domain] = st.number_input(
            domain,
            value=None if pd.isna(original) else float(original),
            step=0.01,
            format="%.4f",
            key=f"target_value_{selected_id}_{domain}",
        )

with right:
    st.subheader("Search settings")
    active_domains = st.multiselect("Domains used for similarity", DOMAINS, default=list(DOMAINS))
    states = sorted(schools["State"].dropna().astype(str).str.strip().loc[lambda values: values.ne("")].unique())
    state_filter = st.selectbox("State filter", ["All states", *states])
    counties = sorted(
        schools.loc[schools["State"].astype(str).str.strip().eq(state_filter), "County"]
        .dropna().astype(str).str.strip().loc[lambda values: values.ne("")].unique()
    ) if state_filter != "All states" else []
    county_filter = st.selectbox("County filter", ["All counties", *counties], disabled=not counties)
    k = st.slider("Number of peer schools (K)", min_value=3, max_value=10, value=5)

state_value = None if state_filter == "All states" else state_filter
county_value = None if county_filter == "All counties" or not county_filter else county_filter
try:
    result = find_similar_schools(
        schools, selected_id, active_domains, k=k, state=state_value, county=county_value,
        target_values=target_inputs,
    )
except SearchError as exc:
    st.warning(str(exc))
    st.stop()

if result.excluded_target_domains:
    st.warning("Target has no value for these domains, so they were omitted from this search: " + ", ".join(result.excluded_target_domains))
st.caption(
    f"Search scope: {result.scope_description}. Equal feature weighting after standardization. "
    f"Rows missing any used domain are excluded from this search ({result.excluded_missing_count:,} peers in scope)."
)
st.dataframe(
    result.matches[["Rank", "Name", "City", "County", "State", IDENTIFIER,
                    RECORD_KEY, *DOMAINS, COMPOSITE, "Distance", "Top contributing domain"]],
    hide_index=True, use_container_width=True,
)

with st.expander("Why these schools matched"):
    for _, row in result.matches.iterrows():
        st.write(f"**{row['Name']}** — strongest distance contribution: {row['Top contributing domain']}")
    st.caption("Feature contributions are squared standardized score differences, normalized within each peer.")

if len(result.matches):
    best = result.matches.iloc[0]
    chart = pd.DataFrame({"Domain": result.domains, "Selected school": [result.target_values[d] for d in result.domains],
                          "Closest peer": [pd.to_numeric(pd.Series([best[d]]), errors="coerce").iloc[0] for d in result.domains]})
    st.subheader("Compare the selected school and closest peer")
    st.bar_chart(chart.set_index("Domain"), y=["Selected school", "Closest peer"])

st.caption("Euclidean distances compare standardized selected-domain scores; smaller distances mean closer measured profiles.")
