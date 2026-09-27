# STEER: explore school community profiles

STEER is a Streamlit prototype for descriptive comparisons of community stress profiles around U.S. public schools. It keeps the three-tab interface and analysis ideas from the `feat/steer-update` branch: nearest-profile search, profile comparisons, a domain-spread leaderboard, and a hypothetical score-adjustment control. The school search updates suggestions as the user types.

The app identifies a record by the source school name, city, state, and district (or county when district is unavailable). It does not read or use NCESSCH. Exact duplicate name/location/profile rows are collapsed. If one name/location key has conflicting score profiles, loading stops rather than choosing a record silently.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app reads `Data/index_scores_v3_2026.csv` relative to the repository. Set `STEER_DATA_PATH` to use another CSV. The app and `clean_data.py` are read-only; they do not rewrite source data.

For the optional generated summary, set `OPENAI_API_KEY`. Without a key, the deterministic text summary is used. The generated text is still exploratory and should be checked before reuse.

## Tests

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

Tests cover name/location identity, duplicate and ambiguous profiles, missing data, distance ordering and self-exclusion, geographic filters, retained analysis features, typeahead search behavior, and app startup.

## How to read the analysis

- Search uses complete records for the selected domains. Candidate rows missing any selected score are excluded and counted; no imputation is performed. A selected school missing a selected score is reported as an error.
- Candidate scores are standardized within the filtered candidate pool. The app returns Euclidean distance in that standardized feature space; it is not a calibrated percentage or probability.
- “Domain-profile spread” is the standard deviation of a school's selected domain z-scores. It is a descriptive dispersion metric (the legacy code called it CMI). It does not establish unmet need, causal effects, composite-score bias, or funding consequences.
- The hypothetical score-adjustment control changes an input score mathematically. It does not estimate an intervention's effect or show that a real school can achieve that change.
- The profile-comparison tab finds nearby records on the four non-Education domains and shows those with higher Education scores. This does not identify effective practices or demonstrate why scores differ.
- A program name shown in the summary is a reference to research further, not a grant recommendation or eligibility determination. Verify current program rules independently.

## Data context

The bundled source documentation describes the Open Data Index for Schools (ODIS), v1, by Hawken, Minar, Choudhary, and Kulick (2026), DOI [10.7281/T170WN53](https://doi.org/10.7281/T170WN53). Refer to `Data/ODIS README v3.pdf` and `Data/ODIS Technical Report March 2026.pdf` for definitions and limitations.

ODIS scores describe community stress indicators synthesized to School Attendance Boundaries. Higher scores indicate more stressful community conditions; they are not measures of school performance. Underlying measures use different sources, geographies, and reference years.
