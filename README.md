# STEER — School Community Profile Explorer

STEER is an interactive data exploration prototype built for the **Carolina Data Challenge 2026**. It helps people inspect how community stress indicators differ across U.S. public school attendance boundaries, look beyond a single composite score, and find other records with nearby multi-domain profiles.

The project started from a practical question: **when two places have similar composite scores, do the underlying domain profiles also look similar?** STEER makes those dimensions visible and provides transparent, descriptive comparisons that can support further questions by researchers, educators, and policymakers.

> **Interpretation:** STEER explores community-context data. It does not measure school quality or student outcomes, explain why differences exist, estimate intervention effects, rank schools by need, or determine grant eligibility.

## Project at a glance

- **Application:** Streamlit multipage web app with an interactive profile explorer and a data story page.
- **Dataset:** Bundled 2026 Open Data Index for Schools (ODIS) v1 data, with 23,597 usable name-and-location records after exact duplicate profiles are collapsed.
- **Coverage:** 52 state and district-level jurisdictions represented in the source, including Washington, D.C. and territories where supplied.
- **Profile dimensions:** Economic, Education, Health, Housing, and Crime, plus the source Composite Score.
- **Core analysis:** Complete-case filtering, cohort-specific standardization, Euclidean nearest-profile search, per-domain distance explanations, profile-spread summaries, and descriptive comparisons.
- **Technology:** Python, pandas, scikit-learn, Streamlit, Plotly, and optionally the OpenAI Python SDK for drafting a caveated natural-language summary.
- **Data handling:** Read-only CSV loading; the application does not modify the bundled source file.

## What the application does

### 1. Search for a school profile

The Explorer page accepts a school name, city, or state and updates matching records as the query changes. Users select a matching school and choose a geographic scope (nationwide or a state), one or more profile domains, and the number of returned peers (`k`, from 3 to 10).

Federal school identifiers are deliberately not read. The source identifier is unreliable in the packaged data, so STEER identifies a row using school name, city, state, and district; county is used when district is missing. Exact duplicate rows with the same identity and scores are collapsed. If one identity key points to conflicting score profiles, loading fails with a clear validation error instead of silently selecting one.

### 2. Find nearby profiles

For the selected domains, STEER filters the candidate pool to records with complete scores and excludes the selected school itself. It fits `sklearn.preprocessing.StandardScaler` on the remaining eligible candidates, transforms the target with those candidate statistics, then calculates Euclidean distance in the selected standardized feature space. The closest records are returned in ascending distance order, with deterministic tie-breaking.

The results table includes the candidate's rank, location, composite and selected domain values, standardized-space distance, and the domain contributing most to that match's squared distance. The app also reports how many candidates were omitted because of missing selected-domain data.

**A distance is not a similarity percentage or probability.** Because scaling is fit separately within the chosen geographic pool, changing state or domain selection can change the distances and ordering. Standardization makes numeric scales comparable for this calculation; it does not make the domains equivalent in meaning.

### 3. Inspect and compare profiles

The Explorer presents the target profile, nearest profile, and candidate-pool mean in a Plotly radar chart when two or more domains are selected. With a single domain, it uses a direct value comparison. An optional overlay compares a selected record with D.H. Conley High School in North Carolina when that record is available.

The **Profile Comparisons** view finds records that are nearby on Economic, Health, Housing, and Crime scores and have a higher Education score. It is an observed comparison only: the data do not contain practices, interventions, or evidence explaining the Education-score difference.

The **Domain Spread Leaderboard** shows records with the largest spread across the selected standardized domain scores, either statewide or nationwide. The metric is the population standard deviation of a row's selected domain z-scores relative to that leaderboard's pool. It is a descriptive way to inspect profile variation, not a validated measure of inequity, need, masking, or harm.

### 4. Explore hypothetical changes and export a summary

The What-If control applies a user-selected numeric reduction to one score (capped at zero) and recalculates the descriptive spread. A tipping-point calculation estimates the reduction in the highest relative domain needed to bring spread to 1.0 standard deviation or less, when that threshold is reachable under the formula. These are mathematical scenarios, not forecasts of achievable change or estimates of a policy's effect.

Users can download a deterministic text summary or request an AI-drafted version. The AI option uses `gpt-4o-mini` through the OpenAI chat completions API when `OPENAI_API_KEY` is configured. Its prompt prohibits causal, eligibility, and unsupported claims; if the key is missing or the request fails, STEER falls back to the deterministic summary. Generated prose is exploratory and should be reviewed before reuse.

### 5. Tell the data story

The **Story** page introduces the motivation, shows exploratory comparisons for selected Triangle-area cities, and presents a Yuma-area example: Gila Ridge High School and San Luis High School have composite scores of 36 and 42, while the listed Education and Housing scores differ substantially. This illustrates why looking at component profiles can add context to a composite. It does not show that either school performs better or has a particular need.

## Data and interpretation

The bundled file is `Data/index_scores_v3_2026.csv`. Its accompanying documentation identifies the source as the **Open Data Index for Schools (ODIS), v1**, by Angela Hawken, Nicholas Minar, Raj Choudhary, and Jonathan Kulick (2026), [DOI: 10.7281/T170WN53](https://doi.org/10.7281/T170WN53). See `Data/ODIS README v3.pdf` and `Data/ODIS Technical Report March 2026.pdf` for source definitions, methods, and limitations.

ODIS scores represent community stress indicators synthesized to School Attendance Boundaries. Higher scores indicate more stressful community conditions; they are not school performance scores. Underlying measures draw on different sources, geographies, and reference years. STEER uses the source scores as supplied and does not impute missing values.

Some domain values are missing in the bundled data (for example, Crime is missing for 3,219 of the 23,597 loaded records). A search excludes candidates missing any selected score and reports the number omitted. If the selected record lacks a score for one of the chosen domains, STEER leaves that domain out of the target comparison and identifies it in the interface. No missing values are synthesized.

The app also displays a potential program area associated with the highest relative domain. This is a navigation aid for independent research—not a recommendation, legal conclusion, allocation model, or eligibility determination. Program requirements and local circumstances are outside the dataset and must be checked with the administering agency.

## Architecture

```text
app.py                 Streamlit page registration and app entry point
pages/explorer.py      Interactive search, analysis controls, charts, tables, export
pages/1_Story.py       Narrative data story and exploratory figures
data.py                Read-only CSV loading, schema checks, record identity, missingness
similarity.py          Matching, profile spread, leaderboard, scenario math, summaries
clean_data.py          Read-only dataset audit command
Data/                  Bundled ODIS CSV and source documentation
tests/                 Automated tests for data, analysis, and app behavior
```

`data.py` normalizes known column aliases, reads only the fields needed by the app, enforces a 100 MiB prototype input limit, coerces score columns to numeric values, and validates ambiguous identities. `similarity.py` contains the analytical functions independently of Streamlit, including nearest-profile retrieval, positive-deviance-style comparison, domain-spread ranking, and optional narrative drafting. The pages compose those functions into interactive views.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

By default, the app reads `Data/index_scores_v3_2026.csv` relative to the repository. To load a compatible CSV from another location, set `STEER_DATA_PATH`:

```bash
STEER_DATA_PATH=/path/to/schools.csv streamlit run app.py
```

The app works without an OpenAI key and uses its deterministic summary. To enable optional AI drafting, export `OPENAI_API_KEY` in your environment before starting Streamlit. Do not commit API keys or other secrets.

To print a read-only dataset audit (resolved source path, record count, and missing domain counts):

```bash
python clean_data.py
```

## Tests

Install the test dependency and run the suite:

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

The tests cover name/location identity, duplicate and conflicting profiles, missing data, distance ordering and target exclusion, geographic filters, retained analysis features, search behavior, and app startup.

## Known limitations and next steps

- **Identity:** Name and location text are not a permanent school identifier. A trusted identity crosswalk would be needed for production use.
- **Distance design:** Equal weighting and complete-case filtering are prototype choices. Policymakers and data owners should evaluate whether these comparisons are meaningful before using them for peer learning.
- **Comparability:** Standardization is cohort-dependent, and source indicators have different underlying definitions and reference periods.
- **No causal analysis:** The repository has no intervention model, counterfactual design, or school outcome evaluation.
- **No decision system:** STEER does not implement authentication, persistent collaboration, grant eligibility checks, or funding recommendations.
- **Validation with users:** A next step is to review identity rules, missing-data handling, example matches, and the usefulness of peer comparisons with school-community stakeholders.

---

STEER was created as a datathon prototype to make profile differences easier to inspect and discuss. Interpret its results alongside the source documentation and independent domain expertise.
