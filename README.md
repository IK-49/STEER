# STEER School Similarity

A local Streamlit prototype for helping policymakers find schools with similar
measured Economic, Education, Health, Housing, and Crime profiles. It is a peer
discovery aid; it does not provide contact details or a collaboration workspace.

## Run locally

Use Python 3.11 or newer. From the project directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

By default, the app reads `pre_cleaned_data.csv` beside `app.py`. To select a
different source file, set `STEER_DATA_PATH` in the process environment before
starting Streamlit. The source must contain `NCESSCH`, `Name`, `State`,
`FIPS County Code`, `County`, `City`, the five domain columns, and
`Composite Score`. NCESSCH is read as text, checked for empties, and never used
as a model feature. Duplicate source IDs are reported in the app and
disambiguated with temporary `rq` row keys.

## Matching behavior

- Every search starts with an independent copy of the full dataset and uses
  only explicitly selected domain columns. Geography is an optional filter.
- Valid selected school scores prefill editable search inputs. A manually
  entered value applies only to that search and never changes the CSV. If a
  selected target value remains blank, that domain is omitted and the UI says
  so.
- Peer rows missing or containing invalid values in any domain used for that
  search are excluded. Missing values in unselected domains do not matter.
- `StandardScaler` is fit on eligible peer schools. Euclidean distances are
  computed from the transformed target vector. Domains have equal weight after
  standardization. Results are ordered by distance, with `rq` as a deterministic
  tie-breaker. K is 3–10 (default 5); the target cannot appear in its own list.
- The table shows all domain values, the distance, and the domain contributing
  most to each distance. A profile chart compares the selected school with the
  closest peer.

## Data limitations and privacy

`pre_cleaned_data.csv` is the repository's prototype input. The repository does
not document its upstream source, update schedule, score definitions, score
direction, or reference year. The interface therefore does not describe high
scores as good or bad. Similarity is descriptive, not causal and not an
official school rating or resource-allocation recommendation.

The current CSV has an identifier quality issue: 19,158 NCESSCH values are in
scientific notation, and 2,635 repeated-ID groups affect 13,001 rows. The app
preserves these source values as-is and adds an in-memory `rq` column (`rq-000001`,
etc.) based on CSV row order to distinguish records during the prototype. Self
exclusion uses `rq`; the original NCESSCH remains visible as supplied. These
temporary row keys can change if the CSV is reordered, so use corrected NCESSCH
values from the source when they are available. The CSV is never rewritten.

The app has no accounts, contact information, free-text fields, or persistent
user data. Peer discovery results are based on the public school metrics in the
CSV. Do not deploy this prototype for policymaker accounts or shared workspace
content without first defining authentication, authorization, hosting, data
retention, and moderation.

## Tests

```sh
python -m pytest
```
