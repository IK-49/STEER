# STEER: explore school community profiles

STEER is a Streamlit prototype for descriptive comparisons of community stress profiles around U.S. public schools. It provides nearest-profile search, profile comparisons, a domain-spread leaderboard, and a hypothetical score-adjustment control.

The app identifies a record by source school name, city, state, and district (or county when district is unavailable). It does not read or use NCESSCH. Exact duplicate name/location/profile rows are collapsed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py