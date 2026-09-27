"""Read-only loading of school profiles keyed by name and location."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DOMAINS = ("Economic", "Education", "Health", "Housing", "Crime")
COMPOSITE = "Composite Score"
RECORD_KEY = "_school_key"
IDENTITY_COLUMNS = ("Name", "City", "State", "School District")
OPTIONAL_COLUMNS = ("County", "FIPS County Code")
SCORE_COLUMNS = (*DOMAINS, COMPOSITE)
REQUIRED_COLUMNS = (*IDENTITY_COLUMNS[:3], *SCORE_COLUMNS)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "Data" / "index_scores_v3_2026.csv"
MAX_DATASET_BYTES = 100 * 1024 * 1024
NULL_TOKENS = {"", "na", "n/a", "nan", "none", "<na>"}

COLUMN_ALIASES = {
    "School_Name": "Name", "County_Name": "County",
    "FIPS_County_Code": "FIPS County Code", "Composite_Score": COMPOSITE,
}


class DatasetValidationError(ValueError):
    """Raised when source rows cannot safely be used as school profiles."""


def configured_data_path() -> Path:
    configured = os.environ.get("STEER_DATA_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_PATH


def _school_key(row: pd.Series, identity_columns: tuple[str, ...]) -> str:
    values = [str(row[column]).strip() if pd.notna(row[column]) else "" for column in identity_columns]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load relevant columns, never reading NCESSCH, and reject ambiguous name keys."""
    source = Path(path).expanduser().resolve() if path else configured_data_path()
    if not source.is_file():
        raise DatasetValidationError(f"Dataset file was not found: {source}")
    if source.stat().st_size > MAX_DATASET_BYTES:
        raise DatasetValidationError("Dataset is larger than the 100 MiB prototype limit.")
    try:
        header = pd.read_csv(source, nrows=0).columns
        renamed = [COLUMN_ALIASES.get(column, column) for column in header]
        available = set(renamed)
        required_missing = set(REQUIRED_COLUMNS) - available
        if required_missing:
            raise DatasetValidationError("Dataset is missing required columns: " + ", ".join(sorted(required_missing)))
        # School District is loaded when present; County supplies the row-level fallback.
        use_columns = set(REQUIRED_COLUMNS) | set(IDENTITY_COLUMNS) | set(OPTIONAL_COLUMNS)
        frame = pd.read_csv(
            source,
            usecols=lambda column: COLUMN_ALIASES.get(column, column) in use_columns,
            dtype="string", keep_default_na=False, low_memory=False,
        ).rename(columns=COLUMN_ALIASES)
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetValidationError(f"Could not read the dataset: {exc}") from exc
    for column in (*IDENTITY_COLUMNS, *OPTIONAL_COLUMNS):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].astype("string").str.strip()
    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column].replace({token: pd.NA for token in NULL_TOKENS}), errors="coerce")
    if frame.empty:
        raise DatasetValidationError("Dataset contains no school records.")
    named = frame["Name"].fillna("").str.strip().ne("")
    frame = frame.loc[named].copy()
    district = frame["School District"].fillna("").str.strip()
    county = frame["County"].fillna("").str.strip()
    frame["_identity_area"] = district.where(district.ne(""), county)
    identity = ("Name", "City", "State", "_identity_area")
    profiles = frame.drop_duplicates(subset=[*identity, *SCORE_COLUMNS])
    counts = profiles.groupby(list(identity), dropna=False).size()
    if (counts > 1).any():
        raise DatasetValidationError(
            f"{int((counts > 1).sum())} name/location keys have different score profiles; clarify names before matching."
        )
    frame = profiles.drop_duplicates(subset=list(identity)).copy()
    frame[RECORD_KEY] = frame.apply(lambda row: _school_key(row, identity), axis=1)
    frame["Display name"] = frame.apply(
        lambda row: f"{row['Name']} — {row['City'] or 'City unavailable'}, {row['State'] or 'State unavailable'}"
        + (f" · District: {row['School District']}" if row["School District"] else
           (f" · County: {row['County']}" if row["County"] else "")), axis=1,
    )
    return frame.reset_index(drop=True)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.loc[:, DOMAINS].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"Domain": DOMAINS, "Missing schools": numeric.isna().sum().to_numpy()})
