"""Read-only loading and validation for the school source dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

IDENTIFIER = "NCESSCH"
RECORD_KEY = "rq"
DOMAINS = ("Economic", "Education", "Health", "Housing", "Crime")
METADATA = ("Name", "State", "FIPS County Code", "County", "City")
COMPOSITE = "Composite Score"
REQUIRED_COLUMNS = (IDENTIFIER, *METADATA, *DOMAINS, COMPOSITE)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "pre_cleaned_data.csv"
MAX_DATASET_BYTES = 100 * 1024 * 1024
SCIENTIFIC_ID_PATTERN = r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][+-]?\d+\s*$"


class DatasetValidationError(ValueError):
    """Raised when the source cannot safely support matching."""


def configured_data_path() -> Path:
    """Resolve the data path without exposing configuration through the UI."""
    configured = os.environ.get("STEER_DATA_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_PATH


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load an independent dataframe, validating identifiers and schema.

    The source file is only read. Feature cleanup happens on per-search copies
    in :mod:`similarity`.
    """
    source = Path(path).expanduser().resolve() if path else configured_data_path()
    if not source.is_file():
        raise DatasetValidationError(f"Dataset file was not found: {source}")
    if source.stat().st_size > MAX_DATASET_BYTES:
        raise DatasetValidationError("Dataset is larger than the 100 MiB prototype limit.")
    try:
        frame = pd.read_csv(source, dtype={IDENTIFIER: "string"}, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetValidationError(f"Could not read the dataset: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetValidationError("Dataset is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise DatasetValidationError("Dataset contains no school records.")

    ids = frame[IDENTIFIER].astype("string").str.strip()
    invalid = ids.isna() | ids.eq("") | ids.str.lower().isin({"nan", "none", "na", "n/a", "<na>"})
    if invalid.any():
        raise DatasetValidationError(f"Dataset has {int(invalid.sum())} empty or invalid NCESSCH identifiers.")
    # NCESSCH is retained exactly as text, even where the source has collisions.
    # rq disambiguates source rows until the upstream identifiers are corrected.
    frame[IDENTIFIER] = ids
    frame.insert(0, RECORD_KEY, [f"rq-{position:06d}" for position in range(1, len(frame) + 1)])
    return frame


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts for the five domain metrics."""
    numeric = frame.loc[:, DOMAINS].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"Domain": DOMAINS, "Missing schools": numeric.isna().sum().to_numpy()})


def identifier_quality(frame: pd.DataFrame) -> dict[str, int]:
    """Summarize source identifier collisions without rewriting identifiers."""
    ids = frame[IDENTIFIER].astype("string").str.strip()
    duplicated = ids.duplicated(keep=False)
    return {
        "duplicate_groups": int(ids[duplicated].nunique()),
        "affected_rows": int(duplicated.sum()),
        "scientific_notation_rows": int(ids.str.contains(SCIENTIFIC_ID_PATTERN, regex=True, na=False).sum()),
    }
