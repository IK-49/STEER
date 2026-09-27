"""
Integration tests for k-NN / similarity tooling.

Authors: Izad Khokhar, Anish Velagapudi, Aurick Smart
"""

from pathlib import Path

import pandas as pd
import pytest

from data import COMPOSITE, DOMAINS, RECORD_KEY, DatasetValidationError, load_dataset
from similarity import (
    SearchError,
    calculate_tipping_point,
    find_positive_deviants,
    find_similar_schools,
    get_systemic_masking_leaderboard,
)


def sample_schools(count: int = 15) -> pd.DataFrame:
    rows = []
    for index in range(count):
        row = {
            "Name": f"School {index}", "City": f"City {index}",
            "State": "CT" if index < 12 else "NY",
            "School District": f"District {index // 2}", "County": "Alpha",
            COMPOSITE: float(30 + index),
        }
        row.update({domain: float(index + offset) for offset, domain in enumerate(DOMAINS)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_loader_uses_name_location_identity_and_never_loads_identifier(tmp_path: Path) -> None:
    frame = sample_schools(3)
    frame["NCESSCH"] = ["damaged", "000000000002", "000000000003"]
    path = tmp_path / "schools.csv"
    frame.to_csv(path, index=False)
    original_bytes = path.read_bytes()

    loaded = load_dataset(path)

    assert "NCESSCH" not in loaded.columns
    assert loaded[RECORD_KEY].is_unique
    assert loaded["Display name"].str.contains("District").all()
    assert path.read_bytes() == original_bytes


def test_loader_collapses_exact_duplicate_and_rejects_conflicting_name_key(tmp_path: Path) -> None:
    frame = sample_schools(3)
    path = tmp_path / "schools.csv"
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    assert len(load_dataset(path)) == 3

    conflict = pd.concat([frame, frame.iloc[[0]].assign(Economic=99)], ignore_index=True)
    conflict.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="different score profiles"):
        load_dataset(path)


def test_neighbors_exclude_target_sort_and_leave_input_unchanged() -> None:
    frame = sample_schools()
    frame[RECORD_KEY] = [f"key-{i}" for i in range(len(frame))]
    original = frame.copy(deep=True)
    result = find_similar_schools(frame, "key-5", ["Economic", "Housing"], k=5)

    assert "key-5" not in set(result.matches[RECORD_KEY])
    assert result.matches["Distance"].is_monotonic_increasing
    assert "Similarity %" not in result.matches
    pd.testing.assert_frame_equal(frame, original)


def test_equal_distance_neighbors_have_stable_name_order() -> None:
    frame = sample_schools(6)
    frame[RECORD_KEY] = [f"key-{i}" for i in range(len(frame))]
    for domain in DOMAINS:
        frame[domain] = 10.0
    result = find_similar_schools(frame, "key-0", ["Economic"], k=3)
    assert result.matches["Name"].tolist() == ["School 1", "School 2", "School 3"]


def test_missing_candidate_rows_and_geographic_filter_are_explicit() -> None:
    frame = sample_schools()
    frame[RECORD_KEY] = [f"key-{i}" for i in range(len(frame))]
    frame["Crime"] = frame["Crime"].astype(object)
    frame.loc[1, "Crime"] = "N/A"

    result = find_similar_schools(frame, "key-0", ["Economic", "Crime"], k=5)
    assert result.excluded_missing_count == 1
    assert "key-1" not in set(result.matches[RECORD_KEY])
    scoped = find_similar_schools(frame, "key-0", ["Economic"], k=3, state="CT")
    assert set(scoped.matches["State"]) == {"CT"}
    with pytest.raises(SearchError, match="outside the chosen"):
        find_similar_schools(frame, "key-0", ["Economic"], k=3, state="NY")


def test_positive_deviance_and_leaderboard_features_return_descriptive_results() -> None:
    frame = sample_schools(20)
    frame[RECORD_KEY] = [f"key-{i}" for i in range(len(frame))]
    deviants = find_positive_deviants(frame, "key-5", k=3)
    assert len(deviants) <= 3
    assert (deviants["Education Outperformance (+pts)"] > 0).all()

    leaders = get_systemic_masking_leaderboard(frame, state="CT", n=5)
    assert len(leaders) == 5
    assert leaders["Domain-profile spread (σ)"].is_monotonic_decreasing
    assert "CMI (σ)" not in leaders


def test_tipping_point_is_minimal_and_noop_for_already_below_threshold() -> None:
    values = {"Economic": 9.0, "Housing": 1.0}
    means = {"Economic": 0.0, "Housing": 0.0}
    stds = {"Economic": 1.0, "Housing": 1.0}
    reduction = calculate_tipping_point(values, ("Economic", "Housing"), means, stds, "Economic")
    assert 0 < reduction <= 9.0
    assert calculate_tipping_point({"Economic": 2.0}, ("Economic",), means, stds, "Economic") == 0.0
    assert calculate_tipping_point(
        {"Economic": 0.0, "Housing": 10.0},
        ("Economic", "Housing"),
        {"Economic": 100.0, "Housing": 5.0},
        stds,
        "Housing",
    ) is None


def test_bundled_data_loads_with_name_keys() -> None:
    frame = load_dataset()
    assert len(frame) > 20_000
    assert frame[RECORD_KEY].is_unique
    assert frame["Name"].notna().all()
    assert not any(column.lower() == "ncessch" for column in frame.columns)
