from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from data import COMPOSITE, DOMAINS, IDENTIFIER, RECORD_KEY, DatasetValidationError, identifier_quality, load_dataset
from similarity import SearchError, find_similar_schools


def schools(count: int = 15) -> pd.DataFrame:
    rows = []
    for index in range(count):
        row = {
            IDENTIFIER: f"{index + 1:012d}",
            "Name": f"School {index}",
            "State": "CT" if index < 12 else "NY",
            "County": "Alpha" if index < 10 else "Beta",
            "City": f"City {index}",
            "FIPS County Code": "001",
            COMPOSITE: 30 + index,
        }
        row.update({domain: float(index + offset) for offset, domain in enumerate(DOMAINS)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_preserves_identifier_leading_zeros(tmp_path):
    frame = schools(3)
    path = tmp_path / "schools.csv"
    frame.to_csv(path, index=False)
    original_bytes = path.read_bytes()
    loaded = load_dataset(path)
    assert loaded.loc[0, IDENTIFIER] == "000000000001"
    assert loaded[IDENTIFIER].is_unique
    assert path.read_bytes() == original_bytes


def test_load_adds_unique_rq_keys_and_reports_duplicate_source_ids(tmp_path):
    frame = schools(3)
    frame.loc[1, IDENTIFIER] = frame.loc[0, IDENTIFIER]
    path = tmp_path / "schools.csv"
    frame.to_csv(path, index=False)
    original_bytes = path.read_bytes()
    loaded = load_dataset(path)
    assert loaded[RECORD_KEY].tolist() == ["rq-000001", "rq-000002", "rq-000003"]
    assert loaded[RECORD_KEY].is_unique
    assert identifier_quality(loaded)["affected_rows"] == 2
    assert path.read_bytes() == original_bytes

    # Empty identifiers remain invalid even though duplicate ones are disambiguated.
    frame.loc[1, IDENTIFIER] = ""
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="empty or invalid"):
        load_dataset(path)


def test_rq_prevents_self_match_when_source_ids_collide():
    frame = schools()
    frame[RECORD_KEY] = [f"rq-{i:06d}" for i in range(len(frame))]
    frame.loc[1, IDENTIFIER] = frame.loc[0, IDENTIFIER]
    result = find_similar_schools(frame, frame.loc[0, RECORD_KEY], ["Economic"], k=3)
    assert frame.loc[0, RECORD_KEY] not in set(result.matches[RECORD_KEY])
    assert frame.loc[1, IDENTIFIER] in set(result.matches[IDENTIFIER])


def test_streamlit_search_flow_runs_with_sample_dataset(tmp_path, monkeypatch):
    frame = schools()
    frame.loc[1, IDENTIFIER] = frame.loc[0, IDENTIFIER]
    path = tmp_path / "app_schools.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("STEER_DATA_PATH", str(path))
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.text_input[0].set_value("School 0").run()
    assert not app.exception
    assert any("Rank" in element.value.columns and "Distance" in element.value.columns for element in app.dataframe)
    assert any(RECORD_KEY in element.value.columns for element in app.dataframe)
    assert any("repeated-ID groups" in element.value for element in app.warning)


def test_k_domain_and_missing_target_validation():
    frame = schools()
    for invalid_k in (2, 11, 3.5, True):
        with pytest.raises(SearchError, match="K must"):
            find_similar_schools(frame, frame.loc[0, IDENTIFIER], DOMAINS, k=invalid_k)
    with pytest.raises(SearchError, match="at least one"):
        find_similar_schools(frame, frame.loc[0, IDENTIFIER], [], k=3)
    with pytest.raises(SearchError, match="unique NCESSCH"):
        find_similar_schools(frame, "absent", DOMAINS, k=3)


def test_results_are_sorted_exclude_target_and_repeatable():
    frame = schools()
    before = frame.copy(deep=True)
    target_id = frame.loc[5, IDENTIFIER]
    first = find_similar_schools(frame, target_id, ["Economic", "Housing"], k=5)
    second = find_similar_schools(frame, target_id, ["Economic", "Housing"], k=5)
    assert target_id not in set(first.matches[IDENTIFIER])
    assert first.matches["Distance"].is_monotonic_increasing
    assert first.matches[IDENTIFIER].tolist() == second.matches[IDENTIFIER].tolist()
    assert first.matches["Distance"].tolist() == second.matches["Distance"].tolist()
    pd.testing.assert_frame_equal(frame, before)


def test_unselected_missing_values_do_not_change_search():
    frame = schools()
    target_id = frame.loc[0, IDENTIFIER]
    baseline = find_similar_schools(frame, target_id, ["Economic", "Housing"], k=4)
    frame["Crime"] = frame["Crime"].astype(object)
    frame.loc[1:8, "Crime"] = "N/A"
    with_unselected_missing = find_similar_schools(frame, target_id, ["Economic", "Housing"], k=4)
    assert baseline.matches[IDENTIFIER].tolist() == with_unselected_missing.matches[IDENTIFIER].tolist()
    assert baseline.matches["Distance"].tolist() == with_unselected_missing.matches["Distance"].tolist()


def test_selected_missing_rows_are_excluded_and_counted():
    frame = schools()
    target_id = frame.loc[0, IDENTIFIER]
    frame["Crime"] = frame["Crime"].astype(object)
    frame.loc[1, "Crime"] = "N/A"
    result = find_similar_schools(frame, target_id, ["Economic", "Crime"], k=5)
    assert result.excluded_missing_count == 1
    assert frame.loc[1, IDENTIFIER] not in set(result.matches[IDENTIFIER])
    # The same missing value is irrelevant when Crime is not selected.
    result_without_crime = find_similar_schools(frame, target_id, ["Economic"], k=5)
    assert result_without_crime.excluded_missing_count == 0


def test_missing_target_domain_is_omitted_or_can_be_entered_manually():
    frame = schools()
    frame["Health"] = frame["Health"].astype(object)
    frame.loc[0, "Health"] = "N/A"
    before = frame.copy(deep=True)
    result = find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Health", "Economic"], k=3)
    assert result.domains == ("Economic",)
    assert result.excluded_target_domains == ("Health",)
    manual = find_similar_schools(
        frame, frame.loc[0, IDENTIFIER], ["Health", "Economic"], k=3,
        target_values={"Health": 12.5},
    )
    assert manual.domains == ("Health", "Economic")
    assert manual.target_values["Health"] == 12.5
    assert manual.excluded_target_domains == ()
    pd.testing.assert_frame_equal(frame, before)


def test_insufficient_complete_peers_fail_clearly():
    frame = schools()
    frame["Health"] = frame["Health"].astype(object)
    frame.loc[1:, "Health"] = "N/A"
    with pytest.raises(SearchError, match="Only 0 eligible peer"):
        find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Health"], k=3, target_values={"Health": 1})


def test_geographic_filter_is_explicit_and_must_include_target():
    frame = schools()
    result = find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Economic"], k=3, state="CT")
    assert set(result.matches["State"]) == {"CT"}
    with pytest.raises(SearchError, match="outside the chosen"):
        find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Economic"], k=3, state="NY")


def test_duplicate_domain_and_zero_complete_candidates_are_rejected():
    frame = schools()
    with pytest.raises(SearchError, match="unique choices"):
        find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Economic", "Economic"], k=3)
    frame["Crime"] = frame["Crime"].astype(object)
    frame["Crime"] = "N/A"
    with pytest.raises(SearchError, match="No selected domains"):
        find_similar_schools(frame, frame.loc[0, IDENTIFIER], ["Crime"], k=3)
