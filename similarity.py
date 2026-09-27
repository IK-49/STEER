"""Per-search school similarity, isolated from the Streamlit interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data import COMPOSITE, DOMAINS, IDENTIFIER, RECORD_KEY


class SearchError(ValueError):
    """Raised when a requested search cannot produce valid matches."""


@dataclass(frozen=True)
class SearchResult:
    target: pd.Series
    target_values: dict[str, float]
    matches: pd.DataFrame
    domains: tuple[str, ...]
    excluded_target_domains: tuple[str, ...]
    excluded_missing_count: int
    scope_description: str


def _check_domains(domains: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = tuple(domains)
    if not selected:
        raise SearchError("Select at least one domain.")
    if len(set(selected)) != len(selected) or any(domain not in DOMAINS for domain in selected):
        raise SearchError("Domains must be unique choices from the five available metrics.")
    return selected


def find_similar_schools(
    frame: pd.DataFrame,
    target_id: str,
    domains: tuple[str, ...] | list[str],
    k: int = 5,
    state: str | None = None,
    county: str | None = None,
    target_values: Mapping[str, float | None] | None = None,
) -> SearchResult:
    """Find K closest complete-case peers in selected standardized domains.

    The target's valid source values are used by default. A numeric override in
    ``target_values`` is used only for this search. A missing target value with
    no override removes that domain from this search and is reported. Candidate
    rows missing any remaining selected domain are excluded. All operations
    use a fresh copy; the input frame and source CSV are never modified.
    """
    selected = _check_domains(domains)
    if not isinstance(k, int) or isinstance(k, bool) or not 3 <= k <= 10:
        raise SearchError("K must be an integer from 3 through 10.")
    identity_column = RECORD_KEY if RECORD_KEY in frame.columns else IDENTIFIER
    required = {identity_column, IDENTIFIER, *selected, "State", "County", "Name", COMPOSITE}
    if not required.issubset(frame.columns):
        raise SearchError("The dataset does not contain all fields required for this search.")
    target_id = str(target_id).strip()
    target_rows = frame.loc[frame[identity_column].astype(str).str.strip() == target_id]
    if len(target_rows) != 1:
        raise SearchError("Select a school with one unique NCESSCH identifier.")
    target = target_rows.iloc[0].copy()

    chosen_values: dict[str, float] = {}
    effective_domains: list[str] = []
    excluded_target_domains: list[str] = []
    overrides = target_values or {}
    for domain in selected:
        raw_value = overrides.get(domain, target[domain])
        value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if pd.isna(value):
            excluded_target_domains.append(domain)
        else:
            chosen_values[domain] = float(value)
            effective_domains.append(domain)
    if not effective_domains:
        raise SearchError("No selected domains have values for the target. Enter a score or select a domain with data.")

    pool = frame.copy(deep=True)
    if state:
        pool = pool.loc[pool["State"].astype(str).str.strip().str.casefold() == state.strip().casefold()].copy()
    if county:
        pool = pool.loc[pool["County"].astype(str).str.strip().str.casefold() == county.strip().casefold()].copy()
    target_mask = pool[identity_column].astype(str).str.strip().eq(target_id)
    if not target_mask.any():
        raise SearchError("The selected school is outside the chosen geographic filter.")

    # Exclude the target before fitting so target values (including manual
    # overrides) cannot affect candidate imputation or scaling statistics.
    pool = pool.loc[~target_mask].copy()
    numeric = pool.loc[:, effective_domains].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    excluded_missing_count = int((~complete).sum())
    pool = pool.loc[complete].copy().reset_index(drop=True)
    numeric = numeric.loc[complete].copy().reset_index(drop=True)
    if len(pool) < k:
        raise SearchError(
            f"Only {len(pool)} eligible peer schools have complete data for the selected domains; "
            "choose a wider area, fewer domains, or smaller K."
        )

    scaler = StandardScaler()
    candidate_scaled = scaler.fit_transform(numeric.to_numpy(dtype=float))
    target_vector = scaler.transform(np.array([[chosen_values[d] for d in effective_domains]], dtype=float))
    neighbors = NearestNeighbors(n_neighbors=len(pool), metric="euclidean", algorithm="auto")
    neighbors.fit(candidate_scaled)
    distances, indices = neighbors.kneighbors(target_vector)
    ordered = sorted(
        ((float(distance), int(position)) for distance, position in zip(distances[0], indices[0])),
        key=lambda item: (item[0], str(pool.iloc[item[1]][identity_column])),
    )[:k]

    result = pool.iloc[[position for _, position in ordered]].copy().reset_index(drop=True)
    result.insert(0, "Rank", range(1, k + 1))
    result["Distance"] = [distance for distance, _ in ordered]
    match_positions = [position for _, position in ordered]
    target_scaled = target_vector[0]
    contributions = np.square(candidate_scaled[match_positions] - target_scaled)
    contribution_total = contributions.sum(axis=1)
    for domain_index, column in enumerate(effective_domains):
        result[f"{column} contribution"] = np.divide(
            contributions[:, domain_index], contribution_total,
            out=np.zeros(len(result), dtype=float), where=contribution_total > 0,
        )
    result["Top contributing domain"] = [
        effective_domains[int(row.argmax())] if total > 0 else "Identical selected scores"
        for row, total in zip(contributions, contribution_total)
    ]
    location_parts = []
    if state:
        location_parts.append(f"state: {state}")
    if county:
        location_parts.append(f"county: {county}")
    scope = ", ".join(location_parts) if location_parts else "all states"
    return SearchResult(target, chosen_values, result, tuple(effective_domains), tuple(excluded_target_domains), excluded_missing_count, scope)
