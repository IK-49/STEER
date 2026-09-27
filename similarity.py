"""Per-search school similarity, mathematical modeling, and statutory grant synthesis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    cmi: float = 0.0
    dominant_domain: str = ""
    dominant_z: float = 0.0
    grant_program: str = ""
    brief_text: str = ""
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    tipping_point_reduction: float = 0.0


def _check_domains(domains: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = tuple(domains)
    if not selected:
        raise SearchError("Select at least one domain.")
    if len(set(selected)) != len(selected) or any(domain not in DOMAINS for domain in selected):
        raise SearchError("Domains must be unique choices from the five available metrics.")
    return selected


def calculate_tipping_point(
    target_values: dict[str, float],
    domains: tuple[str, ...],
    means: dict[str, float],
    stds: dict[str, float],
    dominant_domain: str,
    target_cmi: float = 1.0,
) -> float:
    """Binary search solver for minimum point reduction in dominant domain to reduce CMI <= target_cmi."""
    val_orig = float(target_values.get(dominant_domain, 0.0))
    if val_orig <= 0.0 or not domains or dominant_domain not in stds or stds[dominant_domain] == 0:
        return 0.0

    curr_z = [(target_values[d] - means[d]) / (stds[d] or 1.0) for d in domains]
    if float(np.std(curr_z)) <= target_cmi:
        return 0.0

    low, high = 0.0, val_orig
    best_reduction = val_orig

    for _ in range(25):
        mid = (low + high) / 2.0
        test_z = [
            (max(0.0, val_orig - mid) - means[d]) / (stds[d] or 1.0) if d == dominant_domain
            else (target_values[d] - means[d]) / (stds[d] or 1.0)
            for d in domains
        ]
        if float(np.std(test_z)) <= target_cmi:
            best_reduction = mid
            high = mid
        else:
            low = mid

    return round(float(best_reduction), 1)


def find_similar_schools(
    frame: pd.DataFrame,
    target_id: str,
    domains: tuple[str, ...] | list[str],
    k: int = 5,
    state: str | None = None,
    county: str | None = None,
    target_values: Mapping[str, float | None] | None = None,
) -> SearchResult:
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

    # Exclude target before fitting to eliminate data leakage
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

    # 1. Out-of-sample standardization with NumPy arrays to avoid scikit-learn warnings
    scaler = StandardScaler()
    candidate_scaled = scaler.fit_transform(numeric.to_numpy(dtype=float))
    target_vector = scaler.transform(np.array([[chosen_values[d] for d in effective_domains]], dtype=float))

    # 2. Optimized spatial index query using C-level priority queue heap
    n_query = min(len(pool), max(k * 2, 20))
    neighbors = NearestNeighbors(n_neighbors=n_query, metric="euclidean", algorithm="auto")
    neighbors.fit(candidate_scaled)
    distances, indices = neighbors.kneighbors(target_vector)

    ordered = sorted(
        ((float(distance), int(position)) for distance, position in zip(distances[0], indices[0])),
        key=lambda item: (item[0], str(pool.iloc[item[1]][identity_column])),
    )[:k]

    result = pool.iloc[[position for _, position in ordered]].copy().reset_index(drop=True)
    result.insert(0, "Rank", range(1, k + 1))
    result["Distance"] = [distance for distance, _ in ordered]
    result["Similarity %"] = [max(0.0, round(100.0 * (1.0 - (d / 3.5)), 1)) for d in result["Distance"]]

    # 3. Explainability: Top contributing domain
    match_positions = [position for _, position in ordered]
    target_scaled = target_vector[0]
    contributions = np.square(candidate_scaled[match_positions] - target_scaled)
    contribution_total = contributions.sum(axis=1)
    result["Top contributing domain"] = [
        effective_domains[int(row.argmax())] if total > 0 else "Identical selected scores"
        for row, total in zip(contributions, contribution_total)
    ]

    # 4. Innovation Metrics: CMI, Means, Stds, & Dominant Outlier Decomposition
    means_dict = {d: float(scaler.mean_[i]) for i, d in enumerate(effective_domains)}
    stds_dict = {d: float(scaler.scale_[i]) for i, d in enumerate(effective_domains)}
    target_z_dict = {
        d: (chosen_values[d] - means_dict[d]) / (stds_dict[d] or 1.0)
        for d in effective_domains
    }
    z_values = list(target_z_dict.values())
    cmi = round(float(np.std(z_values)), 2)
    dominant_domain = max(target_z_dict, key=target_z_dict.get)
    dominant_z = round(float(target_z_dict[dominant_domain]), 2)

    tipping_point = calculate_tipping_point(
        chosen_values, tuple(effective_domains), means_dict, stds_dict, dominant_domain, target_cmi=1.0
    )

    # 5. Statutory Federal Grant Crosswalk
    grant_map = {
        "Crime": "Title IV, Part A (Student Support & Academic Enrichment) & BJA STOP School Violence Act",
        "Housing": "McKinney-Vento Homeless Assistance Act",
        "Economic": "Title I, Part A Schoolwide Program Allocation",
        "Health": "HRSA School-Based Health Center Program",
        "Education": "Title III English Language Acquisition & Academic Achievement",
    }
    grant_program = grant_map.get(dominant_domain, "Title I Comprehensive Support & Improvement")

    # 6. Generate Deterministic Statutory Grant Evidence Brief
    peer_bullet_list = "\n".join([
        f"- {r['Name']} ({r['City']}, {r['State']}) | Distance: {r['Distance']:.3f} | Top Driver: {r['Top contributing domain']}"
        for _, r in result.iterrows()
    ])
    brief_text = f"""================================================================================
STATEMENT OF DEMONSTRATED NEED & STATUTORY BENCHMARK REPORT
Target Institution: {target['Name']} ({target['City']}, {target['State']})
Record Identifier: {target_id} | NCESSCH: {target[IDENTIFIER]}
================================================================================

1. EXECUTIVE SUMMARY & COMPOSITE DECOMPOSITION
While {target['Name']} carries an aggregate composite score of {target[COMPOSITE]},
dimensional decomposition reveals an acute localized crisis concentrated in:
DOMAIN: {dominant_domain.upper()} (Measured Score: {chosen_values[dominant_domain]}, Standardized Outlier: +{dominant_z}σ)

COMPOSITE MASKING INDEX (CMI): {cmi}σ
A high CMI demonstrates that scalar composite indices mask extreme operational
disparities, penalizing the school in conventional formula funding.

2. EMPIRICAL SISTER-SCHOOL BENCHMARKS (k-NN Subspace Retrieval)
In continuous standardized civic space, the following statistical twin institutions
operate under equivalent multi-domain environmental profiles:
{peer_bullet_list}

3. STATUTORY LEGISLATIVE ALLOCATION
Under federal administrative guidelines, this empirical profile qualifies for:
RECOMMENDED GRANT PROGRAM: {grant_program}
================================================================================
Generated via STEER (Statistical Twins for Educational Equity & Resources) | CDC 2026
"""

    location_parts = []
    if state:
        location_parts.append(f"state: {state}")
    if county:
        location_parts.append(f"county: {county}")
    scope = ", ".join(location_parts) if location_parts else "all states"

    return SearchResult(
        target, chosen_values, result, tuple(effective_domains),
        tuple(excluded_target_domains), excluded_missing_count, scope,
        cmi=cmi, dominant_domain=dominant_domain, dominant_z=dominant_z,
        grant_program=grant_program, brief_text=brief_text,
        means=means_dict, stds=stds_dict, tipping_point_reduction=tipping_point,
    )


def find_positive_deviants(
    frame: pd.DataFrame,
    target_id: str,
    k: int = 3,
) -> pd.DataFrame:
    """Finds mentor schools facing matching community headwinds with superior education scores."""
    headwind_domains = ["Economic", "Health", "Housing", "Crime"]
    outcome_domain = "Education"
    all_needed = headwind_domains + [outcome_domain]

    # Coerce to numeric FIRST so empty strings ("") become NaN before complete-case filtering
    numeric = frame.loc[:, all_needed].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    pool = frame.loc[complete].copy().reset_index(drop=True)
    numeric = numeric.loc[complete].copy().reset_index(drop=True)

    identity_column = RECORD_KEY if RECORD_KEY in pool.columns else IDENTIFIER
    target_rows = pool.loc[
        pool[identity_column].astype(str).str.strip() == str(target_id).strip()
    ]
    if target_rows.empty:
        return pd.DataFrame()
    target_idx = target_rows.index[0]

    candidates = pool.drop(index=target_idx).copy().reset_index(drop=True)
    cand_numeric = numeric.drop(index=target_idx).copy().reset_index(drop=True)

    if len(candidates) < k:
        return pd.DataFrame()

    scaler = StandardScaler()
    X_headwinds = scaler.fit_transform(
        cand_numeric[headwind_domains].to_numpy(dtype=float)
    )
    target_vec = scaler.transform(
        numeric.loc[target_idx, headwind_domains]
        .to_numpy(dtype=float)
        .reshape(1, -1)
    )

    knn = NearestNeighbors(
        n_neighbors=min(50, len(candidates)), metric="euclidean"
    )
    knn.fit(X_headwinds)
    dists, inds = knn.kneighbors(target_vec)

    mentor_pool = candidates.iloc[inds[0]].copy()
    mentor_pool["Headwind Match Distance"] = dists[0].round(3)
    mentor_pool["Education Score"] = cand_numeric.loc[
        inds[0], outcome_domain
    ].values
    target_edu = float(numeric.loc[target_idx, outcome_domain])
    mentor_pool["Education Outperformance (+pts)"] = (
        mentor_pool["Education Score"] - target_edu
    ).round(1)

    deviants = (
        mentor_pool[mentor_pool["Education Outperformance (+pts)"] > 0]
        .sort_values(by="Education Outperformance (+pts)", ascending=False)
        .head(k)
    )
    return deviants


def get_systemic_masking_leaderboard(
    frame: pd.DataFrame,
    domains: tuple[str, ...] = DOMAINS,
    state: str | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Ranks schools by Composite Masking Index (CMI) to expose systemic algorithmic distortion."""
    pool = frame.copy()
    if state and state != "Nationwide":
        pool = pool.loc[pool["State"].astype(str).str.strip().str.casefold() == state.strip().casefold()]
    numeric = pool.loc[:, list(domains)].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    pool = pool.loc[complete].copy()
    numeric = numeric.loc[complete].copy()
    if pool.empty:
        return pd.DataFrame()

    means = numeric.mean()
    stds = numeric.std().replace(0, 1.0)
    z = (numeric - means) / stds

    pool["CMI (σ)"] = z.std(axis=1).round(2)
    pool["Dominant Crisis"] = z.idxmax(axis=1)
    pool["Outlier (σ)"] = z.max(axis=1).round(2)

    top = pool.sort_values(by="CMI (σ)", ascending=False).head(n)
    cols = ["Name", "City", "State", COMPOSITE, "CMI (σ)", "Dominant Crisis", "Outlier (σ)", *domains]
    return top[[c for c in cols if c in top.columns]].reset_index(drop=True)


def synthesize_llm_grant_narrative(result: SearchResult, target_row: pd.Series) -> str:
    """Generates an executive policy narrative via an LLM, falling back cleanly to the statutory brief."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return result.brief_text

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = f"""
You are a federal educational grant director drafting an urgent Statement of Demonstrated Need.
Rely STRICTLY on the empirical data provided below. Do not fabricate programs, metrics, or citations.

TARGET INSTITUTION: {target_row['Name']} ({target_row['City']}, {target_row['State']})
COMPOSITE HARDSHIP SCORE: {target_row[COMPOSITE]} / 100
DOMINANT CIVIC CRISIS: {result.dominant_domain} (+{result.dominant_z}σ national outlier)
COMPOSITE MASKING INDEX (CMI): {result.cmi}σ (severe dimensional distortion)
STATUTORY GRANT PROGRAM TARGET: {result.grant_program}
STATISTICAL TWIN BENCHMARK PEERS:
{result.matches[['Name', 'State', 'Distance', 'Top contributing domain']].head(3).to_string(index=False)}

Compose a concise, high-impact 3-paragraph executive grant narrative:
Paragraph 1: Deconstruct why the school's composite rating masks its acute operational emergency in {result.dominant_domain}.
Paragraph 2: Cite the statistical twin cohort as empirical proof that distress is driven by macro-environmental determinants.
Paragraph 3: State the statutory request under {result.grant_program} to deliver immediate targeted capital.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content or result.brief_text
    except Exception:
        return result.brief_text