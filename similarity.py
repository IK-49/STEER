"""
Profile comparisons and descriptive domain-spread calculations.

Authors: Izad Khokhar, Anish Velagapudi, Aurick Smart
AI Attribution: Initial code structure, boilerplate definitions, and mathematical
outlines were drafted with AI assistance (GitHub Copilot / LLM). Subspace slicing,
out-of-sample standardization, binary search bounds, tie-breaking logic, and API
handling were refactored, verified, and customized by the project team.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data import COMPOSITE, DOMAINS, RECORD_KEY


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
    tipping_point_reduction: float | None = 0.0


def _check_domains(domains: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(domains))
    if not selected:
        raise SearchError("Select at least one domain.")
    if any(domain not in DOMAINS for domain in selected):
        raise SearchError(f"Domains must be unique selections from: {DOMAINS}")
    return selected


def calculate_tipping_point(
    target_values: dict[str, float],
    domains: tuple[str, ...],
    means: dict[str, float],
    stds: dict[str, float],
    dominant_domain: str,
    target_cmi: float = 1.0,
) -> float | None:
    """Solve for the minimum score reduction in the dominant domain to bring spread <= target_cmi."""
    val_orig = float(target_values.get(dominant_domain, 0.0))
    if len(domains) <= 1 or dominant_domain not in stds or val_orig <= 0.0 or stds[dominant_domain] == 0:
        return 0.0 if len(domains) <= 1 else None

    curr_z = [(target_values[d] - means[d]) / (stds[d] or 1.0) for d in domains]
    if float(np.std(curr_z, ddof=0)) <= target_cmi:
        return 0.0

    other_z = [z for d, z in zip(domains, curr_z) if d != dominant_domain]
    mean_other_z = float(np.mean(other_z))
    scale = stds[dominant_domain] or 1.0
    max_reduction = min(val_orig, max(0.0, (curr_z[domains.index(dominant_domain)] - mean_other_z) * scale))

    low, high = 0.0, max_reduction
    solved = None

    for _ in range(25):
        mid = (low + high) / 2.0
        test_z = [
            (max(0.0, val_orig - mid) - means[d]) / (stds[d] or 1.0) if d == dominant_domain
            else (target_values[d] - means[d]) / (stds[d] or 1.0)
            for d in domains
        ]
        if float(np.std(test_z, ddof=0)) <= target_cmi:
            solved = mid
            high = mid
        else:
            low = mid

    return round(float(solved), 1) if solved is not None else None


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
    if not isinstance(k, int) or isinstance(k, bool) or not (3 <= k <= 10):
        raise SearchError("K must be an integer from 3 through 10.")

    target_rows = frame.loc[frame[RECORD_KEY].astype(str).str.strip() == str(target_id).strip()]
    if len(target_rows) != 1:
        raise SearchError("Select a school with a single unambiguous identity.")
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
        raise SearchError("No selected domains contain valid scores for this target institution.")

    pool = frame
    if state and state != "Nationwide":
        pool = pool.loc[pool["State"].astype(str).str.strip().str.casefold() == state.strip().casefold()]
    if county and "County" in pool:
        pool = pool.loc[pool["County"].astype(str).str.strip().str.casefold() == county.strip().casefold()]

    target_mask = pool[RECORD_KEY].astype(str).str.strip().eq(str(target_id).strip())
    if not target_mask.any():
        raise SearchError("The selected school is outside the chosen geographic filter.")

    # Exclude target school before fitting to eliminate data leakage
    candidates = pool.loc[~target_mask].copy()
    numeric = candidates.loc[:, effective_domains].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    excluded_missing_count = int((~complete).sum())

    candidates = candidates.loc[complete].reset_index(drop=True)
    numeric_arr = numeric.loc[complete].to_numpy(dtype=float)

    if len(candidates) < k:
        raise SearchError(
            f"Only {len(candidates)} candidate schools have complete records in this scope. "
            "Broaden scope or reduce K."
        )

    # Standardize candidate pool
    scaler = StandardScaler()
    cand_scaled = scaler.fit_transform(numeric_arr)
    targ_vec = np.array([[chosen_values[d] for d in effective_domains]], dtype=float)
    target_scaled = scaler.transform(targ_vec)

    # Compute Euclidean distances and sort deterministically
    diffs = cand_scaled - target_scaled[0]
    all_distances = np.linalg.norm(diffs, axis=1)

    names = candidates["Name"].astype(str).str.casefold().to_numpy()
    keys = candidates[RECORD_KEY].astype(str).to_numpy()
    positions = np.lexsort((keys, names, all_distances))[:k]

    result = candidates.iloc[positions].copy().reset_index(drop=True)
    result.insert(0, "Rank", range(1, k + 1))
    result["Distance"] = [float(all_distances[p]) for p in positions]

    # Explainability: Top contributing domain
    contributions = diffs[positions] ** 2
    max_dims = np.argmax(contributions, axis=1)
    result["Top contributing domain"] = [effective_domains[int(idx)] for idx in max_dims]

    # Compute domain-profile spread metric
    means_dict = {d: float(scaler.mean_[i]) for i, d in enumerate(effective_domains)}
    stds_dict = {d: float(scaler.scale_[i]) for i, d in enumerate(effective_domains)}
    target_z_dict = {d: (chosen_values[d] - means_dict[d]) / (stds_dict[d] or 1.0) for d in effective_domains}

    z_vals = list(target_z_dict.values())
    cmi = round(float(np.std(z_vals, ddof=0)), 2)
    dominant_domain = max(target_z_dict, key=target_z_dict.get)
    dominant_z = round(float(target_z_dict[dominant_domain]), 2)

    tipping_point = calculate_tipping_point(
        chosen_values, tuple(effective_domains), means_dict, stds_dict, dominant_domain, target_cmi=1.0
    )

    grant_map = {
        "Crime": "Title IV, Part A & STOP School Violence Act",
        "Housing": "McKinney-Vento Homeless Assistance Act",
        "Economic": "Title I, Part A Schoolwide Program Allocation",
        "Health": "HRSA School-Based Health Center Program",
        "Education": "Title III English Language Acquisition",
    }
    grant_program = grant_map.get(dominant_domain, "General Federal & State Formula Grants")

    peer_lines = "\n".join([
        f"- {r['Name']} ({r['City']}, {r['State']}) | Distance: {r['Distance']:.3f} | Top Driver: {r['Top contributing domain']}"
        for _, r in result.iterrows()
    ])
    brief_text = f"""================================================================================
EXPLORATORY SCHOOL PROFILE COMPARISON
Target Institution: {target['Name']} ({target['City']}, {target['State']})
Identity: school name and location (federal ID not utilized)
================================================================================

1. DESCRIPTIVE DOMAIN PROFILE
Composite score: {target[COMPOSITE]}. Highest relative domain: {dominant_domain.upper()} (score: {chosen_values[dominant_domain]}, z: {dominant_z:+.2f}σ).
Domain-profile spread: {cmi}σ (descriptive standard deviation of selected domain z-scores).

2. NEAREST PROFILE RECORDS (Subspace Retrieval)
{peer_lines}

3. PROGRAM AREA TO RESEARCH
Reference: {grant_program}.
Independent verification required; not a formal determination of funding eligibility.
================================================================================
"""

    scope_parts = [f"state: {state}"] if state and state != "Nationwide" else []
    if county:
        scope_parts.append(f"county: {county}")
    scope = ", ".join(scope_parts) if scope_parts else "all states"

    return SearchResult(
        target=target,
        target_values=chosen_values,
        matches=result,
        domains=tuple(effective_domains),
        excluded_target_domains=tuple(excluded_target_domains),
        excluded_missing_count=excluded_missing_count,
        scope_description=scope,
        cmi=cmi,
        dominant_domain=dominant_domain,
        dominant_z=dominant_z,
        grant_program=grant_program,
        brief_text=brief_text,
        means=means_dict,
        stds=stds_dict,
        tipping_point_reduction=tipping_point,
    )


def find_positive_deviants(
    frame: pd.DataFrame,
    target_id: str,
    k: int = 3,
    state: str | None = None,
) -> pd.DataFrame:
    """Find comparison institutions facing matching neighborhood headwinds with higher Education scores."""
    headwinds = ["Economic", "Health", "Housing", "Crime"]
    outcome = "Education"
    required = headwinds + [outcome]

    pool = frame.copy(deep=False)
    if state and state != "Nationwide":
        pool = pool.loc[pool["State"].astype(str).str.strip().str.casefold() == state.strip().casefold()]

    numeric = pool.loc[:, required].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    pool = pool.loc[complete].reset_index(drop=True)
    numeric = numeric.loc[complete].reset_index(drop=True)

    target_rows = pool.loc[pool[RECORD_KEY].astype(str).str.strip() == str(target_id).strip()]
    if target_rows.empty:
        return pd.DataFrame()
    target_idx = target_rows.index[0]

    candidates = pool.drop(index=target_idx).reset_index(drop=True)
    cand_numeric = numeric.drop(index=target_idx).reset_index(drop=True)

    if len(candidates) < k:
        return pd.DataFrame()

    scaler = StandardScaler()
    X_headwinds = scaler.fit_transform(cand_numeric[headwinds].to_numpy(dtype=float))
    target_vec = scaler.transform(numeric.loc[target_idx, headwinds].to_numpy(dtype=float).reshape(1, -1))

    knn = NearestNeighbors(n_neighbors=min(50, len(candidates)), metric="euclidean")
    knn.fit(X_headwinds)
    dists, inds = knn.kneighbors(target_vec)

    mentor_pool = candidates.iloc[inds[0]].copy()
    mentor_pool["Headwind Match Distance"] = dists[0].round(3)
    mentor_pool["Education Score"] = cand_numeric.loc[inds[0], outcome].values
    target_edu = float(numeric.loc[target_idx, outcome])
    mentor_pool["Education Outperformance (+pts)"] = (mentor_pool["Education Score"] - target_edu).round(1)

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
    """Rank schools by z-score spread across selected domains."""
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
    stds = numeric.std(ddof=0).replace(0, 1.0)
    z = (numeric - means) / stds

    pool["Domain-profile spread (σ)"] = z.std(axis=1, ddof=0).round(2)
    pool["Highest relative domain"] = z.idxmax(axis=1)
    pool["Highest relative z (σ)"] = z.max(axis=1).round(2)

    top = pool.sort_values(by="Domain-profile spread (σ)", ascending=False).head(n)
    cols = ["Name", "City", "State", COMPOSITE, "Domain-profile spread (σ)", "Highest relative domain", "Highest relative z (σ)", *domains]
    return top[[c for c in cols if c in top.columns]].reset_index(drop=True)


def _resolve_gemini_key(user_key: str | None = None) -> str | None:
    """Check user input, Streamlit secrets, and environment for Gemini API key."""
    if user_key and user_key.strip():
        return user_key.strip()
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def synthesize_llm_grant_narrative(
    result: SearchResult,
    target_row: pd.Series,
    user_api_key: str | None = None,
) -> str:
    """Draft a descriptive summary via Gemini API with zero-dependency REST fallback."""
    api_key = _resolve_gemini_key(user_api_key)
    if not api_key:
        return (
            result.brief_text
            + "\n\n[Note: Gemini API Key not detected. Enter your key in the sidebar, "
            "set GEMINI_API_KEY in your shell environment, or configure .streamlit/secrets.toml.]"
        )

    prompt = f"""
Write a concise, neutral summary of exploratory school-profile calculations.
Do not infer causes, school performance, need, intervention effects, funding eligibility,
or legal conclusions. Do not invent citations. State that similarity and score adjustments
are descriptive and require independent validation.

TARGET INSTITUTION: {target_row['Name']} ({target_row['City']}, {target_row['State']})
COMPOSITE HARDSHIP SCORE: {target_row[COMPOSITE]} / 100
Highest relative selected domain: {result.dominant_domain} ({result.dominant_z:+.2f}σ in this comparison cohort)
Domain-profile spread: {result.cmi}σ
Potential program reference for independent research: {result.grant_program}
Nearest comparison records:
{result.matches[['Name', 'State', 'Distance', 'Top contributing domain']].head(3).to_string(index=False)}

Include a short final sentence: "This exploratory output is not a funding recommendation
or eligibility determination; verify all source data and program requirements independently."
"""

    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = ""

    # Primary: Official google-genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_id in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                if response.text:
                    return response.text
            except Exception as e:
                last_error = str(e)
                continue
    except ImportError:
        pass

    # Zero-dependency REST fallback using standard library urllib
    for model_id in models_to_try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                if text:
                    return text
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {err.code}: {err.reason} - {err_body}"
        except Exception as exc:
            last_error = str(exc)

    return (
        result.brief_text
        + f"\n\n[Gemini API Call Failed: {last_error}. Please check your API key and network connection.]"
    )