#!/usr/bin/env python
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(".")
RESULTS_DIR = ROOT / "results" / "tables"
PLAUSIBLY_PATH = ROOT / "int_data" / "4_plausibly_exogenous" / "list_20july2024.xlsx"
N_PERMUTATIONS = 1000
MAX_JW_DISTANCE = 0.10
RNG_SEED = 20260221


def clean_title(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_join(values: pd.Series) -> str:
    out: List[str] = []
    seen = set()
    for v in values:
        t = clean_text(v)
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return "; ".join(out)


def longest_title(values: pd.Series) -> str:
    cleaned = [clean_text(v) for v in values if clean_text(v)]
    if not cleaned:
        return ""
    cleaned.sort(key=len, reverse=True)
    return cleaned[0]


def eo_label(k: int) -> str:
    return f"EO = 9" if k == 9 else f"EO >= {k}"


@dataclass
class MatchResult:
    matched: pd.DataFrame
    log: pd.DataFrame


def resolve_meta_path() -> Path:
    patterns = [
        "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
        "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet",
    ]
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "[EXOGENOUS_BENCHMARK] Missing meta parquet. Tried patterns: "
        + ", ".join(patterns)
    )


def prepare_plausibly() -> pd.DataFrame:
    px = pd.read_excel(PLAUSIBLY_PATH)
    required = {"title", "lhs", "rhs", "source_of_exogenous_variation"}
    missing = required.difference(px.columns)
    if missing:
        raise ValueError(
            f"[EXOGENOUS_BENCHMARK] Missing columns in benchmark file: {sorted(missing)}"
        )

    px = px[list(required)].copy()
    px["cleaned_title"] = px["title"].map(clean_title)
    px = px[px["cleaned_title"] != ""]

    agg = (
        px.groupby("cleaned_title", as_index=False)
        .agg(
            benchmark_title=("title", longest_title),
            rhs=("rhs", unique_join),
            lhs=("lhs", unique_join),
            source_of_exogenous_variation=("source_of_exogenous_variation", unique_join),
        )
    )
    agg["benchmark_id"] = [f"plausibly_{i:05d}" for i in range(1, len(agg) + 1)]
    agg["bench_cause_text"] = "The cause is: " + agg["rhs"].fillna("")
    agg["bench_effect_text"] = "The effect is: " + agg["lhs"].fillna("")
    agg["bench_exo_text"] = "The source(s) of exogenous variation are: " + agg[
        "source_of_exogenous_variation"
    ].fillna("")
    return agg[
        [
            "benchmark_id",
            "cleaned_title",
            "benchmark_title",
            "bench_cause_text",
            "bench_effect_text",
            "bench_exo_text",
        ]
    ].copy()


def prepare_llm_base(meta_path: Path) -> pd.DataFrame:
    cols = ["paper_id", "title", "edge_overlap", "cause", "effect", "sources_of_exogenous_variation"]
    df = pq.read_table(meta_path, columns=cols).to_pandas()
    df["edge_overlap"] = pd.to_numeric(df["edge_overlap"], errors="coerce")
    df = df[df["paper_id"].notna() & df["edge_overlap"].notna()].copy()
    return df


def build_llm_threshold(df: pd.DataFrame, k: int) -> pd.DataFrame:
    d = df[df["edge_overlap"] >= k].copy()
    if d.empty:
        return pd.DataFrame(
            columns=[
                "paper_id",
                "llm_title",
                "cleaned_title",
                "llm_cause_text",
                "llm_effect_text",
                "llm_exo_text",
            ]
        )

    agg = (
        d.groupby("paper_id", as_index=False)
        .agg(
            llm_title=("title", longest_title),
            cause=("cause", unique_join),
            effect=("effect", unique_join),
            exo=("sources_of_exogenous_variation", unique_join),
        )
    )
    agg["cleaned_title"] = agg["llm_title"].map(clean_title)
    agg = agg[agg["cleaned_title"] != ""].copy()
    agg = agg.sort_values(["cleaned_title", "paper_id"]).drop_duplicates("cleaned_title", keep="first")

    agg["llm_cause_text"] = "The causes are: " + agg["cause"].fillna("")
    agg["llm_effect_text"] = "The effects are: " + agg["effect"].fillna("")
    agg["llm_exo_text"] = "The source(s) of exogenous variation are: " + agg["exo"].fillna("")

    return agg[
        ["paper_id", "llm_title", "cleaned_title", "llm_cause_text", "llm_effect_text", "llm_exo_text"]
    ].copy()


def match_titles(llm: pd.DataFrame, bench: pd.DataFrame, max_dist: float = MAX_JW_DISTANCE) -> MatchResult:
    if llm.empty or bench.empty:
        empty = pd.DataFrame()
        return MatchResult(matched=empty, log=empty)

    exact_keys = sorted(set(llm["cleaned_title"]).intersection(set(bench["cleaned_title"])))
    exact_llm = llm[llm["cleaned_title"].isin(exact_keys)].copy()
    exact_bench = bench[bench["cleaned_title"].isin(exact_keys)].copy()
    exact = exact_llm.merge(exact_bench, on="cleaned_title", how="inner")
    exact["match_type"] = "exact"
    exact["distance"] = 0.0

    llm_un = llm[~llm["cleaned_title"].isin(exact_keys)].copy()
    bench_un = bench[~bench["cleaned_title"].isin(exact_keys)].copy()

    fuzzy_matches: List[Dict] = []
    if not llm_un.empty and not bench_un.empty:
        llm_choices = llm_un["cleaned_title"].tolist()
        candidate_rows = []
        for bi, q in enumerate(bench_un["cleaned_title"].tolist()):
            hit = process.extractOne(
                query=q,
                choices=llm_choices,
                scorer=JaroWinkler.normalized_similarity,
                score_cutoff=1.0 - max_dist,
            )
            if hit is None:
                continue
            _, similarity, li = hit
            distance = 1.0 - float(similarity)
            if distance <= max_dist:
                candidate_rows.append((bi, li, distance))

        candidate_rows.sort(key=lambda x: x[2])
        used_llm = set()
        for bi, li, distance in candidate_rows:
            if li in used_llm:
                continue
            used_llm.add(li)
            fuzzy_matches.append(
                {
                    "paper_id": llm_un.iloc[li]["paper_id"],
                    "llm_title": llm_un.iloc[li]["llm_title"],
                    "cleaned_title": llm_un.iloc[li]["cleaned_title"],
                    "llm_cause_text": llm_un.iloc[li]["llm_cause_text"],
                    "llm_effect_text": llm_un.iloc[li]["llm_effect_text"],
                    "llm_exo_text": llm_un.iloc[li]["llm_exo_text"],
                    "benchmark_id": bench_un.iloc[bi]["benchmark_id"],
                    "benchmark_title": bench_un.iloc[bi]["benchmark_title"],
                    "bench_cause_text": bench_un.iloc[bi]["bench_cause_text"],
                    "bench_effect_text": bench_un.iloc[bi]["bench_effect_text"],
                    "bench_exo_text": bench_un.iloc[bi]["bench_exo_text"],
                    "match_type": "fuzzy",
                    "distance": distance,
                }
            )

    fuzzy = pd.DataFrame(fuzzy_matches)
    if fuzzy.empty and exact.empty:
        empty = pd.DataFrame()
        return MatchResult(matched=empty, log=empty)

    matched = pd.concat([exact, fuzzy], ignore_index=True, sort=False)
    log = matched[
        [
            "paper_id",
            "benchmark_id",
            "cleaned_title",
            "llm_title",
            "benchmark_title",
            "match_type",
            "distance",
        ]
    ].copy()
    return MatchResult(matched=matched, log=log)


def rowwise_cosine(a_text: List[str], b_text: List[str], n_perm: int, seed: int) -> Dict[str, float]:
    if len(a_text) == 0:
        return {
            "max_similarity": np.nan,
            "mean_similarity": np.nan,
            "median_similarity": np.nan,
            "std_similarity": np.nan,
            "min_similarity": np.nan,
            "random_baseline_mean": np.nan,
            "random_baseline_std": np.nan,
            "lift_vs_baseline": np.nan,
            "permutation_p_value": np.nan,
        }

    a = pd.Series(a_text).fillna("").astype(str).tolist()
    b = pd.Series(b_text).fillna("").astype(str).tolist()
    n = len(a)

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(a + b)
    A = X[:n]
    B = X[n:]

    observed = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    obs_mean = float(np.mean(observed))

    rng = np.random.default_rng(seed)
    perm_means = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm_idx = rng.permutation(n)
        perm_vals = np.asarray(A.multiply(B[perm_idx]).sum(axis=1)).ravel()
        perm_means[i] = float(np.mean(perm_vals))

    baseline_mean = float(np.mean(perm_means))
    baseline_std = float(np.std(perm_means, ddof=1)) if n_perm > 1 else np.nan
    lift = obs_mean - baseline_mean
    p_value = float((np.sum(perm_means >= obs_mean) + 1.0) / (n_perm + 1.0))

    return {
        "max_similarity": float(np.max(observed)),
        "mean_similarity": obs_mean,
        "median_similarity": float(np.median(observed)),
        "std_similarity": float(np.std(observed, ddof=1)) if n > 1 else 0.0,
        "min_similarity": float(np.min(observed)),
        "random_baseline_mean": baseline_mean,
        "random_baseline_std": baseline_std,
        "lift_vs_baseline": float(lift),
        "permutation_p_value": p_value,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = resolve_meta_path()
    if not PLAUSIBLY_PATH.exists():
        raise FileNotFoundError(
            f"[EXOGENOUS_BENCHMARK] Missing plausibly benchmark: {PLAUSIBLY_PATH}"
        )

    print("[EXOGENOUS_BENCHMARK] Reading inputs...")
    llm_base = prepare_llm_base(meta_path)
    bench = prepare_plausibly()

    similarity_rows = []
    baseline_rows = []
    match_logs = []

    components = [
        ("Cause Similarity", "llm_cause_text", "bench_cause_text", 11),
        ("Effect Similarity", "llm_effect_text", "bench_effect_text", 23),
        ("Exogenous Variation Similarity", "llm_exo_text", "bench_exo_text", 37),
    ]

    for k in range(1, 10):
        lbl = eo_label(k)
        print(f"[EXOGENOUS_BENCHMARK] EO threshold {lbl}...")
        llm_k = build_llm_threshold(llm_base, k)
        m = match_titles(llm_k, bench, MAX_JW_DISTANCE)
        if m.matched.empty:
            continue

        m.log["eo_threshold"] = k
        m.log["eo_label"] = lbl
        match_logs.append(m.log)

        for comp_name, llm_col, bench_col, offset in components:
            stats = rowwise_cosine(
                m.matched[llm_col].tolist(),
                m.matched[bench_col].tolist(),
                n_perm=N_PERMUTATIONS,
                seed=RNG_SEED + offset + 100 * k,
            )
            row = {
                "eo_threshold": k,
                "eo_label": lbl,
                "component": comp_name,
                "n_matched": int(len(m.matched)),
                "n_permutations": int(N_PERMUTATIONS),
            }
            row.update(stats)
            similarity_rows.append(row)

            baseline_rows.append(
                {
                    "eo_threshold": k,
                    "eo_label": lbl,
                    "component": comp_name,
                    "n_matched": int(len(m.matched)),
                    "n_permutations": int(N_PERMUTATIONS),
                    "random_baseline_mean": stats["random_baseline_mean"],
                    "random_baseline_std": stats["random_baseline_std"],
                    "lift_vs_baseline": stats["lift_vs_baseline"],
                    "permutation_p_value": stats["permutation_p_value"],
                }
            )

    if not similarity_rows:
        raise RuntimeError("[EXOGENOUS_BENCHMARK] No similarity rows produced.")

    similarity_df = pd.DataFrame(similarity_rows).sort_values(["eo_threshold", "component"])
    baseline_df = pd.DataFrame(baseline_rows).sort_values(["eo_threshold", "component"])
    log_df = pd.concat(match_logs, ignore_index=True) if match_logs else pd.DataFrame()

    sim_out = RESULTS_DIR / "validation_plausibly_similarity_eo_grid.csv"
    base_out = RESULTS_DIR / "validation_plausibly_baseline_eo_grid.csv"
    log_out = RESULTS_DIR / "validation_plausibly_match_log.csv"

    similarity_df.to_csv(sim_out, index=False)
    baseline_df.to_csv(base_out, index=False)
    log_df.to_csv(log_out, index=False)

    print(f"[EXOGENOUS_BENCHMARK] Wrote: {sim_out}")
    print(f"[EXOGENOUS_BENCHMARK] Wrote: {base_out}")
    print(f"[EXOGENOUS_BENCHMARK] Wrote: {log_out}")
    print("[EXOGENOUS_BENCHMARK] Done.")


if __name__ == "__main__":
    main()
