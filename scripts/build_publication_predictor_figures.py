from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm


ROOT = Path(".")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def _resolve_parquet(patterns: Iterable[str]) -> Path:
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not locate required parquet. Tried: {list(patterns)}")


def _resolve_citations_csv() -> Path:
    patterns = [
        "int_data/paper_level_cites.csv",
        "int_data/*paper_level*cites*.csv",
        "int_data/*cites*.csv",
        "paper_level_cites.csv",
    ]
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Could not locate citation CSV (expected columns: paper_id, coalesced_cites) "
        "under int_data/."
    )


def _load_base_data() -> pd.DataFrame:
    pl_path = _resolve_parquet([
        "int_data/paper_level_data_*_eo3p.parquet",
        "int_data/paper_level_data_eo3p.parquet",
    ])
    pl = pq.read_table(pl_path).to_pandas()
    pl = pl.copy()

    meta_path = _resolve_parquet([
        "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
        "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet",
    ])
    dt = pq.read_table(meta_path).to_pandas()
    dt = dt[(pd.to_numeric(dt["year"], errors="coerce") >= 1980) & (pd.to_numeric(dt["edge_overlap"], errors="coerce") > 3)].copy()
    year_map = (
        dt[["paper_id", "year"]]
        .dropna(subset=["paper_id", "year"])
        .sort_values(["paper_id", "year"])
        .drop_duplicates(subset=["paper_id"], keep="first")
    )

    cites_path = _resolve_citations_csv()
    cites_raw = pd.read_csv(cites_path)
    need_cites_cols = {"paper_id", "coalesced_cites"}
    missing_cites_cols = need_cites_cols.difference(cites_raw.columns)
    if missing_cites_cols:
        raise ValueError(
            f"Citation CSV is missing required columns: {sorted(missing_cites_cols)} "
            f"(file: {cites_path})"
        )
    cites = cites_raw[["paper_id", "coalesced_cites"]].copy()

    out = pl.merge(year_map, on="paper_id", how="left").merge(cites, on="paper_id", how="left")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["journal_rank"] = pd.to_numeric(out["journal_rank"], errors="coerce")
    out["coalesced_cites"] = pd.to_numeric(out["coalesced_cites"], errors="coerce").fillna(0.0)

    out["paper_is_top_5"] = (out["journal_rank"] <= 5).astype(float)
    out["paper_is_top_6_20"] = ((out["journal_rank"] >= 6) & (out["journal_rank"] <= 20)).astype(float)
    out["paper_is_top_21_100"] = ((out["journal_rank"] >= 21) & (out["journal_rank"] <= 100)).astype(float)
    out["transformed_coalesced_cites"] = np.log1p(np.clip(out["coalesced_cites"], a_min=0.0, a_max=None))

    out["log_num_edges_causal"] = np.log1p(pd.to_numeric(out["num_causal_edges"], errors="coerce").fillna(0.0))
    out["log_num_edges_non_causal"] = np.log1p(pd.to_numeric(out["num_non_causal_edges"], errors="coerce").fillna(0.0))
    out["log_num_novel_edges_causal"] = np.log1p(pd.to_numeric(out["num_novel_edges_causal"], errors="coerce").fillna(0.0))
    out["log_num_novel_edges_non_causal"] = np.log1p(pd.to_numeric(out["num_novel_edges_non_causal"], errors="coerce").fillna(0.0))
    out["log_num_unique_paths_causal"] = np.log1p(pd.to_numeric(out["num_unique_paths_causal"], errors="coerce").fillna(0.0))
    out["log_num_unique_paths_non_causal"] = np.log1p(pd.to_numeric(out["num_unique_paths_non_causal"], errors="coerce").fillna(0.0))
    out["log_longest_path_length_causal"] = np.log1p(pd.to_numeric(out["longest_path_length_causal"], errors="coerce").fillna(0.0))
    out["log_longest_path_length_non_causal"] = np.log1p(pd.to_numeric(out["longest_path_length_non_causal"], errors="coerce").fillna(0.0))

    out["mean_eigen_centrality_causal_std"] = _zscore(out["mean_eigen_centrality_causal_RelCumLit_causal"])
    out["mean_eigen_centrality_non_causal_std"] = _zscore(out["mean_eigen_centrality_non_causal_RelCumLit_non_causal"])
    out["var_eigen_centrality_causal_std"] = _zscore(out["var_eigen_centrality_causal_RelCumLit_causal"])
    out["var_eigen_centrality_non_causal_std"] = _zscore(out["var_eigen_centrality_non_causal_RelCumLit_non_causal"])
    out["mean_pagerank_causal_std"] = _zscore(out["mean_pagerank_causal_RelCumLit_causal"])
    out["mean_pagerank_non_causal_std"] = _zscore(out["mean_pagerank_non_causal_RelCumLit_non_causal"])

    gap_non = pq.read_table("int_data/gap_filling_measures_non_causal_cooccurrence_undir_eo3p.parquet").to_pandas()
    gap_cau = pq.read_table("int_data/gap_filling_measures_causal_cooccurrence_undir_eo3p.parquet").to_pandas()
    gap_non = gap_non[["paper_id", "gap_filling_proportion_non_causal"]].drop_duplicates(subset=["paper_id"])
    gap_cau = gap_cau[["paper_id", "gap_filling_proportion_causal"]].drop_duplicates(subset=["paper_id"])
    out = out.merge(gap_non, on="paper_id", how="left").merge(gap_cau, on="paper_id", how="left")

    out = out[(out["year"] >= 1980) & (out["year"] < 2020)].copy()
    return out


def _run_year_fe_reg(df: pd.DataFrame, y_col: str, x_col: str, fill_x_zero: bool = True) -> Dict[str, float]:
    sub = df[[y_col, x_col, "year"]].copy()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
    if fill_x_zero:
        sub[x_col] = sub[x_col].fillna(0.0)
    sub = sub.dropna(subset=[y_col, "year"])
    if sub.empty:
        return {"estimate": np.nan, "std_error": np.nan, "pvalue": np.nan, "n": 0}

    x_dum = pd.get_dummies(sub["year"].astype(int), prefix="year", drop_first=True)
    X = pd.concat([sub[[x_col]], x_dum], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = sub[y_col].astype(float).values

    fit = sm.OLS(y, X.values).fit()
    idx = list(X.columns).index(x_col)
    return {
        "estimate": float(fit.params[idx]),
        "std_error": float(fit.bse[idx]),
        "pvalue": float(fit.pvalues[idx]),
        "n": int(len(sub)),
    }


def _collect_regression_rows(
    df: pd.DataFrame,
    spec: Iterable[Tuple[str, str, str]],
    outcomes: Iterable[Tuple[str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for x_col, measure_label, measure_type in spec:
        for y_col, y_label in outcomes:
            reg = _run_year_fe_reg(df, y_col, x_col)
            est = reg["estimate"]
            se = reg["std_error"]
            rows.append({
                "predictor": x_col,
                "measure_label": measure_label,
                "measure_type": measure_type,
                "outcome": y_label,
                "estimate": est,
                "std_error": se,
                "ci_low": est - 1.96 * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                "ci_high": est + 1.96 * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                "pvalue": reg["pvalue"],
                "n": reg["n"],
            })
    out = pd.DataFrame(rows)
    out["significant_5pct"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def _plot_coef_panel(
    rr: pd.DataFrame,
    measure_order: List[str],
    out_path_base: str,
    figsize: Tuple[float, float],
) -> None:
    outcomes = ["In Top 5", "In Top 6-20", "In Top 21-100", "Citations"]
    palette = {"Causal": "#084594", "Non-Causal": "darkorange"}
    marker = {"Causal": "s", "Non-Causal": "o"}
    y_map = {m: i for i, m in enumerate(measure_order[::-1])}
    offset = {"Causal": 0.15, "Non-Causal": -0.15}

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    def _sym_lim(vals: List[float], min_lim: float = 0.02) -> Tuple[float, float]:
        if not vals:
            return (-min_lim, min_lim)
        lim = max(abs(v) for v in vals) * 1.12
        lim = max(lim, min_lim)
        return (-lim, lim)

    journal_outcomes = {"In Top 5", "In Top 6-20", "In Top 21-100"}
    journal_vals: List[float] = []
    citation_vals: List[float] = []
    for _, r in rr.iterrows():
        if not np.isfinite(r["estimate"]) or not np.isfinite(r["std_error"]):
            continue
        vals = [float(r["estimate"]), float(r["estimate"] - 1.96 * r["std_error"]), float(r["estimate"] + 1.96 * r["std_error"])]
        if r["outcome"] in journal_outcomes:
            journal_vals.extend(vals)
        elif r["outcome"] == "Citations":
            citation_vals.extend(vals)
    journal_xlim = _sym_lim(journal_vals, min_lim=0.02)
    citation_xlim = _sym_lim(citation_vals, min_lim=0.05)

    for ax, out in zip(axes, outcomes):
        sub = rr[rr["outcome"] == out].copy()
        for _, r in sub.iterrows():
            y0 = y_map[r["measure_label"]] + offset.get(str(r["measure_type"]), 0.0)
            ax.errorbar(
                r["estimate"],
                y0,
                xerr=1.96 * r["std_error"],
                fmt=marker.get(str(r["measure_type"]), "o"),
                color=palette.get(str(r["measure_type"]), "#333333"),
                markersize=8.2,
                capsize=3,
                linewidth=1.8,
                alpha=0.95,
            )
        ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(out)
        ax.set_xlabel("Coefficient (year FE)")
        if out in journal_outcomes:
            ax.set_xlim(*journal_xlim)
        else:
            ax.set_xlim(*citation_xlim)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_yticks(range(len(measure_order)))
            ax.set_yticklabels(measure_order[::-1])
        else:
            ax.set_yticks(range(len(measure_order)))
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x")

    handles = [
        plt.Line2D([0], [0], color=palette["Causal"], marker=marker["Causal"], linestyle="", label="Causal"),
        plt.Line2D([0], [0], color=palette["Non-Causal"], marker=marker["Non-Causal"], linestyle="", label="Non-Causal"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.22, right=0.995, bottom=0.12, top=0.90, wspace=0.08)
    plt.savefig(FIG_DIR / f"{out_path_base}.jpg", dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(FIG_DIR / f"{out_path_base}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _build_main_and_variant_figures(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("paper_is_top_5", "In Top 5"),
        ("paper_is_top_6_20", "In Top 6-20"),
        ("paper_is_top_21_100", "In Top 21-100"),
        ("transformed_coalesced_cites", "Citations"),
    ]

    main_spec = [
        ("prop_edges_causal", "Share Causal Claims", "Causal"),
        ("log_num_edges_non_causal", "Log N. Edges", "Non-Causal"),
        ("log_num_edges_causal", "Log N. Edges", "Causal"),
        ("log_num_novel_edges_non_causal", "Log N. New Edges", "Non-Causal"),
        ("log_num_novel_edges_causal", "Log N. New Edges", "Causal"),
        ("mean_eigen_centrality_non_causal_std", "Topic Centrality", "Non-Causal"),
        ("mean_eigen_centrality_causal_std", "Topic Centrality", "Causal"),
        ("cause_effect_ratio_non_causal", "Source-Sink Ratio", "Non-Causal"),
        ("cause_effect_ratio_causal", "Source-Sink Ratio", "Causal"),
    ]
    main_order = [
        "Share Causal Claims",
        "Log N. Edges",
        "Log N. New Edges",
        "Topic Centrality",
        "Source-Sink Ratio",
    ]
    rr_main = _collect_regression_rows(df, main_spec, outcomes)
    _plot_coef_panel(
        rr_main,
        measure_order=main_order,
        out_path_base="publication_predictors_main_five_measures",
        figsize=(18.6, 7.8),
    )

    variant_spec = [
        ("log_num_unique_paths_non_causal", "Log N. Unique Paths", "Non-Causal"),
        ("log_num_unique_paths_causal", "Log N. Unique Paths", "Causal"),
        ("log_longest_path_length_non_causal", "Log Longest Path", "Non-Causal"),
        ("log_longest_path_length_causal", "Log Longest Path", "Causal"),
        ("prop_novel_unique_paths_non_causal", "Prop. Novel Unique Paths", "Non-Causal"),
        ("prop_novel_unique_paths_causal", "Prop. Novel Unique Paths", "Causal"),
        ("gap_filling_proportion_non_causal", "Gap Filling", "Non-Causal"),
        ("gap_filling_proportion_causal", "Gap Filling", "Causal"),
        ("var_eigen_centrality_non_causal_std", "Topic Diversity", "Non-Causal"),
        ("var_eigen_centrality_causal_std", "Topic Diversity", "Causal"),
        ("mean_pagerank_non_causal_std", "Topic Centrality (PageRank)", "Non-Causal"),
        ("mean_pagerank_causal_std", "Topic Centrality (PageRank)", "Causal"),
    ]
    variant_order = [
        "Log N. Unique Paths",
        "Log Longest Path",
        "Prop. Novel Unique Paths",
        "Gap Filling",
        "Topic Diversity",
        "Topic Centrality (PageRank)",
    ]
    rr_var = _collect_regression_rows(df, variant_spec, outcomes)
    _plot_coef_panel(
        rr_var,
        measure_order=variant_order,
        out_path_base="publication_predictors_variants_appendix",
        figsize=(19.0, 10.4),
    )

    return rr_main, rr_var


def _build_eo_grid_figure() -> pd.DataFrame:
    outcomes = [
        ("paper_is_top_5", "In Top 5"),
        ("paper_is_top_6_20", "In Top 6-20"),
        ("paper_is_top_21_100", "In Top 21-100"),
        ("transformed_coalesced_cites", "Citations"),
    ]
    eo_spec = [
        ("prop_edges_causal", "Share Causal Claims", "Causal"),
        ("log_num_edges_non_causal", "Log N. Edges", "Non-Causal"),
        ("log_num_edges_causal", "Log N. Edges", "Causal"),
        ("log_num_novel_edges_non_causal", "Log N. New Edges", "Non-Causal"),
        ("log_num_novel_edges_causal", "Log N. New Edges", "Causal"),
        ("mean_eigen_centrality_RelCumLit_non_causal_std", "Topic Centrality", "Non-Causal"),
        ("mean_eigen_centrality_RelCumLit_causal_std", "Topic Centrality", "Causal"),
        ("cause_effect_ratio_non_causal", "Source-Sink Ratio", "Non-Causal"),
        ("cause_effect_ratio_causal", "Source-Sink Ratio", "Causal"),
    ]
    measure_order = [
        "Share Causal Claims",
        "Log N. Edges",
        "Log N. New Edges",
        "Topic Centrality",
        "Source-Sink Ratio",
    ]

    rows: List[Dict[str, object]] = []
    for thr in range(1, 10):
        p = ROOT / "int_data" / "edge_overlap_runs_snippet_only" / f"eo_ge{thr}" / f"paper_level_data_eo_ge{thr}_snippet_only.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing snippet EO parquet: {p}")
        dt = pq.read_table(p).to_pandas()
        dt["year"] = pd.to_numeric(dt["year"], errors="coerce")
        dt = dt[dt["year"] < 2020].copy()
        for x_col, measure_label, measure_type in eo_spec:
            for y_col, y_label in outcomes:
                reg = _run_year_fe_reg(dt, y_col, x_col, fill_x_zero=True)
                est = reg["estimate"]
                se = reg["std_error"]
                rows.append({
                    "edge_overlap": thr,
                    "predictor": x_col,
                    "measure_label": measure_label,
                    "measure_type": measure_type,
                    "outcome": y_label,
                    "estimate": est,
                    "std_error": se,
                    "ci_low": est - 1.96 * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                    "ci_high": est + 1.96 * se if np.isfinite(est) and np.isfinite(se) else np.nan,
                    "pvalue": reg["pvalue"],
                    "n": reg["n"],
                })

    rr = pd.DataFrame(rows)

    colors = {"Causal": "#084594", "Non-Causal": "darkorange"}
    markers = {"Causal": "s", "Non-Causal": "o"}
    types_for_measure = {
        "Share Causal Claims": ["Causal"],
        "Log N. Edges": ["Non-Causal", "Causal"],
        "Log N. New Edges": ["Non-Causal", "Causal"],
        "Topic Centrality": ["Non-Causal", "Causal"],
        "Source-Sink Ratio": ["Non-Causal", "Causal"],
    }

    fig, axes = plt.subplots(len(measure_order), 4, figsize=(19.6, 17.6), sharey=True)
    if len(measure_order) == 1:
        axes = np.array([axes])
    journal_outcomes = {"In Top 5", "In Top 6-20", "In Top 21-100"}

    def _sym_lim(vals: List[float], min_lim: float = 0.05) -> Tuple[float, float]:
        if not vals:
            return (-min_lim, min_lim)
        lim = max(abs(v) for v in vals) * 1.15
        lim = max(lim, min_lim)
        return (-lim, lim)

    for i, measure in enumerate(measure_order):
        row_sub = rr[rr["measure_label"] == measure].copy()
        journal_vals: List[float] = []
        citation_vals: List[float] = []
        for _, r in row_sub.iterrows():
            if not np.isfinite(r["estimate"]) or not np.isfinite(r["std_error"]):
                continue
            vals = [
                float(r["estimate"]),
                float(r["estimate"] - 1.96 * r["std_error"]),
                float(r["estimate"] + 1.96 * r["std_error"]),
            ]
            if r["outcome"] in journal_outcomes:
                journal_vals.extend(vals)
            elif r["outcome"] == "Citations":
                citation_vals.extend(vals)
        journal_xlim = _sym_lim(journal_vals, min_lim=0.05)
        citation_xlim = _sym_lim(citation_vals, min_lim=0.10)

        for j, out in enumerate(["In Top 5", "In Top 6-20", "In Top 21-100", "Citations"]):
            ax = axes[i, j]
            sub = rr[(rr["measure_label"] == measure) & (rr["outcome"] == out)].copy()
            sub = sub.sort_values("edge_overlap")
            for typ in types_for_measure[measure]:
                s2 = sub[sub["measure_type"] == typ]
                if s2.empty:
                    continue
                ax.errorbar(
                    s2["estimate"],
                    s2["edge_overlap"],
                    xerr=1.96 * s2["std_error"],
                    fmt=markers[typ],
                    color=colors[typ],
                    linewidth=1.2,
                    capsize=2.5,
                    markersize=4.8,
                    alpha=0.95,
                )
                ax.plot(s2["estimate"], s2["edge_overlap"], color=colors[typ], linewidth=1.0, alpha=0.9)

            panel_vals: List[float] = []
            for _, r in sub.iterrows():
                if not np.isfinite(r["estimate"]) or not np.isfinite(r["std_error"]):
                    continue
                panel_vals.extend(
                    [
                        float(r["estimate"]),
                        float(r["estimate"] - 1.96 * r["std_error"]),
                        float(r["estimate"] + 1.96 * r["std_error"]),
                    ]
                )
            if panel_vals:
                lim = float(np.nanquantile(np.abs(np.asarray(panel_vals)), 0.92)) * 1.15
                lim = max(lim, 0.05)
            if out in journal_outcomes:
                ax.set_xlim(*journal_xlim)
            else:
                ax.set_xlim(*citation_xlim)

            ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
            ax.axhline(4, color="#333333", linestyle="--", linewidth=0.9, alpha=0.5)
            ax.set_ylim(0.7, 9.3)
            ax.set_yticks(list(range(1, 10)))
            if j == 0:
                ax.set_ylabel(f"{measure}\nEO threshold")
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(out)
            ax.tick_params(axis="x")
            ax.tick_params(axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#3f3f3f")
            ax.spines["bottom"].set_color("#3f3f3f")

    handles = [
        plt.Line2D([0], [0], color=colors["Causal"], marker=markers["Causal"], linestyle="-", label="Causal"),
        plt.Line2D([0], [0], color=colors["Non-Causal"], marker=markers["Non-Causal"], linestyle="-", label="Non-Causal"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.997))
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.985])
    plt.savefig(FIG_DIR / "main_reg_edge_overlap_grid.jpg", dpi=320)
    plt.savefig(FIG_DIR / "main_reg_edge_overlap_grid.pdf")
    plt.close(fig)

    return rr


def main() -> None:
    df = _load_base_data()
    rr_main, rr_var = _build_main_and_variant_figures(df)
    rr_eo = _build_eo_grid_figure()

    rr_main.to_csv(TABLE_DIR / "publication_predictors_main_five_measures.csv", index=False)
    rr_var.to_csv(TABLE_DIR / "publication_predictors_variants_appendix.csv", index=False)
    rr_eo.to_csv(TABLE_DIR / "publication_predictors_eo_grid.csv", index=False)

    key = rr_main[
        rr_main["measure_label"].isin(
            ["Share Causal Claims", "Log N. Edges", "Log N. New Edges", "Topic Centrality", "Source-Sink Ratio"]
        )
    ][["measure_label", "measure_type", "outcome", "estimate", "std_error", "ci_low", "ci_high", "pvalue", "n"]]
    key.to_csv(TABLE_DIR / "publication_predictors_main_five_measures_key.csv", index=False)
    print("Wrote:")
    print(" - figures/publication_predictors_main_five_measures.jpg/.pdf")
    print(" - figures/publication_predictors_variants_appendix.jpg/.pdf")
    print(" - figures/main_reg_edge_overlap_grid.jpg/.pdf")
    print(" - results/tables/publication_predictors_main_five_measures.csv")
    print(" - results/tables/publication_predictors_variants_appendix.csv")


if __name__ == "__main__":
    main()
