#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D


ROOT = Path(".")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
AEA_INFO_PATH = ROOT / "int_data" / "AEA_Information.csv"


def _set_style(font_scale: float = 1.33) -> None:
    sns.set_theme(style="white", context="talk", font_scale=font_scale)
    plt.rcParams.update({
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
    })


def _apply_classic(ax: plt.Axes | None = None) -> None:
    ax = ax or plt.gca()
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)


def _clean_text_label(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def _compact_concept_label(concept: str, jel: str, max_words: int = 12, width: int = 20) -> str:
    concept = _clean_text_label(concept)
    replacements = {
        r"\bpercentage\b": "pct",
        r"\bpercent\b": "pct",
        r"\bhouseholds\b": "HHs",
        r"\bgovernment\b": "gov",
        r"\bincome\b": "inc",
        r"\beducation\b": "edu",
        r"\bunemployment\b": "unemp",
        r"\bconsumption\b": "cons.",
        r"\binvestment\b": "inv.",
    }
    for pat, rep in replacements.items():
        concept = re.sub(pat, rep, concept, flags=re.IGNORECASE)
    words = concept.split()
    short = " ".join(words[:max_words])
    base = f"{short} [{str(jel).strip()}]"
    return textwrap.fill(base, width=width)


def _load_jel_short_map() -> Dict[str, str]:
    manual = {
        "C62": "Equilibrium",
        "C79": "Game Theory",
        "D14": "Household Finance",
        "D29": "Production",
        "D72": "Political Process",
        "E00": "Macroeconomics",
        "E21": "Consumption",
        "E22": "Investment",
        "E31": "Inflation",
        "E43": "Interest Rates",
        "E49": "Monetary Other",
        "E58": "Central Banks",
        "E63": "Policy Mix",
        "F10": "Trade General",
        "F24": "Remittances",
        "F31": "FX Markets",
        "F32": "Current Account",
        "G10": "Financial Markets",
        "G51": "Household Credit",
        "G52": "Asset Pricing",
        "H19": "Public Finance",
        "H21": "Tax Efficiency",
        "I11": "Health Behavior",
        "I12": "Health Production",
        "I24": "Education-Inequality",
        "I31": "Well-Being",
        "J21": "Labor Force",
        "J23": "Labor Demand",
        "J26": "Retirement",
        "J31": "Wages",
        "J51": "Disability",
        "J64": "Unemployment",
        "K11": "Property Law",
        "K32": "Environmental Law",
        "L15": "Product Quality",
        "L25": "Firm Performance",
        "L26": "Entrepreneurship",
        "L29": "Firm Behavior",
        "O36": "Technology Choice",
        "O49": "Growth",
        "O57": "Country Studies",
        "D91": "Behavior",
        "D79": "Collective Choice",
        "J15": "Minorities",
        "I14": "Health-Inequality",
        "G21": "Banks",
        "J13": "Family",
        "I21": "Education",
    }
    if not AEA_INFO_PATH.exists():
        return manual
    try:
        df = pd.read_csv(AEA_INFO_PATH, usecols=["JEL Code", "Keywords"])
        for _, r in df.iterrows():
            code = str(r.get("JEL Code", "")).strip()
            if not code or code in manual:
                continue
            kw = str(r.get("Keywords", ""))
            kw = kw.replace("Keywords:", "").strip()
            if not kw or "None" in kw:
                continue
            first = kw.split(",")[0].strip()
            first = re.sub(r"\s+", " ", first)
            toks = first.split()[:2]
            if toks:
                manual[code] = "-".join(toks)
    except Exception:
        pass
    return manual


def _jel_label(code: str, jel_map: Dict[str, str]) -> str:
    code = str(code).strip()
    return f"{code}-{jel_map.get(code, 'Topic')}"


def _clean_method(s: str) -> str:
    if not isinstance(s, str):
        return "Other"
    s2 = s.strip().lower()
    mapping = {
        "did": "DiD",
        "iv": "IV",
        "rct": "RCT",
        "rdd": "RDD",
        "event study": "Event Study",
        "simulation": "Simulation",
        "structural estimation": "Structural",
        "theoretical/non-empirical": "Theoretical",
    }
    return mapping.get(s2, "Other")


def _resolve_parquet(patterns: List[str]) -> Path:
    for pat in patterns:
        matches = sorted(ROOT.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate required parquet file. Tried: {patterns}")


def _load_edge_data() -> pd.DataFrame:
    dt_path = _resolve_parquet([
        "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
        "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet",
    ])
    dt = pq.read_table(dt_path).to_pandas()
    dt = dt[(dt["year"] >= 1980) & (dt["edge_overlap"] > 3)].copy()
    return dt


def _load_paper_level() -> pd.DataFrame:
    pl_path = _resolve_parquet([
        "int_data/paper_level_data_*_eo3p.parquet",
        "int_data/paper_level_data_eo3p.parquet",
    ])
    pl = pq.read_table(pl_path).to_pandas()
    return pl


def _find_citations_csv() -> Path | None:
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
    return None


def _load_citations_table() -> pd.DataFrame | None:
    cites_path = _find_citations_csv()
    if cites_path is None:
        print("Skipping citation-based figures: citation CSV not found in int_data/.")
        return None
    cites = pd.read_csv(cites_path)
    if "paper_id" not in cites.columns or "coalesced_cites" not in cites.columns:
        print(
            f"Skipping citation-based figures: missing required columns in {cites_path}."
        )
        return None
    return cites[["paper_id", "coalesced_cites"]].copy()


def _draw_example_graphs(edge_data: pd.DataFrame) -> None:
    paper_map = {
        "w19843": "chetty.png",
        "w18950": "duflo.png",
        "w15286": "xavier.png",
        "w14416": "penny.png",
    }

    _set_style(1.32)

    for pid, out_name in paper_map.items():
        d = edge_data.loc[edge_data["paper_id"] == pid, [
            "cause",
            "effect",
            "jel_cause",
            "jel_effect",
            "is_method_causal_inference",
        ]].dropna(subset=["cause", "effect"])

        if d.empty:
            continue

        d["src"] = d.apply(lambda r: _compact_concept_label(r["cause"], r["jel_cause"], max_words=12, width=20), axis=1)
        d["dst"] = d.apply(lambda r: _compact_concept_label(r["effect"], r["jel_effect"], max_words=12, width=20), axis=1)

        g = nx.DiGraph()
        for _, r in d.iterrows():
            g.add_edge(r["src"], r["dst"], causal=int(r["is_method_causal_inference"] == 1))

        n_nodes = len(g.nodes)
        if n_nodes <= 20:
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = nx.spring_layout(g, seed=42, k=0.58 / math.sqrt(max(len(g.nodes), 2)), iterations=900)

        # Keep Figure 5/6 panels visually comparable and leave enough room for labels.
        fig_size = (11.2, 8.0)
        if n_nodes <= 4:
            scale = 0.56
            node_size = 9800
            label_size = 12.3
            edge_width = 3.8
            arrow_size = 34
        elif n_nodes <= 8:
            scale = 0.68
            node_size = 9200
            label_size = 11.6
            edge_width = 3.5
            arrow_size = 32
        else:
            scale = 0.84
            node_size = 8600
            label_size = 10.9
            edge_width = 3.2
            arrow_size = 30
        pos = {k: np.asarray(v) * scale for k, v in pos.items()}
        if n_nodes <= 3:
            edge_rad = 0.02
        elif n_nodes <= 6:
            edge_rad = 0.07
        else:
            edge_rad = 0.11
        edge_cols = ["#084594" if g[u][v]["causal"] == 1 else "darkorange" for u, v in g.edges()]

        plt.figure(figsize=fig_size)
        nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color="#f2f2f2", edgecolors="#444444", linewidths=2.0)
        nx.draw_networkx_edges(
            g,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrow_size,
            edge_color=edge_cols,
            width=edge_width,
            alpha=0.95,
            connectionstyle=f"arc3,rad={edge_rad}",
            min_source_margin=22,
            min_target_margin=22,
        )
        nx.draw_networkx_labels(g, pos, font_size=label_size, font_weight="bold")
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        xr = max(float(xs.max() - xs.min()), 1e-6)
        yr = max(float(ys.max() - ys.min()), 1e-6)
        base_pad = 0.22 if n_nodes <= 4 else 0.28
        xpad = max(base_pad, base_pad * xr)
        ypad = max(base_pad, base_pad * yr)
        plt.xlim(float(xs.min() - xpad), float(xs.max() + xpad))
        plt.ylim(float(ys.min() - ypad), float(ys.max() + ypad))
        plt.axis("off")
        plt.margins(0.06)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
        plt.savefig(FIG_DIR / out_name, dpi=320)
        plt.close()


def _compute_journal_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["journal_rank"] = pd.to_numeric(out["journal_rank"], errors="coerce")
    out["journal_tier"] = np.select(
        [
            out["journal_rank"] <= 5,
            (out["journal_rank"] >= 6) & (out["journal_rank"] <= 20),
            (out["journal_rank"] >= 21) & (out["journal_rank"] <= 100),
        ],
        ["Top 5", "Top 6-20", "Top 21-100"],
        default="",
    )
    return out


def _build_gap_filling_panels(pl: pd.DataFrame) -> None:
    _set_style(1.35)
    gap_non = pq.read_table("int_data/gap_filling_measures_non_causal_cooccurrence_undir_eo3p.parquet").to_pandas()
    gap_cau = pq.read_table("int_data/gap_filling_measures_causal_cooccurrence_undir_eo3p.parquet").to_pandas()

    j = pl[["paper_id", "coalesced_journal", "journal_rank"]].drop_duplicates().copy()
    j = _compute_journal_tier(j)

    g = j.merge(gap_non[["paper_id", "gap_filling_proportion_non_causal"]], on="paper_id", how="left")
    g = g.merge(gap_cau[["paper_id", "gap_filling_proportion_causal"]], on="paper_id", how="left")

    g["gap_non_pct"] = 100 * pd.to_numeric(g["gap_filling_proportion_non_causal"], errors="coerce")
    g["gap_cau_pct"] = 100 * pd.to_numeric(g["gap_filling_proportion_causal"], errors="coerce")

    # Panel A: journal tiers
    a = g[g["journal_tier"].astype(str).ne("")].copy()
    a_long = a.melt(
        id_vars=["journal_tier"],
        value_vars=["gap_non_pct", "gap_cau_pct"],
        var_name="series",
        value_name="value",
    )
    a_long["series"] = a_long["series"].map({"gap_non_pct": "Non-Causal", "gap_cau_pct": "Causal"})

    # Panel B: top-5 journals
    top5 = [
        "American Economic Review",
        "Quarterly Journal of Economics",
        "Journal of Political Economy",
        "Review of Economic Studies",
        "Econometrica",
    ]
    top5_short = {
        "American Economic Review": "AER",
        "Quarterly Journal of Economics": "QJE",
        "Journal of Political Economy": "JPE",
        "Review of Economic Studies": "ReStud",
        "Econometrica": "ECMA",
    }
    b = g[g["coalesced_journal"].isin(top5)].copy()
    b["journal_short"] = b["coalesced_journal"].map(top5_short)
    b_long = b.melt(
        id_vars=["journal_short"],
        value_vars=["gap_non_pct", "gap_cau_pct"],
        var_name="series",
        value_name="value",
    )
    b_long["series"] = b_long["series"].map({"gap_non_pct": "Non-Causal", "gap_cau_pct": "Causal"})

    # Save panel A + panel B separately using manuscript filenames
    palette = {"Non-Causal": "darkorange", "Causal": "#084594"}

    tier_order = ["Top 5", "Top 6-20", "Top 21-100"]
    plt.figure(figsize=(11.2, 7.2))
    ax = sns.pointplot(
        data=a_long.dropna(subset=["value"]),
        x="journal_tier",
        y="value",
        hue="series",
        order=tier_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        err_kws={"linewidth": 1.6, "color": "#3a3a3a", "alpha": 0.9},
        markers=["o", "s"],
        dodge=0.0,
        palette=palette,
        linewidth=2.2,
        scale=1.1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Gap-filling proportion (%)")
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(title="", fontsize=14)
    _apply_classic(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "gap_filling_journal_tiers_combined_non_causal.jpg", dpi=320, bbox_inches="tight")
    plt.close()

    journal_order = ["AER", "QJE", "JPE", "ReStud", "ECMA"]
    plt.figure(figsize=(11.2, 7.2))
    ax = sns.pointplot(
        data=b_long.dropna(subset=["value"]),
        x="journal_short",
        y="value",
        hue="series",
        order=journal_order,
        estimator=np.mean,
        errorbar=("ci", 95),
        err_kws={"linewidth": 1.6, "color": "#3a3a3a", "alpha": 0.9},
        markers=["o", "s"],
        dodge=0.0,
        palette=palette,
        linewidth=2.2,
        scale=1.1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Gap-filling proportion (%)")
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(title="", fontsize=14)
    _apply_classic(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "gap_filling_top5_journals_combined_non_causal.jpg", dpi=320, bbox_inches="tight")
    plt.close()


def _build_top5_field_method_figures(edge_data: pd.DataFrame, pl: pd.DataFrame) -> None:
    _set_style(1.40)
    top5 = {
        "American Economic Review",
        "Quarterly Journal of Economics",
        "Journal of Political Economy",
        "Review of Economic Studies",
        "Econometrica",
    }

    paper_meta = edge_data[[
        "paper_id",
        "year",
        "causal_inference_method",
        "is_finance",
        "is_development",
        "is_labour",
        "is_public_economics",
        "is_urban_economics",
        "is_macroeconomics",
        "is_behavioral_economics",
        "is_economic_history",
        "is_econometric_theory",
        "is_industrial_organization",
        "is_environmental_economics",
        "is_health_economics",
        "is_political_economy",
    ]].copy()

    paper_meta["method_clean"] = paper_meta["causal_inference_method"].map(_clean_method)
    paper_meta = paper_meta.sort_values(["paper_id", "year"]).drop_duplicates(subset=["paper_id", "method_clean"])

    fields = [
        ("is_finance", "Finance"),
        ("is_development", "Development"),
        ("is_labour", "Labour"),
        ("is_public_economics", "Public"),
        ("is_urban_economics", "Urban"),
        ("is_macroeconomics", "Macro"),
        ("is_behavioral_economics", "Behavioral"),
        ("is_economic_history", "Econ. History"),
        ("is_econometric_theory", "Econometrics"),
        ("is_industrial_organization", "IO"),
        ("is_environmental_economics", "Environmental"),
        ("is_health_economics", "Health"),
        ("is_political_economy", "Pol. Econ."),
    ]

    journal = pl[["paper_id", "coalesced_journal", "journal_rank"]].drop_duplicates().copy()
    journal["is_top5"] = journal["coalesced_journal"].isin(top5).astype(int)

    df = paper_meta.merge(journal[["paper_id", "is_top5"]], on="paper_id", how="left")
    df["is_top5"] = df["is_top5"].fillna(0).astype(int)

    # panel A: arrows pre vs post by field
    field_rows: List[Tuple[str, str, float]] = []
    for fcol, flabel in fields:
        tmp = df[df[fcol] == True].copy()  # noqa: E712
        if tmp.empty:
            continue
        tmp["period"] = np.where(tmp["year"] < 2000, "Pre-2000", "Post-2000")
        s = tmp.groupby("period", dropna=False)["is_top5"].mean()
        field_rows.append((flabel, "Pre-2000", float(s.get("Pre-2000", np.nan))))
        field_rows.append((flabel, "Post-2000", float(s.get("Post-2000", np.nan))))

    fld = pd.DataFrame(field_rows, columns=["field", "period", "share"])
    pre = fld[fld["period"] == "Pre-2000"].set_index("field")["share"]
    post = fld[fld["period"] == "Post-2000"].set_index("field")["share"]
    common_fields = [f for _, f in fields if f in pre.index and f in post.index]

    plt.figure(figsize=(10.4, 8.6))
    y = np.arange(len(common_fields))
    x0 = np.array([pre.loc[f] for f in common_fields]) * 100
    x1 = np.array([post.loc[f] for f in common_fields]) * 100
    for i in range(len(common_fields)):
        plt.arrow(x0[i], y[i], x1[i] - x0[i], 0, length_includes_head=True, head_width=0.15, head_length=0.8, color="#444444", alpha=0.75)
    plt.scatter(x0, y, color="royalblue", label="Pre-2000", zorder=3)
    plt.scatter(x1, y, color="darkorange", label="Post-2000", zorder=3)
    plt.yticks(y, common_fields, fontsize=15)
    plt.xlabel("Top-5 publication share (%)", fontsize=15.2)
    plt.xticks(fontsize=15)
    plt.legend(frameon=False, fontsize=14)
    _apply_classic()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_is_top5_by_field_post_2000_arrow.jpg", dpi=300)
    plt.close()

    # panel B: field x method heatmap
    method_order = ["DiD", "IV", "RCT", "RDD", "Event Study", "Simulation", "Structural", "Theoretical", "Other"]
    recs = []
    for fcol, flabel in fields:
        tmp = df[df[fcol] == True].copy()  # noqa: E712
        if tmp.empty:
            continue
        g = tmp.groupby("method_clean", dropna=False)["is_top5"].mean()
        for m in method_order:
            recs.append((flabel, m, float(g.get(m, np.nan)) * 100))
    hm = pd.DataFrame(recs, columns=["field", "method", "top5_share"])
    piv = hm.pivot(index="field", columns="method", values="top5_share")
    piv = piv.reindex(index=[f for _, f in fields if f in piv.index], columns=method_order)

    plt.figure(figsize=(11.2, 8.8))
    ax = sns.heatmap(
        piv,
        cmap="YlOrBr",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Top-5 share (%)"},
    )
    plt.xlabel("Method", fontsize=15.0)
    plt.ylabel("Field", fontsize=15.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label("Top-5 share (%)", fontsize=13)
    _apply_classic(ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top5_papers_by_field_method_heatmap.jpg", dpi=300)
    plt.close()


def _build_citation_density(pl: pd.DataFrame) -> None:
    _set_style(1.36)
    cites = _load_citations_table()
    if cites is None:
        return

    df = pl[["paper_id", "journal_rank"]].drop_duplicates().merge(
        cites[["paper_id", "coalesced_cites"]], on="paper_id", how="left"
    )
    df["journal_rank"] = pd.to_numeric(df["journal_rank"], errors="coerce")
    df["coalesced_cites"] = pd.to_numeric(df["coalesced_cites"], errors="coerce")
    df = df.dropna(subset=["journal_rank", "coalesced_cites"]).copy()

    df["journal_cat"] = np.select(
        [
            df["journal_rank"] <= 5,
            (df["journal_rank"] >= 6) & (df["journal_rank"] <= 20),
            (df["journal_rank"] >= 21) & (df["journal_rank"] <= 100),
        ],
        ["Top 5", "Top 6-20", "Top 21-100"],
        default="",
    )
    df = df[df["journal_cat"].notna()].copy()
    df["citation_percentile"] = 100 * df["coalesced_cites"].rank(pct=True)

    plt.figure(figsize=(10.8, 6.6))
    palette = {"Top 5": "#084594", "Top 6-20": "#fdb863", "Top 21-100": "#bdbdbd"}
    for cat in ["Top 5", "Top 6-20", "Top 21-100"]:
        sub = df[df["journal_cat"] == cat]["citation_percentile"]
        if sub.notna().sum() > 10:
            sns.kdeplot(sub, label=cat, linewidth=2, fill=False, color=palette[cat])
    plt.xlabel("Citation percentile", fontsize=15.0)
    plt.ylabel("Density", fontsize=15.0)
    plt.xticks(fontsize=14.2)
    plt.yticks(fontsize=14.2)
    plt.legend(title="", fontsize=13.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "citations_by_journal_cat_density.pdf")
    plt.close()


def _build_prop_causal_core_figures(edge_data: pd.DataFrame) -> None:
    _set_style(1.46)
    d = edge_data.copy()
    d["is_method_causal_inference"] = pd.to_numeric(d["is_method_causal_inference"], errors="coerce").fillna(0)

    paper_year = (
        d.groupby(["paper_id", "year"], as_index=False)
        .agg(num_edges=("paper_id", "size"), num_causal=("is_method_causal_inference", "sum"))
    )
    paper_year = paper_year[paper_year["num_edges"] > 0].copy()
    paper_year["prop_causal"] = paper_year["num_causal"] / paper_year["num_edges"]

    ts = (
        paper_year.groupby("year", as_index=False)
        .agg(mean_prop=("prop_causal", "mean"), se_prop=("prop_causal", lambda x: np.nanstd(x, ddof=1) / np.sqrt(len(x))))
    )
    ts["lower"] = ts["mean_prop"] - 1.96 * ts["se_prop"]
    ts["upper"] = ts["mean_prop"] + 1.96 * ts["se_prop"]

    plt.figure(figsize=(11.2, 7.0))
    plt.plot(ts["year"], ts["mean_prop"], color="#084594", linewidth=2.2)
    plt.fill_between(ts["year"], ts["lower"], ts["upper"], color="#084594", alpha=0.15)
    plt.xlabel("Year")
    plt.ylabel("Avg. proportion of causal edges")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    _apply_classic()
    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "prop_causal_edges_over_time.pdf", bbox_inches="tight")
    plt.close()

    fields = [
        ("is_finance", "Finance"),
        ("is_development", "Development"),
        ("is_labour", "Labour"),
        ("is_public_economics", "Public"),
        ("is_urban_economics", "Urban"),
        ("is_macroeconomics", "Macro"),
        ("is_behavioral_economics", "Behavioral"),
        ("is_economic_history", "Econ. History"),
        ("is_econometric_theory", "Econometrics"),
        ("is_industrial_organization", "IO"),
        ("is_environmental_economics", "Environmental"),
        ("is_health_economics", "Health"),
        ("is_political_economy", "Pol. Econ."),
    ]

    paper_fields = d[["paper_id", "year"] + [f for f, _ in fields]].drop_duplicates(subset=["paper_id"])
    pf = paper_year.merge(paper_fields, on=["paper_id", "year"], how="left")
    pf["period"] = np.where(pf["year"] < 2000, "Pre-2000", "Post-2000")

    recs = []
    for fcol, flabel in fields:
        tmp = pf[pf[fcol] == True]  # noqa: E712
        if tmp.empty:
            continue
        m = tmp.groupby("period")["prop_causal"].mean()
        recs.append((flabel, "Pre-2000", float(m.get("Pre-2000", np.nan))))
        recs.append((flabel, "Post-2000", float(m.get("Post-2000", np.nan))))
    fr = pd.DataFrame(recs, columns=["field", "period", "mean_prop"])
    pre = fr[fr["period"] == "Pre-2000"].set_index("field")["mean_prop"]
    post = fr[fr["period"] == "Post-2000"].set_index("field")["mean_prop"]
    field_order = [f for _, f in fields if f in pre.index and f in post.index]
    field_order = sorted(field_order, key=lambda f: float(post.loc[f]), reverse=True)

    plt.figure(figsize=(12.6, 9.4))
    y = np.arange(len(field_order))
    x0 = pre.reindex(field_order).values
    x1 = post.reindex(field_order).values
    for i in range(len(field_order)):
        plt.arrow(
            x0[i],
            y[i],
            x1[i] - x0[i],
            0,
            length_includes_head=True,
            head_width=0.18,
            head_length=0.010,
            linewidth=1.7,
            color="#555555",
            alpha=0.85,
        )
    plt.scatter(x0, y, color="royalblue", s=88, label="Pre-2000", zorder=3)
    plt.scatter(x1, y, color="darkorange", s=88, label="Post-2000", zorder=3)
    plt.yticks(y, field_order)
    plt.gca().invert_yaxis()
    plt.xlabel("Avg. proportion of causal edges")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.legend(frameon=False)
    _apply_classic()
    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "prop_causal_edges_by_field_period.pdf", bbox_inches="tight")
    plt.close()


def _build_source_sink_figures(edge_data: pd.DataFrame, pl: pd.DataFrame) -> None:
    _set_style(1.52)
    fields = [
        ("is_finance", "Finance"),
        ("is_development", "Development"),
        ("is_labour", "Labour"),
        ("is_public_economics", "Public"),
        ("is_urban_economics", "Urban"),
        ("is_macroeconomics", "Macro"),
        ("is_behavioral_economics", "Behavioral"),
        ("is_economic_history", "Econ. History"),
        ("is_econometric_theory", "Econometrics"),
        ("is_industrial_organization", "IO"),
        ("is_environmental_economics", "Environmental"),
        ("is_health_economics", "Health"),
        ("is_political_economy", "Pol. Econ."),
    ]
    pm = edge_data[["paper_id", "year"] + [f for f, _ in fields]].drop_duplicates(subset=["paper_id"])
    d = pl.merge(pm, on="paper_id", how="left")

    d["ratio_non"] = pd.to_numeric(d["cause_effect_ratio_non_causal"], errors="coerce")
    d["ratio_cau"] = pd.to_numeric(d["cause_effect_ratio_causal"], errors="coerce")

    ts = (
        d.groupby("year", as_index=False)
        .agg(
            mean_non=("ratio_non", "mean"),
            se_non=("ratio_non", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))),
            mean_cau=("ratio_cau", "mean"),
            se_cau=("ratio_cau", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))),
        )
    )
    plt.figure(figsize=(12.8, 8.4))
    plt.plot(ts["year"], ts["mean_non"], color="darkorange", linewidth=2.6, label="Non-Causal")
    plt.plot(ts["year"], ts["mean_cau"], color="#084594", linewidth=2.6, label="Causal")
    plt.fill_between(ts["year"], ts["mean_non"] - 1.96 * ts["se_non"], ts["mean_non"] + 1.96 * ts["se_non"], color="darkorange", alpha=0.14)
    plt.fill_between(ts["year"], ts["mean_cau"] - 1.96 * ts["se_cau"], ts["mean_cau"] + 1.96 * ts["se_cau"], color="#084594", alpha=0.14)
    plt.xlabel("Year")
    plt.ylabel("Outdegree-Indegree ratio")
    plt.legend(frameon=False)
    _apply_classic()
    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "source_sink_ratio_over_time_two_series_non_causal.jpg", dpi=320, bbox_inches="tight")
    plt.close()

    recs = []
    for fcol, flabel in fields:
        tmp = d[d[fcol] == True]  # noqa: E712
        if tmp.empty:
            continue
        recs.append((flabel, "Non-Causal", float(np.nanmean(tmp["ratio_non"])), float(np.nanstd(tmp["ratio_non"], ddof=1) / np.sqrt(np.sum(~np.isnan(tmp["ratio_non"]))))))
        recs.append((flabel, "Causal", float(np.nanmean(tmp["ratio_cau"])), float(np.nanstd(tmp["ratio_cau"], ddof=1) / np.sqrt(np.sum(~np.isnan(tmp["ratio_cau"]))))))
    fr = pd.DataFrame(recs, columns=["field", "series", "value", "se"])
    non_rank = (
        fr[fr["series"] == "Non-Causal"][["field", "value"]]
        .dropna(subset=["value"])
        .drop_duplicates(subset=["field"])
        .set_index("field")["value"]
    )
    order = non_rank.sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(14.2, 10.4))
    ax = sns.pointplot(
        data=fr,
        x="value",
        y="field",
        hue="series",
        order=order,
        palette={"Non-Causal": "darkorange", "Causal": "#084594"},
        dodge=0.45,
        errorbar=None,
        linewidth=2.1,
        scale=1.1,
    )
    # Confidence intervals aligned exactly at the plotted points.
    field_to_y = {f: i for i, f in enumerate(order)}
    for _, r in fr.iterrows():
        y0 = field_to_y[r["field"]]
        dy = -0.17 if r["series"] == "Non-Causal" else 0.17
        ax.errorbar(
            r["value"],
            y0 + dy,
            xerr=1.96 * r["se"],
            fmt="none",
            ecolor="#4a4a4a",
            elinewidth=1.4,
            capsize=3.5,
            alpha=0.9,
            zorder=2,
        )
    plt.xlabel("Outdegree-Indegree ratio")
    plt.ylabel("")
    plt.legend(title="")
    _apply_classic(ax)
    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "source_sink_ratio_by_field_combined_non_causal.jpg", dpi=320, bbox_inches="tight")
    plt.close()


def _build_centrality_figures() -> None:
    _set_style(1.48)
    non = pd.read_csv("int_data/node_centrality_non_causal_all_years.csv")
    cau = pd.read_csv("int_data/node_centrality_causal_all_years.csv")
    non = non.dropna(subset=["node", "eigen_centrality", "year"]).copy()
    cau = cau.dropna(subset=["node", "eigen_centrality", "year"]).copy()
    non["year"] = pd.to_numeric(non["year"], errors="coerce")
    cau["year"] = pd.to_numeric(cau["year"], errors="coerce")
    non["eigen_centrality"] = pd.to_numeric(non["eigen_centrality"], errors="coerce")
    cau["eigen_centrality"] = pd.to_numeric(cau["eigen_centrality"], errors="coerce")

    jel_map = _load_jel_short_map()

    m_non = (
        non.groupby("node", as_index=False)
        .agg(non_centrality=("eigen_centrality", "mean"), non_years=("year", "nunique"))
    )
    m_cau = (
        cau.groupby("node", as_index=False)
        .agg(cau_centrality=("eigen_centrality", "mean"), cau_years=("year", "nunique"))
    )
    both = m_non.merge(m_cau, on="node", how="inner")

    # Appendix figure: centrality levels (top 20 by max across the two systems).
    top_levels = both.copy()
    top_levels["max_c"] = top_levels[["non_centrality", "cau_centrality"]].max(axis=1)
    top_levels = top_levels.sort_values("max_c", ascending=False).head(20).copy()
    top_levels["label"] = top_levels["node"].map(lambda x: _jel_label(x, jel_map))
    top_levels = top_levels.sort_values("max_c", ascending=True)

    plt.figure(figsize=(12.2, 9.8))
    y = np.arange(len(top_levels))
    # Dumbbell style to show level differences clearly.
    for i, r in enumerate(top_levels.itertuples(index=False)):
        plt.plot([r.non_centrality, r.cau_centrality], [i, i], color="#9a9a9a", linewidth=1.7, alpha=0.8, zorder=1)
    plt.scatter(top_levels["non_centrality"], y, color="darkorange", s=84, label="Non-Causal", zorder=3)
    plt.scatter(top_levels["cau_centrality"], y, color="#084594", s=84, label="Causal", zorder=3)
    plt.yticks(y, top_levels["label"])
    plt.xlabel("Mean eigenvector centrality")
    plt.legend(frameon=False, loc="lower right")
    _apply_classic()
    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "top_20_jel_nodes_centrality_levels_appendix.jpg", dpi=320, bbox_inches="tight")
    plt.close()

    # Main figure: where causal vs non-causal centrality differs most vs least.
    cmp_df = both.copy()
    cmp_df = cmp_df[(cmp_df["non_years"] >= 15) & (cmp_df["cau_years"] >= 15)].copy()
    cmp_df["non_rank"] = cmp_df["non_centrality"].rank(pct=True)
    cmp_df["cau_rank"] = cmp_df["cau_centrality"].rank(pct=True)
    cmp_df["rank_diff"] = cmp_df["cau_rank"] - cmp_df["non_rank"]
    cmp_df["abs_rank_diff"] = cmp_df["rank_diff"].abs()
    cmp_df["max_rank"] = cmp_df[["non_rank", "cau_rank"]].max(axis=1)
    cmp_df = cmp_df[cmp_df["max_rank"] >= 0.55].copy()

    top_gap = cmp_df.sort_values("abs_rank_diff", ascending=False).head(10).copy()
    # "Most similar" set: avoid very sparse/noisy nodes by requiring high centrality in at least one graph.
    low_gap_pool = cmp_df[cmp_df["max_rank"] >= 0.75].copy()
    low_gap = low_gap_pool.sort_values("abs_rank_diff", ascending=True).head(10).copy()

    top_gap["group"] = "Largest gaps"
    low_gap["group"] = "Smallest gaps"
    comb = pd.concat([top_gap, low_gap], ignore_index=True)
    comb["label"] = comb["node"].map(lambda x: _jel_label(x, jel_map))

    print("Centrality gap figure - largest gaps:", top_gap["node"].tolist())
    print("Centrality gap figure - smallest gaps:", low_gap["node"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 9.8), sharex=False)
    for ax, grp in zip(axes, ["Largest gaps", "Smallest gaps"]):
        dsub = comb[comb["group"] == grp].sort_values("rank_diff")
        y = np.arange(len(dsub))
        colors = ["#c23b23" if x > 0 else "#2f5597" for x in dsub["rank_diff"]]
        ax.barh(y, dsub["rank_diff"], color=colors, alpha=0.90)
        ax.axvline(0, color="#4a4a4a", linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(dsub["label"])
        ax.set_title(grp)
        ax.set_xlabel("Causal rank - Non-causal rank")
        max_abs = float(np.nanmax(np.abs(dsub["rank_diff"]))) if len(dsub) else 0.0
        lim = max_abs + (0.08 if grp == "Largest gaps" else 0.025)
        if grp == "Smallest gaps":
            lim = max(lim, 0.06)
        ax.set_xlim(-lim, lim)
        _apply_classic(ax)
    handles = [
        Line2D([0], [0], color="#c23b23", lw=6, label="Higher in causal graph"),
        Line2D([0], [0], color="#2f5597", lw=6, label="Higher in non-causal graph"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 1.0), pad=0.7)
    plt.savefig(FIG_DIR / "top_20_jel_nodes_centrality_combined_non_causal.jpg", dpi=320, bbox_inches="tight")
    plt.close()

    def _pick_trend_nodes(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        stats = []
        for node, g in df.groupby("node"):
            g = g.dropna(subset=["year", "eigen_centrality"]).sort_values("year")
            if g["year"].nunique() < 20:
                continue
            x = g["year"].to_numpy(dtype=float)
            y = g["eigen_centrality"].to_numpy(dtype=float)
            if np.nanmax(y) - np.nanmin(y) <= 1e-12:
                continue
            slope = np.polyfit(x, y, 1)[0]
            stats.append((node, slope, float(np.nanmean(y)), int(g["year"].nunique())))
        dd = pd.DataFrame(stats, columns=["node", "slope", "mean", "years"])
        if dd.empty:
            return [], []
        # Keep relatively central nodes to reduce noisy tails.
        dd = dd[dd["mean"] >= dd["mean"].quantile(0.65)].copy()
        rise = dd.sort_values("slope", ascending=False).head(3)["node"].tolist()
        fall = dd.sort_values("slope", ascending=True).head(3)["node"].tolist()
        return rise, fall

    def _trend(df: pd.DataFrame, out_name: str, panel_title: str) -> Tuple[List[str], List[str]]:
        rise, fall = _pick_trend_nodes(df)
        sel = rise + fall
        z = df[df["node"].isin(sel)].copy()
        z = z.sort_values(["node", "year"])
        z["norm"] = z.groupby("node")["eigen_centrality"].transform(
            lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
        )
        warm = ["#b2182b", "#d6604d", "#f4a582"]
        cool = ["#2166ac", "#4393c3", "#92c5de"]
        color_map: Dict[str, str] = {}
        for i, n in enumerate(rise):
            color_map[n] = warm[i % len(warm)]
        for i, n in enumerate(fall):
            color_map[n] = cool[i % len(cool)]

        plt.figure(figsize=(14.8, 9.2))
        for node in sel:
            sub = z[z["node"] == node]
            if sub.empty:
                continue
            plt.plot(
                sub["year"],
                sub["norm"],
                linewidth=2.7,
                alpha=0.95,
                color=color_map.get(node, "#555555"),
                label=_jel_label(node, jel_map),
            )
        plt.title(panel_title)
        plt.xlabel("Year")
        plt.ylabel("Normalized eigenvector centrality")
        _apply_classic()
        plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
        plt.tight_layout(pad=0.6)
        plt.savefig(FIG_DIR / out_name, dpi=320, bbox_inches="tight")
        plt.close()
        return rise, fall

    rise_non, fall_non = _trend(
        non,
        "non_causal_node_centrality_selected_trends_with_descriptions.jpg",
        "Non-Causal: Top 3 rising vs declining concepts",
    )
    rise_cau, fall_cau = _trend(
        cau,
        "causal_node_centrality_selected_trends_with_descriptions.jpg",
        "Causal: Top 3 rising vs declining concepts",
    )
    print("Trend nodes non-causal rising:", rise_non, "declining:", fall_non)
    print("Trend nodes causal rising:", rise_cau, "declining:", fall_cau)


def _build_conceptual_importance_figure(edge_data: pd.DataFrame, pl: pd.DataFrame) -> None:
    cites = _load_citations_table()
    if cites is None:
        return

    pm = edge_data[["paper_id", "year"]].drop_duplicates(subset=["paper_id"])
    d = pl.merge(pm, on="paper_id", how="left").merge(cites, on="paper_id", how="left")
    d["journal_rank"] = pd.to_numeric(d["journal_rank"], errors="coerce")
    d["coalesced_cites"] = pd.to_numeric(d["coalesced_cites"], errors="coerce")
    d["paper_is_top_5"] = (d["journal_rank"] <= 5).astype(float)
    d["paper_is_top_6_20"] = ((d["journal_rank"] >= 6) & (d["journal_rank"] <= 20)).astype(float)
    d["paper_is_top_21_100"] = ((d["journal_rank"] >= 21) & (d["journal_rank"] <= 100)).astype(float)
    d["citation_percentile"] = d["coalesced_cites"].rank(pct=True)

    predictors = [
        ("mean_eigen_centrality_non_causal_RelCumLit_non_causal", "Topic Centrality", "Non-Causal"),
        ("mean_eigen_centrality_causal_RelCumLit_causal", "Topic Centrality", "Causal"),
        ("var_eigen_centrality_non_causal_RelCumLit_non_causal", "Topic Diversity", "Non-Causal"),
        ("var_eigen_centrality_causal_RelCumLit_causal", "Topic Diversity", "Causal"),
        ("cause_effect_ratio_non_causal", "Source-Sink Ratio", "Non-Causal"),
        ("cause_effect_ratio_causal", "Source-Sink Ratio", "Causal"),
    ]
    outcomes = [
        ("paper_is_top_5", "In Top 5"),
        ("paper_is_top_6_20", "In Top 6-20"),
        ("paper_is_top_21_100", "In Top 21-100"),
        ("citation_percentile", "Citations"),
    ]

    recs = []
    for ycol, ylab in outcomes:
        for xcol, fam, typ in predictors:
            sub = d[[ycol, xcol, "year"]].dropna().copy()
            if sub.empty:
                continue
            X = pd.get_dummies(sub[["year"]], columns=["year"], drop_first=True)
            X[xcol] = sub[xcol].values
            X = sm.add_constant(X, has_constant="add")
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
            y = pd.to_numeric(sub[ycol], errors="coerce").fillna(0.0).astype(float).values
            m = sm.OLS(y, X.values).fit()
            idx = list(X.columns).index(xcol)
            recs.append((ylab, fam, typ, float(m.params[idx]), float(m.bse[idx])))
    if not recs:
        return
    rr = pd.DataFrame(recs, columns=["outcome", "family", "type", "coef", "se"])
    rr["family"] = pd.Categorical(rr["family"], ["Topic Centrality", "Topic Diversity", "Source-Sink Ratio"], ordered=True)
    rr["outcome"] = pd.Categorical(rr["outcome"], ["In Top 5", "In Top 6-20", "In Top 21-100", "Citations"], ordered=True)

    g = sns.FacetGrid(rr.sort_values(["family", "outcome"]), row="family", col="outcome", margin_titles=True, sharex=False, sharey=False, height=2.5)
    def _panel(data, color, **kwargs):
        ax = plt.gca()
        for _, r in data.iterrows():
            x = r["coef"]
            y = 0 if r["type"] == "Non-Causal" else 1
            c = "darkorange" if r["type"] == "Non-Causal" else "#084594"
            ax.errorbar(x, y, xerr=1.96 * r["se"], fmt="o", color=c, capsize=2)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
        ax.set_yticks([0, 1], ["Non-Causal", "Causal"])

    g.map_dataframe(_panel)
    g.set_axis_labels("Coefficient (year FE)", "")
    g.fig.tight_layout()
    g.fig.savefig(FIG_DIR / "conceptual_importance_variables_combined_plot_non_causal_vs_causal.jpg", dpi=300)
    plt.close(g.fig)


def main() -> None:
    edge_data = _load_edge_data()
    pl = _load_paper_level()
    _build_prop_causal_core_figures(edge_data)
    _build_source_sink_figures(edge_data, pl)
    _build_centrality_figures()
    _draw_example_graphs(edge_data)
    _build_gap_filling_panels(pl)
    _build_top5_field_method_figures(edge_data, pl)
    _build_citation_density(pl)
    _build_conceptual_importance_figure(edge_data, pl)
    print("Done: built core figure set in ./figures")


if __name__ == "__main__":
    main()
