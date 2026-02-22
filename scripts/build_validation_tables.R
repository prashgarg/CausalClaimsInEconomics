#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(arrow)
  library(readxl)
  library(fixest)
  library(broom)
})

dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("logs", recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(x, y) if (is.null(x)) y else x

count_semicolon_items <- function(x) {
  if (is.na(x) || !nzchar(trimws(x))) return(0L)
  parts <- trimws(unlist(strsplit(x, ";", fixed = TRUE)))
  as.integer(sum(nzchar(parts)))
}

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_sd <- function(x) if (sum(!is.na(x)) <= 1L) NA_real_ else sd(x, na.rm = TRUE)

ci95 <- function(x) {
  n <- sum(!is.na(x))
  if (n <= 1L) return(c(NA_real_, NA_real_))
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  e <- 1.96 * s / sqrt(n)
  c(m - e, m + e)
}

message("[VALIDATION] Building snippet validation metrics...")

resp_path <- "int_data/validation/s1_iter1/edge_validation_responses.csv"
parcel_path <- "int_data/paper_parcels_for_validation.json"

if (!file.exists(resp_path)) stop("Missing file: ", resp_path)
if (!file.exists(parcel_path)) stop("Missing file: ", parcel_path)

resp <- fread(resp_path)
required_resp_cols <- c("paper_id", "best_set", "spurious_ids", "missing_edges",
                        "score_accuracy", "score_coverage")
missing_resp_cols <- setdiff(required_resp_cols, names(resp))
if (length(missing_resp_cols)) {
  stop("Missing required columns in snippet responses: ",
       paste(missing_resp_cols, collapse = ", "))
}

resp[, c("base_id", "candidate_set") := tstrsplit(paper_id, "_val_", fixed = TRUE)]
resp[, candidate_set := toupper(candidate_set)]
resp[, best_set := toupper(best_set)]

parcels <- fromJSON(parcel_path, simplifyVector = FALSE)

edge_count <- function(x) {
  if (is.null(x)) return(0L)
  if (is.data.frame(x)) return(as.integer(nrow(x)))
  if (is.list(x) && !is.null(x$id)) return(as.integer(length(x$id)))
  if (is.list(x)) return(as.integer(length(x)))
  0L
}

parcel_counts <- rbindlist(lapply(parcels, function(p) {
  cs <- p$claim_sets %||% list()
  data.table(
    base_id = as.character(p$paper_id %||% NA_character_),
    A = edge_count(cs$A),
    B = edge_count(cs$B),
    C = edge_count(cs$C)
  )
}), fill = TRUE)

counts_long <- melt(
  parcel_counts,
  id.vars = "base_id",
  variable.name = "candidate_set",
  value.name = "candidate_size"
)
counts_long[, candidate_set := as.character(candidate_set)]
counts_long[, candidate_size := as.integer(candidate_size)]

resp <- merge(resp, counts_long, by = c("base_id", "candidate_set"), all.x = TRUE)

abc_map <- resp[candidate_set == "ABC" & best_set %chin% c("A", "B", "C"),
                .(base_id, best_set)]
if (nrow(abc_map)) {
  abc_sizes <- merge(
    abc_map,
    counts_long,
    by.x = c("base_id", "best_set"),
    by.y = c("base_id", "candidate_set"),
    all.x = TRUE
  )
  setnames(abc_sizes, "candidate_size", "candidate_size_abc")
  resp <- merge(resp, abc_sizes[, .(base_id, best_set, candidate_size_abc)],
                by = c("base_id", "best_set"), all.x = TRUE)
  resp[candidate_set == "ABC" & !is.na(candidate_size_abc),
       candidate_size := candidate_size_abc]
  resp[, candidate_size_abc := NULL]
}

resp[is.na(candidate_size), candidate_size := 0L]
resp[, fp_count := vapply(spurious_ids, count_semicolon_items, integer(1))]
resp[, fn_count := vapply(missing_edges, count_semicolon_items, integer(1))]
resp[, score_accuracy := as.numeric(score_accuracy)]
resp[, score_coverage := as.numeric(score_coverage)]

resp[, effective_set := fifelse(candidate_set == "ABC" & best_set %chin% c("A", "B", "C"),
                                best_set, candidate_set)]

resp[, eo_proxy := fcase(
  effective_set == "A", "EO>=1",
  effective_set == "B", "EO>=2",
  effective_set == "C", "EO=3",
  default = "UNMAPPED"
)]

resp[, tp := pmax(as.integer(candidate_size) - fp_count, 0L)]
resp[, support_edges := tp + fn_count]
resp[, audited_total := tp + fp_count + fn_count]
resp[, precision := fifelse(candidate_size > 0L, tp / candidate_size, NA_real_)]
resp[, recall := fifelse((tp + fn_count) > 0L, tp / (tp + fn_count), NA_real_)]
resp[, f1 := fifelse(!is.na(precision) & !is.na(recall) & (precision + recall) > 0,
                     2 * precision * recall / (precision + recall), NA_real_)]
resp[, audited_positive_prevalence :=
       fifelse(audited_total > 0L, support_edges / audited_total, NA_real_)]

snippet_by_set <- resp[, {
  p_ci <- ci95(precision)
  r_ci <- ci95(recall)
  f_ci <- ci95(f1)
  .(
    n_cases = .N,
    micro_tp = sum(tp, na.rm = TRUE),
    micro_fp = sum(fp_count, na.rm = TRUE),
    micro_fn = sum(fn_count, na.rm = TRUE),
    micro_precision = sum(tp, na.rm = TRUE) /
      (sum(tp, na.rm = TRUE) + sum(fp_count, na.rm = TRUE)),
    micro_recall = sum(tp, na.rm = TRUE) /
      (sum(tp, na.rm = TRUE) + sum(fn_count, na.rm = TRUE)),
    micro_f1 = {
      p <- sum(tp, na.rm = TRUE) / (sum(tp, na.rm = TRUE) + sum(fp_count, na.rm = TRUE))
      r <- sum(tp, na.rm = TRUE) / (sum(tp, na.rm = TRUE) + sum(fn_count, na.rm = TRUE))
      ifelse((p + r) > 0, 2 * p * r / (p + r), NA_real_)
    },
    mean_precision = safe_mean(precision),
    mean_recall = safe_mean(recall),
    mean_f1 = safe_mean(f1),
    precision_ci_low = p_ci[1],
    precision_ci_high = p_ci[2],
    recall_ci_low = r_ci[1],
    recall_ci_high = r_ci[2],
    f1_ci_low = f_ci[1],
    f1_ci_high = f_ci[2],
    support_edges = sum(support_edges, na.rm = TRUE),
    mean_audited_positive_prevalence = safe_mean(audited_positive_prevalence),
    mean_score_accuracy = safe_mean(score_accuracy),
    mean_score_coverage = safe_mean(score_coverage)
  )
}, by = .(candidate_set, effective_set, eo_proxy)]

snippet_by_set[, aggregation_level := "candidate_set"]

snippet_by_eo <- resp[, {
  p_ci <- ci95(precision)
  r_ci <- ci95(recall)
  f_ci <- ci95(f1)
  .(
    n_cases = .N,
    micro_tp = sum(tp, na.rm = TRUE),
    micro_fp = sum(fp_count, na.rm = TRUE),
    micro_fn = sum(fn_count, na.rm = TRUE),
    micro_precision = sum(tp, na.rm = TRUE) /
      (sum(tp, na.rm = TRUE) + sum(fp_count, na.rm = TRUE)),
    micro_recall = sum(tp, na.rm = TRUE) /
      (sum(tp, na.rm = TRUE) + sum(fn_count, na.rm = TRUE)),
    micro_f1 = {
      p <- sum(tp, na.rm = TRUE) / (sum(tp, na.rm = TRUE) + sum(fp_count, na.rm = TRUE))
      r <- sum(tp, na.rm = TRUE) / (sum(tp, na.rm = TRUE) + sum(fn_count, na.rm = TRUE))
      ifelse((p + r) > 0, 2 * p * r / (p + r), NA_real_)
    },
    mean_precision = safe_mean(precision),
    mean_recall = safe_mean(recall),
    mean_f1 = safe_mean(f1),
    precision_ci_low = p_ci[1],
    precision_ci_high = p_ci[2],
    recall_ci_low = r_ci[1],
    recall_ci_high = r_ci[2],
    f1_ci_low = f_ci[1],
    f1_ci_high = f_ci[2],
    support_edges = sum(support_edges, na.rm = TRUE),
    mean_audited_positive_prevalence = safe_mean(audited_positive_prevalence),
    mean_score_accuracy = safe_mean(score_accuracy),
    mean_score_coverage = safe_mean(score_coverage)
  )
}, by = .(eo_proxy)]
snippet_by_eo[, `:=`(candidate_set = "ALL", effective_set = "ALL",
                     aggregation_level = "eo_proxy")]

validation_snippet_metrics <- rbindlist(
  list(snippet_by_set, snippet_by_eo),
  fill = TRUE
)

out_snip <- "results/tables/validation_snippet_metrics.csv"
fwrite(validation_snippet_metrics, out_snip)
message("[VALIDATION] Wrote: ", out_snip)


message("[VALIDATION] Building perturbation stability metrics...")
eo_paths <- data.table(
  threshold = 1:9,
  path = sprintf("int_data/edge_overlap_runs/eo_ge%d/paper_level_data_eo_ge%d.parquet", 1:9, 1:9)
)
eo_paths <- eo_paths[file.exists(path)]
if (!nrow(eo_paths)) stop("No EO parquet files found for perturbation summary.")

eo_stats <- rbindlist(lapply(eo_paths$threshold, function(thr) {
  p <- sprintf("int_data/edge_overlap_runs/eo_ge%d/paper_level_data_eo_ge%d.parquet", thr, thr)
  dt <- as.data.table(read_parquet(p))

  required <- c("num_edges", "num_causal_edges", "prop_edges_causal")
  miss <- setdiff(required, names(dt))
  if (length(miss)) stop("Missing columns in ", p, ": ", paste(miss, collapse = ", "))

  data.table(
    threshold = thr,
    perturbation_intensity = thr - 1L,
    n_papers = nrow(dt),
    mean_num_edges = mean(as.numeric(dt$num_edges), na.rm = TRUE),
    mean_num_causal_edges = mean(as.numeric(dt$num_causal_edges), na.rm = TRUE),
    mean_prop_edges_causal = mean(as.numeric(dt$prop_edges_causal), na.rm = TRUE),
    median_prop_edges_causal = median(as.numeric(dt$prop_edges_causal), na.rm = TRUE)
  )
}))

if (!(4 %in% eo_stats$threshold)) {
  stop("EO>=4 baseline not available; cannot compute perturbation-relative metrics.")
}

base <- eo_stats[threshold == 4]
eo_stats[, `:=`(
  baseline_mean_num_edges = base$mean_num_edges[1],
  baseline_mean_num_causal_edges = base$mean_num_causal_edges[1],
  baseline_mean_prop_edges_causal = base$mean_prop_edges_causal[1],
  pct_change_mean_num_edges_vs_eo4 =
    100 * (mean_num_edges - base$mean_num_edges[1]) / base$mean_num_edges[1],
  pct_change_mean_num_causal_edges_vs_eo4 =
    100 * (mean_num_causal_edges - base$mean_num_causal_edges[1]) / base$mean_num_causal_edges[1],
  pct_change_mean_prop_edges_causal_vs_eo4 =
    100 * (mean_prop_edges_causal - base$mean_prop_edges_causal[1]) / base$mean_prop_edges_causal[1]
)]

eo_stats[, within_10pct_prop_edges_causal_vs_eo4 :=
           abs(pct_change_mean_prop_edges_causal_vs_eo4) <= 10]

out_pert <- "results/tables/validation_perturbation_metrics.csv"
fwrite(eo_stats[order(threshold)], out_pert)
message("[VALIDATION] Wrote: ", out_pert)


message("[VALIDATION] Building Brodeur metrics table...")

brodeur_grid_path <- "results/tables/validation_brodeur_class_metrics_eo_grid.csv"
if (!file.exists(brodeur_grid_path)) {
  stop("Missing Brodeur EO-grid file: ", brodeur_grid_path,
       ". Run scripts/validate_brodeur.R first.")
}

brodeur_grid <- fread(brodeur_grid_path)
need_brod <- c("dataset_variant", "eo_threshold", "eo_label", "variable", "class_type",
               "accuracy", "precision", "recall", "f1_score", "support_count",
               "class_prevalence", "tp", "fp", "tn", "fn", "matched_papers",
               "exact_matches", "fuzzy_matches")
miss_brod <- setdiff(need_brod, names(brodeur_grid))
if (length(miss_brod)) {
  stop("Brodeur EO-grid file missing required columns: ", paste(miss_brod, collapse = ", "))
}

selected_thresholds <- c(4L, 7L, 9L)
brodeur_metrics <- brodeur_grid[
  dataset_variant == "primary" & eo_threshold %in% selected_thresholds,
  ..need_brod
]

if (!nrow(brodeur_metrics)) {
  stop("No primary Brodeur metrics found for EO thresholds 4, 7, 9.")
}
if (any(is.na(brodeur_metrics$precision)) || any(is.na(brodeur_metrics$recall))) {
  stop("Brodeur precision/recall contain NA in selected EO thresholds.")
}

setorder(brodeur_metrics, eo_threshold, variable)
out_brod <- "results/tables/validation_brodeur_metrics.csv"
fwrite(brodeur_metrics, out_brod)
message("[VALIDATION] Wrote: ", out_brod)


message("[VALIDATION] Building exogenous-benchmark metrics table...")

plausibly_sim_path <- "results/tables/validation_plausibly_similarity_eo_grid.csv"
if (!file.exists(plausibly_sim_path)) {
  stop("Missing Plausibly EO-grid file: ", plausibly_sim_path,
       ". Run scripts/validate_exogenous_benchmark.py first.")
}

plausibly_grid <- fread(plausibly_sim_path)
need_pl <- c("eo_threshold", "eo_label", "component", "n_matched", "n_permutations",
             "max_similarity", "mean_similarity", "median_similarity", "std_similarity",
             "min_similarity", "random_baseline_mean", "lift_vs_baseline",
             "permutation_p_value")
miss_pl <- setdiff(need_pl, names(plausibly_grid))
if (length(miss_pl)) {
  stop("Plausibly EO-grid file missing required columns: ", paste(miss_pl, collapse = ", "))
}

plausibly_metrics <- plausibly_grid[eo_threshold %in% selected_thresholds, ..need_pl]
if (!nrow(plausibly_metrics)) {
  stop("No Plausibly metrics found for EO thresholds 4, 7, 9.")
}
if (any(is.na(plausibly_metrics$random_baseline_mean)) ||
    any(is.na(plausibly_metrics$lift_vs_baseline))) {
  stop("Plausibly baseline/lift contain NA in selected EO thresholds.")
}

setorder(plausibly_metrics, eo_threshold, component)
out_pl <- "results/tables/validation_plausibly_metrics.csv"
fwrite(plausibly_metrics, out_pl)
message("[VALIDATION] Wrote: ", out_pl)


message("[VALIDATION] Building Plausibly Exogenous coverage summary...")

plausibly_path <- "int_data/4_plausibly_exogenous/list_20july2024.xlsx"
if (file.exists(plausibly_path)) {
  px <- as.data.table(read_excel(plausibly_path))
  setnames(px, names(px), make.names(names(px), unique = TRUE))
  plausibly_summary <- data.table(
    n_rows = nrow(px),
    n_non_missing_title = sum(!is.na(px$title) & nzchar(trimws(as.character(px$title)))),
    n_non_missing_lhs = sum(!is.na(px$lhs) & nzchar(trimws(as.character(px$lhs)))),
    n_non_missing_rhs = sum(!is.na(px$rhs) & nzchar(trimws(as.character(px$rhs)))),
    n_non_missing_exogenous = sum(!is.na(px$source_of_exogenous_variation) &
                                    nzchar(trimws(as.character(px$source_of_exogenous_variation))))
  )
  fwrite(plausibly_summary, "results/tables/validation_plausibly_coverage.csv")
  message("[VALIDATION] Wrote: results/tables/validation_plausibly_coverage.csv")
}

message("[VALIDATION] Done.")
