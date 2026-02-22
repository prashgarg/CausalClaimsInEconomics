#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(readr)
  library(stringdist)
})

dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

message("[BRODEUR] Starting Brodeur validation build...")

find_input_file <- function(patterns) {
  for (pat in patterns) {
    hits <- Sys.glob(pat)
    if (length(hits) > 0) return(hits[[1]])
  }
  stop("Missing required file. Tried patterns: ", paste(patterns, collapse = ", "))
}

meta_path <- find_input_file(c(
  "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
  "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet"
))
s1_path <- find_input_file(c(
  "int_data/dat_*s1_aggregated.parquet",
  "int_data/claim_graph_s1_aggregated.parquet"
))
bench_primary_path <- "abel_etal_EJ_replic_pack/data/bcn_public_v1_as_csv.csv"
bench_sens_path <- "abel_etal_EJ_replic_pack/data/bcn_public_with_wp_v1_as_csv.csv"

required_paths <- c(meta_path, s1_path, bench_primary_path, bench_sens_path)
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths)) {
  stop("[BRODEUR] Missing required file(s): ", paste(missing_paths, collapse = ", "))
}

clean_title <- function(x) {
  x <- as.character(x)
  x <- iconv(x, to = "ASCII//TRANSLIT", sub = "")
  x <- tolower(x)
  x <- gsub("[[:punct:]]", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

to_binary <- function(x) {
  y <- toupper(trimws(as.character(x)))
  out <- rep(NA_integer_, length(y))
  out[y %chin% c("1", "TRUE", "T", "YES", "Y")] <- 1L
  out[y %chin% c("0", "FALSE", "F", "NO", "N", ".", "")] <- 0L
  suppressWarnings(num <- as.numeric(y))
  out[is.na(out) & !is.na(num)] <- as.integer(num[is.na(out) & !is.na(num)] > 0)
  out
}

bin_or <- function(x) {
  xb <- to_binary(x)
  if (all(is.na(xb))) return(NA_integer_)
  as.integer(any(xb == 1L, na.rm = TRUE))
}

safe_div <- function(num, den) {
  ifelse(den > 0, num / den, NA_real_)
}

safe_div_zero <- function(num, den) {
  ifelse(den > 0, num / den, 0)
}

read_benchmark <- function(path, dataset_variant) {
  raw <- suppressMessages(read_csv(path, locale = locale(encoding = "Latin1"), show_col_types = FALSE))
  dt <- as.data.table(raw)

  needed <- c("title", "DID", "IV", "RDD", "EXP", "FINANCE", "MACRO_GROWTH", "DEV", "URB")
  missing_cols <- setdiff(needed, names(dt))
  if (length(missing_cols)) {
    stop("[BRODEUR] Missing benchmark columns in ", path, ": ", paste(missing_cols, collapse = ", "))
  }

  dt <- dt[, ..needed]
  dt[, cleaned_title := clean_title(title)]
  dt <- dt[!is.na(cleaned_title) & nzchar(cleaned_title)]

  agg <- dt[, .(
    benchmark_title = title[which.max(nchar(as.character(title)))][1],
    truth_did = bin_or(DID),
    truth_iv = bin_or(IV),
    truth_rdd = bin_or(RDD),
    truth_rct = bin_or(EXP),
    truth_finance = bin_or(FINANCE),
    truth_macro = bin_or(MACRO_GROWTH),
    truth_dev = bin_or(DEV),
    truth_urban = bin_or(URB)
  ), by = cleaned_title]

  agg[, benchmark_id := sprintf("%s_%05d", dataset_variant, .I)]
  agg[, dataset_variant := dataset_variant]
  setcolorder(agg, c("benchmark_id", "dataset_variant", "cleaned_title", "benchmark_title",
                     "truth_did", "truth_rct", "truth_rdd", "truth_iv",
                     "truth_urban", "truth_finance", "truth_macro", "truth_dev"))
  agg[]
}

message("[BRODEUR] Reading inputs...")

edges <- as.data.table(read_parquet(
  meta_path,
  col_select = c("paper_id", "title", "edge_overlap", "causal_inference_method")
))
s1 <- as.data.table(read_parquet(
  s1_path,
  col_select = c("paper_id", "title", "is_urban_economics", "is_finance", "is_macroeconomics", "is_development")
))

edges[, edge_overlap := as.integer(edge_overlap)]
edges <- edges[!is.na(paper_id) & !is.na(edge_overlap)]

title_map <- rbindlist(
  list(
    s1[, .(paper_id, title)],
    edges[, .(paper_id, title)]
  ),
  fill = TRUE
)
title_map <- title_map[!is.na(title) & nzchar(trimws(title))]
title_map <- title_map[, .(llm_title = title[which.max(nchar(as.character(title)))][1]), by = paper_id]

field_map <- unique(s1[, .(
  paper_id,
  pred_urban = as.integer(to_binary(is_urban_economics) == 1L),
  pred_finance = as.integer(to_binary(is_finance) == 1L),
  pred_macro = as.integer(to_binary(is_macroeconomics) == 1L),
  pred_dev = as.integer(to_binary(is_development) == 1L)
)])
for (f in c("pred_urban", "pred_finance", "pred_macro", "pred_dev")) {
  field_map[is.na(get(f)), (f) := 0L]
}

edges[, method_upper := toupper(trimws(as.character(causal_inference_method)))]
edges[is.na(method_upper), method_upper := ""]
edges[, has_did := grepl("\\bDID\\b|DIFFERENCE\\s*-?\\s*IN\\s*-?\\s*DIFFERENCES", method_upper)]
edges[, has_rct := grepl("\\bRCT\\b|RANDOMI[ZS]ED|EXPERIMENT", method_upper)]
edges[, has_rdd := grepl("\\bRDD\\b|REGRESSION\\s+DISCONTINUITY", method_upper)]
edges[, has_iv := grepl("\\bIV\\b|INSTRUMENTAL", method_upper)]

method_overlap <- edges[, .(
  paper_max_eo = max(edge_overlap, na.rm = TRUE),
  max_did_eo = suppressWarnings(max(ifelse(has_did, as.numeric(edge_overlap), NA_real_), na.rm = TRUE)),
  max_rct_eo = suppressWarnings(max(ifelse(has_rct, as.numeric(edge_overlap), NA_real_), na.rm = TRUE)),
  max_rdd_eo = suppressWarnings(max(ifelse(has_rdd, as.numeric(edge_overlap), NA_real_), na.rm = TRUE)),
  max_iv_eo = suppressWarnings(max(ifelse(has_iv, as.numeric(edge_overlap), NA_real_), na.rm = TRUE))
), by = paper_id]

for (mcol in c("max_did_eo", "max_rct_eo", "max_rdd_eo", "max_iv_eo")) {
  method_overlap[get(mcol) == -Inf, (mcol) := NA_real_]
}

label_spec <- data.table(
  variable = c("Method: DiD", "Method: RCT", "Method: RDD", "Method: IV",
               "Field: Urban Economics", "Field: Finance", "Field: Macroeconomics", "Field: Development"),
  class_type = "One-vs-rest",
  pred_col = c("pred_did", "pred_rct", "pred_rdd", "pred_iv",
               "pred_urban", "pred_finance", "pred_macro", "pred_dev"),
  truth_col = c("truth_did", "truth_rct", "truth_rdd", "truth_iv",
                "truth_urban", "truth_finance", "truth_macro", "truth_dev")
)

make_llm_table <- function(eo_threshold) {
  llm <- copy(method_overlap)[paper_max_eo >= eo_threshold]
  llm[, `:=`(
    pred_did = as.integer(!is.na(max_did_eo) & max_did_eo >= eo_threshold),
    pred_rct = as.integer(!is.na(max_rct_eo) & max_rct_eo >= eo_threshold),
    pred_rdd = as.integer(!is.na(max_rdd_eo) & max_rdd_eo >= eo_threshold),
    pred_iv = as.integer(!is.na(max_iv_eo) & max_iv_eo >= eo_threshold)
  )]
  llm <- merge(llm, field_map, by = "paper_id", all.x = TRUE)
  llm <- merge(llm, title_map, by = "paper_id", all.x = TRUE)
  for (f in c("pred_urban", "pred_finance", "pred_macro", "pred_dev")) {
    llm[is.na(get(f)), (f) := 0L]
  }
  llm <- llm[!is.na(llm_title) & nzchar(trimws(llm_title))]
  llm[, cleaned_title := clean_title(llm_title)]
  llm <- llm[!is.na(cleaned_title) & nzchar(cleaned_title)]
  setorder(llm, -paper_max_eo, paper_id)
  llm <- llm[!duplicated(cleaned_title)]
  llm[]
}

match_titles <- function(llm_tbl, bench_tbl, max_dist = 0.10) {
  exact_keys <- intersect(llm_tbl$cleaned_title, bench_tbl$cleaned_title)

  exact <- merge(
    llm_tbl[cleaned_title %chin% exact_keys],
    bench_tbl[cleaned_title %chin% exact_keys],
    by = "cleaned_title"
  )
  exact[, `:=`(match_type = "exact", distance = 0)]

  llm_un <- llm_tbl[!cleaned_title %chin% exact_keys]
  bench_un <- bench_tbl[!cleaned_title %chin% exact_keys]

  fuzzy <- data.table()
  if (nrow(llm_un) > 0 && nrow(bench_un) > 0) {
    idx <- amatch(
      bench_un$cleaned_title,
      llm_un$cleaned_title,
      method = "jw",
      maxDist = max_dist,
      nomatch = NA_integer_
    )
    cand <- data.table(bench_row = seq_len(nrow(bench_un)), llm_row = idx)
    cand <- cand[!is.na(llm_row)]

    if (nrow(cand) > 0) {
      cand[, distance := stringdist(
        bench_un$cleaned_title[bench_row],
        llm_un$cleaned_title[llm_row],
        method = "jw"
      )]
      cand <- cand[distance <= max_dist]
      setorder(cand, distance)
      cand <- cand[!duplicated(llm_row)]
      if (nrow(cand) > 0) {
        fuzzy <- cbind(llm_un[cand$llm_row], bench_un[cand$bench_row])
        fuzzy[, `:=`(match_type = "fuzzy", distance = cand$distance)]
      }
    }
  }

  matches <- rbindlist(list(exact, fuzzy), fill = TRUE, use.names = TRUE)
  if (nrow(matches) == 0) return(list(matches = matches, match_log = matches))

  matches <- matches[, .(
    paper_id,
    llm_title,
    cleaned_title,
    pred_did, pred_rct, pred_rdd, pred_iv,
    pred_urban, pred_finance, pred_macro, pred_dev,
    benchmark_id,
    benchmark_title,
    truth_did, truth_rct, truth_rdd, truth_iv,
    truth_urban, truth_finance, truth_macro, truth_dev,
    match_type,
    distance
  )]

  log_dt <- copy(matches)[, .(
    paper_id,
    benchmark_id,
    cleaned_title,
    llm_title,
    benchmark_title,
    match_type,
    distance
  )]

  list(matches = matches, match_log = log_dt)
}

calc_class_metrics <- function(matches, variable, class_type, pred_col, truth_col) {
  pred <- as.integer(matches[[pred_col]])
  truth <- as.integer(matches[[truth_col]])
  valid <- !is.na(pred) & !is.na(truth)
  pred <- pred[valid]
  truth <- truth[valid]

  n <- length(pred)
  if (n == 0) {
    return(data.table(
      variable = variable,
      class_type = class_type,
      support_count = 0L,
      class_prevalence = NA_real_,
      tp = NA_integer_, fp = NA_integer_, tn = NA_integer_, fn = NA_integer_,
      accuracy = NA_real_, precision = NA_real_, recall = NA_real_, f1_score = NA_real_
    ))
  }

  tp <- sum(pred == 1L & truth == 1L)
  fp <- sum(pred == 1L & truth == 0L)
  tn <- sum(pred == 0L & truth == 0L)
  fn <- sum(pred == 0L & truth == 1L)

  precision <- safe_div_zero(tp, tp + fp)
  recall <- safe_div_zero(tp, tp + fn)
  f1 <- ifelse((precision + recall) > 0,
               2 * precision * recall / (precision + recall), 0)

  data.table(
    variable = variable,
    class_type = class_type,
    support_count = as.integer(n),
    class_prevalence = mean(truth == 1L),
    tp = as.integer(tp),
    fp = as.integer(fp),
    tn = as.integer(tn),
    fn = as.integer(fn),
    accuracy = safe_div(tp + tn, n),
    precision = precision,
    recall = recall,
    f1_score = f1
  )
}

benchmarks <- list(
  primary = read_benchmark(bench_primary_path, "primary"),
  sensitivity_wp = read_benchmark(bench_sens_path, "sensitivity_wp")
)

thresholds <- 1:9
llm_tables <- lapply(thresholds, make_llm_table)
names(llm_tables) <- as.character(thresholds)

all_metrics <- list()
all_logs <- list()
all_summary <- list()
counter <- 1L

for (dataset_variant in names(benchmarks)) {
  bench_tbl <- benchmarks[[dataset_variant]]
  message("[BRODEUR] Processing benchmark variant: ", dataset_variant)

  for (thr in thresholds) {
    eo_label <- if (thr == 9) "EO = 9" else paste0("EO >= ", thr)
    llm_tbl <- llm_tables[[as.character(thr)]]

    m <- match_titles(llm_tbl, bench_tbl, max_dist = 0.10)
    matches <- m$matches
    match_log <- m$match_log

    if (nrow(matches) == 0) {
      message("[BRODEUR] No matches for ", dataset_variant, " at ", eo_label)
      next
    }

    metrics <- rbindlist(lapply(seq_len(nrow(label_spec)), function(i) {
      calc_class_metrics(
        matches = matches,
        variable = label_spec$variable[i],
        class_type = label_spec$class_type[i],
        pred_col = label_spec$pred_col[i],
        truth_col = label_spec$truth_col[i]
      )
    }), fill = TRUE)

    metrics[, `:=`(
      dataset_variant = dataset_variant,
      eo_threshold = thr,
      eo_label = eo_label,
      matched_papers = nrow(matches),
      exact_matches = sum(matches$match_type == "exact"),
      fuzzy_matches = sum(matches$match_type == "fuzzy")
    )]

    match_log[, `:=`(
      dataset_variant = dataset_variant,
      eo_threshold = thr,
      eo_label = eo_label
    )]

    all_metrics[[counter]] <- metrics
    all_logs[[counter]] <- match_log
    all_summary[[counter]] <- data.table(
      dataset_variant = dataset_variant,
      eo_threshold = thr,
      eo_label = eo_label,
      n_llm_titles = nrow(llm_tbl),
      n_benchmark_titles = nrow(bench_tbl),
      matched_papers = nrow(matches),
      exact_matches = sum(matches$match_type == "exact"),
      fuzzy_matches = sum(matches$match_type == "fuzzy"),
      macro_accuracy = mean(metrics$accuracy, na.rm = TRUE),
      macro_precision = mean(metrics$precision, na.rm = TRUE),
      macro_recall = mean(metrics$recall, na.rm = TRUE),
      macro_f1 = mean(metrics$f1_score, na.rm = TRUE)
    )
    counter <- counter + 1L
  }
}

if (!length(all_metrics)) {
  stop("[BRODEUR] No metrics produced. Matching may have failed.")
}

class_metrics <- rbindlist(all_metrics, fill = TRUE, use.names = TRUE)
match_logs <- rbindlist(all_logs, fill = TRUE, use.names = TRUE)
threshold_summary <- rbindlist(all_summary, fill = TRUE, use.names = TRUE)

setorder(class_metrics, dataset_variant, eo_threshold, variable)
setorder(match_logs, dataset_variant, eo_threshold, match_type, distance)
setorder(threshold_summary, dataset_variant, eo_threshold)

out_class <- "results/tables/validation_brodeur_class_metrics_eo_grid.csv"
out_summary <- "results/tables/validation_brodeur_threshold_summary.csv"
out_log <- "results/tables/validation_brodeur_match_log.csv"

fwrite(class_metrics, out_class)
fwrite(threshold_summary, out_summary)
fwrite(match_logs, out_log)

message("[BRODEUR] Wrote: ", out_class)
message("[BRODEUR] Wrote: ", out_summary)
message("[BRODEUR] Wrote: ", out_log)
message("[BRODEUR] Done.")
