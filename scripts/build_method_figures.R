#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(ggplot2)
  library(stringr)
})

setDTthreads(4)

find_input_file <- function(patterns) {
  for (pat in patterns) {
    hits <- Sys.glob(pat)
    if (length(hits) > 0) return(hits[[1]])
  }
  stop("Missing required input file. Tried patterns: ", paste(patterns, collapse = ", "))
}

method_patterns <- c(
  "did" = "DiD",
  "iv" = "IV",
  "rct" = "RCT",
  "rdd" = "RDD",
  "event study" = "Event Study",
  "fixed effects models" = "TWFE",
  "structural estimation" = "Structural",
  "simulation" = "Simulation",
  "theoretical/non-empirical" = "Theoretical"
)

field_cols <- c(
  "is_finance", "is_development", "is_labour",
  "is_public_economics", "is_urban_economics",
  "is_macroeconomics", "is_behavioral_economics",
  "is_economic_history", "is_econometric_theory",
  "is_industrial_organization", "is_environmental_economics",
  "is_health_economics", "is_political_economy"
)

field_labels <- c(
  is_finance = "Finance",
  is_development = "Development",
  is_labour = "Labour",
  is_public_economics = "Public",
  is_urban_economics = "Urban",
  is_macroeconomics = "Macro",
  is_behavioral_economics = "Behavioral",
  is_economic_history = "Econ. History",
  is_econometric_theory = "Econometrics",
  is_industrial_organization = "IO",
  is_environmental_economics = "Environmental",
  is_health_economics = "Health",
  is_political_economy = "Pol. Econ."
)

save_dual <- function(p, stem, width, height, dpi = 320) {
  ggsave(file.path("figures", paste0(stem, ".jpg")), p, width = width, height = height, dpi = dpi)
  ggsave(file.path("figures", paste0(stem, ".pdf")), p, width = width, height = height, dpi = dpi)
}

build_paper_level <- function(dt) {
  dt[, causal_inference_method := as.character(causal_inference_method)]
  dt <- dt[!is.na(causal_inference_method)]
  out <- dt[, c(
    list(
      methods = tolower(paste(unique(causal_inference_method), collapse = ", "))
    ),
    lapply(.SD, function(x) as.integer(any(as.logical(x))))
  ), by = .(paper_id, year), .SDcols = field_cols]
  out
}

add_method_flags <- function(dt) {
  for (pat in names(method_patterns)) {
    nm <- method_patterns[[pat]]
    dt[, (paste0("m_", nm)) := str_detect(methods, fixed(pat, ignore_case = TRUE))]
  }
  dt
}

build_method_time_plot <- function(paper_dt, out_stem) {
  m_cols <- paste0("m_", unname(method_patterns))
  long <- melt(
    paper_dt,
    id.vars = "year",
    measure.vars = m_cols,
    variable.name = "Method",
    value.name = "is_method"
  )
  long[, Method := gsub("^m_", "", Method)]
  agg <- long[, .(
    total = .N,
    method_n = sum(is_method, na.rm = TRUE),
    share = 100 * sum(is_method, na.rm = TRUE) / .N
  ), by = .(year, Method)]
  agg[, Method := factor(Method, levels = unname(method_patterns))]

  p <- ggplot(agg, aes(x = year, y = share)) +
    geom_smooth(method = "loess", span = 0.70, se = FALSE, linewidth = 0.9, color = "#1f1f1f") +
    geom_point(size = 0.9, color = "#1f1f1f", alpha = 0.60) +
    facet_wrap(~ Method, nrow = 3, ncol = 3, scales = "free_y") +
    labs(x = "Year", y = "Proportion of papers (%)") +
    scale_y_continuous(limits = c(0, NA)) +
    theme_classic(base_size = 26)

  save_dual(p, out_stem, width = 14.2, height = 9.4)
}

build_field_method_plot <- function(paper_dt, out_stem) {
  m_cols <- paste0("m_", unname(method_patterns))
  long <- melt(
    paper_dt,
    id.vars = c("paper_id", "year", "methods", m_cols),
    measure.vars = field_cols,
    variable.name = "Field",
    value.name = "is_field"
  )[is_field == 1]

  stat_rows <- rbindlist(lapply(m_cols, function(mcol) {
    long[, .(
      mean_method = mean(get(mcol), na.rm = TRUE) * 100,
      se_method = sd(get(mcol), na.rm = TRUE) / sqrt(.N) * 100
    ), by = Field][, Method := gsub("^m_", "", mcol)]
  }))

  stat_rows[, `:=`(
    lower_ci = pmax(0, mean_method - 1.96 * se_method),
    upper_ci = pmin(100, mean_method + 1.96 * se_method),
    Field = field_labels[Field],
    Method = factor(Method, levels = unname(method_patterns))
  )]
  stat_rows <- stat_rows[Field != "Econometrics"]

  # Order fields by theoretical share so theory-heavy fields appear at the bottom.
  theory_order <- stat_rows[Method == "Theoretical", .(theory_share = mean_method), by = Field][
    order(-theory_share)
  ]$Field
  stat_rows[, Field := factor(Field, levels = theory_order)]

  pal <- c("#0c4a8a", "#2878b5", "#4aa8d8", "#2c7a3f", "#4aa564", "#e69500", "#d17d00", "#7a5c2e", "#6b6b6b")
  names(pal) <- unname(method_patterns)
  shape_vals <- c(16, 17, 15, 18, 3, 7, 8, 0, 4)
  names(shape_vals) <- unname(method_patterns)

  p <- ggplot(stat_rows, aes(x = Field, y = mean_method, color = Method, shape = Method)) +
    geom_point(position = position_dodge(width = 0.60), size = 3.2) +
    geom_errorbar(
      aes(ymin = lower_ci, ymax = upper_ci),
      width = 0.18,
      position = position_dodge(width = 0.60),
      linewidth = 0.7
    ) +
    coord_flip() +
    labs(x = NULL, y = "Proportion of papers using method (%)") +
    scale_color_manual(values = pal) +
    scale_shape_manual(values = shape_vals) +
    theme_classic(base_size = 24) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank()
    )

  save_dual(p, out_stem, width = 12.8, height = 10.2)
}

run_set <- function(path, eo_threshold, stem_time, stem_field) {
  stopifnot(file.exists(path))
  dt <- as.data.table(read_parquet(path))
  dt[, year := as.numeric(year)]
  dt <- dt[year >= 1980]
  if ("edge_overlap" %in% names(dt)) {
    dt <- dt[edge_overlap >= eo_threshold]
  }

  req <- c("paper_id", "year", "causal_inference_method", field_cols)
  miss <- setdiff(req, names(dt))
  if (length(miss)) stop(sprintf("Missing required columns in %s: %s", path, paste(miss, collapse = ", ")))

  paper <- build_paper_level(dt)
  paper <- add_method_flags(paper)
  build_method_time_plot(paper, stem_time)
  build_field_method_plot(paper, stem_field)
}

main <- function() {
  main_edges_path <- find_input_file(c(
    "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
    "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet"
  ))
  iter1_edges_path <- find_input_file(c(
    "int_data/dat_*s2_3iters_from_s1_iter1.parquet",
    "int_data/claim_graph_s2_3iters_from_s1_iter1.parquet"
  ))

  # Baseline figures in the main text: EO >= 4 from 9-run aggregated data.
  run_set(
    path = main_edges_path,
    eo_threshold = 4,
    stem_time = "proportion_papers_by_method_facet_plot",
    stem_field = "method_usage_by_field_with_CI_and_mixed_methods"
  )

  # Single-iteration sensitivity (Stage-1 iteration 1 branch, 3 Stage-2 runs aggregated).
  run_set(
    path = iter1_edges_path,
    eo_threshold = 2,
    stem_time = "proportion_papers_by_method_facet_plot_iter1",
    stem_field = "method_usage_by_field_with_CI_and_mixed_methods_iter1"
  )
}

main()
