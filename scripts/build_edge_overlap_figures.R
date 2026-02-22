#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(ggplot2)
  library(fixest)
  library(broom)
  library(stringr)
  library(scales)
  library(grid)
})

setDTthreads(4)

dir.create("figures", recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), sprintf(...), "\n")
}

save_plot <- function(plot_obj, stem, width, height, dpi = 300) {
  out_jpg <- file.path("figures", paste0(stem, ".jpg"))
  out_pdf <- file.path("figures", paste0(stem, ".pdf"))
  ggsave(out_jpg, plot_obj, width = width, height = height, dpi = dpi)
  ggsave(out_pdf, plot_obj, width = width, height = height, dpi = dpi)
  log_msg("Wrote %s and %s", out_jpg, out_pdf)
}

find_input_file <- function(patterns) {
  for (pat in patterns) {
    hits <- Sys.glob(pat)
    if (length(hits) > 0) return(hits[[1]])
  }
  stop("Missing required input file. Tried patterns: ", paste(patterns, collapse = ", "))
}

load_edge_data <- function() {
  p <- find_input_file(c(
    "int_data/dat_*all_nine_iter_union_aggregated_meta.parquet",
    "int_data/claim_graph_all_nine_iter_union_aggregated_meta.parquet"
  ))
  dt <- as.data.table(read_parquet(p, as_data_frame = TRUE))
  stopifnot(all(c("paper_id", "year", "edge_overlap", "is_method_causal_inference") %in% names(dt)))
  dt[, year := as.numeric(year)]
  dt
}

load_paper_level_snippet <- function(thr) {
  p <- sprintf(
    "int_data/edge_overlap_runs_snippet_only/eo_ge%d/paper_level_data_eo_ge%d_snippet_only.parquet",
    thr, thr
  )
  stopifnot(file.exists(p))
  dt <- as.data.table(read_parquet(p, as_data_frame = TRUE))
  dt[, year := as.numeric(year)]
  dt
}

make_main_reg_edge_overlap_grid <- function() {
  log_msg("Building main_reg_edge_overlap_grid from snippet-only EO paper-level data")

  dv_list <- c("paper_is_top_5", "paper_is_top_6_20", "paper_is_top_21_100", "transformed_coalesced_cites")
  conceptual_vars <- c(
    "log_num_edges_causal", "log_num_edges_non_causal",
    "log_num_novel_edges_causal", "log_num_novel_edges_non_causal",
    "mean_eigen_centrality_RelCumLit_causal_std",
    "mean_eigen_centrality_RelCumLit_non_causal_std",
    "cause_effect_ratio_causal",
    "cause_effect_ratio_non_causal",
    "prop_edges_causal"
  )

  var_labels <- c(
    log_num_edges_causal = "Nbr. Claims (Causal)",
    log_num_edges_non_causal = "Nbr. Claims (Non-Causal)",
    log_num_novel_edges_causal = "Nbr. Novel Claims (Causal)",
    log_num_novel_edges_non_causal = "Nbr. Novel Claims (Non-Causal)",
    mean_eigen_centrality_RelCumLit_causal_std = "Topic Centrality (Causal)",
    mean_eigen_centrality_RelCumLit_non_causal_std = "Topic Centrality (Non-Causal)",
    cause_effect_ratio_causal = "Source-Sink Ratio (Causal)",
    cause_effect_ratio_non_causal = "Source-Sink Ratio (Non-Causal)",
    prop_edges_causal = "Share of Claims Causal"
  )

  family_of <- function(v) fcase(
    v == "prop_edges_causal", "Share Causal",
    grepl("log_num_edges", v), "N. Claims",
    grepl("log_num_novel_edges", v), "N. Novel Claims",
    grepl("mean_eigen_centrality", v), "Topic Centrality",
    grepl("cause_effect_ratio", v), "Source-Sink Ratio",
    default = NA_character_
  )

  run_regression_for_dv <- function(dv, data, measures) {
    out <- data.table()
    for (v in measures) {
      fit <- feols(as.formula(paste(dv, "~", v, "| year")), data = data)
      coef_dt <- as.data.table(tidy(fit))[term == v]
      if (nrow(coef_dt)) {
        out <- rbind(
          out,
          data.table(
            dv = dv,
            variable = v,
            estimate = coef_dt$estimate[1],
            std.error = coef_dt$std.error[1]
          ),
          fill = TRUE
        )
      }
    }
    out
  }

  results <- data.table()

  for (thr in 9:1) {
    dt <- load_paper_level_snippet(thr)[year < 2020]

    miss <- setdiff(c(dv_list, conceptual_vars, "year"), names(dt))
    if (length(miss)) stop(sprintf("Missing columns in EO>=%d snippet data: %s", thr, paste(miss, collapse = ", ")))

    dt[, (conceptual_vars) := lapply(.SD, function(x) fifelse(is.na(as.numeric(x)), 0, as.numeric(x))), .SDcols = conceptual_vars]

    for (dv in dv_list) {
      dat <- na.omit(copy(dt)[, c(dv, conceptual_vars, "year"), with = FALSE])
      if (!nrow(dat)) next
      reg_res <- run_regression_for_dv(dv, dat, conceptual_vars)
      if (nrow(reg_res)) {
        reg_res[, edge_overlap := thr]
        results <- rbind(results, reg_res, fill = TRUE)
      }
    }
  }

  results[, variable_label := var_labels[variable]]
  results[, is_causal := fifelse(
    variable == "prop_edges_causal" | (str_detect(variable, "_causal") & !str_detect(variable, "_non_causal")),
    "Causal", "Non-Causal"
  )]
  results[, variable_family := factor(
    family_of(variable),
    levels = c("Share Causal", "N. Claims", "N. Novel Claims", "Topic Centrality", "Source-Sink Ratio")
  )]
  results[, dv := factor(
    fcase(
      dv == "paper_is_top_5", "In Top 5",
      dv == "paper_is_top_6_20", "In Top 6-20",
      dv == "paper_is_top_21_100", "In Top 21-100",
      dv == "transformed_coalesced_cites", "Citations"
    ),
    levels = c("In Top 5", "In Top 6-20", "In Top 21-100", "Citations")
  )]

  setorderv(results, c("variable", "is_causal", "dv", "edge_overlap"))

  cols <- c("Causal" = "#084594", "Non-Causal" = "darkorange")

  p <- ggplot(
    results,
    aes(
      x = estimate, y = edge_overlap,
      xmin = estimate - 1.96 * std.error,
      xmax = estimate + 1.96 * std.error,
      colour = is_causal,
      group = interaction(variable, is_causal)
    )
  ) +
    geom_errorbarh(height = 0) +
    geom_path(linewidth = 0.8) +
    geom_point(size = 1.7) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    geom_hline(yintercept = 4, linetype = "dashed", colour = "grey20", alpha = 0.3) +
    scale_y_continuous(breaks = 1:9, labels = paste(">= ", 1:9, sep = "")) +
    scale_colour_manual(values = cols, guide = "none") +
    facet_grid(variable_family ~ dv, scales = "free_x") +
    labs(x = "Coefficient estimate (year-FE spec)", y = "Edge-overlap filter (# iterations)") +
    theme_classic(base_size = 20) +
    theme(strip.text.x = element_text(size = 20), strip.text.y = element_text(size = 20))

  save_plot(p, "main_reg_edge_overlap_grid", width = 12, height = 12)
}

make_prop_causal_ts_and_field <- function(edge_data) {
  log_msg("Building prop_causal_edges_ts_overlay and prop_causal_edges_by_field from aggregated edge data")

  edge_data <- copy(edge_data)
  field_vars <- c(
    "is_finance", "is_development", "is_labour", "is_public_economics",
    "is_urban_economics", "is_macroeconomics", "is_behavioral_economics",
    "is_economic_history", "is_econometric_theory", "is_industrial_organization",
    "is_environmental_economics", "is_health_economics", "is_political_economy"
  )
  miss_fields <- setdiff(field_vars, names(edge_data))
  if (length(miss_fields)) stop(sprintf("Missing field columns: %s", paste(miss_fields, collapse = ", ")))

  field_labels <- c(
    is_finance = "Finance", is_development = "Development", is_labour = "Labour",
    is_public_economics = "Public", is_urban_economics = "Urban", is_macroeconomics = "Macro",
    is_behavioral_economics = "Behavioral", is_economic_history = "Econ. History",
    is_econometric_theory = "Econometrics", is_industrial_organization = "IO",
    is_environmental_economics = "Environmental", is_health_economics = "Health",
    is_political_economy = "Pol. Econ."
  )

  prop_causal_per_paper <- function(dt, thr) {
    dt_thr <- dt[edge_overlap >= thr]
    if (!nrow(dt_thr)) return(data.table())
    dt_thr[, .(num_edges = .N, num_causal_edges = sum(is_method_causal_inference == 1L, na.rm = TRUE)), by = .(paper_id, year)][
      num_edges > 0,
      .(paper_id, year, prop_causal = num_causal_edges / num_edges)
    ]
  }

  ts_store <- rbindlist(lapply(9:1, function(thr) {
    pc <- prop_causal_per_paper(edge_data, thr)
    if (!nrow(pc)) return(NULL)
    pc[, .(mean_prop = mean(prop_causal), se_prop = sd(prop_causal) / sqrt(.N)), by = year][
      , `:=`(lower = mean_prop - 1.96 * se_prop, upper = mean_prop + 1.96 * se_prop, threshold = thr)
    ]
  }), fill = TRUE)

  grey_pal <- gray.colors(9, start = 0.86, end = 0.20)
  names(grey_pal) <- as.character(1:9)
  grey_pal["4"] <- "#2166ac"

  p_ts <- ggplot(ts_store, aes(year, mean_prop, group = factor(threshold), colour = factor(threshold))) +
    geom_line(linewidth = 0.7) +
    geom_ribbon(
      data = ts_store[threshold == 4],
      aes(x = year, ymin = lower, ymax = upper),
      inherit.aes = FALSE,
      fill = alpha("#2166ac", 0.12)
    ) +
    geom_line(data = ts_store[threshold == 4], colour = "#2166ac", linewidth = 1.2, alpha = 0.9) +
    geom_point(data = ts_store[threshold == 4], colour = "#2166ac", size = 1.5, alpha = 0.9) +
    scale_colour_manual(values = grey_pal, guide = "none") +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(y = "Avg. proportion of causal edges", x = "Year") +
    theme_classic(base_size = 22)

  save_plot(p_ts, "prop_causal_edges_ts_overlay", width = 10.8, height = 6.8)

  field_store <- rbindlist(lapply(9:1, function(thr) {
    pc <- prop_causal_per_paper(edge_data, thr)
    if (!nrow(pc)) return(NULL)
    fld <- unique(edge_data[, c("paper_id", field_vars), with = FALSE])
    pc <- merge(pc, fld, by = "paper_id", all.x = TRUE)
    pc[, period := fifelse(year < 2000, "Pre", "Post")]
    pc_long <- melt(pc, id.vars = c("paper_id", "period", "prop_causal"), measure.vars = field_vars, variable.name = "field", value.name = "is_field")
    pc_long <- pc_long[is_field == TRUE]
    pc_long[, .(mean_prop = mean(prop_causal), se_prop = sd(prop_causal) / sqrt(.N)), by = .(field, period)][
      , `:=`(threshold = thr, field = field_labels[field])
    ]
  }), fill = TRUE)

  fld_wide <- dcast(field_store, field + threshold ~ period, value.var = "mean_prop")
  fld_wide[, field := factor(field, levels = field_labels)]
  fld_wide[, threshold_f := factor(threshold, levels = 9:1)]

  p_field <- ggplot(fld_wide, aes(y = threshold_f)) +
    geom_segment(aes(x = Pre, xend = Post, yend = threshold_f), colour = "grey40", arrow = arrow(length = unit(0.25, "cm"), type = "closed")) +
    geom_point(aes(x = Pre), size = 3, colour = "royalblue") +
    geom_point(aes(x = Post), size = 3, colour = "orange") +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    facet_wrap(~ field, ncol = 3, scales = "free_x") +
    labs(x = "Average proportion of causal edges", y = "Edge-overlap filter", colour = NULL) +
    theme_classic(base_size = 20) +
    theme(strip.text = element_text(size = 16))

  save_plot(p_field, "prop_causal_edges_by_field", width = 13.2, height = 14.4)
}

make_method_eo <- function(edge_data) {
  log_msg("Building proportion_papers_by_method_edge_overlap from aggregated edge data")

  method_values <- c(
    "rdd", "did", "rct", "iv", "structural estimation",
    "event study", "simulation", "theoretical/non-empirical",
    "fixed effects models"
  )
  method_labels <- c(
    rdd = "RDD",
    did = "DID",
    rct = "RCT",
    iv = "IV",
    `structural estimation` = "Structural Estimation",
    `event study` = "Event Study",
    simulation = "Simulation",
    `theoretical/non-empirical` = "Theoretical",
    `fixed effects models` = "Fixed Effects"
  )
  rx_methods <- paste0("(", paste(gsub("([/])", "\\\\\\1", method_values), collapse = "|"), ")")

  grey_pal <- gray.colors(9, start = 0.9, end = 0.15)
  names(grey_pal) <- as.character(1:9)
  grey_pal["4"] <- "#2166ac"

  results <- data.table()

  for (thr in 9:1) {
    dt_thr <- edge_data[edge_overlap >= thr]
    if (!nrow(dt_thr)) next

    paper_methods <- dt_thr[
      ,
      .(methods = tolower(paste(unique(causal_inference_method), collapse = ", "))),
      by = .(paper_id, year)
    ]
    year_totals <- paper_methods[, .(total_papers = uniqueN(paper_id)), by = year]

    paper_method_long <- paper_methods[, .(Method = str_extract_all(methods, rx_methods)), by = .(paper_id, year)][
      , .(Method = unlist(Method)), by = .(paper_id, year)
    ]
    by_year_method <- paper_method_long[, .(method_papers = uniqueN(paper_id)), by = .(year, Method)]
    by_year_method <- by_year_method[Method %in% method_values]
    by_year_method <- merge(by_year_method, year_totals, by = "year")
    by_year_method[, `:=`(proportion = (method_papers / total_papers) * 100, threshold = thr)]

    results <- rbind(results, by_year_method, fill = TRUE)
  }

  results[, MethodLabel := factor(method_labels[Method], levels = unname(method_labels))]
  results[, threshold_f := factor(threshold, levels = 1:9)]

  p <- ggplot(results, aes(year, proportion, group = threshold_f, colour = threshold_f)) +
    geom_smooth(method = "loess", span = 0.75, se = FALSE, linewidth = 1) +
    scale_colour_manual(values = grey_pal, guide = "none") +
    facet_wrap(~ MethodLabel, nrow = 5, ncol = 3, scales = "free_y") +
    labs(y = "Proportion of papers (%)", x = "Year") +
    scale_y_continuous(limits = c(0, NA)) +
    theme_classic(base_size = 32) +
    theme(
      strip.text = element_text(size = 19),
      axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)
    )

  save_plot(p, "proportion_papers_by_method_edge_overlap", width = 15, height = 10)
}

main <- function() {
  edge_data <- load_edge_data()
  make_main_reg_edge_overlap_grid()
  make_prop_causal_ts_and_field(edge_data)
  make_method_eo(edge_data)
  log_msg("Edge-overlap figure build complete")
}

main()
