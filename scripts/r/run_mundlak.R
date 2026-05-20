# scripts/r/run_mundlak.R
#
# Fits the Mundlak-type clade-mean control model (model 11).
#
# PURPOSE:
# This script implements the Mundlak (1978) approach to separate
# within-clade from between-clade variation in case inventories. 
#
# DEPENDENCIES:
# This script must be run after scripts/r/run_models.R (or via run_all.R),
# because it reuses the following objects that run_models.R defines in the
# global environment:
#   - df            : language-level data frame from repo_data.csv
#   - model4_data   : Stan data list for the primary model (model 4)
#   - compile_model : helper function to compile Stan models
#   - stan_csv_dir  : path to the CmdStan CSV output directory
#
# INPUTS:
#   data/repo_data.csv       -- language-level outcomes and predictors
#   data/mundlak_data.csv    -- language-to-clade assignments (94 IE languages,
#                               12 clades)
#
# OUTPUTS:
#   output/fit11.rds         -- saved CmdStanMCMC object
#   output/stan-csv/         -- raw CmdStan CSV chains (chain-level draws)
#
# SAMPLING SETTINGS:
# Identical to the primary models: seed = 123, 4 chains, 4000 warmup +
# 4000 post-warmup iterations, no thinning, adapt_delta = 0.95,
# max_treedepth = 15.

library(here)
library(tidyverse)
library(cmdstanr)

# ---- Verify that required objects from run_models.R are in scope ------------
required_objects <- c("df", "model4_data", "compile_model", "stan_csv_dir")
missing_objects  <- required_objects[!sapply(required_objects, exists)]
if (length(missing_objects) > 0) {
  stop(
    "The following objects required from run_models.R are not in scope: ",
    paste(missing_objects, collapse = ", "), ".\n",
    "Source scripts/r/run_models.R first, or run the full pipeline ",
    "via run_all.R."
  )
}

# ---- Load Mundlak data -------------------------------------------------------
# mundlak_data.csv provides the clade assignment for each of the N languages.
mundlak_path <- here("data", "mundlak_data.csv")
if (!file.exists(mundlak_path)) {
  stop("Mundlak data file missing: ", mundlak_path)
}
mundlak_df <- read.csv(mundlak_path, stringsAsFactors = FALSE)
message("Loaded Mundlak data: ", mundlak_path,
        " (", nrow(mundlak_df), " rows, ",
        length(unique(mundlak_df$clade)), " clades)")

# ---- Identify language-identifier column in df ------------------------------
lang_col_df <- if ("lang"     %in% names(df)) "lang"     else
  if ("language" %in% names(df)) "language" else
    stop("df has no 'lang' or 'language' column; ",
         "check repo_data.csv.")

# ---- Merge clade assignments -------------------------------------------------
# Join mundlak_df onto df by language name.
# The merge uses all.x = TRUE so that any language in df but absent from
# mundlak_data.csv raises a detectable NA rather than silently dropping rows.
df_aug <- merge(
  df,
  mundlak_df[, c("language", "clade")],
  by.x   = lang_col_df,
  by.y   = "language",
  all.x  = TRUE,
  sort   = FALSE
)

# Restore the original row order (merge() does not guarantee it).
df_aug <- df_aug[match(df[[lang_col_df]], df_aug[[lang_col_df]]), ]

# Verify that every language received a clade assignment.
n_missing <- sum(is.na(df_aug$clade))
if (n_missing > 0) {
  missing_langs <- df_aug[[lang_col_df]][is.na(df_aug$clade)]
  stop(
    n_missing, " language(s) in repo_data.csv have no clade assignment ",
    "in mundlak_data.csv:\n  ",
    paste(missing_langs, collapse = "\n  "), "\n",
    "Add them to data/mundlak_data.csv and re-run."
  )
}
message("Clade assignments merged successfully; no missing values.")

# ---- Compute clade-mean case inventory (centered at grand mean) --------------
# clade_mean_cases_c[n] = mean(n_case within clade[n]) - grand_mean(n_case)
#
# Centering at the grand mean ensures:
#   (a) the intercepts remain interpretable at an average-clade language 
#   (b) beta_clade_mean_* is on the same logit scale as other fixed effects.
n_cases_vec <- as.integer(df_aug$n_case)
clade_vec   <- as.integer(df_aug$clade)
grand_mean  <- mean(n_cases_vec)

clade_means_raw    <- tapply(n_cases_vec, clade_vec, mean)
clade_mean_cases_c <- as.numeric(
  clade_means_raw[as.character(clade_vec)] - grand_mean
)

message(sprintf(
  paste0("Case-inventory descriptives:\n",
         "  Grand mean:          %.3f\n",
         "  Number of clades:    %d\n",
         "  Clade means (raw):   [%.2f, %.2f]\n",
         "  Centered clade means:[%.3f, %.3f]"),
  grand_mean,
  length(unique(clade_vec)),
  min(clade_means_raw), max(clade_means_raw),
  min(clade_mean_cases_c), max(clade_mean_cases_c)
))

# ---- Build Stan data ---------------------------------------------------------
# Extend model4_data with the Mundlak predictor.
# All other fields (K_phy array, Dgeo, t_lo/t_hi, jitter, log_w) are
# taken directly from model4_data and are therefore guaranteed to be aligned
# with the same N languages in the same order.
mundlak_model_data <- c(
  model4_data,
  list(clade_mean_cases_c = clade_mean_cases_c)
)

# Sanity check: N from model4_data must equal the number of clade values.
stopifnot(
  "Length of clade_mean_cases_c must equal N in model4_data" =
    length(clade_mean_cases_c) == mundlak_model_data$N
)

# ---- Compile model -----------------------------------------------------------
sm11_path <- here("scripts", "stan", "mundlak", "repo_model11_mundlak.stan")
if (!file.exists(sm11_path)) stop("Stan file missing: ", sm11_path)
sm11_mod  <- compile_model(sm11_path)

# ---- Fit model ---------------------------------------------------------------
message("Fitting model 11 (Mundlak clade-mean control) ...")
fit11 <- sm11_mod$sample(
  data            = mundlak_model_data,
  seed            = 123L,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = 4000L,
  iter_sampling   = 4000L,
  thin            = 1L,
  adapt_delta     = 0.95,
  max_treedepth   = 15L,
  refresh         = 250L,
  output_dir      = stan_csv_dir
)

# ---- Save fit ----------------------------------------------------------------
saveRDS(fit11, file = here("output", "fit11.rds"))
message("Model 11 complete; saved fit to output/fit11.rds")

# ---- Brief convergence summary -----------------------------------------------
# Print R-hat and ESS for the primary parameters of interest.
focal_params <- c(
  "theta_def", "theta_indef",
  "beta_clade_mean_def", "beta_clade_mean_indef",
  "beta_def_indef", "tau", "rho"
)
diag_summary <- fit11$summary(
  variables = focal_params,
  .cores    = 1L
)
message("\nConvergence diagnostics (focal parameters):")
print(diag_summary, digits = 3)