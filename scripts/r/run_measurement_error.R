# scripts/r/run_measurement_error.R
#
# Measurement-error sensitivity analysis for the primary model (Model 4).
#
# This script runs the measurement-error Stan model across a grid of fixed
# sigma_cases values. sigma_cases controls how much uncertainty is placed on
# the recorded case inventory counts for each language: sigma_cases = 0
# recovers the hard-assignment behavior of Model 4 exactly, while larger
# values spread probability mass to adjacent case levels.
#
# The grid used in the paper is:
#   sigma_cases in {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00}
#
# This script depends on model4_data, K_phy_arr, M, d_geo, and df being
# present in the environment. It is intended to be sourced after
# run_models.R, which defines all of these objects. run_all.R handles the
# sourcing order.
#
# Results are written to:
#   output/measurement_error/fit_sigma_<value>.rds   (one per grid point)
#   output/measurement_error/sensitivity_summary.csv  (combined table)
#
# The sensitivity_summary.csv is the input to the figures and tables
# reported in Section S3.

library(here)
library(posterior)
library(tidybayes)
library(dplyr)
library(readr)

# ---- Verify that dependencies from run_models.R are available ---------------
required_objects <- c("model4_data", "df")
missing <- required_objects[!sapply(required_objects, exists)]
if (length(missing) > 0) {
  stop(
    "The following objects are missing from the environment:\n  ",
    paste(missing, collapse = ", "),
    "\nThis script must be sourced after run_models.R."
  )
}

# ---- Output directory -------------------------------------------------------
me_output_dir <- here("output", "measurement_error")
if (!dir.exists(me_output_dir)) dir.create(me_output_dir, recursive = TRUE)

me_csv_dir <- here("output", "stan-csv", "measurement_error")
if (!dir.exists(me_csv_dir)) dir.create(me_csv_dir, recursive = TRUE)

# ---- Compile the measurement-error Stan model (once) ------------------------
me_stan_path <- here("scripts", "stan", "sensitivity",
                     "repo_model4_measurement_error.stan")
if (!file.exists(me_stan_path)) {
  stop("Stan file missing: ", me_stan_path)
}
message("Compiling measurement-error model: ", me_stan_path)
me_mod <- cmdstan_model(me_stan_path)
message("Compiled successfully.")

# ---- Grid of sigma_cases values ---------------------------------------------
sigma_grid <- c(0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
message(
  "Running measurement-error sensitivity grid over sigma_cases = ",
  paste(sigma_grid, collapse = ", ")
)

# ---- Helper: extract a tidy summary row from one fit ------------------------
summarize_fit <- function(fit, sigma_val) {
  vars <- c(
    "theta_def", "theta_indef",
    "beta_zero_def", "beta_zero_indef", "beta_def_indef",
    "tau", "rho", "ell_space", "ell_time", "sigma_eps"
  )
  draws <- as_draws_df(fit$draws(vars))
  
  # 89% HPDIs for the three key parameters
  hdi_theta_def      <- hdi(draws$theta_def,      .width = 0.89)
  hdi_theta_indef    <- hdi(draws$theta_indef,    .width = 0.89)
  hdi_beta_def_indef <- hdi(draws$beta_def_indef, .width = 0.89)
  
  tibble::tibble(
    sigma_cases = sigma_val,
    
    # Posterior medians
    theta_def_med        = median(draws$theta_def),
    theta_indef_med      = median(draws$theta_indef),
    beta0_def_med        = median(draws$beta_zero_def),
    beta0_indef_med      = median(draws$beta_zero_indef),
    beta_def_indef_med   = median(draws$beta_def_indef),
    tau_med              = median(draws$tau),
    rho_med              = median(draws$rho),
    ell_space_med        = median(draws$ell_space),
    ell_time_med         = median(draws$ell_time),
    sigma_eps_med        = median(draws$sigma_eps),
    
    # 95% equal-tailed credible intervals
    theta_def_lo_95      = quantile(draws$theta_def,      0.025),
    theta_def_hi_95      = quantile(draws$theta_def,      0.975),
    theta_indef_lo_95    = quantile(draws$theta_indef,    0.025),
    theta_indef_hi_95    = quantile(draws$theta_indef,    0.975),
    beta_def_indef_lo_95 = quantile(draws$beta_def_indef, 0.025),
    beta_def_indef_hi_95 = quantile(draws$beta_def_indef, 0.975),
    
    # 89% equal-tailed credible intervals
    theta_def_lo_89      = quantile(draws$theta_def,      0.055),
    theta_def_hi_89      = quantile(draws$theta_def,      0.945),
    theta_indef_lo_89    = quantile(draws$theta_indef,    0.055),
    theta_indef_hi_89    = quantile(draws$theta_indef,    0.945),
    beta_def_indef_lo_89 = quantile(draws$beta_def_indef, 0.055),
    beta_def_indef_hi_89 = quantile(draws$beta_def_indef, 0.945),
    
    # 89% HPDIs
    theta_def_hdi89_lo      = unname(hdi_theta_def[1]),
    theta_def_hdi89_hi      = unname(hdi_theta_def[2]),
    theta_indef_hdi89_lo    = unname(hdi_theta_indef[1]),
    theta_indef_hdi89_hi    = unname(hdi_theta_indef[2]),
    beta_def_indef_hdi89_lo = unname(hdi_beta_def_indef[1]),
    beta_def_indef_hdi89_hi = unname(hdi_beta_def_indef[2])
  )
}

# ---- Run the grid -----------------------------------------------------------
sens_rows <- vector("list", length(sigma_grid))

for (i in seq_along(sigma_grid)) {
  s <- sigma_grid[i]
  sigma_label <- sprintf("%.2f", s)
  message(sprintf("\n>>> sigma_cases = %s (%d / %d)", sigma_label, i, length(sigma_grid)))

  # Append sigma_cases to model4_data
  data_s <- model4_data
  data_s$sigma_cases <- s

  fit_s <- me_mod$sample(
    data             = data_s,
    seed             = 2025,
    chains           = 4,
    parallel_chains  = min(4, parallel::detectCores()),
    iter_warmup      = 4000,
    iter_sampling    = 4000,
    adapt_delta      = 0.95,
    max_treedepth    = 15,
    thin             = 1,
    output_dir       = me_csv_dir,
    save_warmup      = FALSE,
    refresh          = 250
  )

  # Save full fit object
  rds_path  <- file.path(me_output_dir, paste0("fit_sigma_", sigma_label, ".rds"))
  saveRDS(fit_s, file = rds_path)
  message("Saved: ", rds_path)

  # Save parameter summary CSV
  summ_path <- file.path(me_output_dir, paste0("summary_sigma_", sigma_label, ".csv"))
  write_csv(fit_s$summary() |> as.data.frame(), summ_path)

  # Collect tidy summary row
  sens_rows[[i]] <- summarize_fit(fit_s, s)

  rm(fit_s)
  gc()
}

# ---- Combine and save the sensitivity table ---------------------------------
sens_tbl <- bind_rows(sens_rows)

sens_csv_path <- file.path(me_output_dir, "sensitivity_summary.csv")
write_csv(sens_tbl, sens_csv_path)
message("\nMeasurement-error sensitivity analysis complete.")
message("Summary table written to: ", sens_csv_path)

print(sens_tbl[, c("sigma_cases", "theta_def_med", "theta_indef_med",
                   "beta_def_indef_med")])
