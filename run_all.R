# run_all.R
#
# Master pipeline for all analyses reported in the manuscript.
# Run this script from the repository root to reproduce all results:
#
#   Rscript run_all.R
#
# The script sources three analysis scripts in order:
#
#   1. scripts/r/run_models.R
#      Fits all ten primary models (partial mediation, complete mediation,
#      and flexible random effects variants) corresponding to Table S4. 
#      Saves fits to output/. Defines model4_data, compile_model, 
#      stan_csv_dir, df, and related objects in the
#      environment, which are reused by the scripts below.
#
#   2. scripts/r/run_measurement_error.R
#      Fits the measurement-error sensitivity model across a grid of
#      sigma_cases values (0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00).
#      Depends on model4_data defined by run_models.R above.
#      Saves per-grid fits and a combined summary CSV to
#      output/measurement_error/.
#
#   3. scripts/r/run_mundlak.R
#      Fits the Mundlak-type clade-mean control model (model 11).
#      Extends model 4 by adding clade-mean case inventory (centered at 
#      the grand mean) as a predictor to separate within-clade from 
#      between-clade variation in case inventories. Depends on model4_data,
#      df, compile_model, and stan_csv_dir defined by run_models.R above,
#      and on data/mundlak_data.csv. Saves fit to output/fit11.rds.

if (file.exists("renv.lock") && !requireNamespace("renv", quietly = TRUE)) {
  stop(
    "This project uses renv for package management but the 'renv' package ",
    "is not installed.\nInstall it with: install.packages('renv')"
  )
}

# Declare project root (ensures here() resolves correctly regardless of
# the working directory from which this script is called)
here::i_am("run_all.R")

# Check CmdStan installation
if (!isTRUE(
  tryCatch(!is.null(cmdstanr::cmdstan_version()), error = function(e) FALSE)
)) {
  stop(
    "CmdStan is not installed.\n",
    "Install it with:\n",
    "  cmdstanr::install_cmdstan(cores = parallel::detectCores())"
  )
}

# ---- Step 1: primary models -------------------------------------------------
message("=== Step 1: fitting primary models (1-10) ===")
source(here::here("scripts", "r", "run_models.R"))

# ---- Step 2: measurement-error sensitivity analysis -------------------------
message("\n=== Step 2: measurement-error sensitivity analysis ===")
source(here::here("scripts", "r", "run_measurement_error.R"))

# ---- Step 3: Mundlak clade-mean control model (model 11) --------------------
message("\n=== Step 3: Mundlak clade-mean control model ===")
source(here::here("scripts", "r", "run_mundlak.R"))

message("\n=== All analyses complete ===")
