# scripts/r/run_models.R

library(here)
library(tidyverse)
library(ape)
library(cmdstanr)

options(mc.cores = parallel::detectCores())

message("Using CmdStan: ", cmdstanr::cmdstan_version())
message("here() -> ", here())

# ensure output directory exists
if (!dir.exists(here("output"))) dir.create(here("output"), recursive = TRUE)

# create CSV output directory for CmdStan 
stan_csv_dir <- here("output", "stan-csv")
if (!dir.exists(stan_csv_dir)) dir.create(stan_csv_dir, recursive = TRUE)

# ---- Load data ----
df_path <- here("data", "repo_data.csv")
if (!file.exists(df_path)) stop("Data file missing: ", df_path)
df <- read.csv(df_path, stringsAsFactors = FALSE)
N <- nrow(df)
message("Loaded df: ", df_path)

# geographic distance matrix
d_geo_path <- here("data", "geo_distances_aligned.rds")
if (!file.exists(d_geo_path)) stop("Geo distance file missing: ", d_geo_path)
d_geo <- readRDS(d_geo_path)
message("Loaded geo distances: ", d_geo_path)

# ---- Load phylogenetic trees ----
ieo50_path <- here("data", "ieo_fifty.nex")
if (!file.exists(ieo50_path)) stop("Tree file missing: ", ieo50_path)
ieo50 <- read.nexus(ieo50_path)
message("Loaded phylogenetic trees: ", ieo50_path)

# Rename tip labels to match df
if (!"lang" %in% names(df)) {
  stop("df must contain a 'lang' column; check repo_data.csv.")
}
ieo50 <- lapply(ieo50, function(tr) {
  if (length(tr$tip.label) != N) {
    stop("Tree has ", length(tr$tip.label), " tips, expected ", N)
  }
  tr$tip.label <- df$lang
  tr
})

# ---- Build VCV matrices (symmetrize + scale) ----
Ks <- lapply(ieo50, function(tr) {
  K <- vcv(tr)
  K <- 0.5 * (K + t(K))
  K <- K / mean(diag(K))
  eig_min <- min(eigen(K, symmetric = TRUE, only.values = TRUE)$values)
  if (eig_min < 1e-10) K <- K + diag(1e-8, nrow(K))
  K
})

M <- length(Ks)
if (M < 1) stop("No phylogenetic matrices produced (M < 1).")
K_phy_arr <- array(NA_real_, dim = c(M, N, N))
for (m in seq_len(M)) K_phy_arr[m, , ] <- Ks[[m]]
message("Constructed K_phy array with dims: ", paste(dim(K_phy_arr), collapse = " x "))

# ---- Utility: compile Stan model via cmdstanr ----
compile_model <- function(stan_path) {
  if (!file.exists(stan_path)) stop("Stan file missing: ", stan_path)
  message("Compiling: ", stan_path)
  mod <- cmdstan_model(stan_path)
  message("Compiled: ", stan_path)
  mod
}

# ---- Shared sampler settings ----
SEED          <- 123L
ITER_WARMUP   <- 4000L
ITER_SAMPLING <- 4000L
THIN          <- 1L     # No thinning; 16,000 post-warmup draws per fit
ADAPT_DELTA   <- 0.95   # Higher than the default 0.80; reduces divergences in hierarchical models
MAX_TREEDEPTH <- 15L    # Higher than the default 10; accommodates deep trees sometimes required by GP and BM components
REFRESH       <- 250L

# =============================================================================
# PARTIAL MEDIATION MODELS (models 1-4)
# =============================================================================

# ---- Model 1: no GP, no phylogeny ----
model1_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def),
  indef   = as.integer(df$indef) 
)

sm1_path <- here("scripts", "stan", "partial_mediation", "repo_model1_no_gp_no_phylo.stan")
sm1_mod <- compile_model(sm1_path)

fit1 <- sm1_mod$sample(
  data            = model1_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit1, file = here("output", "fit1.rds"))
message("Model 1 complete; saved fit to output/fit1.rds")

# ---- Model 2: phylogeny only ----
model2_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  M       = M,
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def),
  indef   = as.integer(df$indef),
  K_phy   = K_phy_arr,
  log_w   = rep(-log(M), M)
)

sm2_path <- here("scripts", "stan", "partial_mediation", "repo_model2_phylo.stan")
sm2_mod <- compile_model(sm2_path)

fit2 <- sm2_mod$sample(
  data            = model2_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit2, file = here("output", "fit2.rds"))
message("Model 2 complete; saved fit to output/fit2.rds")

# ---- Model 3: spatiotemporal GP only ----
model3_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def),
  indef   = as.integer(df$indef),
  Dgeo    = d_geo,
  t_lo    = df$t_lo,
  t_hi    = df$t_hi,
  jitter  = 1e-8
)

sm3_path <- here("scripts", "stan", "partial_mediation", "repo_model3_stgp.stan")
sm3_mod <- compile_model(sm3_path)

fit3 <- sm3_mod$sample(
  data            = model3_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit3, file = here("output", "fit3.rds"))
message("Model 3 complete; saved fit to output/fit3.rds")

# ---- Model 4: phylogeny + spatiotemporal GP (primary model) ----
model4_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  M       = M,
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def),
  indef   = as.integer(df$indef),
  K_phy   = K_phy_arr,
  Dgeo    = d_geo,
  log_w   = rep(-log(M), M),
  t_lo    = df$t_lo,
  t_hi    = df$t_hi,
  jitter  = 1e-8
)

sm4_path <- here("scripts", "stan", "partial_mediation", "repo_model4_phylo_stgp.stan")
sm4_mod <- compile_model(sm4_path)

fit4 <- sm4_mod$sample(
  data            = model4_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit4, file = here("output", "fit4.rds"))
message("Model 4 complete; saved fit to output/fit4.rds")

# =============================================================================
# COMPLETE MEDIATION MODELS (models 5-8)
# Each complete mediation model uses the same data as its partial mediation
# counterpart. Only the Stan file differs.
# =============================================================================

# ---- Model 5: no GP, no phylogeny (complete mediation) ----
sm5_path <- here("scripts", "stan", "complete_mediation", "repo_model1_complete_mediation.stan")
sm5_mod <- compile_model(sm5_path)

fit5 <- sm5_mod$sample(
  data            = model1_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit5, file = here("output", "fit5.rds"))
message("Model 5 complete; saved fit to output/fit5.rds")

# ---- Model 6: phylogeny only (complete mediation) ----
sm6_path <- here("scripts", "stan", "complete_mediation", "repo_model2_complete_mediation.stan")
sm6_mod <- compile_model(sm6_path)

# E-BFMI < 0.3 observed across all chains due to the centered phylogenetic
# parameterization under complete mediation. Extended warmup did not resolve
# the issue; standard warmup is retained and the warning is disclosed in
# Section S2.5.
fit6 <- sm6_mod$sample(
  data            = model2_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,  
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit6, file = here("output", "fit6.rds"))
message("Model 6 complete; saved fit to output/fit6.rds")

# ---- Model 7: spatiotemporal GP only (complete mediation) ----
sm7_path <- here("scripts", "stan", "complete_mediation", "repo_model3_complete_mediation.stan")
sm7_mod <- compile_model(sm7_path)

fit7 <- sm7_mod$sample(
  data            = model3_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit7, file = here("output", "fit7.rds"))
message("Model 7 complete; saved fit to output/fit7.rds")

# ---- Model 8: phylogeny + spatiotemporal GP (complete mediation) ----
sm8_path <- here("scripts", "stan", "complete_mediation", "repo_model4_complete_mediation.stan")
sm8_mod <- compile_model(sm8_path)

fit8 <- sm8_mod$sample(
  data            = model4_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit8, file = here("output", "fit8.rds"))
message("Model 8 complete; saved fit to output/fit8.rds")

# =============================================================================
# FLEXIBLE RANDOM EFFECTS MODELS (models 9-10)
# Both are variants of model 4 (phylogeny + spatiotemporal GP, partial
# mediation) that relax the shared-field assumption. Both use model4_data
# unchanged.
# =============================================================================

# ---- Model 9: separate outcome-specific scaling ----
# Relaxes the constraint that the phylogenetic and spatiotemporal random
# effects enter both article equations with identical magnitude by assigning
# independent scale parameters to each outcome.
sm9_path <- here("scripts", "stan", "flexible_random_effects", "repo_model4_sep_scaling.stan")
sm9_mod <- compile_model(sm9_path)

fit9 <- sm9_mod$sample(
  data            = model4_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit9, file = here("output", "fit9.rds"))
message("Model 9 complete; saved fit to output/fit9.rds")

# ---- Model 10: full bivariate random effects ----
# Further relaxes the shared-field assumption by introducing independent
# cross-outcome correlation parameters (rho_phy, rho_st) for the
# phylogenetic and spatiotemporal components, respectively.
sm10_path <- here("scripts", "stan", "flexible_random_effects", "repo_model4_bivariate.stan")
sm10_mod <- compile_model(sm10_path)

fit10 <- sm10_mod$sample(
  data            = model4_data,
  seed            = SEED,
  chains          = 4,
  parallel_chains = min(4L, parallel::detectCores()),
  iter_warmup     = ITER_WARMUP,
  iter_sampling   = ITER_SAMPLING,
  thin            = THIN,
  adapt_delta     = ADAPT_DELTA,
  max_treedepth   = MAX_TREEDEPTH,
  refresh         = REFRESH,
  output_dir      = stan_csv_dir
)
saveRDS(fit10, file = here("output", "fit10.rds"))
message("Model 10 complete; saved fit to output/fit10.rds")

message("All models finished.")