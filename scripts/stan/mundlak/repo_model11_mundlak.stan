// repo_model11_mundlak.stan
//
// MUNDLAK-TYPE CLADE-MEAN CONTROL MODEL
//
// Extends the primary model (model 4: phylogenetic BM mixture +
// spatiotemporal GP, partial mediation) by adding a clade-mean case
// inventory predictor following the Mundlak (1978) approach.
//
// PARAMETERIZATION:
// The linear predictor for definite articles becomes:
//
//   eta_def[n] = beta_zero_def
//              + g_def[n_cases[n]]                     (ordinal, within-clade)
//              + beta_clade_mean_def * cm[n]            (between-clade control)
//              + f_phy[n] + f_st[n]                    (random effects)
//
// where cm[n] = clade_mean_cases[n] - grand_mean_cases (centered at grand mean).
// An analogous term is added to the indefinite-article predictor.
// All other model components are unchanged from model 4.
//
// The clade-mean predictor clade_mean_cases_c is computed in R by
// scripts/r/run_mundlak.R before being passed to Stan.
//
// INTERPRETATION:
// - theta_def / theta_indef: within-clade ordinal associations between case 
//   inventory and article presence after controlling for clade-level means.
// - beta_clade_mean_def / beta_clade_mean_indef: residual between-clade
//   associations not absorbed by the random effects. Values near zero
//   indicate that between-clade variation adds little beyond what the
//   random effects already capture.
// - If theta_def (and theta_indef) under this model are similar in
//   magnitude to the primary model, the primary results are robust to the
//   within/between distinction. Substantial attenuation would indicate that
//   the primary results were driven partly by between-clade contrasts.
//
// GENERATED QUANTITIES:
// - Population-level predicted probabilities are evaluated at
//   clade_mean_cases_c = 0 (the grand-mean clade), which provides
//   a natural reference comparable across models.
// - Sample-marginal predicted probabilities use the clade mean
//   and latent random effects of each language.
//
// Priors follow the published model throughout. See model 4 for full
// prior justification.

functions {
  // Matern 3/2 covariance kernel.
  // D   : symmetric matrix of pairwise distances
  // ell : length-scale parameter
  matrix matern32_cov(matrix D, real ell) {
    int N = rows(D);
    matrix[N, N] K;
    real c = sqrt(3.0) / ell;
    for (i in 1:N) {
      K[i, i] = 1.0;
      for (j in (i + 1):N) {
        real r = D[i, j];
        real a = 1.0 + c * r;
        real v = a * exp(-c * r);
        K[i, j] = v;
        K[j, i] = v;
      }
    }
    return K;
  }
}

data {

  // ---- Sample size and ordinal range ----
  int<lower=1> N;                              // number of languages
  int<lower=2> K;                              // max case-inventory size

  // ---- Outcome and predictor variables ----
  array[N] int<lower=1, upper=K> n_cases;      // observed case inventory
  array[N] int<lower=0, upper=1> def;          // definite article present
  array[N] int<lower=0, upper=1> indef;        // indefinite article present

  // ---- Mundlak predictor: clade-mean case inventory (centered) ----
  // cm[n] = mean(n_cases within clade of language n) - grand_mean(n_cases).
  // Centering ensures that the intercepts represent the baseline at the
  // average clade and that beta_clade_mean_* is on the same scale as
  // other fixed-effect coefficients.
  // Computed by scripts/r/run_mundlak.R using mundlak_data.csv.
  vector[N] clade_mean_cases_c;

  // ---- Phylogenetic component: BM VCV mixture ----
  int<lower=1> M;                              // number of posterior trees
  array[M] matrix[N, N] K_phy;                // scaled VCV matrices
  vector[M] log_w;                             // log mixture weights

  // ---- Spatial component: pairwise great-circle distances (1000 km) ----
  matrix[N, N] Dgeo;

  // ---- Temporal component: date bounds per language (centuries CE/BCE) ----
  vector[N] t_lo;                              // lower date bound
  vector[N] t_hi;                              // upper date bound

  // ---- Numerical stability ----
  real<lower=0> jitter;

}

transformed data {
  // Pre-compute Cholesky factors of the BM VCV matrices once.
  // A small diagonal jitter (1e-8) is added to each matrix before
  // decomposition to guarantee positive definiteness.
  array[M] matrix[N, N] L_phy;
  for (m in 1:M) {
    matrix[N, N] Km = K_phy[m] + diag_matrix(rep_vector(1e-8, N));
    L_phy[m] = cholesky_decompose(Km);
  }
}

parameters {

  // ---- Intercepts ----
  real beta_zero_def;
  real beta_zero_indef;

  // ---- Between-clade (Mundlak) coefficients ----
  // Additional logit-scale association with clade-mean case inventory,
  // after controlling for the ordinal within-clade effect and the
  // phylogenetic / spatiotemporal random effects.
  real beta_clade_mean_def;
  real beta_clade_mean_indef;

  // ---- Cross-article coefficient ----
  real beta_def_indef;

  // ---- Ordinal / monotone case effects (within-clade after Mundlak control) ----
  real theta_def;
  simplex[K-1] w_def;
  real theta_indef;
  simplex[K-1] w_indef;

  // ---- Phylogenetic latent field (BM mixture prior) ----
  vector[N] f_phy;

  // ---- Spatiotemporal GP: non-centered parameterization ----
  vector[N] eta_st;
  real<lower=0> sigma_eps;      // nugget / observation noise
  real<lower=0> ell_space;      // spatial length-scale
  real<lower=0> ell_time;       // temporal length-scale

  // ---- Latent dates for languages with imprecise attestation ----
  // Uniform on [0, 1]; mapped to [t_lo, t_hi] in transformed parameters.
  vector<lower=0, upper=1>[N] t_u;

  // ---- Shared variance decomposition ----
  real<lower=0> tau;            // total random-effect scale
  real<lower=0, upper=1> rho;   // fraction allocated to phylogenetic component

}

transformed parameters {

  // ---- Ordinal cumulative case-effect increments ----
  // g[k] is the log-odds increment at case-inventory level k relative to k = 1.
  // The simplex w ensures the cumulative profile is monotone whenever
  // theta > 0 (or monotone-decreasing whenever theta < 0).
  vector[K] g_def;
  vector[K] g_indef;
  g_def[1] = 0;
  g_indef[1] = 0;
  for (k in 2:K) {
    g_def[k]   = g_def[k-1]   + theta_def   * w_def[k-1];
    g_indef[k] = g_indef[k-1] + theta_indef * w_indef[k-1];
  }

  // ---- Latent language dates ----
  vector[N] t;
  for (i in 1:N)
    t[i] = t_lo[i] + t_u[i] * (t_hi[i] - t_lo[i]);

  // ---- Derived random-effect scales ----
  real<lower=0> sigma_phy = tau * sqrt(rho);
  real<lower=0> sigma_st  = tau * sqrt(1.0 - rho);

  // ---- Spatiotemporal GP draw ----
  // Separable Matern 3/2 kernel over space x time.
  // Non-centered: f_st = chol(SIGMA) * eta_st, eta_st ~ N(0, I).
  vector[N] f_st;
  {
    matrix[N, N] Dtime;
    for (i in 1:N) {
      Dtime[i, i] = 0.0;
      for (j in (i + 1):N) {
        real d = abs(t[i] - t[j]);
        Dtime[i, j] = d;
        Dtime[j, i] = d;
      }
    }
    matrix[N, N] Ks    = matern32_cov(Dgeo,  ell_space);
    matrix[N, N] Kt    = matern32_cov(Dtime, ell_time);
    matrix[N, N] K_sep = elt_multiply(Ks, Kt);
    matrix[N, N] SIGMA =
        square(sigma_st) * K_sep
      + diag_matrix(rep_vector(square(sigma_eps), N))
      + diag_matrix(rep_vector(jitter, N));
    f_st = cholesky_decompose(SIGMA) * eta_st;
  }

}

model {

  // ---- Priors: intercepts ----
  // Grambank global base-rates, as in models 1-10.
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);

  // ---- Priors: Mundlak between-clade coefficients ----
  // Weakly informative, which matches the scale of all other fixed effects.
  beta_clade_mean_def   ~ normal(0, 1);
  beta_clade_mean_indef ~ normal(0, 1);

  // ---- Priors: remaining fixed effects ----
  beta_def_indef ~ normal(0, 1);
  theta_def      ~ normal(0, 1);
  theta_indef    ~ normal(0, 1);
  w_def          ~ dirichlet(rep_vector(2.0, K-1));
  w_indef        ~ dirichlet(rep_vector(2.0, K-1));

  // ---- Priors: spatiotemporal GP hyperparameters ----
  eta_st    ~ std_normal();
  tau       ~ normal(0, 0.25) T[0, ];
  rho       ~ beta(2, 2);
  sigma_eps ~ normal(0, 0.05) T[0, ];
  ell_space ~ lognormal(log(0.3), 0.5);
  ell_time  ~ lognormal(log(4),   0.5);

  // ---- BM mixture prior on phylogenetic latent field ----
  {
    array[M] real lps;
    for (m in 1:M) {
      lps[m] = log_w[m]
             + multi_normal_cholesky_lpdf(f_phy | rep_vector(0.0, N),
                                          sigma_phy * L_phy[m]);
    }
    target += log_sum_exp(lps);
  }

  // ---- Likelihood ----
  // Mundlak term beta_clade_mean_* * clade_mean_cases_c[n] added to each
  // linear predictor; all other terms identical to model 4.
  for (n in 1:N) {
    real eta_def_n =
        beta_zero_def
      + g_def[n_cases[n]]
      + beta_clade_mean_def * clade_mean_cases_c[n]
      + f_phy[n] + f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n =
        beta_zero_indef
      + g_indef[n_cases[n]]
      + beta_clade_mean_indef * clade_mean_cases_c[n]
      + beta_def_indef * def[n]
      + f_phy[n] + f_st[n];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }

}

generated quantities {

  // ---- Pointwise log-likelihoods (for LOO-CV) ----
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;

  // ---- Posterior predictive samples ----
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  for (n in 1:N) {
    real eta_def_n =
        beta_zero_def
      + g_def[n_cases[n]]
      + beta_clade_mean_def * clade_mean_cases_c[n]
      + f_phy[n] + f_st[n];
    real eta_indef_n =
        beta_zero_indef
      + g_indef[n_cases[n]]
      + beta_clade_mean_indef * clade_mean_cases_c[n]
      + beta_def_indef * def[n]
      + f_phy[n] + f_st[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // ---- Variance decomposition ----
  // Empirical variances of the latent fields, used to compute the
  // fraction of random-effect variance attributable to each component.
  real mean_f_phy = 0;
  real mean_f_st  = 0;
  for (n in 1:N) {
    mean_f_phy += f_phy[n];
    mean_f_st  += f_st[n];
  }
  mean_f_phy /= N;
  mean_f_st  /= N;

  real ss_f_phy = 0;
  real ss_f_st  = 0;
  for (n in 1:N) {
    ss_f_phy += square(f_phy[n] - mean_f_phy);
    ss_f_st  += square(f_st[n]  - mean_f_st);
  }

  real var_f_phy = (N > 1) ? ss_f_phy / (N - 1) : 0;
  real var_f_st  = (N > 1) ? ss_f_st  / (N - 1) : 0;

  real tiny = 1e-12;
  real denom = var_f_phy + var_f_st + square(sigma_eps) + tiny;
  real share_phy     = var_f_phy / denom;
  real share_st      = var_f_st  / denom;
  real prior_share_st = 1.0 - rho;

  // ---- Population-level predicted probabilities ----
  // Evaluated at clade_mean_cases_c = 0 (grand-mean clade) and latent
  // effects = 0, providing a clean reference comparable across models.
  vector[K] p_def_pop;
  vector[K] p_indef_pop_def0;
  vector[K] p_indef_pop_def1;
  vector[K-1] delta_def_pop;
  vector[K-1] delta_indef_pop_def0;
  vector[K-1] delta_indef_pop_def1;
  array[K] int p_def_pop_gt_half;
  array[K] int p_indef_pop_def0_gt_half;
  array[K] int p_indef_pop_def1_gt_half;
  int kstar_def_pop;
  int kstar_indef_pop_def0;
  int kstar_indef_pop_def1;
  vector[K] cumdef_pop;

  for (k in 1:K) {
    // clade_mean_cases_c = 0 => no between-clade adjustment needed
    p_def_pop[k]        = inv_logit(beta_zero_def   + g_def[k]);
    p_indef_pop_def0[k] = inv_logit(beta_zero_indef + g_indef[k]);
    p_indef_pop_def1[k] = inv_logit(beta_zero_indef + g_indef[k] + beta_def_indef);
    p_def_pop_gt_half[k]        = p_def_pop[k]        > 0.5 ? 1 : 0;
    p_indef_pop_def0_gt_half[k] = p_indef_pop_def0[k] > 0.5 ? 1 : 0;
    p_indef_pop_def1_gt_half[k] = p_indef_pop_def1[k] > 0.5 ? 1 : 0;
    cumdef_pop[k] = p_def_pop[k] - p_def_pop[1];
  }
  for (k in 1:(K-1)) {
    delta_def_pop[k]        = p_def_pop[k+1]        - p_def_pop[k];
    delta_indef_pop_def0[k] = p_indef_pop_def0[k+1] - p_indef_pop_def0[k];
    delta_indef_pop_def1[k] = p_indef_pop_def1[k+1] - p_indef_pop_def1[k];
  }
  kstar_def_pop        = K + 1;
  kstar_indef_pop_def0 = K + 1;
  kstar_indef_pop_def1 = K + 1;
  for (k in 1:K) {
    if (kstar_def_pop        == K + 1 && p_def_pop[k]        > 0.5)
      kstar_def_pop        = k;
    if (kstar_indef_pop_def0 == K + 1 && p_indef_pop_def0[k] > 0.5)
      kstar_indef_pop_def0 = k;
    if (kstar_indef_pop_def1 == K + 1 && p_indef_pop_def1[k] > 0.5)
      kstar_indef_pop_def1 = k;
  }

  // ---- Sample-marginal predicted probabilities ----
  // Uses clade_mean_cases_c and latent random effects of each
  // language, averaging over the observed distribution of both.
  vector[K] p_def_avg;
  vector[K] p_indef_avg;
  vector[K-1] delta_def_avg;
  vector[K-1] delta_indef_avg;
  array[K] int p_def_avg_gt_half;
  array[K] int p_indef_avg_gt_half;
  int kstar_def_avg;
  int kstar_indef_avg;
  vector[K] cumdef_avg;

  {
    vector[K] sum_p_def   = rep_vector(0.0, K);
    vector[K] sum_p_indef = rep_vector(0.0, K);

    for (i in 1:N) {
      real lat_i = f_phy[i] + f_st[i];
      real cm_i  = clade_mean_cases_c[i];
      for (k in 1:K) {
        sum_p_def[k] += inv_logit(
            beta_zero_def
          + g_def[k]
          + beta_clade_mean_def * cm_i
          + lat_i);
        sum_p_indef[k] += inv_logit(
            beta_zero_indef
          + g_indef[k]
          + beta_clade_mean_indef * cm_i
          + beta_def_indef * def[i]
          + lat_i);
      }
    }
    for (k in 1:K) {
      p_def_avg[k]   = sum_p_def[k]   / N;
      p_indef_avg[k] = sum_p_indef[k] / N;
    }
  }

  for (k in 1:(K-1)) {
    delta_def_avg[k]   = p_def_avg[k+1]   - p_def_avg[k];
    delta_indef_avg[k] = p_indef_avg[k+1] - p_indef_avg[k];
  }

  cumdef_avg = rep_vector(0.0, K);
  for (k in 1:K) {
    p_def_avg_gt_half[k]   = p_def_avg[k]   > 0.5 ? 1 : 0;
    p_indef_avg_gt_half[k] = p_indef_avg[k] > 0.5 ? 1 : 0;
    cumdef_avg[k] = p_def_avg[k] - p_def_avg[1];
  }

  kstar_def_avg   = K + 1;
  kstar_indef_avg = K + 1;
  for (k in 1:K) {
    if (kstar_def_avg   == K + 1 && p_def_avg[k]   > 0.5) kstar_def_avg   = k;
    if (kstar_indef_avg == K + 1 && p_indef_avg[k] > 0.5) kstar_indef_avg = k;
  }

}
