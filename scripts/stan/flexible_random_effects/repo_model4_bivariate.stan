// ============================================================
// Model 10: Phylogenetic term + spatiotemporal GP, full bivariate random effects
//
// Bivariate logistic regression with ordinal case effect,
// a phylogenetic random effect (Brownian-motion mixture over M
// posterior trees), and a separable spatiotemporal Gaussian
// process (Matern-3/2 in space and time).
//
// This model relaxes both assumptions of the shared-field
// structure in Model 4 (repo_model4_phylo_stgp.stan): that the
// phylogenetic and spatiotemporal random effects enter both
// article equations with (a) identical magnitude and (b) perfect
// cross-outcome correlation. Both constraints are removed here.
//
// The bivariate extension uses a conditional decomposition.
// For the phylogenetic component:
//
//   f_phy_raw ~ MVN(0, K_phy[m])     [unit-scale; primary direction]
//   z_phy     ~ MVN(0, K_phy[m])     [unit-scale; independent residual]
//
//   contrib_phy_def[i]   = sigma_phy_def * f_phy_raw[i]
//   contrib_phy_indef[i] = sigma_phy_indef *
//                          (rho_phy * f_phy_raw[i]
//                           + sqrt(1 - rho_phy^2) * z_phy[i])
//
// This implies:
//   Var(contrib_phy_def)            = sigma_phy_def^2  * K[i,j]
//   Var(contrib_phy_indef)          = sigma_phy_indef^2 * K[i,j]
//   Cov(contrib_phy_def, contrib_phy_indef) = rho_phy
//                                             * sigma_phy_def
//                                             * sigma_phy_indef * K[i,j]
//
// An identical structure is applied to the spatiotemporal GP.
// The covariance matrix K is replaced with the separable
// Matérn-3/2 kernel, and two independent standard-normal
// latent vectors (eta_st_1, eta_st_2) are mapped through the
// same Cholesky factor. This avoids computing a second
// Cholesky decomposition.
//
// The parameterization is numerically stable because it never
// divides by a scale parameter. Both f_phy_raw and z_phy receive
// independent mixture priors (same M trees, same log-weights).
// As a result, the cost of the phylogenetic term is exactly
// double that of Model 4.
//
// New parameters relative to Model 4:
//   sigma_phy_def, sigma_phy_indef  (replace tau * sqrt(rho))
//   sigma_st_def, sigma_st_indef    (replace tau * sqrt(1-rho))
//   rho_phy  (cross-outcome phylogenetic correlation)
//   rho_st   (cross-outcome spatiotemporal correlation)
//
// If rho_phy or rho_st is near 1 a posteriori, the data are
// consistent with Model 4's shared-field assumption for that
// component. LOO results and comparison with Models 4 and 9 are
// reported in Section S2.5 (Table S4).
//
// Mediation structure: partial
//   Case -> Definite Article
//   Case -> Indefinite Article  (direct)
//   Definite Article -> Indefinite Article  (cross-link)
//
// Corresponds to Table S4, row:
//   "Phylogenetic term and spatiotemporal GP (bivariate) | Partial"
// ============================================================

functions {
  // Matern-3/2 covariance matrix from a matrix of pairwise distances
  // D must already be scaled to the appropriate units
  matrix matern32_cov(matrix D, real ell) {
    int N = rows(D);
    matrix[N, N] K;
    real c = sqrt(3.0) / ell;
    for (i in 1:N) {
      K[i, i] = 1.0;
      for (j in (i + 1):N) {
        real r  = D[i, j];
        real a  = 1.0 + c * r;
        real v  = a * exp(-c * r);
        K[i, j] = v;
        K[j, i] = v;
      }
    }
    return K;
  }
}

data {
  int<lower=1> N;                        // number of languages
  int<lower=2> K;                        // max number of cases
  array[N] int<lower=1, upper=K> n_cases;

  array[N] int<lower=0, upper=1> def;    // definite article present (1) or absent (0)
  array[N] int<lower=0, upper=1> indef;  // indefinite article present (1) or absent (0)

  // Phylogeny: M trees (Brownian-motion VCVs), scaled so mean(diag) = 1
  int<lower=1> M;
  array[M] matrix[N, N] K_phy;
  vector[M] log_w;                       // log mixture weights; equal weights: rep_vector(-log(M), M)

  // Pairwise great-circle distances in units of 1000 km
  matrix[N, N] Dgeo;

  // Date bounds per language in centuries before 2000 CE
  // For contemporary languages set t_lo[i] = t_hi[i] = 0
  vector[N] t_lo;
  vector[N] t_hi;

  real<lower=0> jitter;                  // fixed diagonal jitter for Cholesky stability
}

transformed data {
  // Precompute Cholesky factors for the phylogenetic VCV matrices
  // A small jitter (1e-8) is added to each diagonal for numerical stability
  array[M] matrix[N, N] L_phy;
  for (m in 1:M) {
    matrix[N, N] Km = K_phy[m] + diag_matrix(rep_vector(1e-8, N));
    L_phy[m] = cholesky_decompose(Km);
  }
}

parameters {
  // Intercepts (Grambank-informed priors; see Section S1.2)
  real beta_zero_def;
  real beta_zero_indef;

  // Definite -> Indefinite cross-link
  real beta_def_indef;

  // Ordinal / monotone case effects (per outcome)
  real theta_def;
  simplex[K-1] w_def;
  real theta_indef;
  simplex[K-1] w_indef;

  // Phylogenetic latent fields (both unit-scale)
  // f_phy_raw: primary direction, directly used for the definite article contribution
  // z_phy:     orthogonal residual direction, used to construct the indef contribution
  // Both receive independent mixture priors over the M trees
  vector[N] f_phy_raw;
  vector[N] z_phy;

  // Spatiotemporal GP: two independent unit-scale latent vectors
  // eta_st_1: primary direction (shared between outcomes, weighted by rho_st)
  // eta_st_2: orthogonal direction (indef residual)
  vector[N] eta_st_1;
  vector[N] eta_st_2;

  // GP hyperparameters
  real<lower=0> sigma_eps;              // nugget standard deviation (common to both outcomes)
  real<lower=0> ell_space;             // spatial length-scale (1000 km units)
  real<lower=0> ell_time;              // temporal length-scale (century units)

  // Latent dates for date-uncertain languages
  // t_u[i] ~ Uniform(0, 1) maps linearly to [t_lo[i], t_hi[i]]
  vector<lower=0, upper=1>[N] t_u;

  // Outcome-specific scale parameters
  real<lower=0> sigma_phy_def;
  real<lower=0> sigma_phy_indef;
  real<lower=0> sigma_st_def;
  real<lower=0> sigma_st_indef;

  // Cross-outcome correlation parameters
  real<lower=-1, upper=1> rho_phy;      // phylogenetic cross-outcome correlation
  real<lower=-1, upper=1> rho_st;       // spatiotemporal cross-outcome correlation
}

transformed parameters {
  // Cumulative monotone case effects: g[1] = 0 by construction
  vector[K] g_def;
  vector[K] g_indef;
  g_def[1]   = 0;
  g_indef[1] = 0;
  for (k in 2:K) {
    g_def[k]   = g_def[k-1]   + theta_def   * w_def[k-1];
    g_indef[k] = g_indef[k-1] + theta_indef * w_indef[k-1];
  }

  // Realized language dates
  vector[N] t;
  for (i in 1:N) {
    t[i] = t_lo[i] + t_u[i] * (t_hi[i] - t_lo[i]);
  }

  // Unit-scale spatiotemporal GP: build Cholesky once, map both
  // eta_st_1 and eta_st_2 through it to obtain two orthogonal draws
  vector[N] f_st_1;                      // L_st * eta_st_1 (primary direction)
  vector[N] f_st_2;                      // L_st * eta_st_2 (orthogonal direction)
  {
    matrix[N, N] Dtime;
    for (i in 1:N) {
      Dtime[i, i] = 0;
      for (j in (i + 1):N) {
        real d     = abs(t[i] - t[j]);
        Dtime[i, j] = d;
        Dtime[j, i] = d;
      }
    }

    // Separable kernel: K_space o K_time (Hadamard product)
    matrix[N, N] K_sep = elt_multiply(matern32_cov(Dgeo,  ell_space),
                                      matern32_cov(Dtime, ell_time));

    // Unit-scale covariance: outcome-specific sigma_st scaling enters the likelihood
    matrix[N, N] SIGMA =
        K_sep
        + diag_matrix(rep_vector(square(sigma_eps), N))
        + diag_matrix(rep_vector(jitter, N));

    matrix[N, N] L_st = cholesky_decompose(SIGMA);
    f_st_1 = L_st * eta_st_1;
    f_st_2 = L_st * eta_st_2;
  }

  // Outcome-specific latent contributions via conditional decomposition
  // Precompute sqrt(1 - rho^2) terms for numerical clarity
  real sqrt1m_rho_phy_sq = sqrt(1.0 - square(rho_phy));
  real sqrt1m_rho_st_sq  = sqrt(1.0 - square(rho_st));

  // Phylogenetic contributions
  vector[N] contrib_phy_def   = sigma_phy_def * f_phy_raw;
  vector[N] contrib_phy_indef = sigma_phy_indef *
                                 (rho_phy * f_phy_raw + sqrt1m_rho_phy_sq * z_phy);

  // Spatiotemporal contributions
  vector[N] contrib_st_def    = sigma_st_def * f_st_1;
  vector[N] contrib_st_indef  = sigma_st_indef *
                                 (rho_st * f_st_1 + sqrt1m_rho_st_sq * f_st_2);
}

model {
  // Intercept priors informed by cross-linguistic base rates (Grambank)
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);

  beta_def_indef ~ normal(0, 1);
  theta_def      ~ normal(0, 1);
  theta_indef    ~ normal(0, 1);
  w_def   ~ dirichlet(rep_vector(2.0, K - 1));
  w_indef ~ dirichlet(rep_vector(2.0, K - 1));

  eta_st_1  ~ std_normal();
  eta_st_2  ~ std_normal();
  sigma_eps ~ normal(0, 0.05) T[0,];
  ell_space ~ lognormal(log(0.3), 0.5);  // median ~300 km
  ell_time  ~ lognormal(log(4),   0.5);  // median ~4 centuries

  // Half-normal(0, 0.25) priors on outcome-specific scales: consistent with
  // the original tau prior while placing no prior assumption that def and
  // indef are equally affected by phylogenetic or areal structure
  sigma_phy_def   ~ normal(0, 0.25) T[0,];
  sigma_phy_indef ~ normal(0, 0.25) T[0,];
  sigma_st_def    ~ normal(0, 0.25) T[0,];
  sigma_st_indef  ~ normal(0, 0.25) T[0,];

  // normal(0, 0.5) prior on correlations: weakly favors independence while
  // allowing strong correlation if supported by the data
  rho_phy ~ normal(0, 0.5);
  rho_st  ~ normal(0, 0.5);

  // Independent mixture-of-Gaussians priors for both phylogenetic latent fields
  // Both marginalise over tree uncertainty. Doubling the mixture evaluations
  // is the only additional cost over Model 4
  {
    array[M] real lps_raw;
    array[M] real lps_z;
    for (m in 1:M) {
      lps_raw[m] = log_w[m]
                   + multi_normal_cholesky_lpdf(f_phy_raw | rep_vector(0, N), L_phy[m]);
      lps_z[m]   = log_w[m]
                   + multi_normal_cholesky_lpdf(z_phy     | rep_vector(0, N), L_phy[m]);
    }
    target += log_sum_exp(lps_raw);
    target += log_sum_exp(lps_z);
  }

  // Likelihood
  for (n in 1:N) {
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + contrib_phy_def[n]
                     + contrib_st_def[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + contrib_phy_indef[n]
                       + contrib_st_indef[n];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }
}

generated quantities {
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  for (n in 1:N) {
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + contrib_phy_def[n]
                     + contrib_st_def[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + contrib_phy_indef[n]
                       + contrib_st_indef[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // Variance and share summaries
  // Sample means of the realized outcome-specific latent contributions
  real mean_cpd = 0;
  real mean_cpi = 0;
  real mean_csd = 0;
  real mean_csi = 0;
  for (n in 1:N) {
    mean_cpd += contrib_phy_def[n];
    mean_cpi += contrib_phy_indef[n];
    mean_csd += contrib_st_def[n];
    mean_csi += contrib_st_indef[n];
  }
  mean_cpd /= N;
  mean_cpi /= N;
  mean_csd /= N;
  mean_csi /= N;

  // Sample variances of the realized latent contributions
  // These are the variances of each component's contribution to
  // cross-language variability in the log-odds of each outcome
  real ss_cpd = 0;
  real ss_cpi = 0;
  real ss_csd = 0;
  real ss_csi = 0;
  for (n in 1:N) {
    ss_cpd += square(contrib_phy_def[n]   - mean_cpd);
    ss_cpi += square(contrib_phy_indef[n] - mean_cpi);
    ss_csd += square(contrib_st_def[n]    - mean_csd);
    ss_csi += square(contrib_st_indef[n]  - mean_csi);
  }

  real tiny = 1e-12;
  real var_phy_def   = (N > 1) ? ss_cpd / (N - 1) : 0;
  real var_phy_indef = (N > 1) ? ss_cpi / (N - 1) : 0;
  real var_st_def    = (N > 1) ? ss_csd / (N - 1) : 0;
  real var_st_indef  = (N > 1) ? ss_csi / (N - 1) : 0;

  // Outcome-specific variance shares
  real denom_def       = var_phy_def   + var_st_def   + tiny;
  real share_phy_def   = var_phy_def   / denom_def;
  real share_st_def    = var_st_def    / denom_def;

  real denom_indef     = var_phy_indef + var_st_indef + tiny;
  real share_phy_indef = var_phy_indef / denom_indef;
  real share_st_indef  = var_st_indef  / denom_indef;

  // Empirical Pearson correlation between total latent contributions of the two outcomes
  // This is the realized cross-outcome correlation, analogous to the parametric rho_phy
  // and rho_st combined. If near 1.0, the data do not require the additional flexibility
  // of this model over Model 9 (repo_model4_sep_scaling.stan)
  real lat_total_def_mean   = 0;
  real lat_total_indef_mean = 0;
  for (n in 1:N) {
    lat_total_def_mean   += contrib_phy_def[n]   + contrib_st_def[n];
    lat_total_indef_mean += contrib_phy_indef[n] + contrib_st_indef[n];
  }
  lat_total_def_mean   /= N;
  lat_total_indef_mean /= N;

  real ss_def_total   = 0;
  real ss_indef_total = 0;
  real ss_cross       = 0;
  for (n in 1:N) {
    real d_def   = (contrib_phy_def[n]   + contrib_st_def[n])   - lat_total_def_mean;
    real d_indef = (contrib_phy_indef[n] + contrib_st_indef[n]) - lat_total_indef_mean;
    ss_def_total   += square(d_def);
    ss_indef_total += square(d_indef);
    ss_cross       += d_def * d_indef;
  }

  real empirical_latent_corr =
      ss_cross / (sqrt(ss_def_total) * sqrt(ss_indef_total) + tiny);

  // ---- Population-level predicted probabilities (fixed effects only; random effects at zero) ----
  vector[K] p_def_pop;
  vector[K] p_indef_pop_def0;           // P(indef=1 | case=k, def=0)
  vector[K] p_indef_pop_def1;           // P(indef=1 | case=k, def=1)

  vector[K-1] delta_def_pop;
  vector[K-1] delta_indef_pop_def0;
  vector[K-1] delta_indef_pop_def1;

  array[K] int p_def_pop_gt_half;
  array[K] int p_indef_pop_def0_gt_half;
  array[K] int p_indef_pop_def1_gt_half;

  int kstar_def_pop;                     // smallest k with P(def=1) > 0.5; K+1 if none
  int kstar_indef_pop_def0;
  int kstar_indef_pop_def1;

  // ---- Sample-marginal predicted probabilities (averaged over observed languages) ----
  vector[K] p_def_avg;
  vector[K] p_indef_avg;

  vector[K-1] delta_def_avg;
  vector[K-1] delta_indef_avg;

  array[K] int p_def_avg_gt_half;
  array[K] int p_indef_avg_gt_half;

  int kstar_def_avg;
  int kstar_indef_avg;

  vector[K] cumdef_pop;                  // cumulative change from k=1 (population)
  vector[K] cumdef_avg;                  // cumulative change from k=1 (sample-marginal)

  {
    // Initialize accumulators
    for (i in 1:K) {
      p_def_pop[i]               = 0;
      p_indef_pop_def0[i]        = 0;
      p_indef_pop_def1[i]        = 0;
      p_def_avg[i]               = 0;
      p_indef_avg[i]             = 0;
      p_def_pop_gt_half[i]       = 0;
      p_indef_pop_def0_gt_half[i] = 0;
      p_indef_pop_def1_gt_half[i] = 0;
      p_def_avg_gt_half[i]       = 0;
      p_indef_avg_gt_half[i]     = 0;
      cumdef_pop[i]              = 0;
      cumdef_avg[i]              = 0;
    }

    // Population-level predictions (random effects held at zero)
    for (i in 1:K) {
      p_def_pop[i] = inv_logit(beta_zero_def + g_def[i]);

      p_indef_pop_def0[i] = inv_logit(beta_zero_indef + g_indef[i]
                                       + beta_def_indef * 0);
      p_indef_pop_def1[i] = inv_logit(beta_zero_indef + g_indef[i]
                                       + beta_def_indef * 1);
    }

    for (i in 1:(K-1)) {
      delta_def_pop[i]        = p_def_pop[i+1]        - p_def_pop[i];
      delta_indef_pop_def0[i] = p_indef_pop_def0[i+1] - p_indef_pop_def0[i];
      delta_indef_pop_def1[i] = p_indef_pop_def1[i+1] - p_indef_pop_def1[i];
    }

    for (i in 1:K) {
      p_def_pop_gt_half[i]         = p_def_pop[i]        > 0.5 ? 1 : 0;
      p_indef_pop_def0_gt_half[i]  = p_indef_pop_def0[i] > 0.5 ? 1 : 0;
      p_indef_pop_def1_gt_half[i]  = p_indef_pop_def1[i] > 0.5 ? 1 : 0;
    }

    kstar_def_pop        = K + 1;
    kstar_indef_pop_def0 = K + 1;
    kstar_indef_pop_def1 = K + 1;
    for (i in 1:K) {
      if (kstar_def_pop        == K + 1 && p_def_pop[i]        > 0.5) kstar_def_pop        = i;
      if (kstar_indef_pop_def0 == K + 1 && p_indef_pop_def0[i] > 0.5) kstar_indef_pop_def0 = i;
      if (kstar_indef_pop_def1 == K + 1 && p_indef_pop_def1[i] > 0.5) kstar_indef_pop_def1 = i;
    }

    for (i in 1:K) cumdef_pop[i] = p_def_pop[i] - p_def_pop[1];

    // Sample-marginal predictions: average P(article | case=k) over observed languages
    // Uses outcome-specific realized latent contributions (contrib_phy_* + contrib_st_*)
    {
      vector[K] sum_p_def_i;
      vector[K] sum_p_indef_i;
      for (i in 1:K) {
        sum_p_def_i[i]   = 0;
        sum_p_indef_i[i] = 0;
      }

      for (i in 1:N) {
        real lat_def_i   = contrib_phy_def[i]   + contrib_st_def[i];
        real lat_indef_i = contrib_phy_indef[i] + contrib_st_indef[i];

        for (k in 1:K) {
          sum_p_def_i[k]   += inv_logit(beta_zero_def   + g_def[k]   + lat_def_i);
          // Use observed def[i] to preserve the likelihood's conditional structure
          sum_p_indef_i[k] += inv_logit(beta_zero_indef + g_indef[k]
                                        + beta_def_indef * def[i]
                                        + lat_indef_i);
        }
      }

      for (i in 1:K) {
        p_def_avg[i]   = sum_p_def_i[i]   / N;
        p_indef_avg[i] = sum_p_indef_i[i] / N;
      }
    }

    for (i in 1:(K-1)) {
      delta_def_avg[i]   = p_def_avg[i+1]   - p_def_avg[i];
      delta_indef_avg[i] = p_indef_avg[i+1] - p_indef_avg[i];
    }

    for (i in 1:K) {
      p_def_avg_gt_half[i]   = p_def_avg[i]   > 0.5 ? 1 : 0;
      p_indef_avg_gt_half[i] = p_indef_avg[i] > 0.5 ? 1 : 0;
    }

    kstar_def_avg   = K + 1;
    kstar_indef_avg = K + 1;
    for (i in 1:K) {
      if (kstar_def_avg   == K + 1 && p_def_avg[i]   > 0.5) kstar_def_avg   = i;
      if (kstar_indef_avg == K + 1 && p_indef_avg[i] > 0.5) kstar_indef_avg = i;
    }

    for (i in 1:K) cumdef_avg[i] = p_def_avg[i] - p_def_avg[1];

  } // end local block
}
