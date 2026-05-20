// ============================================================
// Model 9: Phylogenetic term + spatiotemporal GP, separate outcome scaling
//
// Bivariate logistic regression with ordinal case effect,
// a phylogenetic random effect (Brownian-motion mixture over M
// posterior trees), and a separable spatiotemporal Gaussian
// process (Matern-3/2 in space and time).
//
// This model relaxes one assumption of Model 4
// (repo_model4_phylo_stgp.stan): in that model, the phylogenetic
// and spatiotemporal random effects enter both article equations
// with identical magnitude (a consequence of the shared tau/rho
// parameterization). Here, each outcome receives independent
// scale parameters for each random-effects component:
//   sigma_phy_def, sigma_phy_indef  (phylogenetic contribution)
//   sigma_st_def,  sigma_st_indef   (spatiotemporal contribution)
//
// These four half-normal parameters replace tau and rho.
// The latent fields f_phy and f_st remain single shared vectors,
// so the two outcomes are still correlated through the same
// underlying directions. Only their magnitudes are free to differ.
// Cross-outcome correlation in the latent direction therefore
// remains implicitly 1.0. For a full bivariate treatment that
// additionally frees the cross-outcome correlation, see the
// full bivariate model (repo_model4_bivariate.stan).
//
// Both latent fields are unit-scale: f_phy is drawn from
// MVN(0, K_phy[m]) and f_st from a unit-scale separable GP.
// Outcome-specific scaling is applied in the likelihood.
//
// LOO results and comparison with Model 4 are reported in
// Section 11.3 of the Supplement (Table 7).
//
// Mediation structure: partial
//   Case -> Definite Article
//   Case -> Indefinite Article  (direct)
//   Definite Article -> Indefinite Article  (cross-link)
//
// Corresponds to Table 7, row:
//   "Phylogenetic term and spatiotemporal GP (separate scaling) | Partial"
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

  // Phylogenetic latent field (unit-scale mixture-of-Gaussians prior over M trees)
  // Outcome-specific magnitudes are applied in the likelihood via sigma_phy_*
  vector[N] f_phy;

  // Spatiotemporal GP: unit-scale non-centered parameterization
  // Outcome-specific magnitudes are applied in the likelihood via sigma_st_*
  vector[N] eta_st;                      // standard-normal latent vector
  real<lower=0> sigma_eps;              // nugget standard deviation (common to both outcomes)
  real<lower=0> ell_space;             // spatial length-scale (1000 km units)
  real<lower=0> ell_time;              // temporal length-scale (century units)

  // Latent dates for date-uncertain languages
  // t_u[i] ~ Uniform(0, 1) maps linearly to [t_lo[i], t_hi[i]]
  vector<lower=0, upper=1>[N] t_u;

  // Outcome-specific scale parameters (replace tau and rho from Model 4)
  real<lower=0> sigma_phy_def;          // phylogenetic scale, definite article
  real<lower=0> sigma_phy_indef;        // phylogenetic scale, indefinite article
  real<lower=0> sigma_st_def;           // spatiotemporal scale, definite article
  real<lower=0> sigma_st_indef;         // spatiotemporal scale, indefinite article
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

  // Unit-scale separable spatiotemporal GP draw
  // SIGMA = K_sep + sigma_eps^2 * I + jitter * I
  // No sigma_st scaling in SIGMA; outcome-specific scaling is applied in the likelihood
  vector[N] f_st;
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

    matrix[N, N] SIGMA =
        K_sep
        + diag_matrix(rep_vector(square(sigma_eps), N))
        + diag_matrix(rep_vector(jitter, N));

    f_st = cholesky_decompose(SIGMA) * eta_st;
  }
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

  eta_st    ~ std_normal();
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

  // Unit-scale mixture-of-Gaussians prior for f_phy: marginalizes over tree uncertainty
  // Outcome-specific magnitudes (sigma_phy_def, sigma_phy_indef) are applied
  // in the likelihood, not in the prior
  {
    array[M] real lps;
    for (m in 1:M) {
      lps[m] = log_w[m]
               + multi_normal_cholesky_lpdf(f_phy | rep_vector(0, N), L_phy[m]);
    }
    target += log_sum_exp(lps);
  }

  // Likelihood: each outcome uses its own phylogenetic and spatiotemporal scales
  for (n in 1:N) {
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + sigma_phy_def * f_phy[n]
                     + sigma_st_def  * f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef  * def[n]
                       + sigma_phy_indef * f_phy[n]
                       + sigma_st_indef  * f_st[n];
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
                     + sigma_phy_def * f_phy[n]
                     + sigma_st_def  * f_st[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef  * def[n]
                       + sigma_phy_indef * f_phy[n]
                       + sigma_st_indef  * f_st[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // Variance and share summaries
  // Step 1: sample variances of the unit-scale latent fields
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

  // Step 2: outcome-specific scaled variances
  // Variance of each component's contribution to each outcome's linear predictor
  real var_phy_def_scaled   = square(sigma_phy_def)   * var_f_phy;
  real var_phy_indef_scaled = square(sigma_phy_indef) * var_f_phy;
  real var_st_def_scaled    = square(sigma_st_def)    * var_f_st;
  real var_st_indef_scaled  = square(sigma_st_indef)  * var_f_st;

  real tiny = 1e-12;

  // Step 3: outcome-specific variance shares
  // Fraction of structured latent variance attributable to each component
  real denom_def       = var_phy_def_scaled   + var_st_def_scaled   + tiny;
  real share_phy_def   = var_phy_def_scaled   / denom_def;
  real share_st_def    = var_st_def_scaled    / denom_def;

  real denom_indef     = var_phy_indef_scaled + var_st_indef_scaled + tiny;
  real share_phy_indef = var_phy_indef_scaled / denom_indef;
  real share_st_indef  = var_st_indef_scaled  / denom_indef;

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
    // These are unaffected by the separate-scaling change: no latent terms enter here
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
    // Each outcome uses its own outcome-specific latent contribution
    {
      vector[K] sum_p_def_i;
      vector[K] sum_p_indef_i;
      for (i in 1:K) {
        sum_p_def_i[i]   = 0;
        sum_p_indef_i[i] = 0;
      }

      for (i in 1:N) {
        real lat_def_i   = sigma_phy_def   * f_phy[i] + sigma_st_def   * f_st[i];
        real lat_indef_i = sigma_phy_indef * f_phy[i] + sigma_st_indef * f_st[i];

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
