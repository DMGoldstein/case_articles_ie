// ============================================================
// Model 8: Phylogenetic term + spatiotemporal GP, complete mediation
//
// Bivariate logistic regression with ordinal case effect,
// a phylogenetic random effect (Brownian-motion mixture over M
// posterior trees), a separable spatiotemporal Gaussian process
// (Matern-3/2 in space and time), and complete mediation: case
// inventory affects indefinite articles only through definite
// articles.
//
// The total structured residual scale tau is split between the
// phylogenetic and spatiotemporal components by the allocation
// parameter rho (Beta(2,2) prior):
//   sigma_phy = tau * sqrt(rho)
//   sigma_st  = tau * sqrt(1 - rho)
//
// Mediation structure: complete
//   Case -> Definite Article
//   Definite Article -> Indefinite Article  (cross-link only)
//   Case -/-> Indefinite Article            (no direct path)
//
// Structural consequence: theta_indef, w_indef, and g_indef
// are absent. The indefinite article equation contains only
// an intercept, the cross-link from def, and the shared
// phylogenetic and spatiotemporal latent fields.
//
// Note on population-level predictions: because indefinite
// articles have no direct case effect, p_indef_pop_def0[k] and
// p_indef_pop_def1[k] are constant across k by construction,
// and the corresponding delta vectors are identically zero.
// These quantities are retained for structural consistency with
// the partial mediation models.
//
// Corresponds to Table S4, row:
//   "Phylogenetic term and spatiotemporal GP | Complete"
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
  int<lower=1> M;                        // number of trees in the mixture
  array[M] matrix[N, N] K_phy;          // one VCV matrix per tree
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

  // Ordinal / monotone case effect for definite article only
  // (No corresponding parameters for indefinite article under complete mediation)
  real theta_def;
  simplex[K-1] w_def;

  // Phylogenetic latent field (mixture-of-Gaussians prior over M trees)
  vector[N] f_phy;

  // Spatiotemporal GP: non-centered 
  vector[N] eta_st;                      // standard-normal latent vector
  real<lower=0> sigma_eps;              // nugget standard deviation
  real<lower=0> ell_space;             // spatial length-scale (1000 km units)
  real<lower=0> ell_time;              // temporal length-scale (century units)

  // Latent dates for date-uncertain languages
  // t_u[i] ~ Uniform(0, 1) maps linearly to [t_lo[i], t_hi[i]]
  vector<lower=0, upper=1>[N] t_u;

  // Total structured residual scale and phylogenetic allocation share
  real<lower=0> tau;
  real<lower=0, upper=1> rho;           // share allocated to phylogeny
}

transformed parameters {
  // Cumulative monotone case effect for def: g_def[1] = 0 by construction
  vector[K] g_def;
  g_def[1] = 0;
  for (k in 2:K) {
    g_def[k] = g_def[k-1] + theta_def * w_def[k-1];
  }

  // Realized language dates
  vector[N] t;
  for (i in 1:N) {
    t[i] = t_lo[i] + t_u[i] * (t_hi[i] - t_lo[i]);
  }

  // Component scales derived from tau and rho
  real<lower=0> sigma_phy = tau * sqrt(rho);
  real<lower=0> sigma_st  = tau * sqrt(1.0 - rho);

  // Separable spatiotemporal GP draw via Cholesky non-centered 
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
        square(sigma_st) * K_sep
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
  w_def          ~ dirichlet(rep_vector(2.0, K - 1));

  eta_st    ~ std_normal();
  tau       ~ normal(0, 0.25) T[0,];
  rho       ~ beta(2, 2);
  sigma_eps ~ normal(0, 0.05) T[0,];
  ell_space ~ lognormal(log(0.3), 0.5);  // median ~300 km
  ell_time  ~ lognormal(log(4),   0.5);  // median ~4 centuries

  // Mixture-of-Gaussians prior for f_phy: marginalizes over tree uncertainty
  {
    array[M] real lps;
    for (m in 1:M) {
      lps[m] = log_w[m]
               + multi_normal_cholesky_lpdf(f_phy | rep_vector(0, N),
                                            sigma_phy * L_phy[m]);
    }
    target += log_sum_exp(lps);
  }

  // Likelihood
  for (n in 1:N) {
    // Case -> Definite Article (ordinal case effect + both latent fields)
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + f_phy[n] + f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    // Definite Article -> Indefinite Article (no direct case effect)
    real eta_indef_n = beta_zero_indef
                       + beta_def_indef * def[n]
                       + f_phy[n] + f_st[n];
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
                     + f_phy[n] + f_st[n];
    real eta_indef_n = beta_zero_indef
                       + beta_def_indef * def[n]
                       + f_phy[n] + f_st[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // Sample variances of the latent fields
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

  // Empirical variance shares of each structured component
  real tiny = 1e-12;
  real denom     = var_f_phy + var_f_st + square(sigma_eps) + tiny;
  real share_phy = var_f_phy / denom;
  real share_st  = var_f_st  / denom;

  // Prior allocation of structured variance to spatiotemporal GP
  real prior_share_st = 1.0 - rho;

  // ---- Population-level predicted probabilities (fixed effects only; random effects at zero) ----
  // Note: p_indef_pop_def0[k] and p_indef_pop_def1[k] do not vary with k under
  // complete mediation (no direct case effect on indef). delta_indef_pop values
  // are therefore identically zero. These quantities are retained for structural
  // consistency with the partial mediation models.
  vector[K] p_def_pop;
  vector[K] p_indef_pop_def0;           // P(indef=1 | case=k, def=0) -- constant across k
  vector[K] p_indef_pop_def1;           // P(indef=1 | case=k, def=1) -- constant across k

  vector[K-1] delta_def_pop;
  vector[K-1] delta_indef_pop_def0;     // identically zero under complete mediation
  vector[K-1] delta_indef_pop_def1;     // identically zero under complete mediation

  array[K] int p_def_pop_gt_half;
  array[K] int p_indef_pop_def0_gt_half;
  array[K] int p_indef_pop_def1_gt_half;

  int kstar_def_pop;
  int kstar_indef_pop_def0;
  int kstar_indef_pop_def1;

  // ---- Sample-marginal predicted probabilities (averaged over observed languages) ----
  // p_indef_avg[k] varies with k only through the case effect on def, not directly.
  vector[K] p_def_avg;
  vector[K] p_indef_avg;

  vector[K-1] delta_def_avg;
  vector[K-1] delta_indef_avg;

  array[K] int p_def_avg_gt_half;
  array[K] int p_indef_avg_gt_half;

  int kstar_def_avg;
  int kstar_indef_avg;

  vector[K] cumdef_pop;
  vector[K] cumdef_avg;

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

      // Indef has no direct case effect; these values are the same for all k
      p_indef_pop_def0[i] = inv_logit(beta_zero_indef + beta_def_indef * 0);
      p_indef_pop_def1[i] = inv_logit(beta_zero_indef + beta_def_indef * 1);
    }

    for (i in 1:(K-1)) {
      delta_def_pop[i]        = p_def_pop[i+1]        - p_def_pop[i];
      delta_indef_pop_def0[i] = p_indef_pop_def0[i+1] - p_indef_pop_def0[i];  // = 0
      delta_indef_pop_def1[i] = p_indef_pop_def1[i+1] - p_indef_pop_def1[i];  // = 0
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
    // For indef, uses observed def[i] to preserve the likelihood's conditional structure.
    // p_indef_avg[k] is not constant across k because the loop averages over
    // observed def[i], which in turn depends on n_cases[i].
    {
      vector[K] sum_p_def_i;
      vector[K] sum_p_indef_i;
      for (i in 1:K) {
        sum_p_def_i[i]   = 0;
        sum_p_indef_i[i] = 0;
      }

      for (i in 1:N) {
        real lat_i = f_phy[i] + f_st[i];
        for (k in 1:K) {
          sum_p_def_i[k]   += inv_logit(beta_zero_def   + g_def[k] + lat_i);
          sum_p_indef_i[k] += inv_logit(beta_zero_indef
                                        + beta_def_indef * def[i] + lat_i);
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
