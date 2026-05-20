// ============================================================
// Model 2: Phylogenetic term only (no spatiotemporal GP)
//
// Bivariate logistic regression with ordinal case effect and
// a phylogenetic random effect modelled as a Brownian-motion
// process. Phylogenetic uncertainty is handled by a finite
// mixture over M posterior trees.
//
// The single scale parameter tau controls the overall magnitude
// of phylogenetic structured residual variation.
//
// Mediation structure: partial
//   Case -> Definite Article
//   Case -> Indefinite Article  (direct)
//   Definite Article -> Indefinite Article  (cross-link)
//
// Corresponds to Table S4, row:
//   "Phylogenetic term | Partial"
// ============================================================

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

  // Phylogenetic latent field (mixture-of-Gaussians prior over M trees)
  vector[N] f_phy;

  // Overall phylogenetic scale
  real<lower=0> tau;
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

  // In this model tau is the phylogenetic scale directly (no GP to share with)
  real<lower=0> sigma_phy = tau;
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

  // Half-normal prior on phylogenetic scale
  tau ~ normal(0, 0.25) T[0,];

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
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + f_phy[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n];
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
                     + f_phy[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // Sample variance of the phylogenetic latent field
  real mean_f_phy = 0;
  for (n in 1:N) mean_f_phy += f_phy[n];
  mean_f_phy /= N;

  real ss_f_phy = 0;
  for (n in 1:N) ss_f_phy += square(f_phy[n] - mean_f_phy);

  real var_f_phy = (N > 1) ? ss_f_phy / (N - 1) : 0;
}
