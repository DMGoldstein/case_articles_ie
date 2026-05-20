// ============================================================
// Model 1: Baseline (no phylogenetic term, no spatiotemporal GP)
//
// Bivariate logistic regression with ordinal case effect.
// No random effects: case inventory is the sole predictor of
// both definite and indefinite article presence.
//
// Mediation structure: partial
//   Case -> Definite Article
//   Case -> Indefinite Article  (direct)
//   Definite Article -> Indefinite Article  (cross-link)
//
// Corresponds to Table S4, row:
//   "No phylogenetic term or spatiotemporal GP | Partial"
// ============================================================

data {
  int<lower=1> N;                        // number of languages
  int<lower=2> K;                        // max number of cases
  array[N] int<lower=1, upper=K> n_cases;

  array[N] int<lower=0, upper=1> def;    // definite article present (1) or absent (0)
  array[N] int<lower=0, upper=1> indef;  // indefinite article present (1) or absent (0)
}

parameters {
  // Intercepts (Grambank-informed priors; see Section S1.2)
  real beta_zero_def;
  real beta_zero_indef;

  // Definite -> Indefinite cross-link
  real beta_def_indef;

  // Ordinal / monotone case effects (per outcome)
  // theta_* is the total signed magnitude; w_* distributes it across steps
  real theta_def;
  simplex[K-1] w_def;
  real theta_indef;
  simplex[K-1] w_indef;
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

  // Likelihood
  for (n in 1:N) {
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n];
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
                     + g_def[n_cases[n]];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }
}
