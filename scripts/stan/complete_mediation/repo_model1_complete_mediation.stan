// ============================================================
// Model 5: Baseline, complete mediation (no phylogenetic term, no spatiotemporal GP)
//
// Bivariate logistic regression with ordinal case effect under
// the complete mediation assumption: case inventory affects
// indefinite articles only through its effect on definite
// articles. There is no direct path from case to indefinite
// articles and no random effects.
//
// Mediation structure: complete
//   Case -> Definite Article
//   Definite Article -> Indefinite Article  (cross-link only)
//   Case -/-> Indefinite Article            (no direct path)
//
// Structural consequence: theta_indef, w_indef, and g_indef
// are absent. The indefinite article equation contains only
// an intercept and the cross-link from def.
//
// Corresponds to Table S4, row:
//   "No phylogenetic term or spatiotemporal GP | Complete"
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

  // Ordinal / monotone case effect for definite article only
  // (No corresponding parameters for indefinite article under complete mediation)
  real theta_def;
  simplex[K-1] w_def;
}

transformed parameters {
  // Cumulative monotone case effect for def: g_def[1] = 0 by construction
  vector[K] g_def;
  g_def[1] = 0;
  for (k in 2:K) {
    g_def[k] = g_def[k-1] + theta_def * w_def[k-1];
  }
}

model {
  // Intercept priors informed by cross-linguistic base rates (Grambank)
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);

  beta_def_indef ~ normal(0, 1);
  theta_def      ~ normal(0, 1);
  w_def          ~ dirichlet(rep_vector(2.0, K - 1));

  // Likelihood
  for (n in 1:N) {
    // Case -> Definite Article
    real eta_def_n = beta_zero_def + g_def[n_cases[n]];
    def[n] ~ bernoulli_logit(eta_def_n);

    // Definite Article -> Indefinite Article (no direct case effect)
    real eta_indef_n = beta_zero_indef + beta_def_indef * def[n];
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
    real eta_def_n   = beta_zero_def + g_def[n_cases[n]];
    real eta_indef_n = beta_zero_indef + beta_def_indef * def[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }
}
