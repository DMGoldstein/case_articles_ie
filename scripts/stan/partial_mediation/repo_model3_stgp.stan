// ============================================================
// Model 3: Spatiotemporal GP only (no phylogenetic term)
//
// Bivariate logistic regression with ordinal case effect and
// a separable spatiotemporal Gaussian process (Matern-3/2 in
// space and time, combined via Hadamard product). No explicit
// phylogenetic random effect is included.
//
// The single scale parameter tau controls the overall magnitude
// of spatiotemporal structured residual variation.
//
// Mediation structure: partial
//   Case -> Definite Article
//   Case -> Indefinite Article  (direct)
//   Definite Article -> Indefinite Article  (cross-link)
//
// Corresponds to Table S4, row:
//   "Spatiotemporal GP | Partial"
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

  // Pairwise great-circle distances in units of 1000 km
  matrix[N, N] Dgeo;

  // Date bounds per language in centuries before 2000 CE
  // For contemporary languages set t_lo[i] = t_hi[i] = 0
  vector[N] t_lo;
  vector[N] t_hi;

  real<lower=0> jitter;                  // fixed diagonal jitter for Cholesky stability
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

  // Spatiotemporal GP: non-centered parameterization
  vector[N] eta_st;                      // standard-normal latent vector
  real<lower=0> sigma_eps;              // nugget standard deviation
  real<lower=0> ell_space;             // spatial length-scale (1000 km units)
  real<lower=0> ell_time;              // temporal length-scale (century units)

  // Latent dates for date-uncertain languages
  // t_u[i] ~ Uniform(0, 1) maps linearly to [t_lo[i], t_hi[i]]
  vector<lower=0, upper=1>[N] t_u;

  // Overall spatiotemporal scale
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

  // Realized language dates
  vector[N] t;
  for (i in 1:N) {
    t[i] = t_lo[i] + t_u[i] * (t_hi[i] - t_lo[i]);
  }

  // In this model tau is the spatiotemporal scale directly (no phylogeny to share with)
  real<lower=0> sigma_st = tau;

  // Separable spatiotemporal GP draw via Cholesky non-centered parameterization
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
  theta_indef    ~ normal(0, 1);

  w_def   ~ dirichlet(rep_vector(2.0, K - 1));
  w_indef ~ dirichlet(rep_vector(2.0, K - 1));

  eta_st    ~ std_normal();
  tau       ~ normal(0, 0.25) T[0,];
  sigma_eps ~ normal(0, 0.05) T[0,];
  ell_space ~ lognormal(log(0.3), 0.5);  // median ~300 km
  ell_time  ~ lognormal(log(4),   0.5);  // median ~4 centuries

  // Likelihood (no phylogenetic term)
  for (n in 1:N) {
    real eta_def_n = beta_zero_def
                     + g_def[n_cases[n]]
                     + f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_st[n];
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
                     + f_st[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_st[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // Sample variance of the spatiotemporal GP latent field
  real mean_f_st = 0;
  for (n in 1:N) mean_f_st += f_st[n];
  mean_f_st /= N;

  real ss_f_st = 0;
  for (n in 1:N) ss_f_st += square(f_st[n] - mean_f_st);

  real var_f_st = (N > 1) ? ss_f_st / (N - 1) : 0;
}
