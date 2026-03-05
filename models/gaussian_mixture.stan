// ============================================================
// Gaussian Mixture Model in Stan
// ============================================================
// This model fits a K-component Gaussian mixture to 1D data.
// Key challenge: label switching — components can swap identities
// across chains. Solved by constraining mu to be ordered<>.

data {
  int<lower=1> N;          // number of observations
  int<lower=1> K;          // number of mixture components
  vector[N] y;             // observed data
}

parameters {
  // ordered<> enforces mu[1] < mu[2] < ... < mu[K]
  // This kills label switching 
  ordered[K] mu;

  // simplex sums to 1 and each element is >= 0
  // Perfect type for mixture weights
  simplex[K] theta;

  // Standard deviations must be positive
  vector<lower=0>[K] sigma;
}

model {
  // ---- PRIORS ----
  // Weakly informative priors centered on the data range
  mu    ~ normal(0, 10);        // broad prior on component means
  theta ~ dirichlet(rep_vector(1.0, K));  // uniform over simplex
  sigma ~ exponential(1);       // prefer smaller sigmas, but flexible

  // ---- LIKELIHOOD ----
  // For each observation, sum over K components (marginalize out
  // the latent component assignment z_i).
  //
  // log p(y_i) = log Σ_k theta[k] * Normal(y_i | mu[k], sigma[k])
  //
  // Stan's log_sum_exp is numerically stable log of sum of exponentials.
  // We work in log space to avoid underflow with small probabilities.

  for (n in 1:N) {
    vector[K] log_contributions;

    for (k in 1:K) {
      // log(theta[k]) + log Normal(y[n] | mu[k], sigma[k])
      log_contributions[k] = log(theta[k])
                             + normal_lpdf(y[n] | mu[k], sigma[k]);
    }

    // Add log p(y_n) to the accumulated log posterior
    target += log_sum_exp(log_contributions);
  }
}

generated quantities {
  // SOFT ASSIGNMENTS: posterior probability that observation n
  // belongs to component k. Useful for clustering.
  matrix[N, K] soft_z;

  // LOG LIKELIHOOD: used for model comparison (LOO-CV, WAIC)
  vector[N] log_lik;

  for (n in 1:N) {
    vector[K] log_contributions;

    for (k in 1:K) {
      log_contributions[k] = log(theta[k])
                             + normal_lpdf(y[n] | mu[k], sigma[k]);
    }

    // Normalize to get posterior component probabilities
    soft_z[n] = to_row_vector(softmax(log_contributions));

    // Store log-likelihood for each observation
    log_lik[n] = log_sum_exp(log_contributions);
  }
}
