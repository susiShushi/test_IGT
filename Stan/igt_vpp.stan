data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int choice[N, T];
  real outcome[N, T];
}

transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[8] mu_pr;
  vector<lower=0>[8] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] A_pr;
  vector[N] alpha_pr;
  vector[N] cons_pr;
  vector[N] lambda_pr;
  vector[N] epP_pr;
  vector[N] epN_pr;
  vector[N] K_pr;
  vector[N] w_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N]  A;
  vector<lower=0, upper=2>[N]  alpha;
  vector<lower=0, upper=5>[N]  cons;
  vector<lower=0, upper=10>[N] lambda;
  vector[N] epP;
  vector[N] epN;
  vector<lower=0, upper=1>[N] K;
  vector<lower=0, upper=1>[N] w;

  for (i in 1:N) {
    A[i]      = Phi_approx(mu_pr[1] + sigma[1] * A_pr[i]);
    alpha[i]  = Phi_approx(mu_pr[2] + sigma[2] * alpha_pr[i]) * 2;
    cons[i]   = Phi_approx(mu_pr[3] + sigma[3] * cons_pr[i]) * 5;
    lambda[i] = Phi_approx(mu_pr[4] + sigma[4] * lambda_pr[i]) * 10;
    K[i]      = Phi_approx(mu_pr[7] + sigma[7] * K_pr[i]);
    w[i]      = Phi_approx(mu_pr[8] + sigma[8] * w_pr[i]);
  }
  epP = mu_pr[5] + sigma[5] * epP_pr;
  epN = mu_pr[6] + sigma[6] * epN_pr;
}

model {
  // Hyperparameters
  target += normal_lpdf(mu_pr|0, 1.0);
  target += normal_lpdf(sigma[1:4]|0, 0.2);
  target += cauchy_lpdf(sigma[5:6]|0, 1.0);
  target += normal_lpdf(sigma[7:8]|0, 0.2);

  // individual parameters
  target += normal_lpdf(A_pr|0, 1.0);
  target += normal_lpdf(alpha_pr|0, 1.0);
  target += normal_lpdf(cons_pr|0, 1.0);
  target += normal_lpdf(lambda_pr|0, 1.0);
  target += normal_lpdf(epP_pr|0, 1.0);
  target += normal_lpdf(epN_pr|0, 1.0);
  target += normal_lpdf(K_pr|0, 1.0);
  target += normal_lpdf(w_pr|0, 1.0);

  for (i in 1:N) {
    // Define values
    vector[4] ev;
    vector[4] p_next;
    vector[4] str;
    vector[4] pers;   // perseverance
    vector[4] V;   // weighted sum of ev and pers

    real curUtil;     // utility of curFb
    real theta;       // theta = 3^c - 1

    // Initialize values
    theta = pow(3, cons[i]) -1;
    ev    = initV; // initial ev values
    pers  = initV; // initial pers values
    V     = initV;

    for (t in 1:Tsubj[i]) {
      // softmax choice
      target += categorical_logit_lpmf(choice[i, t]|theta * V);

      // perseverance decay
      pers *= K[i]; // decay

      if (outcome[i, t] >= 0) {  // x(t) >= 0
        curUtil = pow(outcome[i, t], alpha[i]);
        pers[choice[i, t]] += epP[i];  // perseverance term
      } else {                  // x(t) < 0
        curUtil = -1 * lambda[i] * pow(-1 * outcome[i, t], alpha[i]);
        pers[choice[i, t]] += epN[i];  // perseverance term
      }

      ev[choice[i, t]] += A[i] * (curUtil - ev[choice[i, t]]);
      // calculate V
      V = w[i] * ev + (1-w[i]) * pers;
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_A;
  real<lower=0, upper=2>  mu_alpha;
  real<lower=0, upper=5>  mu_cons;
  real<lower=0, upper=10> mu_lambda;
  real mu_epP;
  real mu_epN;
  real<lower=0, upper=1> mu_K;
  real<lower=0, upper=1> mu_w;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_A      = Phi_approx(mu_pr[1]);
  mu_alpha  = Phi_approx(mu_pr[2]) * 2;
  mu_cons   = Phi_approx(mu_pr[3]) * 5;
  mu_lambda = Phi_approx(mu_pr[4]) * 10;
  mu_epP    = mu_pr[5];
  mu_epN    = mu_pr[6];
  mu_K      = Phi_approx(mu_pr[7]);
  mu_w      = Phi_approx(mu_pr[8]);

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ev;
      vector[4] p_next;
      vector[4] str;
      vector[4] pers;   // perseverance
      vector[4] V;   // weighted sum of ev and pers

      real curUtil;     // utility of curFb
      real theta;       // theta = 3^c - 1

      // Initialize values
      log_lik[i] = 0;
      theta      = pow(3, cons[i]) -1;
      ev         = initV; // initial ev values
      pers       = initV; // initial pers values
      V          = initV;

      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | theta * V);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(softmax(theta * V));

        // perseverance decay
        pers *= K[i]; // decay

        if (outcome[i, t] >= 0) {  // x(t) >= 0
          curUtil = pow(outcome[i, t], alpha[i]);
          pers[choice[i, t]] += epP[i];  // perseverance term
        } else {                  // x(t) < 0
          curUtil = -1 * lambda[i] * pow(-1 * outcome[i, t], alpha[i]);
          pers[choice[i, t]] += epN[i];  // perseverance term
        }

        ev[choice[i, t]] += A[i] * (curUtil - ev[choice[i, t]]);
        // calculate V
        V = w[i] * ev + (1-w[i]) * pers;
      }
    }
  }
}
