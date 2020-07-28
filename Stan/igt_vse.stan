//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N; // nrow(all)
  int<lower=1> T; // 100
  int<lower=1, upper=T> Tsubj[N]; //subjID
  int choice[N, T]; //Deck
  real gain[N, T]; //これどうする？
  real loss[N, T]; //abs 入れる
}
transformed data{
  vector[4] initV; //各Deckの初期値
  initV = rep_vector(0.0, 4); //0を4個並べたベクトルを作る（初期値はとりま0）
}
parameters {
  // ベクトル化のために全部ベクトルで宣言
  //集団レベルのパラメータ
  vector[5] mu_raw; //モデルで計算する各パラメータと対応
  vector<lower=0>[5] sigma; //階層モデルでのmu_prの集団分散
  
  // 参加者レベルのraw パラメータ 
  // Matt trick = https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html
  // beta ~ normal(mu_beta, sigma_beta) -> beta = mu_beta + sigma_beta * beta_raw; + beta_raw ~ std_normal()
  vector[N] theta_raw;
  vector[N] delta_raw;
  vector[N] alpha_raw;
  vector[N] phi_raw;
  vector[N] cons_raw;
}
transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N]  theta;
  vector<lower=0, upper=1>[N]  delta;
  vector<lower=0, upper=1>[N]  alpha;
  vector<lower=0, upper=5>[N]  cons;
  vector[N] phi;
  
  for (i in 1:N) {
    theta[i]      = Phi_approx(mu_raw[1] + sigma[1] * theta_raw[i]);
    delta[i]  = Phi_approx(mu_raw[2] + sigma[2] * delta_raw[i]);
    alpha[i]   = Phi_approx(mu_raw[3] + sigma[3] * alpha_raw[i]);
    cons[i] = Phi_approx(mu_raw[5] + sigma[5] * cons_raw[i]) * 5;
  }
  phi = mu_raw[4] + sigma[4] * phi_raw;
}

model {
  // Hyperparameters
  target += normal_lpdf(mu_raw|0, 1.0);
  target += normal_lpdf(sigma[1:3]|0, 0.2);
  target += cauchy_lpdf(sigma[4]|0, 1.0);
  target += normal_lpdf(sigma[5]|0, 0.2);

  // individual parameters
  target += normal_lpdf(theta_raw|0, 1.0);
  target += normal_lpdf(delta_raw|0, 1.0);
  target += normal_lpdf(alpha_raw|0, 1.0);
  target += normal_lpdf(phi_raw|0, 1.0);
  target += normal_lpdf(cons_raw|0, 1.0);
  

  for (i in 1:N) {//個人ごとに回す
    // Define values
    vector[4] ev; //
    vector[4] explore;   // explore
    vector[4] V;   // sum of ev and explore

    real curUtil;     // utility of curFb
    real rand;       // rand = 3^c - 1

    // Initialize values
    rand = pow(3, cons[i]) -1;
    ev    = initV; // initial ev values
    explore  = initV; // initial explore values
    V     = initV;
    
    
    for (t in 1:Tsubj[i]) {
      // softmax choice
      target += categorical_logit_lpmf(choice[i, t]|rand * V);
      
      curUtil = pow(gain[i, t], theta[i]) - pow(loss[i,t], theta[i]);

      explore += alpha[i] * (phi[i] - explore); // 
      explore[choice[i,t]]  = 0; // 
      
      
      ev *= delta[i];
      ev[choice[i, t]] += curUtil;
    
      
      // calculate V
      V = ev + explore;
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_theta;
  real<lower=0, upper=1>  mu_delta;
  real<lower=0, upper=1>  mu_alpha;
  real<lower=0, upper=5>  mu_cons;
  real mu_phi;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  //real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  //for (i in 1:N) {
  //  for (t in 1:T) {
  //    y_pred[i, t] = -1;
  //  }
  //}

  mu_theta  = Phi_approx(mu_raw[1]);
  mu_delta  = Phi_approx(mu_raw[2]);
  mu_alpha  = Phi_approx(mu_raw[3]);
  mu_cons = Phi_approx(mu_raw[5]) * 5;
  mu_phi = mu_raw[4];

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ev; //
      vector[4] explore;   // explore
      vector[4] V;   // sum of ev and explore

      real curUtil;     // utility of curFb
      real rand;       // rand = 3^c - 1

      // Initialize values
      rand = pow(3, cons[i]) -1;
      ev    = initV; // initial ev values
      explore  = initV; // initial explore values
      V     = initV;

      log_lik[i] = 0;
      
      
      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_logit_lpmf(choice[i, t]|rand * V);
      
        curUtil = pow(gain[i, t], theta[i]) - pow(loss[i,t], theta[i]);

        explore += alpha[i] * (phi[i] - explore); // to zero
        explore[choice[i,t]]  = 0; //+= alpha[i] * (phi[i] - explore[choice[i,t]]); 
      
      
        ev *= delta[i];
        ev[choice[i, t]] += curUtil;
    
      
        // calculate V
        V = ev + explore;
      }
    }
  }
}
