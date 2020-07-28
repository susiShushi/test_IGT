// https://github.com/CCS-Lab/hBayesDM/tree/master/commons/stan_files
data {
  int<lower=1> N; //Sample size
  int<lower=1> T; //Trial数
  int<lower=1, upper=T> Tsubj[N]; //各Sampleの試行数
  int choice[N, T]; //N×Tの行列＝個人nの試行tでの選択
  real outcome[N, T]; //得られた額
}
transformed data{
  vector[4] initV; //各パラメータの初期値
  initV = rep_vector(0.0, 4); //0を4個並べたベクトルを作る（初期値はとりま0）
}
parameters {
  // ベクトル化のために全部ベクトルで宣言
  //集団レベルのパラメータ
  vector[4] mu_raw; //これがデルタモデルで計算する各パラメータと対応
  vector<lower=0>[4] sigma; //階層モデルでのmu_prの集団分散
  
  // 参加者レベルのraw パラメータ 
  // Matt trick = https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html
  // beta ~ normal(mu_beta, sigma_beta) -> beta = mu_beta + sigma_beta * beta_raw; + beta_raw ~ std_normal()
  vector[N] A_raw;
  vector[N] alpha_raw;
  vector[N] cons_raw;
  vector[N] lambda_raw;
}
transformed parameters {
  // rawパラメータの変換
  vector<lower=0, upper=1>[N]  A; //この周辺のパラメータ範囲はモデルに基づいてる
  vector<lower=0, upper=2>[N]  alpha;
  vector<lower=0, upper=5>[N]  cons;
  vector<lower=0, upper=10>[N] lambda;

  for (i in 1:N) { // ここはまじで再パラメタ化の処理
    A[i]      = Phi_approx(mu_raw[1] + sigma[1] * A_raw[i]); //Phi_approx = スケーリングされたロジスティック関数による近似値0-1でいい感じにする
    alpha[i]  = Phi_approx(mu_raw[2] + sigma[2] * alpha_raw[i]) * 2;
    cons[i]   = Phi_approx(mu_raw[3] + sigma[3] * cons_raw[i]) * 5;
    lambda[i] = Phi_approx(mu_raw[4] + sigma[4] * lambda_raw[i]) * 10;
  }
}
model {
  // Hyper parameters
  target += normal_lpdf(mu_raw|0,1); //標準正規分布
  target += normal_lpdf(sigma|0, 0.2); //これなんで0.2なんかかわからん。多分0-1の範囲やからかな？
  
  // raw parameter
  target += normal_lpdf(A_raw|0,1);
  target += normal_lpdf(alpha_raw|0,1);
  target += normal_lpdf(cons_raw|0,1);
  target += normal_lpdf(lambda_raw|0,1);
  
  for(i in 1:N){
    // 変数定義、こっから試行数の要素がゴリゴリ入ってくる＝強化学習の肝
    vector[4] ev;
    real curUtil; //都度計算される効用関数
    real theta; // theta = 3^cons - 1; モデルのアレ
    
    //初期値
    theta = pow(3, cons[i]) - 1; //pow() = 第一引数を字数にしたやつpow(x, y) = x^y
    ev = initV; // 効用関数の初期値をここに
    for(t in 1:Tsubj[i]){//個人の数だけ全試行分の計算ぶんまわす
    // softmax 選択ルール
      target += categorical_logit_lpmf(choice[i, t] | theta * ev);
      
      if(outcome[i, t] >= 0){// x(t) >= 0
        curUtil = pow(outcome[i,t], alpha[i]); 
      }else{ // x(t) < 0
        curUtil = -1*lambda[i]*pow(-1*outcome[i,t], alpha[i]); //pow内の-1は絶対値の処理用
      }
      
      // delta rule
      ev[choice[i,t]] += A[i] * (curUtil - ev[choice[i,t]]); // 更新、この段階でできあがるのがE_i(t+1) 
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_A;
  real<lower=0, upper=2>  mu_alpha;
  real<lower=0, upper=5>  mu_cons;
  real<lower=0, upper=10> mu_lambda;

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

  mu_A      = Phi_approx(mu_raw[1]);
  mu_alpha  = Phi_approx(mu_raw[2]) * 2;
  mu_cons   = Phi_approx(mu_raw[3]) * 5;
  mu_lambda = Phi_approx(mu_raw[4]) * 10;

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ev;
      real curUtil;     // utility of curFb
      real theta;       // theta = 3^c - 1

      // Initialize values
      log_lik[i] = 0;
      theta      = pow(3, cons[i]) -1;
      ev         = initV; // initial ev values

      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | theta * ev);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(softmax(theta * ev));

        if (outcome[i, t] >= 0) {  // x(t) >= 0
          curUtil = pow(outcome[i, t], alpha[i]);
        } else {                  // x(t) < 0
          curUtil = -1 * lambda[i] * pow(-1 * outcome[i, t], alpha[i]);
        }

        // delta
        ev[choice[i, t]] += A[i] * (curUtil - ev[choice[i, t]]);
      }
    }
  }
}
