library(rstan)

behav_dat <- read.csv("all100.csv", header=T) # from Streingover et al. (2014)

N = nrow(behav_dat)/100

pre_choice = t(matrix(0, N, 100))
pre_gain = t(matrix(0, N, 100))
pre_loss = t(matrix(0, N, 100))
pre_outcome = t(matrix(0, N, 100))

for(n in 1:504){
  for(t in 1:100){
    pre_choice[t, n] <- behav_dat$choice[t+(n-1)*100]
    pre_gain[t, n] <- behav_dat$gain[t+(n-1)*100]
    pre_loss[t, n] <- abs(behav_dat$loss[t+(n-1)*100])
    pre_outcome[t, n] <- behav_dat$gain[t+(n-1)*100]- abs(behav_dat$loss[t+(n-1)*100])
  }
}

payscale = 100

stan_data <- list(N = nrow(behav_dat)/100, T = 100,
                  Tsubj = rep(100,N), choice = t(pre_choice), gain = t(pre_gain)/payscale,
                  loss = t(pre_loss)/payscale,
                  outcome = t(pre_outcome)/payscale,
                  sign_out = sign(t(pre_outcome)/payscale))


model_vse <- stan_model("Stan/igt_vse.stan")
model_delta <- stan_model("Stan/igt_delta.stan")
model_vpp <- stan_model("Stan/igt_vpp.stan")

fit_vse <- sampling(model_vse, data=stan_data, cores = 4,
                    control=list(adapt_delta=0.99))

fit_delta <- sampling(model_delta, data=stan_data, cores = 4,
                    control=list(adapt_delta=0.99))

fit_vpp <- sampling(model_vpp, data=stan_data, cores = 4,
                      control=list(adapt_delta=0.99))
