D0 <- seq(100, 1000, 100)
l_allot <- c(rep(0,33), rep(1,33), rep(2,34))
N <- 100

Results_d <- matrix(0, 10, 100)
for (i in 1:10){
  D <- D0[i]
  X <- matrix(0, N, D)
  for (n in 1:N){
    X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
  }
  
  m <- rep(0, 100)
  for (j in 1:100){
    m0 <- as.numeric(Sys.time())
    R0 <- vimixr::cvi_npmm(X, variational_params = 10, prior_shape_alpha = 0.001, 
                           prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
                           post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
                           post_mean_eta = matrix(0, 10, ncol(X)),
                           log_prob_matrix = NULL,
                           maxit = 1000,
                           n_inits = 1,
                           covariance_type="full",fixed_variance=FALSE,
                           cluster_specific_covariance = TRUE,
                           variance_prior_type = "sparse",
                           prior_shape_d_cs_cov = matrix(15199.11, 1, 10),
                           prior_rate_d_cs_cov = matrix(15199.11, 10, ncol(X)),
                           prior_var_offd_cs_cov = 100000,
                           post_shape_d_cs_cov = matrix(0.001, 1, 10),
                           post_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
                           post_var_offd_cs_cov = matrix(0.001, 10, 3),
                           scaling_cov_eta = (nrow(X)+1))
    m1 <- as.numeric(Sys.time())
    
    m[j] <- (m1 - m0)/R0$optimisation$Iterations
  }
  
  Results_d[i,] <- m
}  
  

N0 <- seq(100, 1000, 100)
D <- 100

Results_n <- matrix(0, 10, 100)
for (i in 1:10){
  N <- N0[i]
  l_allot <- c(rep(0,N/2), rep(1,N/2))
  X <- matrix(0, N, D)
  for (n in 1:N){
    X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
  }
  
  m <- rep(0, 100)
  for (j in 1:100){
    m0 <- as.numeric(Sys.time())
    R0 <- vimixr::cvi_npmm(X, variational_params = 10, prior_shape_alpha = 0.001, 
                           prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
                           post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
                           post_mean_eta = matrix(0, 10, ncol(X)),
                           log_prob_matrix = NULL,
                           maxit = 1000,
                           n_inits = 1,
                           covariance_type="full",fixed_variance=FALSE,
                           cluster_specific_covariance = TRUE,
                           variance_prior_type = "sparse",
                           prior_shape_d_cs_cov = matrix(0.001, 1, 10),
                           prior_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
                           prior_var_offd_cs_cov = 100000,
                           post_shape_d_cs_cov = matrix(0.001, 1, 10),
                           post_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
                           post_var_offd_cs_cov = matrix(0.001, 10, 3),
                           scaling_cov_eta = (nrow(X)+1))
    m1 <- as.numeric(Sys.time())
    
    m[j] <- (m1 - m0)/R0$optimisation$Iterations
  }
  
  Results_n[i,] <- m
} 

#plots
library(ggplot2)
df_n <- data.frame(N    = N0[as.vector(row(Results_n))],
                   time = as.vector(Results_n))

p1 <- ggplot(df_n, aes(N, time)) +
  geom_boxplot(aes(group = N), fill          = "lightgrey",
               colour        = "maroon") +
  geom_smooth(method = "lm", se = TRUE, colour = "darkblue", linewidth = 1) +
  labs(x = "N", y = "Time per iteration (s)")+
  ggtitle("(a) Dependency on N ~ O(N)")+
  scale_x_continuous(labels = scales::label_scientific())+
  theme_minimal()

df_d <- data.frame(D    = D0[as.vector(row(Results_d))],
                   time = as.vector(Results_d))

p2<- ggplot(df_d, aes(D^2, time)) +
  geom_boxplot(aes(group = D), fill          = "lightgrey",
               colour        = "maroon") +                 
  geom_smooth(method = "lm", se = TRUE, colour = "darkblue", linewidth = 1) +        
  labs(x = expression(d^2), y = "Time per iteration (s)")+
  ggtitle("(b) Dependency on d ~ O(d^2)")+
  scale_x_continuous(labels = scales::label_scientific())+
  theme_minimal()

library(patchwork)
p1+p2



  
