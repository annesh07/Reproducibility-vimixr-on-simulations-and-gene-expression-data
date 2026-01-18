#Here are sample R codes used in the Curta cluster for result generation

#excel files for generating fig 1, fig 2 and fig S2
#the code:

# # Set the personal library path
# .libPaths(c("~/R/x86_64-pc-linux-gnu-library/4.4", .libPaths()))
# 
# # Load the package
# library(vimixr)
# library(openxlsx)
# library(mclust)
# # Create a new workbook
# wb <- createWorkbook()
# 
# # packages for parallelising
# library(doParallel)
# library(foreach)
# k0 <- seq(100, 1000, 100)
# K <- 2
# D <- 2
# Results <- array(0, c(100, 4, 10))
# 
# for (k in 1:10){
#   n_cores <- makeCluster(2)
#   registerDoParallel(n_cores)
#   Results[,,k] <- foreach(j=1:100, .combine=rbind, 
#                           .packages = c("vimixr", "mclust")) %dopar%
#     { 
#       N <- k0[k]
#       l_allot <- rbinom(N, K, 0.5)
#       X <- matrix(0, N, D)
#       for (n in 1:N){
#         X[n,] <- rnorm(D, 0, 0.5) + l_allot[n]*7
#       }
#       t0 <- as.integer(Sys.time())
#       R0 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001, 
#                              prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                              post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
#                              post_mean_eta = matrix(0.001, 20, ncol(X)),
#                              log_prob_matrix = t(apply(matrix(0.001, nrow(X), 20), 1, 
#                                                        function(x){x/sum(x)})), maxit = 1000,
#                              covariance_type="full",fixed_variance=FALSE,
#                              cluster_specific_covariance = TRUE,
#                              variance_prior_type = "sparse",
#                              prior_shape_d_cs_cov = 100000000,
#                              prior_rate_d_cs_cov = 100000000,
#                              prior_var_offd_cs_cov = 100000000,
#                              post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                              post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                              post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), 20)),
#                              scaling_cov_eta = 1)
#       t1 <- as.numeric(Sys.time())
#       tym <- t1 - t0
#       M0 <- length(R0$optimisation$ELBO) - 1
#       av_tym <- tym/M0
#       cl <- R0$posterior$'Cluster number'
#       pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#       ari <- mclust::adjustedRandIndex(l_allot, pred)
#       c(tym, av_tym, cl, ari) 
#     }
#   stopCluster(n_cores)
#   
#   sheet_name <- paste("Sheet", k)  
#   addWorksheet(wb, sheet_name)  
#   writeData(wb, sheet_name, Results[,,k])  
# }
# 
# 
# # Save the workbook
# saveWorkbook(wb, "varied_csSparse_N.xlsx", overwrite = TRUE)
# cat("Simulation complete. Results saved to varied_csSparse.xlsx\n")
# q("no")

#the above code generates varied.csSparse_N.xlsx. For different model choices,
#the user has to change the variables of cvi_npmm function in the vimixr 
#package. For other variables (different D and different K), one can fix N and
#change D or K accordingly. (For different N as above, D=2 and K=2 was fixed;
#for different D, N=100 and K=2 was fixed; and for different K, N=100 and 
#D=100 was fixed)


#excel files for figure 3
#the code:

# # Set the personal library path
# .libPaths(c("/gpfs/home/apal/R/x86_64-pc-linux-gnu-library/4.4", .libPaths()))
# 
# # Load the package
# library(vimixr)
# library(openxlsx)
# library(mclust)
# 
# library(doParallel)
# library(foreach)
# 
# # Create a new workbook
# wb <- createWorkbook()
# 
# n_cores <- makeCluster(4)
# registerDoParallel(n_cores)
# 
# N <- 100
# D <- 1000
# Results <- matrix(0, nrow = 1000, ncol = 4)
# 
#   l_allot <- rbinom(N, 2, 0.5)
#   X <- matrix(0, N, D)
#   for (n in 1:N){
#     X[n,] <- rnbinom(D, prob=0.5, size=1) + l_allot[n]*5
#   }
#   Plog <- matrix(runif(20*nrow(X), -1, -0.0006), nrow = nrow(X))
#   Results <- foreach(j=1:1000, .combine=rbind, 
#                          .packages = c("vimixr", "mclust"), .export = c("X","Plog","l_allot")) %dopar%
#     {
#       a <- runif(1, 0.001, 2000)
#       R0 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001, 
#                              prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                              post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
#                              post_mean_eta = matrix(0, 20, ncol(X)),
#                              log_prob_matrix = Plog, 
#                              maxit = 1000,
#                              covariance_type="full",fixed_variance=FALSE,
#                              cluster_specific_covariance = TRUE,
#                              variance_prior_type = "sparse",
#                              prior_shape_d_cs_cov = matrix(0.001, 1, 20),
#                              prior_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                              prior_var_offd_cs_cov = 100000,
#                              post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                              post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                              post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), 20)),
#                              scaling_cov_eta = a)
#       pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#       ari <- mclust::adjustedRandIndex(l_allot, pred)
#       last_opt <- R0$optimisation$ELBO[[length(R0$optimisation$ELBO)]]
#       VLL <- unname(last_opt)[4]
#       cl <- R0$posterior$`Cluster number`
#       
#       c(a, VLL, ari, cl)
#     }
#   
# stopCluster(n_cores)
# 
# sheet_name <- paste("Sheet", 1)  
# addWorksheet(wb, sheet_name)  
# writeData(wb, sheet_name, Results)  
# 
# # Save the workbook
# saveWorkbook(wb, "VLL_diffk0_NB.xlsx", overwrite = TRUE)
# cat("Simulation complete. Results saved to VLL_diffk0_NB.xlsx\n")
# q("no")

#the above code generates VLL_diffk0_NB.csv file (initially generated a 
#.xlsx file, processed later into .csv file). Similarly, for 1000 random values
#of a0=b0, similar results are generated


#excel files for figure 3
#the code:

# # Set the personal library path
# .libPaths(c("/gpfs/home/apal/R/x86_64-pc-linux-gnu-library/4.4", .libPaths()))
# 
# # Load the package
# library(vimixr)
# library(openxlsx)
# library(mclust)
# 
# library(doParallel)
# library(foreach)
# 
# # Create a new workbook
# wb <- createWorkbook()
# 
# n_cores <- makeCluster(2)
# registerDoParallel(n_cores)
# 
# N <- 100
# D <- 1000
# T0 <- 20
# l_allot <- rbinom(N, 2, 0.5)
# X <- matrix(0, N, D)
# for (n in 1:N){
#   X[n,] <- rnorm(D, 0, 0.5) + l_allot[n]*5
# }
# Results <- matrix(0, nrow = 1000, ncol = 4)
# 
# Results <- foreach(j=1:1000, .combine=rbind, 
#                    .packages = c("vimixr", "mclust")) %dopar%
#   {
#     Plog <- matrix(runif(T0*nrow(X), -1, -0.0006), nrow = nrow(X))
#     R0 <- vimixr::cvi_npmm(X, variational_params = T0, prior_shape_alpha = 0.001, 
#                            prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                            post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
#                            post_mean_eta = matrix(0, T0, ncol(X)),
#                            log_prob_matrix = Plog, 
#                            maxit = 1000,
#                            covariance_type="full",fixed_variance=FALSE,
#                            cluster_specific_covariance = TRUE,
#                            variance_prior_type = "sparse",
#                            prior_shape_d_cs_cov = matrix(0.001, 1, T0),
#                            prior_rate_d_cs_cov = matrix(0.001, T0, ncol(X)),
#                            prior_var_offd_cs_cov = 100000,
#                            post_shape_d_cs_cov = matrix(0.001, 1, T0),
#                            post_rate_d_cs_cov = matrix(0.001, T0, ncol(X)),
#                            post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), T0)),
#                            scaling_cov_eta = 1)
#     pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#     ari <- mclust::adjustedRandIndex(l_allot, pred)
#     last_opt <- R0$optimisation$ELBO[[length(R0$optimisation$ELBO)]]
#     ELBO <- sum(last_opt)
#     VLL <- unname(last_opt)[4]
#     cl <- R0$posterior$`Cluster number`
#     
#     c(ELBO, VLL, ari, cl)
#   }
# stopCluster(n_cores)
# 
# sheet_name <- paste("Sheet", 1)  # Naming sheet
# addWorksheet(wb, sheet_name)  # Add a new sheet
# writeData(wb, sheet_name, Results)  # Write the 2D matrix to the sheet
# 
# 
# # Save the workbook
# saveWorkbook(wb, "ELBOvsVLL.xlsx", overwrite = TRUE)
# cat("Simulation complete. Results saved to ELBOvsVLL.xlsx\n")
# q("no")