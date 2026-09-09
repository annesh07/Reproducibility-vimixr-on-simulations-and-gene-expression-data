#install required packages
packages <- c("MetBrewer", "patchwork", "mclust", "readxl", 
              "dplyr", "ggplot2", "tidyr", "scales", "NPflow", "colorspace", 
              "readr", "viridis", "circlize", "rlang", "gridtext",
              "grid", "gridExtra", "dbscan", "HDclassif", "FNN", "cluster", 
              "reticulate", "igraph", "utils", "GEOquery", "Matrix", "reshape2",
              "irlba")

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}
for (p in packages){
  library(p, character.only = TRUE)
}

#install vimixr latest version
remotes::install_github("annesh07/vimixr", force = TRUE)
library(vimixr)

#install biomaRt, ComplexHeatmap and GEOquery
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(ask = FALSE)
BiocManager::install("biomaRt", ask = FALSE, force = TRUE)
BiocManager::install("ComplexHeatmap", ask = FALSE, force = TRUE)
BiocManager::install("GEOquery", ask = FALSE, force = TRUE)

library(biomaRt)
library(ComplexHeatmap)
library(GEOquery)
##fig 1
#results from Curta Cluster 
#for different N=100 to N=1000 (10 sheets each), with fixed D=2 and K=2, 
#and 100 different initialisations for the log of latent  probability 
#allocation matrix Plog; the output contains 4 columns namely total run-time, 
#average run-time per iteration, number of clusters & ARI respectively. 

fixed_diagonal <- read_excel("Results/fixed_diagonal_N.xlsx", sheet = 1)
fixed_full <- read_excel("Results/fixed_full_N.xlsx", sheet = 1)
varied_IW <- read_excel("Results/varied_IW_N.xlsx", sheet = 1)
varied_decomposed <- read_excel("Results/varied_decomposed_N.xlsx", sheet = 1)
varied_diagonal <- read_excel("Results/varied_diagonal_N.xlsx", sheet = 1)
varied_csIW <- read_excel("Results/varied_csIW_N.xlsx", sheet = 1)
varied_csSparse <- read_excel("Results/varied_csSparse_N.xlsx", sheet = 1)
varied_csoffD <- read_excel("Results/varied_csoffD_N.xlsx", sheet = 1)

a_dat <- data.frame(c(fixed_diagonal[1:100,3], fixed_full[1:100,3], varied_diagonal[1:100,3], varied_IW[1:100,3],
                      varied_decomposed[1:100,3], varied_csIW[1:100,3], varied_csSparse[1:100,3], varied_csoffD[1:100,3]))
colnames(a_dat) <- c("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")

#create a long data frame with all the columns one after another
a_dat_df <- pivot_longer(a_dat, cols = everything(), names_to = "Model", values_to = "Value")

my_col <- c(met.brewer("Signac")[3],met.brewer("Signac")[6],met.brewer("Signac")[4],met.brewer("Signac")[5],met.brewer("Signac")[10],met.brewer("Signac")[13],met.brewer("Signac")[12],met.brewer("Signac")[11])
p1 <- ggplot(a_dat_df, aes(x = Model, y = Value, color = Model)) +
  geom_boxplot(fill = "grey88") +
  ggtitle(expression("(a) " * K[post] * " boxplots")) +
  scale_color_manual(values = my_col) +
  theme_minimal() +
  labs(x = "",
       y = "Posterior Cluster number (averaged)") +
  theme(plot.title = element_text(size = 18)) +
  theme(axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 12),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank()) +
  theme(legend.position = "none")

a2_dat <- data.frame(c(fixed_diagonal[1:100,4], fixed_full[1:100,4], varied_diagonal[1:100,4], varied_IW[1:100,4],
                       varied_decomposed[1:100,4], varied_csIW[1:100,4], varied_csSparse[1:100,4], varied_csoffD[1:100,4]))
colnames(a2_dat) <- c("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")


df_long <- pivot_longer(a2_dat, cols = everything(), names_to = "Model", values_to = "Value")

# Boxplot
p2 <- ggplot(df_long, aes(x = Model, y = Value, color = Model)) +
  geom_boxplot(fill = "grey88") +
  ggtitle("(b) ARI boxplots") +
  scale_color_manual(values = my_col) + 
  theme_minimal() +
  labs(x = "", y = "ARI")+
  theme(plot.title = element_text(size = 18)) +
  theme(axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 12),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank())
Fig_1 <- p1|p2

# ggsave("Fig_1.pdf", plot = Fig_1, device = "pdf", path = "Results/Figures",
#        width = 5.76, height = 4.20, units = "in")
Fig_1
#generates Fig_1.pdf in Results/Figures folder


##fig 2
#results from Curta Cluster
#Models M6, M7 and M8 are compared for their performances over different true
#cluster K = {4,6,8,10} (corresponding to 4 sheets), with fixed N=100 and D=100
varied_csIW <- read_excel("Results/varied_csIW_K.xlsx", sheet = 1)
varied_csSparse <- read_excel("Results/varied_csSparse_K.xlsx", sheet = 1)
varied_csoffD <- read_excel("Results/varied_csoffD_K.xlsx", sheet = 1)
df1 <- data.frame(c(varied_csIW[1:100,4], varied_csSparse[1:100,4], varied_csoffD[1:100,4]))
colnames(df1) <- c("M6", "M7", "M8")

varied_csIW <- read_excel("Results/varied_csIW_K.xlsx", sheet = 2)
varied_csSparse <- read_excel("Results/varied_csSparse_K.xlsx", sheet = 2)
varied_csoffD <- read_excel("Results/varied_csoffD_K.xlsx", sheet = 2)
df2 <- data.frame(c(varied_csIW[1:100,4], varied_csSparse[1:100,4], varied_csoffD[1:100,4]))
colnames(df2) <- c("M6", "M7", "M8")

varied_csIW <- read_excel("Results/varied_csIW_K.xlsx", sheet = 3)
varied_csSparse <- read_excel("Results/varied_csSparse_K.xlsx", sheet = 3)
varied_csoffD <- read_excel("Results/varied_csoffD_K.xlsx", sheet = 3)
df3 <- data.frame(c(varied_csIW[1:100,4], varied_csSparse[1:100,4], varied_csoffD[1:100,4]))
colnames(df3) <- c("M6", "M7", "M8")

varied_csIW <- read_excel("Results/varied_csIW_K.xlsx", sheet = 4)
varied_csSparse <- read_excel("Results/varied_csSparse_K.xlsx", sheet = 4)
varied_csoffD <- read_excel("Results/varied_csoffD_K.xlsx", sheet = 4)
df4 <- data.frame(c(varied_csIW[1:100,4], varied_csSparse[1:100,4], varied_csoffD[1:100,4]))
colnames(df4) <- c("M6", "M7", "M8")

df1$Source <- "DF1"
df2$Source <- "DF2"
df3$Source <- "DF3"
df4$Source <- "DF4"

#combine for a long format
combined_df <- bind_rows(df1, df2, df3, df4)

#create a long data frame with all the columns one after another
long_df <- pivot_longer(combined_df, 
                        cols = -Source, 
                        names_to = "Model", 
                        values_to = "Value")

my_cols <- c(met.brewer("Signac")[13],met.brewer("Signac")[12],met.brewer("Signac")[11])

Fig_2 <- ggplot(long_df, aes(x = Source, y = Value, color = Model)) +
  geom_boxplot(fill = "grey88",position = position_dodge(0.8), width = 0.7) +
  theme_minimal() +
  labs(title = "ARI boxplots for cluster-specific models",
       x = " ",
       y = "ARI") +
  scale_fill_manual(values = my_cols) +
  scale_color_manual(values = my_cols) +
  scale_x_discrete(labels = c(expression(K[true]==4), expression(K[true]==6),
                              expression(K[true]==8), expression(K[true]==10))) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  theme(plot.title = element_text(size = 18),
        axis.text.x = element_text(size = 14, face = "bold"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14), 
        legend.title = element_text(size = 12),             
        legend.text  = element_text(size = 10),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank())

# ggsave("Fig_2.pdf", plot = Fig_2, device = "pdf", path = "Results/Figures",
#        width = 5.76, height = 4.20, units = "in")
Fig_2
#generates Fig_2.pdf in Results/Figures folder



##fig Supplementary 1
#results from Curta Cluster
# comparing ELBO vs VLL scores for a fixed N=100, D=1000, K=3 Gaussian 
#simulations for 1000 random initialisations of Plog. Output are 4 columns 
#namely ELBO values, VLL values, ARI scores and #clusters estimated
elbovsvll <- read_excel("Results/ELBOvsVLL.xlsx", sheet = 1)
df <- data.frame(elbovsvll)

my_col2 <- met.brewer("Nizami")[c(2, 8, 6)]
p1 <- ggplot(df, aes(x = as.factor(Cluster), y = ELBO, color = as.factor(Cluster))) +
  geom_boxplot(fill = "grey88") +
  ggtitle("ELBO values for random initialisations") +
  theme_minimal() +
  labs(x = expression(K[post]),
       y = "ELBO") +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  
        axis.title.x = element_text(size = 16, hjust = 0.5), 
        axis.title.y = element_text(size = 14, hjust = 0.5),
        axis.text.x = element_text(size = 12, face="bold", hjust = 0.5),
        legend.title = element_text(size = 14),             
        legend.text  = element_text(size = 14),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank()) +
  theme(legend.position = "none")+
  scale_color_manual(values = my_col2)

p2 <- ggplot(df, aes(x = as.factor(Cluster), y = VLL, color = as.factor(Cluster))) +
  geom_boxplot(fill = "grey88") +
  ggtitle("Variational log likelihood for random initialisations") +
  theme_minimal() +
  labs(x = expression(K[post]),
       y = expression(sum(E[q] * "[ " * log * "(p(X | Z)) ]")),
       colour = expression(K[post])) +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  
        axis.title.x = element_text(size = 16, hjust = 0.5),
        axis.title.y = element_text(size = 14, hjust = 0.5),
        axis.text.x = element_text(size = 12, face="bold", hjust = 0.5),
        legend.title = element_text(size = 14),             
        legend.text  = element_text(size = 12),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank()) +
  scale_color_manual(values = my_col2)
Fig_S1 <- p1|p2

# ggsave("Fig_S1.pdf", plot = Fig_S1, device = "pdf", path = "Results/Figures",
#        width = 12.50, height = 6.75, units = "in")
Fig_S1
#generates Fig_S1.pdf in Results/Figures folder


##fig Supplementary 2
#results from Curta Cluster
#for negative binomial simulations with fixed N = 100, D = 1000 and K = 3,
#hyper-parameters a0=b0 and k0 are varied from 0.001 to 2000 for 1000 times
#and corresponding VLL values calculated. Output are 3 columns, namely 
#x = random a0=b0 (correspondingly k0) values, y = VLL value and 
#Cluster = number of posterior clusters
vlla0 <- read.csv("Results/VLL_diffa0b0_NB.csv")
dfa0 <- data.frame(vlla0)

vllk0 <- read.csv("Results/VLL_diffk0_NB.csv")
dfk0 <- data.frame(vllk0)

my_col1 <- met.brewer("Nizami")[c(1,2,6,8,5)]
p1 <- ggplot(dfa0, aes(x = x, y = y, color = as.factor(Cluster))) +
  geom_point(size = 3, alpha = 0.9) +
  theme_minimal() +
  labs(x = expression("(a) " *a[0] == b[0] %in% "(" * 0.001 * ", " * 2000 * ")"),
       y = "Variational Log Likelihood (VLL)",
       colour = expression(K[post])) +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  
        axis.title.x = element_text(size = 16, hjust = 0.5),
        axis.title.y = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 14),             
        legend.text  = element_text(size = 14),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank()) +
  scale_color_manual(values = my_col1)

my_col2 <- met.brewer("Nizami")[c(1,2,6)]
p2 <- ggplot(dfk0, aes(x = x, y = y, color = as.factor(Cluster))) +
  geom_point(size = 3, alpha = 0.8) +
  theme_minimal() +
  labs(x = expression("(b) " *k[0] %in% "(" * 0.001 * ", " * 2000 * ")"),
       y = "",
       colour = expression(K[post])) +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  
        axis.title.x = element_text(size = 16, hjust = 0.5),             
        legend.title = element_text(size = 14),             
        legend.text  = element_text(size = 14),
        panel.grid.major.x = element_blank(),   
        panel.grid.minor.x = element_blank()) +
  scale_color_manual(values = my_col2)
Fig_S2 <- p1|p2

# ggsave("Fig_S2.pdf", plot = Fig_3, device = "pdf", path = "Results/Figures",
#        width = 8.75, height = 5.0, units = "in")
Fig_S2
#generates Fig_S2.pdf in Results/Figures folder

##fig Supplementary 3
# N <- 100
# D <- 1000
# l_allot <- c(rep(0,33), rep(1,33), rep(2,34))
# set.seed(30032026)
# X <- matrix(0, N, D)
# for (n in 1:N){
#   X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
# }
# 
# Results <- matrix(0, 1000, 7)
# for (m in 1:1000){
#     a <- runif(1, 0.001, 2000)
#     b <- runif(1, 0.001, 2000)
#     Plog <- matrix(runif(20*nrow(X), -1, -0.0006), nrow = nrow(X))
#     R0 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001,
#                            prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                            post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                            post_mean_eta = matrix(0, 20, ncol(X)),
#                            log_prob_matrix = Plog, maxit = 1000, n_inits = 1,
#                            covariance_type="full",fixed_variance=FALSE,
#                            cluster_specific_covariance = TRUE,
#                            variance_prior_type = "sparse",
#                            prior_shape_d_cs_cov = matrix(a, 1, 20),
#                            prior_rate_d_cs_cov = matrix(b, 20, ncol(X)),
#                            prior_var_offd_cs_cov = 100000,
#                            post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                            post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                            post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), 20)),
#                            scaling_cov_eta = (nrow(X)+1))
#     pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#     ari <- mclust::adjustedRandIndex(l_allot, pred)
#     last_opt <- R0$optimisation$ELBO[[length(R0$optimisation$ELBO)]]
#     VLL <- unname(last_opt)[4]
#     cl <- R0$posterior$`Cluster number`
#     elb <- sum(last_opt)
#     k0 <- cl*D + cl*(D*(D+1)/2) + cl-1
#     Bic <- -2*VLL + k0*log(N)
# 
# 
#     Results[m, ] <- c(a, b, VLL, elb, ari, cl, Bic)
# }
# write.csv(Results, file = "Grid_diffa0b0_fixedk0.csv", row.names = FALSE)
# 
# Results <- matrix(0, 1000, 7)
# for (m in 1:1000){
#     a <- runif(1, 0.001, 2000)
#     k <- runif(1, 0.001, 2000)
#     Plog <- matrix(runif(20*nrow(X), -1, -0.0006), nrow = nrow(X))
#     R0 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001,
#                            prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                            post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                            post_mean_eta = matrix(0, 20, ncol(X)),
#                            log_prob_matrix = Plog, maxit = 1000, n_inits = 1,
#                            covariance_type="full",fixed_variance=FALSE,
#                            cluster_specific_covariance = TRUE,
#                            variance_prior_type = "sparse",
#                            prior_shape_d_cs_cov = matrix(a, 1, 20),
#                            prior_rate_d_cs_cov = matrix(a, 20, ncol(X)),
#                            prior_var_offd_cs_cov = 100000,
#                            post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                            post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                            post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), 20)),
#                            scaling_cov_eta = k)
#     pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#     ari <- mclust::adjustedRandIndex(l_allot, pred)
#     last_opt <- R0$optimisation$ELBO[[length(R0$optimisation$ELBO)]]
#     VLL <- unname(last_opt)[4]
#     cl <- R0$posterior$`Cluster number`
#     elb <- sum(last_opt)
#     k0 <- cl*D + cl*(D*(D+1)/2) + cl-1
#     Bic <- -2*VLL + k0*log(N)
# 
# 
#     Results[m, ] <- c(a, k, VLL, elb, ari, cl, Bic)
# }
# write.csv(Results, file = "Grid_diffa0=b0_diffk0.csv", row.names = FALSE)

Results <- read.csv("Results/Grid_diffa0b0_fixedk0.csv")
Results$Clusters <- as.factor(Results$CL)
Results1 <- read.csv("Results/Grid_diffa0=b0_diffk0.csv")
Results1$Clusters <- as.factor(Results1$Clusters)

library(ggplot2)
library(MetBrewer)
my_col1 <- met.brewer("Nizami")[c(1,2,6,8)]
p1 <- ggplot(Results, aes(x = a0, y = b0)) +
  
  geom_point(aes(colour = Clusters, shape = Clusters), size = 2.75, alpha = 0.8) +
  geom_rug(aes(colour = Clusters), show.legend=FALSE, linewidth=0.0002)+
  scale_color_manual(values = my_col1) +
  scale_shape_manual(values=c(15, 4,3,20)) +
  labs(
    title = expression(K[post] * " corresponding to different " * a[0] * ", " * 
                         b[0] * " (fixed " * k[0] * ")"),
    x = expression(a[0]),
    y = expression(b[0])) +
  theme_minimal() +
  theme(
    legend.position = "right",             
    legend.box = "vertical",               
    aspect.ratio = 1                       
  )

my_col2 <- met.brewer("Nizami")[c(2,6,8,3,5,7,1)]
p2 <- ggplot(Results1, aes(x = A, y = K)) +
  
  geom_point(aes(colour = Clusters, shape = Clusters), size = 2.75, alpha = 0.8) +
  geom_rug(aes(colour = Clusters), show.legend=FALSE, linewidth=0.0002)+
  scale_color_manual(values = my_col2) +
  scale_shape_manual(values=c( 4,3,20, 17, 19, 6, 15)) +
  labs(
    title = expression(K[post] * " corresponding to different " * a[0] *"="*b[0] * 
                         " and " * k[0]),
    x = expression(a[0]*"="*b[0]),
    y = expression(k[0])) +
  theme_minimal() +
  theme(
    legend.position = "right",             
    legend.box = "vertical",               
    aspect.ratio = 1                       
  )

Fig_S3 <- p1+p2

# ggsave("Fig_S3.pdf", plot = Fig_S3, device = "pdf", path = "Results/Figures",
#        width = 12, height = 6, units = "in")
Fig_S3
#generates Fig_S3.pdf in Results/Figures folder


##fig Supplementary 4
#mean average run time of Sparse DPMM calculated by taking the sample average
#of 100 simulation runs for different N, values can be found in 
#Results_n.csv corresponding to 10 sample size N; similar results for comparing 
#speed across dimension D in Results_d.csv

# D0 <- seq(100, 1000, 100)
# l_allot <- c(rep(0,33), rep(1,33), rep(2,34))
# N <- 100
# 
# Results_d <- matrix(0, 10, 100)
# for (i in 1:10){
#   D <- D0[i]
#   X <- matrix(0, N, D)
#   for (n in 1:N){
#     X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
#   }
#   
#   m <- rep(0, 100)
#   for (j in 1:100){
#     m0 <- as.numeric(Sys.time())
#     R0 <- vimixr::cvi_npmm(X, variational_params = 10, prior_shape_alpha = 0.001, 
#                            prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                            post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
#                            post_mean_eta = matrix(0, 10, ncol(X)),
#                            log_prob_matrix = NULL,
#                            maxit = 1000,
#                            n_inits = 1,
#                            covariance_type="full",fixed_variance=FALSE,
#                            cluster_specific_covariance = TRUE,
#                            variance_prior_type = "sparse",
#                            prior_shape_d_cs_cov = matrix(15199.11, 1, 10),
#                            prior_rate_d_cs_cov = matrix(15199.11, 10, ncol(X)),
#                            prior_var_offd_cs_cov = 100000,
#                            post_shape_d_cs_cov = matrix(0.001, 1, 10),
#                            post_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
#                            post_var_offd_cs_cov = matrix(0.001, 10, 3),
#                            scaling_cov_eta = (nrow(X)+1))
#     m1 <- as.numeric(Sys.time())
#     
#     m[j] <- (m1 - m0)/R0$optimisation$Iterations
#   }
#   
#   Results_d[i,] <- m
# }  
# 
# 
# N0 <- seq(100, 1000, 100)
# D <- 100
# 
# Results_n <- matrix(0, 10, 100)
# for (i in 1:10){
#   N <- N0[i]
#   l_allot <- c(rep(0,N/2), rep(1,N/2))
#   X <- matrix(0, N, D)
#   for (n in 1:N){
#     X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
#   }
#   
#   m <- rep(0, 100)
#   for (j in 1:100){
#     m0 <- as.numeric(Sys.time())
#     R0 <- vimixr::cvi_npmm(X, variational_params = 10, prior_shape_alpha = 0.001, 
#                            prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                            post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)), 
#                            post_mean_eta = matrix(0, 10, ncol(X)),
#                            log_prob_matrix = NULL,
#                            maxit = 1000,
#                            n_inits = 1,
#                            covariance_type="full",fixed_variance=FALSE,
#                            cluster_specific_covariance = TRUE,
#                            variance_prior_type = "sparse",
#                            prior_shape_d_cs_cov = matrix(0.001, 1, 10),
#                            prior_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
#                            prior_var_offd_cs_cov = 100000,
#                            post_shape_d_cs_cov = matrix(0.001, 1, 10),
#                            post_rate_d_cs_cov = matrix(0.001, 10, ncol(X)),
#                            post_var_offd_cs_cov = matrix(0.001, 10, 3),
#                            scaling_cov_eta = (nrow(X)+1))
#     m1 <- as.numeric(Sys.time())
#     
#     m[j] <- (m1 - m0)/R0$optimisation$Iterations
#   }
#   
#   Results_n[i,] <- m
# } 

N0 <- seq(100, 1000, 100)
Results_n <- as.matrix(read.csv("Results/Results_n.csv"))
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

D0 <- N0
Results_d <- as.matrix(read.csv("Results/Results_d.csv"))
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
Fig_S4 <- p1|p2

# ggsave("Fig_S4.pdf", plot = Fig_S4, device = "pdf", path = "Results/Figures",
#        width = 7.50, height = 5.00, units = "in")
Fig_S4
#generates Fig_S4.pdf in Results/Figures folder


##fig Supplementary 5
#comparison between vimixr and an MCMC splice sampling technique, 
#implemented using DPMGibbsN function from NPflow package; due to time taken
#for microbenching, the results are provided as violinplot.csv
#the code
set.seed(08012026)
N <- 100
D <- 100
T0 <- 20
l_allot <- rbinom(N, 1, 0.5)
X <- matrix(0, N, D)
for (n in 1:N){
  X[n,] <- rnorm(D, 0, 0.5) + l_allot[n]*5
}
Plog <- matrix(runif(2000, -5, -0.001), nrow = N)

#NPflow (Boris)
hyperG0 <- list()
hyperG0[["mu"]] <- rep(0,D)
hyperG0[["kappa"]] <- 0.001
hyperG0[["nu"]] <- D+2
hyperG0[["lambda"]] <- diag(D)/10
a <- 0.0001
b <- 0.0001
N <- 1000
nbclust_init <- 20
# microbencmarking (commented code below) takes ~2-3 hours, so results
# of microbenchmarking attached and used. The code for microbenchmarking results:
# violinplot <- microbenchmark::microbenchmark(DPMGibbsN(t(X), hyperG0, a, b, N, doPlot = F),
#                                              vimixr::cvi_npmm(X, variational_params = T0, prior_shape_alpha = 0.001,
#                                                               prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                                                               post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                                                               post_mean_eta = matrix(0, T0, ncol(X)),
#                                                               log_prob_matrix = Plog,
#                                                               maxit = 1000,
#                                                               covariance_type="full",fixed_variance=FALSE,
#                                                               cluster_specific_covariance = TRUE,
#                                                               variance_prior_type = "sparse",
#                                                               prior_shape_d_cs_cov = matrix(0.001, 1, T0),
#                                                               prior_rate_d_cs_cov = matrix(0.001, T0, ncol(X)),
#                                                               prior_var_offd_cs_cov = 100000,
#                                                               post_shape_d_cs_cov = matrix(0.001, 1, T0),
#                                                               post_rate_d_cs_cov = matrix(0.001, T0, ncol(X)),
#                                                               post_var_offd_cs_cov = matrix(0.001, T0, 3),
#                                                               scaling_cov_eta = 1))
# levels(violinplot$expr) <- c("DPMGibbsN", "Sparse DPMM")
#available results
violinplot = read.csv("Results/violinplot.csv")

violinplot$time <- violinplot$time/1e+9
df <- as.data.frame(violinplot)
violin_col <- met.brewer("Hokusai2")[c(2,5)]
Fig_S5 <- ggplot(df, aes(x=expr, y=time, fill = expr)) +
  geom_violin(trim=FALSE) +
  scale_y_log10(
    breaks = scales::log_breaks(base = 10),
    labels = scales::label_number()
  ) +
  annotation_logticks(
    sides = "l",
    scaled = TRUE
  ) +
  labs(x = "",
       y = "Execution time in seconds (log-scale)") + 
  scale_fill_manual(values = violin_col) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12, face = "bold"))

# ggsave("Fig_S5.pdf", plot = Fig_S5, device = "pdf", path = "Results/Figures",
#        width = 5.76, height = 4.20, units = "in")
Fig_S5
#generates Fig_S5.pdf in Results/Figures folder


#Leukemia data implementation
data1 <- read_delim("https://schlieplab.org/Static/Supplements/CompCancer/Affymetrix/armstrong-2002-v2/armstrong-2002-v2_database.txt", delim = "\t", col_names = TRUE)
gene1 <- data1[2:2195,1]
Y0 <- data1[2:2195, 2:73]
Y <- (matrix(as.numeric(t(Y0)), nrow = dim(Y0)[2]))
#normalising the data
Y3 <- t(apply(log2(Y), 1, FUN = function(x){(x-mean(x))/sqrt(var(x))}))
#labelled Leukemia subtypes
tag1 <- as.character(data1[1, 2:73])

##fig 3
#code for implementation with empirical Bayes hyper-parameters
# R0 <- vimixr::cvi_npmm(Y3, variational_params = 20, prior_shape_alpha = 0.001, 
#                          prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                          post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(Y3)), 
#                          post_mean_eta = matrix(0, 20, ncol(Y3)),
#                          log_prob_matrix = NULL,
#                          maxit = 1000,
#                          n_inits = 5,
#                          Seed = c(1106663, 7735431, 4490956, 322613, 3476196),
#                          covariance_type="full",fixed_variance=FALSE,
#                          cluster_specific_covariance = TRUE,
#                          variance_prior_type = "sparse",
#                          prior_shape_d_cs_cov = matrix(28.90762, 1, 20),
#                          prior_rate_d_cs_cov = matrix(28.90762, 20, ncol(Y3)),
#                          prior_var_offd_cs_cov = 100000,
#                          post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                          post_rate_d_cs_cov = matrix(0.001, 20, ncol(Y3)),
#                          post_var_offd_cs_cov = matrix(0.001, 20, 3),
#                          scaling_cov_eta = (nrow(Y3)+1))
# pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)

pred <- c(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
          2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1)
#label correction for plot conjugacy 
pred[which(pred==1)]="A"
pred[which(pred==2)]="B"
pred[which(pred=="A")]=2
pred[which(pred=="B")]=1

pca <- prcomp(Y3)
#variance
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
pc1_pct <- round(var_explained[1] * 100, 2)
pc2_pct <- round(var_explained[2] * 100, 2)
#the plot
pca_df <- data.frame("PC1" = pca$x[,1], "PC2" = pca$x[,2], 
                     "Cluster" = as.factor(tag1))
my_col_pca <- c(met.brewer("Benedictus")[3], met.brewer("Renoir")[9], 
                met.brewer("Signac")[10])
ggplot_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, 
                                 color = Cluster, shape = Cluster)) +
  geom_point(size = 3, alpha = 1) +
  labs(title = "(a) PCA projection: Labelled sub-types", 
       x = paste0("PC 1 (", pc1_pct, "%)"), 
       y = paste0("PC 2 (", pc2_pct, "%)")) +
  theme_minimal() +
  scale_color_manual(values = my_col_pca) + 
  theme(plot.title = element_text(face = "bold"))

pca_df_pred <- data.frame("PC1" = pca$x[,1], "PC2" = pca$x[,2], 
                          "Cluster" = as.factor(pred))
my_col_pca_pred <- c(met.brewer("Signac")[4], met.brewer("VanGogh2")[4], 
                     met.brewer("Manet")[11])
ggplot_pca_pred <- ggplot(pca_df_pred, aes(x = PC1, y = PC2, 
                                           color = Cluster, shape = Cluster)) +
  geom_point(size = 3, alpha = 1) +
  labs(title = "(b) PCA projection: Sparse DPMM clusters", 
       x = paste0("PC 1 (", pc1_pct, "%)"), 
       y = paste0("PC 2 (", pc2_pct, "%)")) +
  theme_minimal() +
  scale_color_manual(values = my_col_pca_pred)+ 
  theme(plot.title = element_text(face = "bold"))
Fig_3 <- ggplot_pca | ggplot_pca_pred 

# ggsave("Fig_3.pdf", plot = Fig_4, device = "pdf", path = "Results/Figures",
#        width = 8.75, height = 5, units = "in")
Fig_3
#generates Fig_3.pdf in Results/Figures folder


#fig 4a
#code for implementation with empirical Bayes hyper-parameters
# R0 <- vimixr::cvi_npmm(Y3, variational_params = 20, prior_shape_alpha = 0.001, 
#                          prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
#                          post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(Y3)), 
#                          post_mean_eta = matrix(0, 20, ncol(Y3)),
#                          log_prob_matrix = NULL,
#                          maxit = 1000,
#                          n_inits = 5,
#                          Seed = c(2484881, 7893038, 8404945, 4769778, 1279802),
#                          covariance_type="full",fixed_variance=FALSE,
#                          cluster_specific_covariance = TRUE,
#                          variance_prior_type = "sparse",
#                          prior_shape_d_cs_cov = matrix(10, 1, 20),
#                          prior_rate_d_cs_cov = matrix(10, 20, ncol(Y3)),
#                          prior_var_offd_cs_cov = 100000,
#                          post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                          post_rate_d_cs_cov = matrix(0.001, 20, ncol(Y3)),
#                          post_var_offd_cs_cov = matrix(0.001, 20, 3),
#                          scaling_cov_eta = (nrow(Y3)+1))
# pred4 <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)

pred4 <- c(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
           2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3,
           1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1)
pred4[which(pred4==1)]="A"
pred4[which(pred4==2)]="B"
pred4[which(pred4=="A")]=2
pred4[which(pred4=="B")]=1

pca_df_pred4 <- data.frame("PC1" = pca$x[,1], "PC2" = pca$x[,2], 
                           "Cluster" = as.factor(pred4))
my_col_pca_pred4 <- c(met.brewer("Signac")[4], met.brewer("VanGogh2")[4], 
                      met.brewer("Manet")[11], lighten(met.brewer("Klimt")[6], 
                                                       amount = 0.3))
Fig_4a <- ggplot(pca_df_pred4, aes(x = PC1, y = PC2, 
                                             color = Cluster, shape = Cluster)) +
  geom_point(size = 3, alpha = 0.9) +
  labs(title = "PCA projection: Sparse DPMM clusters", 
       x = paste0("PC 1 (", pc1_pct, "%)"), 
       y = paste0("PC 2 (", pc2_pct, "%)")) +
  theme_minimal() +
  scale_color_manual(values = my_col_pca_pred4) + 
  scale_shape_manual(values = c(16,17,15,18)) +
  theme(plot.title = element_text(face = "bold"))

# ggsave("Fig_4a.pdf", plot = Fig_5, device = "pdf", path = "Results/Figures",
#        width = 5.76, height = 4.20, units = "in")
Fig_4a
#generates Fig_4a.pdf in Results/Figures folder


##fig 4b
#using BIOCONDUCTR for full signature of labelled sub-types as per Armstrong 
#and comparing those signatures for the estimated clusters with weaker 
#hyper-priors ,i.e., pred4
pred <- pred4
data1 <- read_delim("https://schlieplab.org/Static/Supplements/CompCancer/Affymetrix/armstrong-2002-v2/armstrong-2002-v2_database.txt", delim = "\t", col_names = TRUE)
fulld <- read_delim("https://pubs.broadinstitute.org/mpr/projects/Leukemia/expression_data.txt", delim = "\t", col_names = TRUE)
gene <- fulld[1:12582,1]$Name
Y0 <- fulld[1:12582, 3:74]
Y <- (matrix(as.numeric(t(Y0)), nrow = dim(Y0)[2]))
Y <- t(apply(Y, 1, FUN = function(x){(x-mean(x))/sqrt(var(x))}))
tag1 <- as.character(data1[1, 2:73])

ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
results <- getBM(
  attributes = c(
    "affy_hg_u95av2",        
    "external_gene_name",    
    "ensembl_gene_id",
    "embl"                   
  ),
  filters = "affy_hg_u95av2",
  values = gene,    
  mart = ensembl
)
embl <- results$embl
gname <- results$external_gene_name
avl_probe_ind0 <- lapply(c("J03779", "L33930", "Y12735", "M11722", "X83441", "AF032885", "M96803", "AB020674", "X59350", "Z49194", "U48959", "U29175", "AF054825",
                           "U02687", "AB007888", "AJ001687", "AF009615", "AF027208", "AB028948", "AL050157", "Z48579", "AF026816", "AB023137", "X61118",
                           "X04325", "X64364", "X99906", "M63138", "M84526", "U35117", "U41843", "Y08134", "M22324", "AC005787", "AF004222", "U05569"), function(x){which(results$embl==x)})

avl_probe_ind <- c(avl_probe_ind0[1:24], list(which(gname=="CCNA1")), avl_probe_ind0[25:52])
avl_probe_ind <- unlist(avl_probe_ind)
avl_probe <- results$affy_hg_u95av2[avl_probe_ind][c(c(1:9),c(11:26),c(28:38),c(40:51))]
d_ALL <- t(Y[, unlist(lapply(avl_probe, function(x){which(gene==x)}))])
d_ALL[,c(17,24)] <- d_ALL[,c(24,17)]
d_ALL[,c(41,44)] <- d_ALL[,c(44,41)]
d_ALL[,c(45,52)] <- d_ALL[,c(52,45)]
d_ALL[,c(46,55)] <- d_ALL[,c(55,46)]
pred[c(17,24)] <- pred[c(24,17)]
pred[c(41,44)] <- pred[c(44,41)]
pred[c(45,52)] <- pred[c(52,45)]
pred[c(46,55)] <- pred[c(55,46)]
rownames(d_ALL) <- c("MME(CD10) 1389_at", "CD24 266_s_at", "DYRK3 39931_at", "DNTT(TDT) 34168_at", "LIG4 963_at", "FOXO1A(FKHR) 40570_at", "SPTBN1 39556_at", "SPTBN1 39452_s_at", "KIAA0867 35260_at", "CD22 38522_s_at", "CD22 38521_at", "POU2AF1 36239_at", "MYLK(MLCK) 32847_at", "SMARCA4 32579_at", "VAMP5 32533_s_at",
                     "FLT3 34583_at", "FLT3 1065_at", "KIAA0428 34306_at", "NKG2D 36777_at", "ADAM10 40798_s_at", "ADAM10 40797_at", "PROML1(AC133) 41470_at", "KIAA1025 34785_at", "CCNA1 34833_at", "DKFZp586o0120 40798_s_at", "ADAM10 35801_at", "ITPA 35985_at", "K1AA0920 32184_at", "LMO2 1914_at",
                     "GJB1 39598_at", "BSG 36162_at", "ENSA 39011_at", "ENSA 39010_at", "ENSA 39012_g_at", "CTSD 239_at", "DF 40282_s_at", "TFDP2 34741_at", "TFDP2 633_s_at", "TFDP2 2013_at", "DRAP1 39076_s_at", "PDE3B 37779_at", "ANPEP 39385_at", "FZR1(Chromosome 19 clone) 41623_s_at", "FZR1(Chromosome 19 clone) 41624_r_at", "FZR1(Chromosome 19 clone) 39855_at", "RTN2 34408_at", "CRYAA 33311_at", "CRYAA 33313_g_at")
colnames(d_ALL) <- as.factor(pred)

make_html_label <- function(x) {
  parts <- strsplit(x, " ")[[1]]
  last  <- parts[length(parts)]
  first <- paste(parts[-length(parts)], collapse = " ")
  
  if (first == "") return(sprintf('<b>%s</b>', last))
  
  sprintf('<b>%s</b> <i>%s</i>', first, last)
}

row_html <- sapply(rownames(d_ALL), make_html_label)

levs <- levels(as.factor(pred))
col_map <- setNames(scales::hue_pal()(length(levs)), levs)

met_colors_disc <- met.brewer("OKeeffe1")[rev(c(2,3,4,6,8,9,10))]
col_fun <- colorRamp2(breaks = seq(-6.1, 6.1, length.out = 7),
                      colors = met_colors_disc)

ht <- Heatmap(d_ALL, name = "Expression level",
              column_dend_reorder = FALSE,
              col = col_fun, 
              column_split = factor(c(rep("ALL", 24), rep("MLL", 20), rep("AML", 28)), levels = c("ALL", "MLL", "AML")),
              cluster_columns = FALSE,
              cluster_column_slices = FALSE,
              cluster_rows = FALSE,
              height = unit(16, "cm"),                 
              width  = unit(20, "cm"),
              row_labels = gt_render(row_html),   
              row_names_gp = gpar(fontsize = 9, parse = TRUE),
              column_names_gp = gpar(
                col = my_col_pca_pred4[as.factor(pred)],
                fontsize = 11.5), 
              column_names_rot = 0,
              column_names_centered = TRUE,
              heatmap_legend_param = list(
                title_position = "topcenter",
                title_gp = gpar(fontsize = 11),   # legend title size
                labels_gp = gpar(fontsize = 10),
                at = c(-6.1, -4, -2, 0, 2, 4, 6.1),
                labels = c("-6", "-4", "-2", "0", "2", "4", "6") # legend labels size
              )
)

Fig_4b <- draw(ht, column_title = "Sparse DPMM clusters",
     column_title_side = "bottom",
     column_title_gp = gpar(fontsize = 18, fontface = "bold"))

# pdf("Results/Figures/Fig_4b.pdf", width = 11.75, height = 7.50)
# dev.off()

#generates Fig_4b.pdf in Results/Figures folder


##comparison with s.o.t.a techniques 
X <- dist(Y3)
X0 <- prcomp(Y3, rank. = 10)$x
sil_width_kmeans <- c()
sil_width_dbscan <- rep(0, 9)
sil_width_hdbscan <- rep(0, 9)
sil_width_sNNclust <- rep(0, 9)
bic_hddc <- rep(0, 9)
bic_hddc_kmeans <- rep(0, 9)
mod_leiden <- rep(0, 9)

set.seed(05122005)
for(k in 1:9){
  k0 <- k + 1
  #Silhouette coeff. for k-means
  km <- kmeans(Y3, centers = k0, nstart = 1)
  sil <- silhouette(km$cluster, X)
  sil_width_kmeans[k] <- mean(sil[, 3])
  
  #dbscan
  kdist <- kNNdist(Y3, k = k0)
  kdist_sorted <- sort(kdist, decreasing = FALSE)
  curv <- c(0, abs(diff(kdist_sorted, differences = 2)), 0)
  knee_idx <- which.max(curv)
  eps_curvature <- kdist_sorted[knee_idx]
  cl_dbscan <- dbscan(Y3, eps = eps_curvature, minPts = (k0+1))$cluster
  sil1 <- silhouette(cl_dbscan, X)
  if (length(sil1)==1&all(is.na(sil1))){
    sil_width_dbscan[k] <- NA
  } else {sil_width_dbscan[k] <- mean(sil1[, 3])}
   
  
  #hdbscan
  cl_hdbscan <- hdbscan(Y3, minPts = k0)$cluster
  sil2 <- silhouette(cl_hdbscan, X)
  sil_width_hdbscan[k] <- mean(sil2[, 3]) 
  
  #sNNclust
  snn_mat <- sNN(Y3, k = k0)$shared
  eps <- floor(k0/2)
  cl_sNNclust <- sNNclust(Y3, k=k0, eps=eps, minPts=eps+1, borderPoints = T)$cluster 
  sil3 <- silhouette(cl_sNNclust, X)
  sil_width_sNNclust[k] <- mean(sil3[, 3])
  
  #hddc
  r_hddc <- hddc(Y3, K = k0, model="ALL", init = "random")
  bic_hddc[k] <- r_hddc$BIC
  
  r_hddc_kmeans <- hddc(Y3, K=k0, model="ALL", init = "kmeans")
  bic_hddc_kmeans[k] <- r_hddc_kmeans$BIC
  
  #Leiden, modularity based k selection
  knn_res <- get.knn(Y3, k = k0)
  edges <- cbind(
    rep(1:nrow(Y3), each = k0),
    as.vector(t(knn_res$nn.index))
  )
  g <- graph_from_edgelist(edges, directed = FALSE)
  g <- simplify(g)
  mod_leiden[k] <- cluster_leiden(g, objective_function = "modularity")$quality
}

algo_names <- c("DBSCAN", "HDBSCAN", "sNNclust", "HDDC (random)", "HDDC (k-means)", 
                "Leiden", "K-means")
opt_k <- rep(0, 7)
opt_k[1] <- which.max(sil_width_dbscan) + 1
opt_k[2] <- which.max(sil_width_hdbscan) + 1
opt_k[3] <- which.max(sil_width_sNNclust) + 1
opt_k[4] <- which.max(bic_hddc) + 1
opt_k[5] <- which.max(bic_hddc_kmeans) + 1
opt_k[6] <- which.max(mod_leiden) + 1
opt_k[7] <- which.max(sil_width_kmeans) + 1

#supplementary table tab_s3
tab_S3 <- data.frame(algo_names, opt_k)
names(tab_S3) <- c("Clustering method", "k_opt")
# write.csv(tab_S3, file="Results/Tables/tab_S3.csv")
View(tab_S3)

#these provide the optimal k corresponding to every method 
#(Supplementary Table S1), which is used to evaluate the Leukemia data, 
#and compared based on external metrics: posterior clusters and ARI


cl_models <- rep(0 , 8)
ari_models <- rep(0 , 8)
time_models <- rep(0 , 8)
iteration_models <- rep(0 , 8)

#dbscan
set.seed(05122005)
t0 <- as.numeric(Sys.time())
k=opt_k[1]
kdist <- kNNdist(Y3, k = k)      
kdist_sorted <- sort(kdist, decreasing = FALSE)
curv <- c(0, abs(diff(kdist_sorted, differences = 2)), 0)   
knee_idx <- which.max(curv)
eps_curvature <- kdist_sorted[knee_idx]
cl_dbscan <- dbscan(Y3, eps = eps_curvature, minPts = k+1)$cluster
t1 <- as.numeric(Sys.time())
cl_models[1] <- length(unique(cl_dbscan))
time_models[1] <- t1 - t0
iteration_models[1] <- 1 #non-iterative implementation
ari_models[1] <- mclust::adjustedRandIndex(tag1, cl_dbscan)

#hdbscan
set.seed(05122005)
t0 <- as.numeric(Sys.time())
k=opt_k[2]
cl_hdbscan <- hdbscan(Y3, minPts = k)$cluster
t1 <- as.numeric(Sys.time())
cl_models[2] <- length(unique(cl_hdbscan))
time_models[2] <- t1 - t0
iteration_models[2] <- 1 #non-iterative implementation
ari_models[2] <- mclust::adjustedRandIndex(tag1, cl_hdbscan)

#sNNclust
set.seed(05122005)
t0 <- as.numeric(Sys.time())
k=opt_k[3]
snn_mat <- sNN(Y3, k = k)$shared
eps <- floor(quantile(snn_mat, 0.9))
cl_sNNclust <- sNNclust(Y3, k=k, eps=eps, minPts=eps+3, borderPoints = T)$cluster 
t1 <- as.numeric(Sys.time())
cl_models[3] <- length(unique(cl_sNNclust))
time_models[3] <- t1 - t0
iteration_models[3] <- 1 #non-iterative implementation
ari_models[3] <- mclust::adjustedRandIndex(tag1, cl_sNNclust)

#hddc
set.seed(05122005)
k=opt_k[4]
cl_hddc_m <- hddc(Y3, K = k, model="ALL", init = "random")$model #best model
t0 <- as.numeric(Sys.time())
cl_hddc <- hddc(Y3, K = k, model=cl_hddc_m, init = "random")
t1 <- as.numeric(Sys.time())
cl_models[4] <- length(unique(cl_hddc$class))
time_models[4] <- t1 - t0
iteration_models[4] <- length(cl_hddc$loglik_all)
ari_models[4] <- mclust::adjustedRandIndex(tag1, cl_hddc$class)

set.seed(05122005)
k=opt_k[5]
cl_hddc_km <- hddc(Y3, K = k, model="ALL", init = "kmeans")$model #best model
t0 <- as.numeric(Sys.time())
cl_hddc_kmeans <- hddc(Y3, K = k, model="ABKQKD", init = "kmeans")
t1 <- as.numeric(Sys.time())
cl_models[5] <- length(unique(cl_hddc_kmeans$class))
time_models[5] <- t1 - t0
iteration_models[5] <- length(cl_hddc_kmeans$loglik_all)
ari_models[5] <- mclust::adjustedRandIndex(tag1, cl_hddc_kmeans$class)

#Leiden
set.seed(05122005)
t0 <- as.numeric(Sys.time())
k=opt_k[6]
knn_res <- get.knn(Y3, k = k)
edges <- cbind(
  rep(1:nrow(Y3), each = k),
  as.vector(t(knn_res$nn.index))
)
g <- graph_from_edgelist(edges, directed = FALSE)
g <- simplify(g)
cl_leiden <- cluster_leiden(g, objective_function = "modularity")
t1 <- as.numeric(Sys.time())
cl_models[6] <- cl_leiden$nb_clusters
time_models[6] <- t1 - t0
iteration_models[6] <- 2 #by default
ari_models[6] <- mclust::adjustedRandIndex(tag1, cl_leiden$membership)

#k-means
set.seed(05122005)
t0 <- as.numeric(Sys.time())
k=opt_k[7]
Km <- kmeans(Y3, centers = k, nstart = 1, trace = 1)
t1 <- as.numeric(Sys.time())
cl_models[7] <- length(unique(Km$cluster))
time_models[7] <- t1 - t0
iteration_models[7] <- Km$iter
ari_models[7] <- mclust::adjustedRandIndex(tag1, Km$cluster)

##fig 5
#comparison plots based on average run-time, number of posterior clusters 
#and ARI scores; the metrics values used based on the above results and 
#pred output (for Sparse DPMM, implementation below)
set.seed(05122005)
M0 <- as.integer(Sys.time())
R0 <- vimixr::cvi_npmm(Y3, variational_params = 20, prior_shape_alpha = 0.001,
                      prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
                      post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(Y3)),
                      post_mean_eta = matrix(0, 20, ncol(Y3)),
                      log_prob_matrix = NULL,
                      maxit = 1000,
                      n_inits = 1,
                      Seed = c(1106663),
                      covariance_type="full",fixed_variance=FALSE,
                      cluster_specific_covariance = TRUE,
                      variance_prior_type = "sparse",
                      prior_shape_d_cs_cov = matrix(28.90762, 1, 20),
                      prior_rate_d_cs_cov = matrix(28.90762, 20, ncol(Y3)),
                      prior_var_offd_cs_cov = 100000,
                      post_shape_d_cs_cov = matrix(0.001, 1, 20),
                      post_rate_d_cs_cov = matrix(0.001, 20, ncol(Y3)),
                      post_var_offd_cs_cov = matrix(0.001, 20, 3),
                      scaling_cov_eta = (nrow(Y3)+1))
M1 <- as.integer(Sys.time())
pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
cl_models[8] <- length(unique(pred))
time_models[8] <- M1-M0
iteration_models[8] <- R0$optimisation$Iterations - 1 #1st iteration is random starting point
ari_models[8] <- mclust::adjustedRandIndex(tag1, pred)

algo_names[8] <- "Sparse DPMM"
time_models <- round(time_models, digits = 2)

#generating figure 5 
n_algos <- length(ari_models)

data <- data.frame(
  algorithm = factor(algo_names, levels = algo_names),
  ari = ari_models,
  clusters = cl_models,
  time = time_models,
  iterations = iteration_models
)

algo_colors <- met.brewer("Hiroshige", n = 10)[c(1,2,3,4,7,8,9,10)]
names(algo_colors) <- algo_names

create_runtime_inset <- function(algo_name, time, iterations, color) {
  bar_data <- data.frame(
    x = 1:iterations,
    y = 1
  )
  
  bar_positions <- seq(3, 3 + (iterations - 1) * 0.17, length.out = iterations)
  bar_data_fixed <- data.frame(
    x = bar_positions,
    y = 1
  )
  
  bar_width <- 0.1
  
  p <- ggplot(bar_data_fixed, aes(x = x, y = y)) +
    geom_col(fill = color, width = bar_width, color = NA) +
    annotate("text", x = 1.5, y = 0.75, label = paste0(algo_name, ":"), 
             hjust = 0.5, size = 3.5, fontface = "bold", color = "gray20") +
    annotate("text", x = 5.5, y = 0.75, 
             label = time, 
             hjust = 0.5, size = 3, color = "gray20") +
    ylim(0, 1.5) +
    xlim(0, 6) +  
    theme_void() +
    theme(
      plot.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      plot.margin = margin(3, 3, 3, 3)
    ) +
    coord_cartesian(clip = "off")  
  
  return(p)
}

inset_plots <- list()
for (i in 1:n_algos) {
  inset_plots[[i]] <- create_runtime_inset(
    algo_names[i], 
    data$time[i], 
    data$iterations[i],
    algo_colors[i]
  )
}

inset_grid <- wrap_plots(inset_plots[c(1,5,2,6,3,7,4,8)], nrow = 4, ncol = 2)

p_clusters <- ggplot(data, aes(x = algorithm, y = clusters, fill = algorithm)) +
  geom_segment(aes(xend = algorithm, yend = 0), color = algo_colors, linewidth = 0.8, alpha = 0.6) +
  geom_point(color = algo_colors, size = 6, alpha = 1) +
  labs(x = NULL, y = "Number of Clusters", title = "b) Estimated Clusters") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 10, face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0, size = 14),
    plot.margin = margin(10, 10, 10, 10)
  )

p_ari <- ggplot(data, aes(x = algorithm, y = ari, fill = algorithm)) +
  geom_col(width = 0.7, color = "white", linewidth = 0.5) +
  scale_fill_manual(values = algo_colors) +
  labs(x = NULL, y = "ARI", title = "(c) ARI Score") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 10, face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0, size = 14),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  ylim(0, 1)

p0 <- p_clusters | p_ari

#final plot
Fig_5 <- inset_grid / p0 + 
  plot_layout(heights = c(1, 3)) +
  plot_annotation(
    title = "Benchmark of Clustering Techniques",
    subtitle = "a) Runtime (in seconds)",
    theme = theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, face = "bold", hjust = 0, 
                                   margin = margin(t = 5, b = 5))
    )
  )

# ggsave("Fig_5.pdf", plot = Fig_7, device = "pdf", path = "Results/Figures",
#        width = 12, height = 7.75, units = "in")
Fig_5
#generates Fig_5.pdf in Results/Figures folder

##fig S6
#the data is stored as getGEO("GSE116256", GSEMatrix = FALSE)
m <- as.matrix(read.table(gzfile("GSE116256/raw/GSM3587950_AML419A-D0.dem.txt.gz"), header = TRUE, row.names = 1, sep = "\t", 
                          check.names = FALSE))
anno <- read.table(gzfile("GSE116256/raw/GSM3587951_AML419A-D0.anno.txt.gz"), header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, check.names = FALSE)
E <- t(t(m) / anno$TranscriptomeUMIs) * 10000
E_log <- log1p(E)
m0 <- scale(t(E_log))

g180 <- read_excel("Results/mmc3.xlsx")
g180 <- c(g180$`Tumor-derived, per cell-type`[2:31], g180$...9[2:31], 
          g180$...10[2:31], g180$...11[2:31], g180$...12[2:31], g180$...13[2:31])


sel_C  <- grepl("FLT3.*N841K", anno$MutTranscripts) & 
  grepl("malignant", anno$PredictionRF2) |
  grepl("FLT3.*N841K", anno$NanoporeTranscripts)
sel_B  <- grepl("FLT3.*ITD", anno$MutTranscripts) & 
  grepl("malignant", anno$PredictionRF2) |
  (grepl("FLT3\\.ITD/", anno$NanoporeTranscripts) & 
     !grepl("N841K", anno$NanoporeTranscripts))
sel_AB <- grepl("FLT3.*A680V", anno$MutTranscripts) & 
  !grepl("N841K|ITD", anno$MutTranscripts) |
  (grepl("A680V", anno$NanoporeTranscripts) & 
     !grepl("FLT3\\.ITD/", anno$NanoporeTranscripts) &
     !grepl("N841K", anno$NanoporeTranscripts))

sel <- sel_C | sel_B | sel_AB
anno40 <- anno[sel,]
m40 <- m0[sel,g180]
nan_genes <- !is.na(colSums(m40))
Y3 <- m40[,nan_genes]

R1 <- vimixr::cvi_npmm(Y3, variational_params = 20, prior_shape_alpha = 0.001, 
                       prior_rate_alpha = 0.001, post_shape_alpha = 0.001, 
                       post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(Y3)), 
                       post_mean_eta = matrix(0, 20, ncol(Y3)),
                       log_prob_matrix = NULL,
                       maxit = 500,
                       n_inits = 1,
                       Seed = c("80290"),
                       covariance_type="full",fixed_variance=FALSE,
                       cluster_specific_covariance = TRUE,
                       variance_prior_type = "sparse",
                       prior_shape_d_cs_cov = matrix(100, 1, 20),
                       prior_rate_d_cs_cov = matrix(100, 20, ncol(Y3)),
                       prior_var_offd_cs_cov = 1000000,
                       post_shape_d_cs_cov = matrix(0.001, 1, 20),
                       post_rate_d_cs_cov = matrix(0.001, 20, ncol(Y3)),
                       post_var_offd_cs_cov = matrix(0.001, 20, 3),
                       scaling_cov_eta = (nrow(Y3)+1))

#pca 
cl_pred <- apply(R1$posterior$`log Probability matrix`, 1, which.max)
pca <- prcomp_irlba(Y3,2)
#variation explained
var_explained <- pca$sdev^2 / pca$totalvar
pc1_pct <- round(var_explained[1] * 100, 2)
pc2_pct <- round(var_explained[2] * 100, 2)
#the plot
pca_df <- data.frame("PC1" = pca$x[,1], "PC2" = pca$x[,2], "Cluster" = as.factor(cl_pred))
my_col <- MetBrewer::met.brewer("Hiroshige", n = 10)
col1 <- my_col[c(1,3,5,7,9)]
x_range <- range(pca_df$PC1)
y_range <- range(pca_df$PC2)
x_lim <- x_range + c(-0.5, 0.5)
y_lim <- y_range + c(-0.5, 0.5)
ggplot_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, fill = Cluster,  shape = Cluster)) +
  ggrastr::geom_point_rast(shape = 21, size = 3, alpha = 0.75, stroke = 0.25,
    color = "#1a1a1a", raster.dpi = 300) +
  scale_fill_manual(values = col1, name = "Cluster") +
  scale_x_continuous(limits = x_lim, expand = expansion(0)) +
  scale_y_continuous(limits = y_lim, expand = expansion(0)) +
  labs(title = "PCA projection of SparseDPMM clusters",
    x = paste0("PC 1 (", pc1_pct, "%)"),
    y = paste0("PC 2 (", pc2_pct, "%)")) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "grey75", fill = NA, linewidth = 0.4),
    axis.title.x = element_text(size = 10, margin = margin(t = 6)),
    axis.title.y = element_text(size = 10, margin = margin(r = 6)),
    axis.text = element_text(size = 8, color = "grey40"),
    axis.ticks = element_line(color = "grey70", linewidth = 0.3),
    axis.ticks.length = unit(3, "pt"),
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold",
                                    margin = margin(b = 8)),
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.margin = margin(15, 15, 15, 15)) +
  guides(fill = guide_legend(override.aes = list(size = 3, alpha = 1, stroke = 0.4),
    ncol = 1))

m40_raw <- m[g180, sel]
dot_data <- melt(m40_raw, varnames = c("gene", "cell"), value.name = "count")

dot_data$gene <- factor(dot_data$gene, levels = unique(dot_data$gene))

genes <- levels(dot_data$gene)

facet_labels <- c("HSC-like", "Progenitor-like", "GMP-like", "Promono-like",
  "Monocyte-like", "cDC-like")
gene_facet_df <- data.frame(gene = genes, facet = rep(facet_labels, each = 30))
dot_data <- merge(dot_data, gene_facet_df, by = "gene")
dot_data$color_group <- ifelse(dot_data$count == 0, "Zero", "Non-zero")

p2 <- ggplot(dot_data, aes(x = cell, y = gene, size = count, color = color_group)) +
  geom_point(alpha = 0.6, shape = 16) +
  scale_size_continuous(range = c(0.5, 6),
    name = "Raw count",
    breaks = function(x) {
      b <- pretty(x)
      b[b > 0]
    }) +
  scale_color_manual(values = c("Zero" = "#8B0000","Non-zero" = "steelblue"),
    name = "Count status") +
  facet_grid(rows = vars(facet), scales = "free_y", space = "free_y", switch = "y") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.x = element_text(face = "bold", size = 16),
    axis.title.y = element_text(face = "bold", size = 16),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.box = "vertical",
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, size = 8),
    strip.background = element_blank()) +
  labs(x = "Cells", y = "Genes",
    title = "Raw gene count per AML malignant cell") +
  guides(size = guide_legend(override.aes = list(color = "steelblue")),
    color = guide_legend(override.aes = list(size = 3))
  )

Fig_S6 <- p2/ggplot_pca

# ggsave("Fig_S6.pdf", plot = Fig_S6, device = "pdf", path = "Results/Figures",
#        width = 10.00, height = 7.00, units = "in")
Fig_S6
#generates Fig_S6.pdf in Results/Figures folder

##fig S7
col_fun <- colorRamp2(c(-3, 0, 3), c("#2166AC", "white", "#B2182B"))

gene_block <- rep(c("HSC-like","Prog-like","GMP-like","Promono-like","Mono-like",
                    "cDC-like"), each = 30)
gene_block_present <- gene_block[nan_genes]
gene_block_present <- factor(gene_block_present, 
                             levels = c("HSC-like","Prog-like","GMP-like",
                                        "Promono-like","Mono-like","cDC-like"))

subclone <- ifelse(sel_C,  "Subclone C", "Subclones A/B and B")
subclone <- factor(subclone, levels = c("Subclones A/B and B", "Subclone C"))

has_A680V <- as.integer(grepl("FLT3.*A680V", anno$MutTranscripts) & 
                          !grepl("N841K|ITD", anno$MutTranscripts) |
                          (grepl("A680V", anno$NanoporeTranscripts) & 
                             !grepl("FLT3\\.ITD/", anno$NanoporeTranscripts) &
                             !grepl("N841K", anno$NanoporeTranscripts)))
has_ITD   <- as.integer(grepl("FLT3.*ITD", anno$MutTranscripts) & 
                          grepl("malignant", anno$PredictionRF2) |
                          (grepl("FLT3\\.ITD/", anno$NanoporeTranscripts) & 
                             !grepl("N841K", anno$NanoporeTranscripts)))
has_N841K <- as.integer(grepl("FLT3.*N841K", anno$MutTranscripts) & 
                          grepl("malignant", anno$PredictionRF2) |
                          grepl("FLT3.*N841K", anno$NanoporeTranscripts))

top_anno <- HeatmapAnnotation(`FLT3 A680V` = has_A680V, `FLT3 ITD`   = has_ITD,
  `FLT3 N841K` = has_N841K,
  col = list(`FLT3 A680V` = c("0" = "grey", "1" = "black"),
    `FLT3 ITD`   = c("0" = "grey", "1" = "black"),
    `FLT3 N841K` = c("0" = "grey", "1" = "black")),
  annotation_name_side = "left", show_legend = FALSE)

bottom_anno <- HeatmapAnnotation(
  `Labelled Cell types`   = factor(anno40$CellType, 
                                   levels = c("Prog-like", "Mono-like", "cDC-like", "ProMono-like", "HSC-like", "")),
  `SparseDPMM clusters` = as.factor(apply(R1$posterior$`log Probability matrix`, 1, which.max)),
  col = list(
    `Labelled Cell types` = c("Prog-like" = "#FF7F00", "Mono-like" = "#377EB8",
      "cDC-like" = "#984EA3", "ProMono-like" = "#4DAF4A", "HSC-like" = "#E41A1C"),
    `SparseDPMM clusters` = c("1" = "#FF7F00", "2" = "#377EB8", "3" = "#984EA3",
      "4" = "#4DAF4A", "5" = "#E41A1C")),
  annotation_name_side = "left")

ht <- Heatmap(
  t(Y3),
  name = "Expression-level",
  col = col_fun,
  
  column_split = subclone[sel],
  cluster_columns = FALSE,
  
  row_split = gene_block_present,
  cluster_rows = FALSE,
  row_title = levels(gene_block_present), 
  row_title_side = "left",
  row_title_rot = 0,
  row_title_gp = gpar(fontsize = 8),
  
  top_annotation = top_anno[sel],
  bottom_annotation = bottom_anno,
  
  show_row_names = FALSE,
  show_column_names = FALSE,
  
  row_gap = unit(2, "mm"),
  column_gap = unit(2, "mm"),
  border = TRUE
)

draw(ht,
     column_title = "AML419A Subclone Hierarchy: Mallignant Cell-types Clustering",
     column_title_gp = gpar(fontsize = 20, fontface = "bold"),
     column_title_side = "top",
     row_title = "Genes",
     row_title_side = "left",
     row_title_gp = gpar(fontsize = 16, fontface = "bold"),
     padding = unit(c(10, 3, 10, 3), "mm")
)

grid::grid.text("Cells", x = 0.45, y = unit(5, "mm"), 
                gp = gpar(fontsize = 16, fontface = "bold"))
#generate Fig_S7

##fig S8
#Fig_1 with different covariance per cluster
#results from Curta Cluster 
#for different N=100 to N=1000 with fixed D=2 and K=2, 
#and 100 different initialisations for the log of latent  probability 
#allocation matrix Plog; the output contains 2 csv files corresponding to 
#number of clusters & ARI respectively for the 8 models considered
k_post_df <- as.data.frame(read.csv("Results/k_post_diff_sigma_per_cluster.csv"))
ari_df <- as.data.frame(read.csv("Results/ari_diff_sigma_per_cluster.csv"))

k_dat_df <- pivot_longer(k_post_df, cols = everything(), names_to = "Model", values_to = "Value")
ari_dat_df <- pivot_longer(ari_df, cols = everything(), names_to = "Model", values_to = "Value")

my_col <- c(met.brewer("Signac")[3],met.brewer("Signac")[6],met.brewer("Signac")[4],met.brewer("Signac")[5],met.brewer("Signac")[10],met.brewer("Signac")[13],met.brewer("Signac")[12],met.brewer("Signac")[11])
p1 <- ggplot(k_dat_df, aes(x = Model, y = Value, color = Model)) +
       geom_boxplot(fill = "grey88") +
       ggtitle(expression("(a) " * K[post] * " boxplots")) +
       scale_color_manual(values = my_col) +
       theme_minimal() +
       labs(x = "",
            y = "Posterior Cluster number (averaged)") +
       theme(plot.title = element_text(size = 18)) +
       theme(axis.title.x = element_text(size = 10),
                         axis.title.y = element_text(size = 12),
                         panel.grid.major.x = element_blank(),   
                         panel.grid.minor.x = element_blank()) +
     theme(legend.position = "none")
p2 <- ggplot(ari_dat_df, aes(x = Model, y = Value, color = Model)) +
       geom_boxplot(fill = "grey88") +
       ggtitle("(b) ARI boxplots") +
       scale_color_manual(values = my_col) + 
       theme_minimal() +
       labs(x = "", y = "ARI")+
       theme(plot.title = element_text(size = 18)) +
       theme(axis.title.x = element_text(size = 10),
                         axis.title.y = element_text(size = 12),
                         panel.grid.major.x = element_blank(),   
                         panel.grid.minor.x = element_blank())

Fig_S8 <- p1+p2

# ggsave("Fig_S8.pdf", plot = Fig_S8, device = "pdf", path = "Results/Figures",
#        width = 6.00, height = 4.19, units = "in")
Fig_S8
#generates Fig_S8.pdf in Results/Figures folder

#supplementary table S1
#The following code generated the data for the comparison table as Results_bic.csv
# N <- 100
# D <- 1000
# l_allot <- c(rep(0,33), rep(1,33), rep(2,34))
# set.seed(24042026)
# X <- matrix(0, N, D)
# for (n in 1:N){
#   X[n,] <- rnorm(D, 0, 1) + l_allot[n]*5
# }
# 
# 
# Results <- matrix(0, 1000, 15)
# for (m in 1:1000){
#   Plog <- matrix(runif(20*nrow(X), -1, -0.0006), nrow = nrow(X))
#   R0 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001,
#                          prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                          post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                          post_mean_eta = matrix(0, 20, ncol(X)),
#                          log_prob_matrix = Plog, maxit = 1000, n_inits = 1,
#                          covariance_type="full",fixed_variance=FALSE,
#                          cluster_specific_covariance = TRUE,
#                          variance_prior_type = "sparse",
#                          prior_shape_d_cs_cov = matrix(rep(100, 20), nrow = 1, ncol = 20),
#                          prior_rate_d_cs_cov = matrix(rep(100, 20*ncol(X)), 20, ncol(X)),
#                          prior_var_offd_cs_cov = 1000,
#                          post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                          post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                          post_var_offd_cs_cov = array(0.001, c(ncol(X), ncol(X), 20)),
#                          scaling_cov_eta = nrow(X))
#   pred <- apply(R0$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#   ari <- mclust::adjustedRandIndex(l_allot, pred)
#   last_opt <- R0$optimisation$ELBO[[length(R0$optimisation$ELBO)]]
#   VLL <- unname(last_opt)[4]
#   cl <- R0$posterior$`Cluster number`
#   elb <- sum(last_opt)
#   k0 <- cl*D + cl*(D*(D+1)/2) + cl-1
#   Bic <- -2*VLL + k0*log(N)
# 
#   R1 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001,
#                          prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                          post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                          post_mean_eta = matrix(0, 20, ncol(X)),
#                          log_prob_matrix = Plog, maxit = 1000, n_inits = 1,
#                          covariance_type="full",fixed_variance=FALSE,
#                          cluster_specific_covariance = TRUE,
#                          variance_prior_type = "IW",
#                          prior_df_cs_cov = ncol(X) + 2,
#                          prior_scale_cs_cov = diag(ncol(X)),
#                          post_df_cs_cov = matrix(rep(ncol(X) + 2, 20), nrow = 1),
#                          post_scale_cs_cov = array(rep(diag(ncol(X)), 20), c(ncol(X), ncol(X), 20)),
#                          scaling_cov_eta = nrow(X))
#   pred1 <- apply(R1$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#   ari1 <- mclust::adjustedRandIndex(l_allot, pred1)
#   last_opt1 <- R1$optimisation$ELBO[[length(R1$optimisation$ELBO)]]
#   VLL1 <- unname(last_opt1)[4]
#   cl1 <- R1$posterior$`Cluster number`
#   elb1 <- sum(last_opt1)
#   k1 <- cl1*D + cl1*(D*(D+1)/2) + cl1-1
#   Bic1 <- -2*VLL1 + k1*log(N)
# 
#   R2 <- vimixr::cvi_npmm(X, variational_params = 20, prior_shape_alpha = 0.001,
#                          prior_rate_alpha = 0.001, post_shape_alpha = 0.001,
#                          post_rate_alpha = 0.001, prior_mean_eta = matrix(0, 1, ncol(X)),
#                          post_mean_eta = matrix(0, 20, ncol(X)),
#                          log_prob_matrix = Plog, maxit = 1000, n_inits = 1,
#                          covariance_type="full",fixed_variance=FALSE,
#                          cluster_specific_covariance = TRUE,
#                          variance_prior_type = "off-diagonal normal",
#                          prior_shape_d_cs_cov = matrix(rep(100, 20), nrow = 1, ncol = 20),
#                          prior_rate_d_cs_cov = 100,
#                          prior_var_offd_cs_cov = 0.0001,
#                          post_shape_d_cs_cov = matrix(0.001, 1, 20),
#                          post_rate_d_cs_cov = matrix(0.001, 20, ncol(X)),
#                          post_mean_offd_cs_cov = array(rep(diag(ncol(X)), 20), c(ncol(X), ncol(X), 20)),
#                          scaling_cov_eta = nrow(X))
#   pred2 <- apply(R2$posterior$'log Probability matrix', MARGIN = 1, FUN=which.max)
#   ari2 <- mclust::adjustedRandIndex(l_allot, pred2)
#   last_opt2 <- R2$optimisation$ELBO[[length(R2$optimisation$ELBO)]]
#   VLL2 <- unname(last_opt2)[4]
#   cl2 <- R2$posterior$`Cluster number`
#   elb2 <- sum(last_opt2)
#   k2 <- cl2*D + cl2*(D*(D+1)/2) + cl2-1
#   Bic2 <- -2*VLL2 + k2*log(N)
# 
# 
#   Results[m, ] <- c(VLL, elb, ari, cl, Bic, VLL1, elb1, ari1, cl1, Bic1, VLL2, elb2, ari2, cl2, Bic2)
# }
#Taking average (mean) of the corresponding columns we get the table S1

#sessionInfo
S <- sessionInfo()
# cat(capture.output(sessionInfo()), file = "sessionInfo.txt", sep = "\n")
S