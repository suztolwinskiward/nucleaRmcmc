#' @useDynLib nucleaRmcmc
#' @importFrom Rcpp sourceCpp
NULL
# -----------------------------------------------------------------------------


#####################
#' r.cond.mvn.Nuclear
#' 
#' Faster conditional sampling multivar. normal vector entry given all others.
#' 
#' \code{r.cond.mvn.Nuclear} conditional sampling of a single component 
#' of a multivariate normal given all other vector components. ``Faster''
#' compared to the straightforward conditional formula. 
#'
#' @param mu Mean vector of MVN distribution
#' @param cholSigma Cholesky decomposition of the covariance matrix of the MVN 
#' @param iSigma Precision matrix of the MVN
#' @param x Vector realization of MVN, with some conponents on which we want 
#' to condition
#' @param jOmit The index of the vector x which we'd like to omit from 
#' conditioning and sample 
#' 
#' @export
r.cond.mvn.Nuclear <- function(mu,cholSigma,iSigma,x,jOmit) {
  
  #
  iSigma.jj <- iSigma[jOmit, jOmit]
  weights<- -1*iSigma[-jOmit,jOmit]/iSigma.jj
  adjust<- sum( weights*((x-mu)[-jOmit]) )
  mustar <- mu[jOmit] + adjust
  #
  xUncond <- t(cholSigma) %*% rnorm(length(x))
  error1 <- xUncond[jOmit] - sum( weights * xUncond[-jOmit] )
  # NOTE draw from the conditional is  mustar + error1
  return(list(samp = c(mustar + error1),
              mustar = mustar,
              adjust = adjust,
              sig2.jj = 1/iSigma.jj))
}

#####################
#
##
## test libraries and functions
##
#
## Load the Rcpp mcmc code
# Rcpp::sourceCpp('./src/mvn_tools.cpp')
#
#
# ##
# ## Simulate some data
# ##
# 
# set.seed(106)
# n <- 5
# X <- matrix(rnorm(n * n), ncol = n)
# Sigma <- X %*% t(X);
# mu <- matrix(rnorm(n),n,1);
# 
# ## generate a random vector with distribution N(mu,Sigma):
# N <- 100
# X <- matrix(NA,n,N)
# for(k in 1:N){
#   X[,k] <- rmvn_bayes_cpp(solve(Sigma), Sigma%*%mu) 
# }
# plot(X[,1])
# for(k in 1:N){
#   points(X[,k])
# }
# pairs(t(X))
# 
# #############
# 
# Nsamp <- 33
# cholSigma <- chol(Sigma);
# iSigma <- solve(Sigma);
# 
# foo1 <- rmvn_cond_cpp(mu,cholSigma,iSigma,X[,Nsamp],2)
# foo2 <- r.cond.mvn.Nuclear(mu,cholSigma,iSigma,X[,Nsamp], 2)
# 
# N <- 1000
# m <- 2
# foo1 <- foo2 <- err1 <- err2 <- matrix(NA,N,100)
# for(k in 1:100){
#   for(n in 1:N){
#     foo1[n,k] <- rmvn_cond_cpp(mu,cholSigma,iSigma,X[,k],m)$samp
#     err1[n,k] <- X[m,k] - foo1[n,k]  
#     foo2[n,k] <- r.cond.mvn.Nuclear(mu,cholSigma,iSigma,X[,k],m)$samp
#     err2[n,k] <- X[m,k] - foo2[n,k]  
#   }
# 
# }
# 
# 
# hist(foo)
# cat(X[m,Nsamp])
# cat(median(foo))
# #####################
# # Now do partial conditioning:
# i <- 1
# for(k in 1:N){
#   if(k != i){
#     X[,k] <- rmvncond_cpp(Sigma, mu, x, i) 
#   }
# }
# plot(X[,1])
# for(k in 1:N){
#   points(X[,k])
# }
# pairs(t(X))
# 
