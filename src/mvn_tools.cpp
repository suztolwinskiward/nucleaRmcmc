// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
//' rmvn_bayes_cpp 
//' 
//' Fast sampling from multivar. normal dist'ns for Bayesian hierarchical
//' models. 
//'  
//' \code{rmvn_bayes_cpp} samples from a MV normal of the form 
//' N(A^(-1) * b, A^(-1)).
//' 
//' @param A Precision matrix of the desired sample. 
//' @param b Vector s.t. mean of desired sample is A^(-1) * b 
//' 
//' @export 
// [[Rcpp::export]]
arma::vec rmvn_bayes_cpp(arma::mat A, arma::vec b) {
  int ncols = A.n_cols;
  arma::mat A_chol = chol(A);
  arma::vec devs = randn(ncols);
	arma::vec temp = solve(trimatl(A_chol.t()), b);
	return arma::vec(solve(trimatu(A_chol), temp + devs));
}

///////////////////////////////////////////////////////////////////////////////
//' rm_sngl_vec_entry
//'
//' Fast conditional sampling multivar. normal vector entry given all others.
//' 
//' \code{rm_sngl_vec_entry} 
//' 
//' @param x vector
//' @param jOmit The index of the vector x which we'd like to omit from 
//' conditioning and sample) 
//' 
//' @export
// [[Rcpp::export]]
arma::vec rm_sngl_vec_entry(arma::vec x, int jOmit) {

  int N = x.n_rows;
  arma::vec y;
  if( (jOmit > 1) & (jOmit < N) ){
    arma::vec foo1 = x(span(0,jOmit-2));
    arma::vec foo2 = x(span(jOmit,N-1));
    y = join_vert(foo1,foo2);
  }else{
    if(jOmit  == 1){y = x(span(1,N-1));}
    if(jOmit  == N){y = x(span(0,N-2));}
  }

  return y;

}


///////////////////////////////////////////////////////////////////////////////
//' Fast conditional sampling multivar. normal vector entry given all others.
//' 
//' \code{r_cond_mvn_cpp2} 
//' 
//' @param mu Mean vector of MVN distribution
//' @param cholSigma Cholesky decomposition of the covariance matrix of the MVN 
//' @param iSigma Precision matrix of the MVN
//' @param x Vector realization of MVN, with some conponents on which we want 
//' to condition
//' @param jOmit The index of the vector x which we'd like to omit from 
//' conditioning and sample) 
//' 
//' @export
// [[Rcpp::export]]
List rmvn_cond_cpp(arma::vec mu, arma::mat cholSigma, arma::mat iSigma, 
                          arma::vec x, int jOmit) {

  int N = mu.n_rows;
  double iSigma_jj = as_scalar(iSigma(jOmit-1,jOmit-1));
  arma::vec preweights = iSigma.col(jOmit-1);
  arma::vec weights = -1 * rm_sngl_vec_entry(preweights,jOmit)/ iSigma_jj;
  double adjust = sum(weights % rm_sngl_vec_entry(x-mu,jOmit));
  double mustar = mu(jOmit-1) + as_scalar(adjust);
  
  arma::vec xUncond = trans(cholSigma) * randn(N);
  double error1 = as_scalar(xUncond(jOmit) -
          sum(weights % rm_sngl_vec_entry(xUncond,jOmit)));

  return List::create(Named("samp") = mustar + error1,
                      Named("mustar") = mustar,
                      Named("adjust") = adjust,
                      Named("sig.jj") = iSigma_jj);
}




/*
r.cond.mvn.Nuclear <- function(mu,SigmaCholesky,SigmaInverse,x, jOmit) {
  
  #
  SigmaI.jj <- SigmaInverse[jOmit, jOmit]
  weights<- -1*SigmaInverse[-jOmit,jOmit]/SigmaI.jj
  adjust<- sum( weights*((x-mu)[-jOmit]) )
  mustar <- mu[jOmit] + adjust
  #
  xUncond <- t(SigmaCholesky) %*% rnorm(length(x))
  error1 <- xUncond[jOmit] - sum( weights * xUncond[-jOmit] )
  #error2 <- mustar - x[-i]
  # NOTE draw from the conditional is  mustar + error1
  return(c(mustar + error1) )
}
*/
