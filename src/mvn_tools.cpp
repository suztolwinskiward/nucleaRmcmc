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
//' \code{rmvn_cond_cpp} 
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

///////////////////////////////////////////////////////////////////////////////
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
using namespace std;

//' Following Rodriguez-Yam, http://www.stat.columbia.edu/~rdavis/papers/CLR.pdf
//'
//'  Result from Rodriguez-Yam:
//'  Given
//'      X ~ N_T(mu,Sigma), where T = {x : Bx <= b}
//'  Define 
//'      Z := AX, with A s.t. A %*% Sigma %*% t(A) = I (eg. the lower-triangular
//'                                                     cholesky decomposition)
//'  Then Z ~ N_S(A%*% mu, I) ~ N(alpha,I) 
//'          where S = {z: Dz <= b}, D = B %*% inverse(A), and alpha = A %*% mu.
//'  
//'  Then marginals have structure
//'    z_j|z_-j ~ N_S_j(alpha_j,1) 
//'          where S_j = {z_j : d_j z_j <= b - D_-j z_-j}
//'                      
//' @param x
//' @param mu
//' @param Sig The covariance matrix
//' @param B Transformation matrix for x to give constraints BX <= b
//' @param b The vector giving the constraints BX <= b 
//'
// [[Rcpp::export()]]
arma::mat r_truncmvn_cpp(NumericMatrix x, NumericMatrix mu,
NumericMatrix Sig, NumericMatrix B, NumericMatrix b){
  
  int N = mu.nrow();
  
  // Now simulate:
  arma::mat A = arma::trans(arma::inv(trimatu(arma::chol(as<arma::mat>(Sig)))));
  arma::vec alpha = A * as<arma::vec>(mu);
  arma::mat D = as<arma::mat>(B) * arma::inv(A);
  arma::vec z = A * as<arma::vec>(x);
  
  for(int j = 0; j < N; j++){
    
    // Matrix/vector subsets needed:
    arma::mat Dmj = D; // "D minus j"
    Dmj.shed_col(j); 
    arma::vec zmj = z; // "z minus j"
    zmj.shed_row(j); 
    
    //  
    
    arma::mat bds = (as<arma::mat>(b) - Dmj * zmj)/D.col(j); 
    arma::mat lhs_bds = D.col(j);
    
    uvec ub_i = find(lhs_bds > 0);
    NumericVector uj;
    if(ub_i.size()==0){
      uj = datum::inf;
    }else{
      uj = min(bds(ub_i));
    }
    
    uvec lb_i = find(lhs_bds < 0);
    NumericVector lj;
    if(lb_i.size()==0){
      lj = -datum::inf;
    }else{
      lj = max(bds(lb_i));
    }
    
    NumericVector u_lj = pnorm(lj,alpha[j],1.0);
    NumericVector u_uj = pnorm(uj,alpha[j],1.0);
    NumericVector u = runif(1,u_lj[0],u_uj[0]);
    
    if(u[0] == 1){z[j] <- lj;}
    if(u[0] == 0){z[j] <- uj;}
    if((u[0] > 0) & (u[0] < 1)){
      NumericVector tmp = qnorm(u,alpha[j],1.0);
      z[j] = tmp[0];}
  }
  
  return(arma::solve(A,z));
  
}

///////////////////////////////////////////////////////////////////////////////





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
