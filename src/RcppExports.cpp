// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// rmvn_bayes_cpp
arma::vec rmvn_bayes_cpp(arma::mat A, arma::vec b);
RcppExport SEXP nucleaRmcmc_rmvn_bayes_cpp(SEXP ASEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b(bSEXP);
    __result = Rcpp::wrap(rmvn_bayes_cpp(A, b));
    return __result;
END_RCPP
}
// rm_sngl_vec_entry
arma::vec rm_sngl_vec_entry(arma::vec x, int jOmit);
RcppExport SEXP nucleaRmcmc_rm_sngl_vec_entry(SEXP xSEXP, SEXP jOmitSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type jOmit(jOmitSEXP);
    __result = Rcpp::wrap(rm_sngl_vec_entry(x, jOmit));
    return __result;
END_RCPP
}
// rmvn_cond_cpp
List rmvn_cond_cpp(arma::vec mu, arma::mat cholSigma, arma::mat iSigma, arma::vec x, int jOmit);
RcppExport SEXP nucleaRmcmc_rmvn_cond_cpp(SEXP muSEXP, SEXP cholSigmaSEXP, SEXP iSigmaSEXP, SEXP xSEXP, SEXP jOmitSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cholSigma(cholSigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type iSigma(iSigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type jOmit(jOmitSEXP);
    __result = Rcpp::wrap(rmvn_cond_cpp(mu, cholSigma, iSigma, x, jOmit));
    return __result;
END_RCPP
}
