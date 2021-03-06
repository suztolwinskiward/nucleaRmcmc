% Generated by roxygen2 (4.1.1): do not edit by hand
% Please edit documentation in R/script.r
\name{r.cond.mvn.Nuclear}
\alias{r.cond.mvn.Nuclear}
\title{r.cond.mvn.Nuclear}
\usage{
r.cond.mvn.Nuclear(mu, cholSigma, iSigma, x, jOmit)
}
\arguments{
\item{mu}{Mean vector of MVN distribution}

\item{cholSigma}{Cholesky decomposition of the covariance matrix of the MVN}

\item{iSigma}{Precision matrix of the MVN}

\item{x}{Vector realization of MVN, with some conponents on which we want
to condition}

\item{jOmit}{The index of the vector x which we'd like to omit from
conditioning and sample}
}
\description{
Faster conditional sampling multivar. normal vector entry given all others.
}
\details{
\code{r.cond.mvn.Nuclear} conditional sampling of a single component
of a multivariate normal given all other vector components. ``Faster''
compared to the straightforward conditional formula.
}

