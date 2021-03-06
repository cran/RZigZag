% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{EstimateESS}
\alias{EstimateESS}
\title{EstimateESS}
\usage{
EstimateESS(skeletonList, n_batches = 100L, coordinate = -1L,
  zeroMeans = FALSE)
}
\arguments{
\item{skeletonList}{a piecewise deterministic skeleton (consisting of Times, Points and Velocities) returned by a sampler}

\item{n_batches}{optional argument indicating the number of batches to use in the batch means estimation method}

\item{coordinate}{if specified, only estimate the ESS of the specified coordinate, otherwise estimate the ESS of all coordinates}

\item{zeroMeans}{if TRUE do not estimate means but assume a centered distribution}
}
\value{
Returns a list containing the estimated asymptotic variance, ESS and estimated covariance matrix
}
\description{
Estimates the effective sample size (ESS) of a piecewise deterministic skeleton
}
