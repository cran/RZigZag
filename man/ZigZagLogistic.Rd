% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{ZigZagLogistic}
\alias{ZigZagLogistic}
\title{ZigZagLogistic}
\usage{
ZigZagLogistic(dataX, dataY, n_epochs, subsampling = TRUE,
  controlvariates = TRUE, beta0 = numeric(0), n_samples = 0L,
  n_batches = 0L, computeCovariance = FALSE, upperbound = FALSE)
}
\arguments{
\item{dataX}{Matrix containing the independent variables x. The i-th column represents the i-th observation with components x_{1,i}, ..., x_{d-1,i}.}

\item{dataY}{Vector of length n containing {0, 1}-valued observations of the dependent variable y.}

\item{n_epochs}{Integer indicating how many times the algorithm processes an equivalent of the full dataset. For basic zig-zag this is identical to the number of iterations of the algorithm. For subsampling and control variates zig-zag this is n times the number of iterations, since every iteration has cost 1/n of an epoch. Here n is the number of observations.}

\item{subsampling}{Boolean. Use Zig-Zag with subsampling if TRUE.}

\item{controlvariates}{Boolean. Use Zig-Zag with control variates if TRUE (overriding any value of \code{subsampling}).}

\item{beta0}{Optional argument indicating the starting point for the Zig-Zag sampler}

\item{n_samples}{Number of discrete time samples to extract from the Zig-Zag skeleton.}

\item{n_batches}{If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.}

\item{computeCovariance}{Boolean indicating whether to estimate the covariance matrix.}

\item{upperbound}{Boolean. If TRUE, sample without subsampling and using a constant upper bound instead of a linear Hessian dependent upper bound}
}
\value{
Returns a list with the following objects:

\code{skeletonTimes} Vector of switching times

\code{skeletonPoints} Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.

\code{samples} If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples at fixed intervals along the Zig-Zag trajectory.

\code{mode} If \code{controlvariates = TRUE}, this is a vector containing the posterior mode obtained using Newton's method.

\code{batchMeans} If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means

\code{means} If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory

\code{covariance} If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory

\code{asVarEst} If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component

\code{ESS} If \code{n_batches > 0} this is an estimate of the effective sample size along each component
}
\description{
Applies the Zig-Zag Sampler to logistic regression, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
}
\examples{
require("RZigZag")
generate.logistic.data <- function(beta, nobs) {
  ncomp <- length(beta)
  dataX <- matrix(rnorm((ncomp -1) * nobs), nrow = ncomp -1);
  vals <- beta[1] + colSums(dataX * as.vector(beta[2:ncomp]))
  generateY <- function(p) { rbinom(1, 1, p)}
  dataY <- sapply(1/(1 + exp(-vals)), generateY)
  return(list(dataX, dataY))
}

beta <- c(1,2)
data <- generate.logistic.data(beta, 1000)
result <- ZigZagLogistic(data[[1]], data[[2]], 1000, n_samples = 100)
plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
points(result$samples[1,], result$samples[2,], col='magenta')
}