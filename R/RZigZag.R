#' RZigZag
#' 
#' Implements the Zig-Zag algorithm with subsampling and control variates (ZZ-CV) of (Bierkens, Fearnhead, Roberts, 2016) \url{https://arxiv.org/abs/1607.03188} as applied to Bayesian logistic regression, as well as basic Zig-Zag for a Gaussian target distribution.
#' 
#' This package currently consists of two functions, \code{\link{ZigZagLogistic}} for logistic regression and \code{\link{ZigZagGaussian}} for multivariate Gaussian.
#' 
#' @docType package
#' @author Joris Bierkens
#' @author With thanks to Matt Moores, \url{https://mattstats.wordpress.com/}, for his help in getting from C++ code to a CRAN-ready Rcpp based package.
#' @import Rcpp 
#' @importFrom Rcpp evalCpp
#' @useDynLib RZigZag
#' @name RZigZag
NULL  
