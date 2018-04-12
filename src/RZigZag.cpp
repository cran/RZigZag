// RZigZag.cpp : implements Zig-Zag with sub-sampling and control variates (ZZ-CV)
//
// Copyright (C) 2017--2018 Joris Bierkens
//
// This file is part of RZigZag.
//
// RZigZag is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RZigZag is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RZigZag.  If not, see <http://www.gnu.org/licenses/>.


#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#include "RZigZag.h"
#include <stdio.h>

VectorXi sample(const unsigned int n, const unsigned int k) {
  // crappy inefficient implementation. see http://stackoverflow.com/questions/33802205
  RNGScope scp; // initialize random number generator
  VectorXi samples(k);
  for (unsigned int i = 0; i < k; ++i) {
    bool found_sample = false;
    int proposed_sample;
    while (!found_sample) {
      proposed_sample = floor(runif(1)(0) * n);
      found_sample = true;
      for (unsigned int j = 0; j < i; ++j)
        if (samples[j] == proposed_sample) {
          found_sample = false;
          break;
        }
    }
    if (found_sample)
      samples[i] = proposed_sample;
  }
  return samples;
}
void Skeleton::computeBatchMeans(const unsigned int n_batches) {
  if (n_batches == 0)
    stop("n_batches should be positive.");
  const unsigned int n_skeletonPoints = skeletonTimes.rows();
  const unsigned int dim = skeletonPoints.rows();
  const double t_max = skeletonTimes[n_skeletonPoints-1];
  const double batch_length = t_max / n_batches;

  double t0 = skeletonTimes[0];
  VectorXd x0 = skeletonPoints.col(0);

  batchMeans = MatrixXd(dim, n_batches);

  unsigned int batchNr = 0;
  double t_intermediate = batch_length;
  VectorXd currentBatchMean = VectorXd::Zero(dim);

  for (unsigned int i = 1; i < n_skeletonPoints; ++i) {
    double t1 = skeletonTimes[i];
    VectorXd x1 = skeletonPoints.col(i);

    while (batchNr < n_batches - 1 && t1 > t_intermediate) {
      VectorXd x_intermediate = x0 + (t_intermediate - t0) / (t1 - t0) * (x1 - x0);
      batchMeans.col(batchNr) = currentBatchMean + (t_intermediate - t0) * (x_intermediate + x0)/(2 * batch_length);

      // initialize next batch
      currentBatchMean = VectorXd::Zero(dim);
      batchNr++;
      t0 = t_intermediate;
      x0 = x_intermediate;
      t_intermediate = batch_length * (batchNr + 1);
    }
    currentBatchMean += (t1 - t0) * (x1 + x0)/(2 * batch_length);
    t0 = t1;
    x0 = x1;
  }
  batchMeans.col(batchNr) = currentBatchMean;

  computeCovariance();

  MatrixXd meanZeroBatchMeans = batchMeans.colwise() - means;
  asVarEst = batch_length * meanZeroBatchMeans.rowwise().squaredNorm()/(n_batches - 1);
  ESS = (covarianceMatrix.diagonal().array()/asVarEst.array() * t_max).matrix();
}

void Skeleton::computeCovariance() {
  const unsigned int n_skeletonPoints = skeletonTimes.rows();
  const unsigned int dim = skeletonPoints.rows();
  const double t_max = skeletonTimes[n_skeletonPoints-1];

  double t0 = skeletonTimes[0];
  VectorXd x0 = skeletonPoints.col(0);
  MatrixXd cov_current = x0 * x0.transpose();

  covarianceMatrix = MatrixXd::Zero(dim, dim);
  means = VectorXd::Zero(dim);

  for (unsigned int i = 1; i < n_skeletonPoints; ++i) {
    double t1 = skeletonTimes[i];
    VectorXd x1 = skeletonPoints.col(i);
    // the following expression equals \int_{t_0}^{t_1} x(t) (x(t))^T d t
    covarianceMatrix += (t1 - t0) * (2 * x0 * x0.transpose() + x0 * x1.transpose() + x1 * x0.transpose() + 2 * x1 * x1.transpose())/(6 * t_max);
    means += (t1 - t0) * (x1 + x0) /(2 * t_max);
    t0 = t1;
    x0 = x1;
  }
  covarianceMatrix -= means * means.transpose();
}

LogisticData::LogisticData(const MatrixXd* dataXptr, const VectorXi* dataYptr) : dataXptr(dataXptr), dataYptr(dataYptr) {
  dim = dataXptr->rows();
  n_observations = dataXptr->cols();
}

double LogisticData::potential(const VectorXd& beta) const {
  double val = 0;
  for (unsigned int j = 0; j < n_observations; ++j) {
    double innerproduct = beta.dot(dataXptr->col(j));
    val += log(1 + exp(innerproduct)) - (*dataYptr)(j) * innerproduct;
  }
  return val;
}

VectorXd LogisticData::gradient(const VectorXd& beta) const {
  VectorXd grad(VectorXd::Zero(dim));
  for (unsigned int j = 0; j < n_observations; ++j) {
    double val = exp(dataXptr->col(j).dot(beta));
    grad += dataXptr->col(j) * (val/(1+val) - (*dataYptr)(j));
  }
  return grad;
}

MatrixXd LogisticData::hessian(const VectorXd& beta) const {
  MatrixXd hess(MatrixXd::Zero(dim,dim));
  for (unsigned int j = 0; j < n_observations; ++j) {
    double innerproduct = beta.dot(dataXptr->col(j));
    hess += (dataXptr->col(j) * dataXptr->col(j).transpose())* exp(innerproduct)/((1+exp(innerproduct)*(1+exp(innerproduct))));
  }
  return hess;
}

void Skeleton::LogisticBasicZZ(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_iter, VectorXd beta) {

  const MatrixXd dataXpp(preprocessLogistic(dataX));
  const unsigned int n_components = dataXpp.rows();

  MatrixXd Q(domHessianLogistic(dataXpp));

  if (beta.rows()==0)
    beta = VectorXd::Zero(n_components);
  VectorXd theta(VectorXd::Constant(n_components, 1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;

  const VectorXd b(sqrt((double)n_components) * Q.rowwise().norm());
  VectorXd derivative_upperbound(n_components);
  VectorXd a(n_components);
  for (unsigned int k = 0; k < n_components; ++k) {
    derivative_upperbound(k) = theta(k) * derivativeLogistic(dataXpp, dataY, beta, k);
    if (derivative_upperbound(k) > 0)
      a(k) = derivative_upperbound(k);
    else
      a(k) = 0;
  }

  unsigned int switches = 0;

  RNGScope scp; // initialize random number generator

  double minTime, simulatedTime;

  int i0;
  skeletonPoints = MatrixXd(n_components,n_iter);
  skeletonPoints.col(0) = beta;
  skeletonTimes = ArrayXd(n_iter);

  for (unsigned int step = 1; step < n_iter; ++step) {
    NumericVector U(runif(n_components));
    i0 = -1;
    for (unsigned int i = 0; i < n_components; ++i) {
      simulatedTime = getRandomTime(a(i), b(i), U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      stop("Zigzag wandered off to infinity.");
    }
    else {
      currentTime = currentTime + minTime;
      beta = beta + minTime * theta;
      for (unsigned int i = 0; i < n_components; ++i) {
        if (i != i0) {
          derivative_upperbound(i) = derivative_upperbound(i) + b(i) * minTime;
          if (derivative_upperbound(i) > 0)
            a(i) = derivative_upperbound(i);
          else
            a(i) = 0;
        }
      }
      double derivative = derivativeLogistic(dataXpp, dataY, beta, i0);
      double V = runif(1)(0);
      if (V <= theta(i0) * derivative/(a(i0)+b(i0)*minTime)) {
        theta(i0) = -theta(i0);
        ++switches;
      }

      derivative_upperbound(i0) = derivative * theta(i0);
      if (derivative_upperbound(i0) > 0)
        a(i0) = derivative_upperbound(i0);
      else
        a(i0) = 0;

      skeletonTimes(step) = currentTime;
      skeletonPoints.col(step) = beta;
    }
  }
  Rprintf("LogisticBasicZZ: Fraction of accepted switches: %g\n", double(switches)/(n_iter));
}
void Skeleton::LogisticUpperboundZZ(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_iter, VectorXd beta) {
  
  const MatrixXd dataXpp(preprocessLogistic(dataX));
  const unsigned int n_components = dataXpp.rows();
  const unsigned int n_observations = dataXpp.cols();
  VectorXd theta(VectorXd::Constant(n_components, 1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  
  const VectorXd a(n_observations * logisticUpperbound(dataXpp));
  
  unsigned int switches = 0;
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  
  int i0;
  skeletonPoints = MatrixXd(n_components,n_iter);
  skeletonPoints.col(0) = beta;
  skeletonTimes = ArrayXd(n_iter);
  
  for (unsigned int step = 1; step < n_iter; ++step) {
    NumericVector U(runif(n_components));
    i0 = -1;
    for (unsigned int i = 0; i < n_components; ++i) {
      simulatedTime = getRandomTime(a(i), 0, U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      Rprintf("Zig zag wandered off to infinity.\n");
      break;
    }
    else {
      currentTime = currentTime + minTime;
      beta = beta + minTime * theta;
      double derivative = derivativeLogistic(dataXpp, dataY, beta, i0);
      double V = runif(1)(0);
      if (V <= theta(i0) * derivative/a(i0)) {
        theta(i0) = -theta(i0);
        ++switches;
      }
      skeletonTimes(step) = currentTime;
      skeletonPoints.col(step) = beta;
    }
  }
  Rprintf("LogisticUpperboundZZ: Fraction of accepted switches: %g\n", double(switches)/(n_iter));
}



void Skeleton::LogisticSubsamplingZZ(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_iter, VectorXd beta) {
  
  const MatrixXd dataXpp(preprocessLogistic(dataX));
  const unsigned int dim = dataXpp.rows();
  const unsigned int n_observations = dataXpp.cols();
  
  if (beta.rows()==0)
    beta = VectorXd::Zero(dim);
  
  
  VectorXd theta(VectorXd::Constant(dim,1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  VectorXd upperbound(n_observations * logisticUpperbound(dataXpp));
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  const unsigned int MAX_SIZE = 1e4;
  unsigned int max_switches = (n_iter < MAX_SIZE ? n_iter : MAX_SIZE);
  
  MatrixXd skeletonPointsTemp = MatrixXd(dim, max_switches);
  skeletonPointsTemp.col(0) = beta;
  ArrayXd skeletonTimesTemp = ArrayXd(max_switches);
  skeletonTimesTemp[0] = currentTime;
  
  
  unsigned int i0;
  unsigned int switches = 0;
  for (unsigned int step = 1; step < n_iter; ++step) {
    for (unsigned int i = 0; i < dim; ++i) {
      simulatedTime = rexp(1, upperbound(i))(0);
      if (i == 0 || simulatedTime < minTime) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    currentTime = currentTime + minTime;
    beta = beta + minTime * theta;
    unsigned int J = floor(n_observations*runif(1)(0)); // randomly select observation
    double derivative = n_observations * dataXpp(i0,J) * (1.0/(1.0+exp(-dataXpp.col(J).dot(beta))) - dataY(J));
    double V = runif(1)(0);
    if (derivative > upperbound(i0)) {
      Rprintf("LogisticSubsamplingZZ:: Error: derivative larger than its supposed upper bound.\n");
      Rprintf("  Upper bound: %g, actual derivative: %g.\n", upperbound(i0), derivative);
      Rprintf("  Index: %d, Beta(0): %g, Beta(1): %g\n", i0, beta(0), beta(1));
      break;
    }
    if (V <= theta(i0) * derivative/upperbound(i0)) {
      theta(i0) = -theta(i0);
      ++switches;
      if (switches >= max_switches) {
        max_switches *= 2;
        //        Rprintf("Resizing to size %d...\n", max_switches);
        skeletonTimesTemp.conservativeResize(max_switches);
        skeletonPointsTemp.conservativeResize(dim, max_switches);
      }
      skeletonTimesTemp[switches] = currentTime;
      skeletonPointsTemp.col(switches) = beta;
    }
  }
  Rprintf("LogisticSubSampling: Fraction of accepted switches: %g\n", double(switches)/n_iter);
  skeletonTimes = skeletonTimesTemp.head(switches + 1);
  skeletonPoints = skeletonPointsTemp.leftCols(switches + 1);
}


void Skeleton::LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_iter, VectorXd beta) {
  
  const MatrixXd dataXpp(preprocessLogistic(dataX));
  LogisticData data(&dataXpp, &dataY);
  const unsigned int dim = dataXpp.rows();
  const unsigned int n_observations = dataXpp.cols();
  const double precision = 1e-10;
  const unsigned int max_iter = 1e2;
  mode = VectorXd::Zero(dim);
  newtonLogistic(data, mode, precision, max_iter);
  if (beta.rows()==0)
    beta = mode;
  
  VectorXd theta(VectorXd::Constant(dim,1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  const VectorXd uniformBound(cvBound(dataXpp) * n_observations);
  const VectorXd b(sqrt((double)dim) * uniformBound);
  VectorXd a((beta-mode).norm() * uniformBound);
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  
  const unsigned int MAX_SIZE = 1e4;
  unsigned int max_switches = (n_iter < MAX_SIZE ? n_iter : MAX_SIZE);
  
  MatrixXd skeletonPointsTemp = MatrixXd(dim, max_switches);
  skeletonPointsTemp.col(0) = beta;
  ArrayXd skeletonTimesTemp = ArrayXd(max_switches);
  skeletonTimesTemp[0] = currentTime;
  
  int i0;
  unsigned int switches = 0;
  for (unsigned int step = 1; step < n_iter; ++step) {
    NumericVector U(runif(dim));
    i0 = -1;
    for (unsigned int i = 0; i < dim; ++i) {
      simulatedTime = getRandomTime(a(i), b(i), U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      // this means that we simulated T = inf
      stop("Zig zag wandered off to infinity.");
    }
    else {
      currentTime = currentTime + minTime;
      beta = beta + minTime * theta;
      unsigned int J = floor(n_observations*runif(1)(0)); // randomly select observation
      double switch_rate = theta(i0) * n_observations * dataXpp(i0,J) * (1.0/(1.0+exp(-dataXpp.col(J).dot(beta)))-1.0/(1.0+exp(-dataXpp.col(J).dot(mode))));
      double simulated_rate = a(i0) + b(i0) * minTime;
      if (switch_rate > simulated_rate) {
        stop("LogisticCVZZ:: Error: switch rate larger than its supposed upper bound.");
        //        Rprintf("  Step: %d, Upper bound: %g, actual switch rate: %g.\n", step, simulated_rate, switch_rate);
        //        Rprintf("  Index: %d, Beta(0): %g, Beta(1): %g\n", i0, beta(0), beta(1));
        break;
      }
      double V = runif(1)(0);
      if (V <= switch_rate/simulated_rate) {
        theta(i0)=-theta(i0);
        ++switches;
        if (switches >= max_switches) {
          max_switches *= 2;
          //          Rprintf("Resizing to size %d...\n", max_switches);
          skeletonTimesTemp.conservativeResize(max_switches);
          skeletonPointsTemp.conservativeResize(dim, max_switches);
        }
        skeletonTimesTemp[switches] = currentTime;
        skeletonPointsTemp.col(switches) = beta;
      }
      a = (beta-mode).norm() * uniformBound;
    }
  }
  Rprintf("LogisticCVZZ: Fraction of accepted switches: %g\n", double(switches)/n_iter);
  skeletonTimes = skeletonTimesTemp.head(switches + 1);
  skeletonPoints = skeletonPointsTemp.leftCols(switches + 1);
}

void Skeleton::GaussianZZ(const MatrixXd& V, const VectorXd& mu, const unsigned int n_steps, const VectorXd& x0) {
  // Gaussian skeleton
  // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_steps number of switches
  // invariant: w = V theta, z = V (x-mu)

  const unsigned int dim = V.cols();
  VectorXd x(x0);
  VectorXd theta = VectorXd::Constant(dim, 1); // initialize theta at (+1,...,+1)
  ArrayXd w(V * theta), z(V * (x-mu));
  ArrayXd a(theta.array() * z), b(theta.array() * w); // convert to array for pointwise multiplication

  RNGScope scp; // initialize random number generator

  double minTime, simulatedTime;
  int i0;
  skeletonPoints = MatrixXd(dim,n_steps);
  skeletonPoints.col(0) = x0;
  skeletonTimes = ArrayXd(n_steps);
  double currentTime = 0;

  for (unsigned int step = 1; step < n_steps; ++step) {
    NumericVector U(runif(dim));
    i0 = -1;
    for (unsigned int i = 0; i < dim; ++i) {
      simulatedTime = getRandomTime(a(i), b(i), U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      // this means that we simulated T = inf
      stop("Zig zag wandered off to infinity.");
    }
    else {
      currentTime = currentTime + minTime;
      x = x + minTime * theta;
      theta(i0) = -theta(i0);
      z = z + w * minTime; // preserve invariant  z = V (x-mu)
      w = w + 2 * theta(i0) * V.col(i0).array(); // preserve invariant w = V theta
      a = theta.array() * z;
      b = theta.array() * w;
      skeletonTimes(step) = currentTime;
      skeletonPoints.col(step) = x;
    }
  }
}

VectorXd resampleVelocity(const unsigned int dim, const bool unit_velocity = true) {
  // helper function for GaussianBPS
  VectorXd v = as<Eigen::Map<VectorXd> >(rnorm(dim));
//  std::cout << v(0) << ", " << v(1) << std::endl;
  if (unit_velocity)
    v.normalize();
  return v;
}

void Skeleton::GaussianBPS(const MatrixXd& V, const VectorXd& mu, const unsigned int n_steps, const VectorXd& x0, const double refresh_rate, const bool unit_velocity) {
  
  // Gaussian skeleton using BPS
  // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_steps number of switches
  
  if (refresh_rate < 0)
    stop("GaussianBPS error: refresh_rate should be non-negative.");

  const unsigned int dim = V.cols();
  VectorXd x(x0);
//  VectorXd v = VectorXd::Constant(dim, 1/sqrt(dim)); // initialize speed at (1/sqrt(d), ..., 1/sqrt(d))
  VectorXd v = resampleVelocity(dim, unit_velocity);

  RNGScope scp; // initialize random number generator
  
  double t, t_reflect, t_refresh;
  skeletonPoints = MatrixXd(dim,n_steps);
  skeletonPoints.col(0) = x0;
  skeletonTimes = ArrayXd(n_steps);
  double currentTime = 0;
  
  VectorXd gradient = V * (x - mu); // gradient
  VectorXd w = V * v; // useful invariant

  double a = v.dot(gradient);
  double b = v.dot(w);
  
  for (unsigned int step = 1; step < n_steps; ++step) {
    NumericVector U(runif(2));
    t_reflect = getRandomTime(a, b, U(0));
    if (refresh_rate <= 0) {
      t_refresh = -1; // indicating refresh rate = infinity
      t = t_reflect;
    }
    else {
      t_refresh = -log(U(1))/refresh_rate;
      t = (t_reflect < t_refresh ? t_reflect : t_refresh);
    }
    currentTime = currentTime + t;
    x = x + t * v;
    gradient = gradient + t * w;

    if (t_refresh < 0 || t_reflect < t_refresh) {
      VectorXd normalized_gradient = gradient.normalized(); // for projection
      VectorXd delta_v = -2 * (v.dot(normalized_gradient)) * normalized_gradient;
      v = v + delta_v;
    }
    else
      v = resampleVelocity(dim, unit_velocity);
//    Rprintf("%g\n", v.norm());
    w = V * v; // preserves invariant for w
    a = v.dot(gradient);
    b = v.dot(w);
    skeletonTimes(step) = currentTime;
    skeletonPoints.col(step) = x;
  }
}

List Skeleton::toR() {
  // output: R list consisting of skeletonTimes and skeletonPoints, and if samples are collected these too
  return List::create(Named("skeletonTimes") = skeletonTimes, Named("skeletonPoints") = skeletonPoints, Named("samples") = samples, Named("mode") = mode, Named("batchMeans") = batchMeans, Named("means") = means, Named("covariance") = covarianceMatrix, Named("asVarEst") = asVarEst, Named("ESS") = ESS);
}

Eigen::MatrixXd Skeleton::sample(const unsigned int n_samples) {

  const unsigned int n_steps = skeletonTimes.size();
  const unsigned int dim = skeletonPoints.rows();
  const double t_max = skeletonTimes(n_steps-1);
  const double dt = t_max / (n_samples+1);

  double t_current = dt;
  double t0 = skeletonTimes(0);
  double t1;
  VectorXd x0(skeletonPoints.col(0));
  VectorXd x1(dim);
  samples = MatrixXd(dim, n_samples);
  unsigned int n_sampled = 0; // number of samples collected

  for (unsigned int i = 1; i < n_steps; ++i) {
    x1 = skeletonPoints.col(i);
    t1 = skeletonTimes(i);
    while (t_current < t1 && n_sampled < n_samples) {
      samples.col(n_sampled) = x0 + (x1-x0) * (t_current - t0)/(t1-t0);
      ++n_sampled;
      t_current = t_current + dt;
    }
    x0 = x1;
    t0 = t1;
  }

  return samples;
}

//' ZigZagLogistic
//' 
//' Applies the Zig-Zag Sampler to logistic regression, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
//'
//'
//' @param dataX Matrix containing the independent variables x. The i-th column represents the i-th observation with components x_{1,i}, ..., x_{d-1,i}.
//' @param dataY Vector of length n containing {0, 1}-valued observations of the dependent variable y.
//' @param n_iter Integer indicating the number of iterations, i.e. the number of proposed switches.
//' @param subsampling Boolean. Use Zig-Zag with subsampling if TRUE. 
//' @param controlvariates Boolean. Use Zig-Zag with control variates if TRUE (overriding any value of \code{subsampling}).
//' @param beta0 Optional argument indicating the starting point for the Zig-Zag sampler
//' @param n_samples Number of discrete time samples to extract from the Zig-Zag skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @param upperbound Boolean. If TRUE, sample without subsampling and using a constant upper bound instead of a linear Hessian dependent upper bound
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes} Vector of switching times
//' @return \code{skeletonPoints} Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{samples} If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples at fixed intervals along the Zig-Zag trajectory. 
//' @return \code{mode} If \code{controlvariates = TRUE}, this is a vector containing the posterior mode obtained using Newton's method. 
//' @return \code{batchMeans} If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means} If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance} If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst} If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS} If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' require("RZigZag")
//' generate.logistic.data <- function(beta, nobs) {
//'   ncomp <- length(beta)
//'   dataX <- matrix(rnorm((ncomp -1) * nobs), nrow = ncomp -1);
//'   vals <- beta[1] + colSums(dataX * as.vector(beta[2:ncomp]))
//'   generateY <- function(p) { rbinom(1, 1, p)}
//'   dataY <- sapply(1/(1 + exp(-vals)), generateY)
//'   return(list(dataX, dataY))
//' }
//'
//' beta <- c(1,2)
//' data <- generate.logistic.data(beta, 1000)
//' result <- ZigZagLogistic(data[[1]], data[[2]], 1000, n_samples = 100)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List ZigZagLogistic(const Eigen::MatrixXd dataX, const Eigen::VectorXi dataY, const unsigned int n_iter, const bool subsampling = true, const bool controlvariates = true, const NumericVector beta0 = NumericVector(0), const unsigned int n_samples = 0, const unsigned int n_batches = 0, const bool computeCovariance = false, const bool upperbound = false) {
  const int dim = beta0.length();
  VectorXd beta(dim);
  for (int i = 0; i < dim; ++i)
    beta[i] = beta0[i];
  Skeleton skeleton;
  if (upperbound) {
//    Rprintf("ZZ-UB\n");
    skeleton.LogisticUpperboundZZ(dataX, dataY, n_iter, beta);
  }
  else if (controlvariates) {
//    Rprintf("ZZ-CV\n");
    skeleton.LogisticCVZZ(dataX, dataY, n_iter, beta);
  }
  else if (subsampling && !controlvariates) {
//    Rprintf("ZZ-SS\n");
    skeleton.LogisticSubsamplingZZ(dataX, dataY, n_iter, beta);
  }
  else {
//    Rprintf("ZZ\n");
    skeleton.LogisticBasicZZ(dataX, dataY, n_iter, beta);
  }
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}

//' ZigZagGaussian
//' 
//' Applies the Zig-Zag Sampler to a Gaussian target distribution, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
//' Assume potential of the form \code{Psi(x) = (x - mu)^T V (x - mu)/2}, i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
//'
//' @param V the inverse covariance matrix of the Gaussian target distribution
//' @param mu mean of the Gaussian target distribution
//' @param n_steps Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param x0 starting point
//' @param n_samples Number of discrete time samples to extract from the Zig-Zag skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @param c optional argument, specifies which fraction of the data is used to determine a reference point for ZZ-CV. Values for c < 1 give a suboptimal reference point.
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes}: Vector of switching times
//' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}
//' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples along the Zig-Zag trajectory.
//' @return \code{mode}: If \code{controlvariates = TRUE}, this is a vector containing the posterior mode obtained using Newton's method. 
//' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance} :If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' V <- matrix(c(3,1,1,3),nrow=2)
//' mu <- c(2,2)
//' x0 <- c(0,0)
//' result <- ZigZagGaussian(V, mu, 100, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List ZigZagGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, const unsigned int n_steps, const Eigen::VectorXd x0, const unsigned int n_samples=0, const unsigned int n_batches=0, bool computeCovariance=false, const double c = 1) {
/*  const unsigned int dim = V.cols();
  if (x0.size() == 0)
    x = VectorXd::Zero(dim); // start from origin by default
  else
    for (int i = 0; i < dim; ++i)
      x[i] = x0[i];
*/
  Skeleton skeleton;
  skeleton.GaussianZZ(V, mu, n_steps, x0);
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}


//' BPSGaussian
//' 
//' Applies the BPS Sampler to a Gaussian target distribution, as detailed in Bouchard-Côté et al, 2017.
//' Assume potential of the form \code{Psi(x) = (x - mu)^T V (x - mu)/2}, i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
//'
//' @param V the inverse covariance matrix of the Gaussian target distribution
//' @param mu mean of the Gaussian target distribution
//' @param n_steps Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param x0 starting point
//' @param refresh_rate \code{lambda_refresh}
//' @param unit_velocity TRUE indicates velocities uniform on unit sphere, FALSE indicates standard normal velocities
//' @param n_samples Number of discrete time samples to extract from the Zig-Zag skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @param c optional argument, specifies which fraction of the data is used to determine a reference point for ZZ-CV. Values for c < 1 give a suboptimal reference point.
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes}: Vector of switching times
//' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}
//' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples along the Zig-Zag trajectory.
//' @return \code{mode}: If \code{controlvariates = TRUE}, this is a vector containing the posterior mode obtained using Newton's method. 
//' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance} :If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' V <- matrix(c(3,1,1,3),nrow=2)
//' mu <- c(2,2)
//' x0 <- c(0,0)
//' result <- BPSGaussian(V, mu, 100, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List BPSGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, const unsigned int n_steps, const Eigen::VectorXd x0, const double refresh_rate = 1, const bool unit_velocity = true, const unsigned int n_samples=0, const unsigned int n_batches=0, bool computeCovariance=false, const double c = 1) {
  /*  const unsigned int dim = V.cols();
   if (x0.size() == 0)
   x = VectorXd::Zero(dim); // start from origin by default
   else
   for (int i = 0; i < dim; ++i)
   x[i] = x0[i];
   */
  Skeleton skeleton;
  skeleton.GaussianBPS(V, mu, n_steps, x0, refresh_rate, unit_velocity);
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}


double getRandomTime(double a, double b, double u) {
  // NOTE: Return value -1 indicates +Inf!
  if (b > 0) {
    if (a < 0)
      return -a/b + getRandomTime(0, b, u);
    else       // a >= 0
      return -a/b + sqrt(a*a/(b * b) - 2 * log(u)/b);
  }
  else if (b == 0) {
    if (a > 0)
      return -log(u)/a;
    else
      return -1; // infinity
  }
  else {
    // b  < 0
    if (a <= 0)
      return -1; // infinity
    else {
      // a > 0
      double t1 = -a/b;
      if (-log(u) <= a * t1 + b * t1 * t1/2)
        return -a/b - sqrt(a*a/(b * b) - 2 * log(u)/b);
      else
        return -1;
    }
  }
}

MatrixXd preprocessLogistic(const MatrixXd& dataX) {
  const unsigned int n_observations = dataX.cols();
  const unsigned int n_components = dataX.rows() + 1; // we will add a row of constant, so for practical purposes +1

  // CURRENTLY OFF: re-center the data around the origin. TODO: How does this affect the results?
  // VectorXd meanX = dataX.rowwise().sum()/n_observations;
  VectorXd meanX(VectorXd::Zero(n_components));
  MatrixXd dataXpp(n_components, n_observations);
  dataXpp.topRows(1) = MatrixXd::Constant(1, n_observations,1);
  for (unsigned int i = 0; i < n_observations; ++i)
    dataXpp.bottomRows(n_components - 1).col(i) = dataX.col(i) - meanX;
  return dataXpp;
}

MatrixXd domHessianLogistic(const MatrixXd& dataX) {
  const unsigned int n_observations = dataX.cols();
  const unsigned int dim = dataX.rows();

  MatrixXd domHessian(MatrixXd::Zero(dim,dim));
  for (unsigned int j = 0; j < n_observations; ++j) {
    domHessian += 0.25 * (dataX.col(j) * dataX.col(j).transpose());
  }
  return domHessian;
}


VectorXd cvBound(const MatrixXd& dataX) {
  const unsigned int dim = dataX.rows();
  const unsigned int n_observations = dataX.cols();
  const VectorXd norms (dataX.colwise().norm());
  VectorXd bounds(dim);

  for (unsigned int k =0; k < dim; ++k) {
    bounds(k) = 0.0;
    for (unsigned int l = 0; l < n_observations; ++l) {
      double val = fabs(dataX(k,l) * norms(l));
      if (bounds(k) < val)
        bounds(k) = val;
    }
  }
  return 0.25 * bounds;
}

double derivativeLogistic(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& beta, unsigned int k) {
// compute dPsi/dbeta_k for logistic regression

  const unsigned int n_observations = dataX.cols();
  const unsigned int n_components = dataX.rows();
  double derivative = 0;

  for (unsigned int j = 0; j < n_observations; ++j) {
    double val = exp(dataX.col(j).dot(beta));
    derivative += dataX(k,j) * (val/(1+val) - dataY(j));
  }
  return derivative;
}

VectorXd logisticUpperbound(const MatrixXd& dataX) {
  return dataX.array().abs().rowwise().maxCoeff();
}

double newtonLogistic(const LogisticData& data, VectorXd& beta, double precision, const unsigned int max_iter) {
  VectorXd grad(data.gradient(beta));
  unsigned int i = 0;
  for (i = 0; i < max_iter; ++i) {
    if (grad.norm() < precision)
      break;
    MatrixXd H(data.hessian(beta));
    beta -= H.ldlt().solve(grad);
    grad = data.gradient(beta);
  }
  if (i == max_iter) {
    Rprintf("Maximum number of iterations (", max_iter, ") reached without convergence in Newton's method in computing control variate.");
  }
  else
    Rprintf("Newton: Converged to local minimum in %d iterations.\n", i);
  return data.potential(beta);
}

