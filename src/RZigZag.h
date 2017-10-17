// RZigZag.h : implements Zig-Zag with sub-sampling and control variates (ZZ-CV)
//
// Copyright (C) 2017 Joris Bierkens
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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
double getRandomTime(double a, double b, double u); // simulate T such that P(T>= t) = exp(-at-bt^2/2), using uniform random input u

MatrixXd preprocessLogistic(const MatrixXd& dataX); // center data and add a row of ones
MatrixXd domHessianLogistic(const MatrixXd& dataX); // compute dominating Hessian for logistic regression
double derivativeLogistic(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& beta, unsigned int k); // k-th derivative of potential
VectorXd logisticUpperbound(const MatrixXd& dataX);
VectorXd cvBound(const MatrixXd& dataX);

class ZigZagSkeleton {
public:
  MatrixXd sample(const unsigned int n_samples);
  List toR();
  void LogisticBasic(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_epochs, VectorXd beta = VectorXd::Zero(0)); // logistic regression with zig zag
  void LogisticUpperbound(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_epochs, VectorXd beta0 = VectorXd::Zero(0)); 
  void LogisticSubsampling(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_epochs, VectorXd beta = VectorXd::Zero(0));
  void LogisticControlVariates(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_epochs, VectorXd beta = VectorXd::Zero(0));
  void GaussianBasic(const MatrixXd& V, const VectorXd& mu, const unsigned int n_steps, const VectorXd& x0); // sample Gaussian with precision matrix V
  void computeBatchMeans(const unsigned int n_batches);
  void computeCovariance();

private:
  ArrayXd skeletonTimes;
  MatrixXd skeletonPoints;
  MatrixXd samples;
  VectorXd mode;
  MatrixXd batchMeans;
  VectorXd means;
  MatrixXd covarianceMatrix;
  VectorXd asVarEst;
  VectorXd ESS;
};

class LogisticData {
public:
  LogisticData(const MatrixXd* dataXptr, const VectorXi* dataYptr);
  double potential(const VectorXd& beta) const;
  VectorXd gradient(const VectorXd& beta) const;
  MatrixXd hessian(const VectorXd& beta) const;
private:
  unsigned int dim, n_observations;
  const MatrixXd* dataXptr;
  const VectorXi* dataYptr;
};

double newtonLogistic(const LogisticData& data, VectorXd& beta, double precision, const unsigned int max_iter);

MatrixXd LogisticMALA(const MatrixXd& dataX, const VectorXi& dataY, const unsigned int n_epochs, const VectorXd& beta0, const double stepsize);
