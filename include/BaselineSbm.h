///
/// @file    BaselineSBM.h
/// @author  Bryan Roger Leon Guillaume
///

#ifndef SBM_H
#define SBM_H

#include <armadillo>
#include "VaryingTau.h"
#include "VaryingAlpha.h"
#include "VaryingPi.h"

using namespace std;
using namespace arma;
/// @brief base class of SBM
/// Extended documentation for this class.
/// @author Bryan Roger Leon Guillaume

// enum DENSITY
// {
//   Bernouilli,
//   Binomial,
//   Normal,
//   Poisson
// };

class BaselineSbm 
{
private:
  // members that are fixed
  uword nClusters;
  bool directed;
  const ucube* pAdjacencyMatrices;
  const umat* pStartingNodeAssignment;
  bool verbose;

  // convergence criteria
  bool convergence;
  uword nIterSBM;
  double relConvTol;
  // double convergenceDeltaForTau;
  // double convergenceDeltaForPi;
  // double convergenceDeltaForAlpha;
  // double convergenceDeltaForVarBound;
  
  // members to update
  VaryingTau varyingTau;
  VaryingAlpha varyingAlpha;
  VaryingPi varyingPi;

  colvec icl;
  double iclPenalisation;
public:
  BaselineSbm(const ucube* pAdjacencyMatrices, bool directed, const mat* pDesignMatrixAlpha, const mat* pDesignMatrixPi, uword nClusters, uword nIterSBM, uword nIterTau, uword nIterAlpha, uword nIterPi, uword nIterHalvingAlpha, uword nIterHalvingPi, double maxDeltaBeta, double minTau, double minAlpha, double minDeltaTau, const umat* pStartingNodeAssignment, double relTolConvergenceIcl, bool verbose);
  void estimateModel();
  void estimateNodeAssignement();
  void computeIcl(uword iIterSBM);
  void saveInDirectory(string dirName);

};

#endif // SBM_H
