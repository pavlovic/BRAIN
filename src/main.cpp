///
/// @file    main.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "main.h"
#include "SbmVersion.h"
#include "BaselineSbm.h"
#include "VaryingTau.h"
#include "VaryingAlpha.h"
#include "VaryingPi.h"

#include <iostream>
#include <string>

#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv) {
    
    cout << "SBM Project v" << SBM_VERSION << endl;
    
    uword nClusters = atoi(argv[5]);
    uword nIterSBM = 100;
    uword nIterTau = 100;
    uword nIterAlpha= 1000;
    uword nIterPi= 1000;
    uword nIterHalvingAlpha = 200;
    uword nIterHalvingPi = 200;
    double maxDeltaBeta = 5;
    double minAlpha = 1.0e-10;
    double minTau = 1.0e-10;
    double minDeltaTau = 1.0e-10;
    bool directed = false;
    bool verbose = true;
    double relTolConvergenceIcl = 1.0e-10;

    if (verbose)
        cout << "Loading data..." << endl;
    umat startingNodeAssignment; // = repmat(regspace<uvec>(1,nClusters), nNodes / nClusters, 1)
    startingNodeAssignment.load(argv[1], arma_ascii);
    ucube adjacencyMatrices;
    adjacencyMatrices.load(argv[2], arma_ascii);
    mat designMatrix;
    designMatrix.load(argv[3], arma_ascii);

    if (verbose)
        cout << "Initialising SBM model..." << endl;   
    BaselineSbm sbm(&adjacencyMatrices, directed, &designMatrix, &designMatrix, 
                    nClusters, nIterSBM, nIterTau, nIterAlpha, nIterPi, nIterHalvingAlpha, nIterHalvingPi, 
                    maxDeltaBeta, minTau, minAlpha, minDeltaTau, &startingNodeAssignment, relTolConvergenceIcl, verbose);

    if (verbose)
        cout << "Estimating SBM model..." << endl;   
    sbm.estimateModel();

    sbm.saveInDirectory(argv[4]);

    return MAIN_SUCCESS;
}

