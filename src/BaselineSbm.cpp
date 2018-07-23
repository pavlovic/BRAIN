///
/// @file    BaselineSBM.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "BaselineSbm.h"

BaselineSbm::BaselineSbm(const ucube* pAdjacencyMatrices, bool directed, const mat* pDesignMatrixAlpha, const mat* pDesignMatrixPi, uword nClusters, uword nIterSBM, uword nIterTau, uword nIterAlpha, uword nIterPi, uword nIterHalvingAlpha, uword nIterHalvingPi, double maxDeltaBeta, double minTau, double minAlpha, double minDeltaTau, const umat* pStartingNodeAssignment, double relConvTol, bool verbose) 
: pAdjacencyMatrices(pAdjacencyMatrices), directed(directed), nClusters(nClusters), nIterSBM(nIterSBM), pStartingNodeAssignment(pStartingNodeAssignment), convergence(false), verbose(verbose),
    varyingTau(pAdjacencyMatrices, pStartingNodeAssignment, nClusters, nIterTau, minTau, relConvTol),
    varyingAlpha(pDesignMatrixAlpha, nIterAlpha, nIterHalvingAlpha, maxDeltaBeta, minAlpha, relConvTol, varyingTau.getTau()),
    varyingPi(pDesignMatrixPi, nIterPi, nIterHalvingPi, maxDeltaBeta, relConvTol, varyingTau.getTau(), pAdjacencyMatrices),
    relConvTol(relConvTol)
{
    uword nNodes = (*pAdjacencyMatrices).n_slices;
    uword nSubjects =  (*pDesignMatrixPi).n_rows;
    icl = zeros<colvec>(1);
    iclPenalisation = 0.25  * nClusters * (nClusters + 1) * (*pDesignMatrixPi).n_cols  * log(0.5 * nNodes * (nNodes - 1) * nSubjects)
                     + 0.5 * (nClusters - 1) * (*pDesignMatrixAlpha).n_cols * nNodes * log (nNodes * nSubjects);
    variationalBound = zeros<colvec>(nIterSBM);
    varyingTau.assignPointers(varyingAlpha.getAugmentedAlpha(), varyingPi.getPi());
}

void BaselineSbm::estimateModel()
{
    // Optimise the SBM model until convergence
    uword iIterSBM = 0;
    while (!convergence && iIterSBM < nIterSBM)
    {
        if(verbose)
            cout << "SBM iteration #" << iIterSBM << "\nComputing tau... "<< endl;
        if(iIterSBM > 0)
            varyingTau.updateTau();

        if(verbose)
            cout << "Computing parameters for alpha..." << endl;
        varyingAlpha.computeBeta();

        if(verbose)
            cout << "Computing parameters for pi..." << endl;
        varyingPi.computeBeta();

        // compute the variational Bound
        BaselineSbm::computeVariationalBound(iIterSBM);
        if(verbose)
            cout << "Variational bound: " << variationalBound[iIterSBM] << endl;  

        // evaluate if there is convergence
        if(iIterSBM > 0 && (variationalBound[iIterSBM] - variationalBound[iIterSBM - 1]) < ( abs(variationalBound[iIterSBM - 1]) * relConvTol ) )
            convergence = true;
        
        iIterSBM++;
    }

    if(iIterSBM < nIterSBM)
    {
        // trim variationalBound 
        variationalBound.resize(iIterSBM);
    }

	BaselineSbm::computeIcl();

}
void BaselineSbm::computeVariationalBound(uword iIterSBM)
{
    variationalBound[iIterSBM] = accu( trimatu( *(varyingPi.getBlockLogLiks()) ) ) 
                    + accu( *(varyingAlpha.getNodalLogLiks()) )
                    - accu ( *(varyingTau.getTau()) % log( *(varyingTau.getTau()) ) );
}

void BaselineSbm::computeIcl()
{
    icl = accu( trimatu( *(varyingPi.getBlockLogLiks()) ) ) 
                    + accu( *(varyingAlpha.getNodalLogLiks()) )
                    - iclPenalisation;
}

void BaselineSbm::saveInDirectory(string dirName)
{
    varyingTau.printTau(dirName + "/tau.txt");
    varyingTau.computeNodeAssignment().save(dirName + "/nodeAssignmentBasedOnTau.txt", arma_ascii);

    varyingAlpha.printAlpha(dirName + "/alpha.txt");
    varyingAlpha.printAugmentedAlpha(dirName + "/augmentedAlpha.txt");
    varyingAlpha.printBeta(dirName + "/beta_alpha.txt");
    varyingAlpha.computeNodeAssignment().save(dirName + "/nodeAssignmentBasedOnAlpha.txt", arma_ascii);

    varyingPi.printPi(dirName + "/pi.txt");
    varyingPi.printBeta(dirName + "/beta_pi.txt");
    
    variationalBound.save(dirName + "/variationalBound.txt", arma_ascii);
    icl.save(dirName + "/icl.txt", arma_ascii);

}
