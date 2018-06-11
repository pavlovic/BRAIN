///
/// @file    BaselineSBM.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "BaselineSbm.h"

BaselineSbm::BaselineSbm(const ucube* pAdjacencyMatrices, bool directed, const mat* pDesignMatrixAlpha, const mat* pDesignMatrixPi, uword nClusters, uword nIterSBM, uword nIterTau, uword nIterAlpha, uword nIterPi, uword nIterHalvingAlpha, uword nIterHalvingPi, double maxDeltaBeta, double minTau, double minAlpha, double minDeltaTau, const umat* pStartingNodeAssignment, double relTolConvergenceIcl, bool verbose) 
: pAdjacencyMatrices(pAdjacencyMatrices), directed(directed), nClusters(nClusters), nIterSBM(nIterSBM), pStartingNodeAssignment(pStartingNodeAssignment), convergence(false), verbose(verbose),
    varyingTau(pAdjacencyMatrices, pStartingNodeAssignment, nClusters, nIterTau, minTau, minDeltaTau),
    varyingAlpha(pDesignMatrixAlpha, nIterAlpha, nIterHalvingAlpha, maxDeltaBeta, minAlpha, varyingTau.getTau()),
    varyingPi(pDesignMatrixPi, nIterPi, nIterHalvingPi, maxDeltaBeta, varyingTau.getTau(), pAdjacencyMatrices),
    relTolConvergenceIcl(relTolConvergenceIcl)
{
    uword nNodes = (*pAdjacencyMatrices).n_slices;
    uword nSubjects =  (*pDesignMatrixPi).n_rows;
    iclPenalisation = 0.25  * nClusters * (nClusters + 1) * (*pDesignMatrixPi).n_cols  * log(0.5 * nNodes * (nNodes - 1) * nSubjects)
                     + 0.5 * (nClusters - 1) * (*pDesignMatrixAlpha).n_cols * nNodes * log (nNodes * nSubjects);
    icl = zeros<colvec>(nIterSBM);
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

        // compute alpha and pi model parameters given tau        
        if(verbose)
            cout << "Computing parameters for alpha..." << endl;
        varyingAlpha.computeBeta();
        if(verbose)
            cout << "Computing parameters for pi..." << endl;
        varyingPi.computeBeta();

        // compute ICL
        BaselineSbm::computeIcl(iIterSBM);
        if(verbose)
            cout << "ICL score: " << icl[iIterSBM] << endl;  

        // evaluate if there is convergence
        if(iIterSBM > 0 && (icl[iIterSBM] - icl[iIterSBM - 1]) < ( abs(icl[iIterSBM - 1]) * relTolConvergenceIcl ) )
            convergence = true;
        
        iIterSBM++;
    }

    if(iIterSBM < nIterSBM)
    {
        // trim icl 
        icl.resize(iIterSBM);
    }
	
}

void BaselineSbm::computeIcl(uword iIterSBM)
{
    icl[iIterSBM] = accu( trimatu( *(varyingPi.getBlockLogLiks()) ) ) 
                    + accu( *(varyingAlpha.getNodalLogLiks()) )
                    - accu ( *(varyingTau.getTau()) % log( *(varyingTau.getTau()) ) )  
                    - iclPenalisation;
                    // cout << accu( trimatu( *(varyingPi.getBlockLogLiks()) ) ) << endl;
                    // cout << accu( *(varyingAlpha.getNodalLogLiks()) ) << endl;
                    // cout << -accu ( *(varyingTau.getTau()) % log( *(varyingTau.getTau()) ) )  << endl;
                    // cout << -iclPenalisation << endl;
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
    
    icl.save(dirName + "/icl.txt", arma_ascii);

}
