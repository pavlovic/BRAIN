///
/// @file    VaryingPi.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "VaryingPi.h"
#include <iostream>
#include <string>
using namespace std;

VaryingPi::VaryingPi(const mat* pDesignMatrix, uword nIterNR, uword nIterHalving, double maxDeltaBeta, const cube *pTau, const ucube *pAdjacencyMatrices) 
    : pDesignMatrix(pDesignMatrix), nIterNR(nIterNR), nIterHalving(nIterHalving), pTau(pTau), pAdjacencyMatrices(pAdjacencyMatrices), maxDeltaBeta(maxDeltaBeta)
{
    nCovariates = (*pDesignMatrix).n_cols;
    nSubjects  =  (*pDesignMatrix).n_rows;
    nClusters = (*pTau).n_rows;    
    nNodes = (*pTau).n_slices;

    VaryingPi::computeSumGamma();

    // set beta to 0 as starting beta; might be changed later by something better    
    beta = zeros<mat>(nCovariates, nClusters * nClusters);
    pi.set_size(nSubjects, nClusters * nClusters);
    VaryingPi::computePi();

    blockLogLiks.set_size(nClusters, nClusters);

    score.set_size(nCovariates, nClusters * nClusters);
    fisherInformation.set_size(nCovariates, nCovariates, nClusters * nClusters);
    blockConvergences = zeros<umat>(nClusters, nClusters);

    designMatrix2.set_size(nCovariates, nCovariates, nSubjects);
    for (uword i = 0; i < nSubjects; i++)
        designMatrix2.slice(i) = (*pDesignMatrix).row(i).t() * (*pDesignMatrix).row(i);
    
}
void VaryingPi::computeBlockPi(uword index1, uword index2)
{
    uword index = index1 + index2 * nClusters;
    uword indexBis = index2 + index1 * nClusters;  
    colvec tmp = 1.0 / (1.0 + exp(-(*pDesignMatrix) * beta.col(index)));   
    pi.col(index) = tmp;
    if (index != indexBis)
        pi.col(indexBis) = tmp;
}

void VaryingPi::computePi()
{
    pi = 1.0 / (1.0 + exp(-(*pDesignMatrix) * beta));
}

void VaryingPi::computeSumGamma()
{
    sumHalfGamma = zeros<mat>(nClusters * nClusters, nSubjects);
    sumHalfMaskedGamma = zeros<mat>(nClusters * nClusters, nCovariates);
    mat tmpSlice1, tmpSlice2, tmp;
    umat tmpAdj1;
    ucolvec tmpAdj2;
    for (uword i = 0; i < nNodes; i++)
    {
        tmpSlice1 = (*pTau).slice(i);
        tmpAdj1 = (*pAdjacencyMatrices).slice(i);
        for (uword ii = (i + 1); ii < nNodes; ii++)
        {
            tmpSlice2 = (*pTau).slice(ii);
            tmpAdj2 = tmpAdj1.col(ii);
            for (uword iii = 0; iii < nSubjects; iii++)
            {
                tmp = tmpSlice1.col(iii) * tmpSlice2.col(iii).t();
                tmp.diag() *= 0.5;
                tmp = tmp + tmp.t();
                sumHalfGamma.col(iii) += vectorise(tmp);       
                for (uword iiii = 0; iiii < nCovariates; iiii++)
                    sumHalfMaskedGamma.col(iiii) += vectorise(tmp) * tmpAdj2[iii] * (*pDesignMatrix).at(iii, iiii);
            }
        }
     }
}

double VaryingPi::computeBlockLogLik(uword index1, uword index2, const colvec &deltaBeta)
{
    uword index = index1 + index2 * nClusters;    
    colvec newBeta = beta.col(index) + deltaBeta;
    double blockLogLik = dot(sumHalfMaskedGamma.row(index), newBeta) - dot( sumHalfGamma.row(index), log( 1 + exp( (*pDesignMatrix) * newBeta ) ) );
    return blockLogLik; 
}
void VaryingPi::computeBlockFisherInformation(uword index1, uword index2)
{
    uword index = index1 + index2 * nClusters;
    colvec tmp = pi.col(index);
    tmp = ( sumHalfGamma.row(index).t() % ( tmp % (1.0 - tmp) ) );
    mat blockFisherInformation = zeros<mat>(nCovariates, nCovariates);
    for (uword i = 0; i < nSubjects; i++)
        blockFisherInformation += tmp[i] * designMatrix2.slice(i);
    fisherInformation.slice(index) = blockFisherInformation;
}

void VaryingPi::computeBlockScore(uword index1, uword index2)
{
    uword index = index1 + index2 * nClusters;
    score.col(index) = sumHalfMaskedGamma.row(index).t() - 
                    (*pDesignMatrix).t() * (sumHalfGamma.row(index).t() % pi.col(index)); 
}

void VaryingPi::computeBlockBeta(uword index1, uword index2)
{
    uword index = index1 + index2 * nClusters; 
    uword indexBis = index2 + index1 * nClusters; 
    blockConvergences.at(index1, index2) = 0;
    colvec deltaBeta =  zeros<colvec>(nCovariates);
    uword iIterNR = 0; // NR for Newton-Raphson

    // compute the block log-likilihood based on the updated tau and the previous pi
    double blockLogLik = VaryingPi::computeBlockLogLik(index1, index2, deltaBeta);
    blockLogLiks.at(index1, index2) = blockLogLik;
    if (index1 != index2)
        blockLogLiks.at(index2, index1) = blockLogLik;
        
    while (blockConvergences.at(index1, index2) == 0 && iIterNR < nIterNR)
    {
        VaryingPi::computeBlockScore(index1, index2);
        VaryingPi::computeBlockFisherInformation(index1, index2);
        // compute the change in beta and clamp it to avoid overshooting away from the solution
        deltaBeta = clamp(solve(fisherInformation.slice(index), score.col(index)), -maxDeltaBeta, maxDeltaBeta);
        // use halving procedure if the change in beta do not increase the log-likelihood
        blockLogLik = computeBlockLogLik(index1, index2, deltaBeta);
        uword iIterHalving = 0;
        while (blockLogLik < blockLogLiks.at(index1, index2) && iIterHalving < nIterHalving)
        {
            deltaBeta = 0.5 * deltaBeta;
            blockLogLik = computeBlockLogLik(index1, index2, deltaBeta);
            iIterHalving++;
        }
        // update beta if there is an improvement
        if (blockLogLik > blockLogLiks.at(index1, index2))
        {
            beta.col(index) += deltaBeta; // TODO FI and score needs to be updated somewhere unless no clamping an no halving was used
            // update the symmetric value if needed
            if (indexBis != index) 
                beta.col(indexBis) = beta.col(index);
            
            // update pi fitted
            VaryingPi::computeBlockPi(index1, index2);
            blockLogLiks.at(index1, index2) = blockLogLik;
            blockLogLiks.at(index2, index1) = blockLogLik;
        }
        // state convergence if there is no improvement and do not update beta (i.e. keep the previous value)
        else
        {
            blockConvergences.at(index1, index2) = 1;
            blockConvergences.at(index2, index1) = 1;
        }
        iIterNR++;
    }
}

void VaryingPi::computeBeta()
{
    // update some variables depending on tau
    VaryingPi::computeSumGamma();
    for (uword i = 0; i < nClusters; i++)
        for (uword ii = i; ii < nClusters; ii++)
            VaryingPi::computeBlockBeta(i, ii);
}

const mat* VaryingPi::getPi() const
{
    return &pi;
}

const mat* VaryingPi::getBlockLogLiks() const
{
    return &blockLogLiks;
}

// member print 
void VaryingPi::printPi()
{
    VaryingPi::reshapePi().print();
}

void VaryingPi::printPi(string filename)
{
    VaryingPi::reshapePi().save(filename, arma_ascii);
}

void VaryingPi::printBeta()
{
    VaryingPi::reshapeBeta().print();
}

void VaryingPi::printBeta(string filename)
{
    VaryingPi::reshapeBeta().save(filename, arma_ascii);
}

cube VaryingPi::reshapePi()
{
    cube cubicPi(nSubjects, nClusters, nClusters);
    uword it = 0;
    for (uword i = 0; i < nClusters; i++)
      for (uword ii = 0; ii < nClusters; ii++)
      {
        cubicPi.slice(i).col(ii) = pi.col(it);
        it++;
      }

    return cubicPi;
}

cube VaryingPi::reshapeBeta()
{
    cube cubicBeta(nCovariates, nClusters, nClusters);
    uword it = 0;
    for (uword i = 0; i < nClusters; i++)
      for (uword ii = 0; ii < nClusters; ii++)
      {
        cubicBeta.slice(i).col(ii) = beta.col(it);
        it++;
      }

    return cubicBeta;
}
