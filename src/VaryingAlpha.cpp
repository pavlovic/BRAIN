///
/// @file    VaryingAlpha.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "VaryingAlpha.h"
#include <iostream>
#include <string>
using namespace std;

VaryingAlpha::VaryingAlpha(const mat* pDesignMatrix, uword nIterNR, uword nIterHalving, double maxDeltaBeta, double minAlpha, const cube* pTau)
: pDesignMatrix(pDesignMatrix), nIterNR(nIterNR), nIterHalving(nIterHalving), maxDeltaBeta(maxDeltaBeta), minAlpha(minAlpha), pTau(pTau) 
{
    nSubjects = (*pDesignMatrix).n_rows;
    nCovariates = (*pDesignMatrix).n_cols;
    nClusters = (*pTau).n_rows;
    nNodes = (*pTau).n_slices;
    uword nPairCovariates = (nCovariates * (nCovariates + 1)) / 2;
    designMatrix2.set_size(nCovariates, nCovariates, nSubjects);
    for (uword i = 0; i < nSubjects; i++)
        designMatrix2.slice(i) = (*pDesignMatrix).row(i).t() * (*pDesignMatrix).row(i);

    nodalConvergences = zeros<ucolvec>(nNodes);

    // set beta to 0 as starting beta; might be changed later by something else
    beta = zeros<mat>((nClusters - 1) * nCovariates, nNodes);
    alpha.set_size(nClusters - 1, nSubjects, nNodes);
    VaryingAlpha::computeAlpha();
    augmentedAlpha.set_size(nClusters, nSubjects, nNodes);

    nodalLogLiks.set_size(nNodes);

    score.set_size((nClusters - 1) * nCovariates, nNodes);
    fisherInformation.set_size((nClusters - 1) * nCovariates, (nClusters - 1) * nCovariates, nNodes);
  
}

void VaryingAlpha::computeNodalFisherInformation(uword index)
{
    mat nodalFisherInformation = zeros<mat>((nClusters - 1) * nCovariates, (nClusters - 1) * nCovariates);
    mat nodalAlpha = alpha.slice(index);
    colvec tmp;
    for (uword ii = 0; ii < nSubjects; ii++)
    {
        tmp = nodalAlpha.col(ii);
        nodalFisherInformation += kron(designMatrix2.slice(ii), diagmat(tmp) - (tmp * tmp.t()));
    }
    fisherInformation.slice(index) = nodalFisherInformation;
}

void VaryingAlpha::computeFisherInformation()
{
    for (uword i = 0; i < nNodes; i++)
        VaryingAlpha::computeNodalFisherInformation(i);
}

// Code related to the computation of the score
void VaryingAlpha::computeNodalScore(uword index)
{
    mat tmp = ((*pTau).slice(index).head_rows(nClusters-1) - alpha.slice(index)) * (*pDesignMatrix);
    score.col(index) = vectorise(tmp);
}
void VaryingAlpha::computeScore()
{
    for (uword i = 0; i < (nClusters - 1); i++)
    {
        VaryingAlpha::computeNodalScore(i);
    }
}
void VaryingAlpha::computeNodalBeta(uword index)
{
    nodalConvergences[index] = 0;
    colvec deltaBeta = zeros<colvec>((nClusters - 1) * nCovariates);
    uword iIterNR = 0; // NR for Newton-Raphson

    // compute the initial nodal Log-likelihood based on the last updated tau and the previous alpha
    nodalLogLiks[index] = VaryingAlpha::computeNodalLogLik(index, deltaBeta);
    double nodalLogLik;
    while (nodalConvergences[index] == 0 && iIterNR < nIterNR)
    {
        VaryingAlpha::computeNodalScore(index);
        VaryingAlpha::computeNodalFisherInformation(index);
        // compute the change in beta and clamp it to avoid overshooting away from the solution
        deltaBeta = clamp(solve(fisherInformation.slice(index), score.col(index)), -maxDeltaBeta, maxDeltaBeta);
        // use halving procedure if the change in beta do not increase the log-likelihood
        nodalLogLik = computeNodalLogLik(index, deltaBeta);
        uword iIterHalving = 0;
        while (nodalLogLik < nodalLogLiks[index] && iIterHalving < nIterHalving)
        {
            deltaBeta = 0.5 * deltaBeta;
            nodalLogLik = computeNodalLogLik(index, deltaBeta);
            iIterHalving++;
        }
        // update beta if there is an improvement
        if (nodalLogLik > nodalLogLiks[index])
        {
            beta.col(index) += deltaBeta; // TODO FI and score needs to be updated somewhere unless no clamping an no halving was used
            VaryingAlpha::computeNodalAlpha(index);
            nodalLogLiks[index] = nodalLogLik;
        }
        // state convergence if there is no improvement and do not update beta (i.e. keep the previous value)
        else
        {
            nodalConvergences[index] = 1;
        }
        iIterNR++;
    }
}
void VaryingAlpha::computeBeta()
{
    // reset beta to zero; this seems to be more stable than starting with the previous beta
    beta.zeros();
    VaryingAlpha::computeAlpha();
    for (uword i = 0; i < nNodes; i++)
        VaryingAlpha::computeNodalBeta(i);
    
    // update also augmented alpha as it might be used by other classes
    VaryingAlpha::computeAugmentedAlpha();
}
void VaryingAlpha::computeNodalAlpha(uword index)
{
    // note that the last row for the last cluster is not computed
    mat tmpBeta = reshape(beta.col(index), nClusters - 1, nCovariates);
    mat tmpAlpha = exp(tmpBeta * (*pDesignMatrix).t());
    rowvec sumTmpAlpha = sum(tmpAlpha) + 1;
    tmpAlpha.each_row() /= sumTmpAlpha;
  
    // ajust to avoid zero values
    alpha.slice(index) = tmpAlpha / (1 + nClusters * minAlpha) + minAlpha / (1 + nClusters * minAlpha);
}

void VaryingAlpha::computeAlpha()
{
    for (uword i = 0; i < nNodes; i++)
        VaryingAlpha::computeNodalAlpha(i);
}

void VaryingAlpha::computeAugmentedAlpha()
{
    augmentedAlpha(span(0, nClusters -2), span::all, span::all) = alpha;
    for (uword i = 0; i < nNodes; i++)
        augmentedAlpha(span(nClusters -1), span::all, span(i)) = 1.0-sum(alpha.slice(i));
}

double VaryingAlpha::computeNodalLogLik(const uword index, const colvec &deltaBeta)
{
    colvec newBeta = beta.col(index) + deltaBeta;
    mat tmp = (*pTau).slice(index).head_rows(nClusters-1) * (*pDesignMatrix);
    double nodalLogLik = dot(vectorise(tmp), newBeta);
    mat tmpBeta = reshape(newBeta, nClusters - 1, nCovariates);
    rowvec tmpSumAlpha = sum(exp(tmpBeta * (*pDesignMatrix).t()));
    nodalLogLik -= accu(log(tmpSumAlpha + 1.0));
    return nodalLogLik;
}

// get pointers
const cube* VaryingAlpha::getAlpha() const
{
    return &alpha;
}
const cube* VaryingAlpha::getAugmentedAlpha() const
{
    return &augmentedAlpha;
}
const mat* VaryingAlpha::getBeta() const
{
    return &beta;
}
const colvec* VaryingAlpha::getNodalLogLiks() const
{
    return &nodalLogLiks;   
}

// member print 
void VaryingAlpha::printAlpha()
{
    alpha.print();
}

void VaryingAlpha::printAlpha(string filename)
{
    alpha.save(filename, arma_ascii);
}

void VaryingAlpha::printAugmentedAlpha()
{
    augmentedAlpha.print();
}

void VaryingAlpha::printAugmentedAlpha(string filename)
{
    augmentedAlpha.save(filename, arma_ascii);
}

void VaryingAlpha::printBeta()
{
    beta.print();
}

void VaryingAlpha::printBeta(string filename)
{
    beta.save(filename, arma_ascii);
}

umat VaryingAlpha::computeNodeAssignment()
{
    mat tmpSlice;
    colvec tmpVec;
    umat nodeAssignment(nSubjects, nNodes);
    for (uword i = 0; i < nNodes; i++)
    {
        tmpSlice = augmentedAlpha.slice(i);
        for (uword ii = 0; ii < nSubjects; ii++)
        {
            tmpVec = tmpSlice.col(ii);
            nodeAssignment.at(ii,i) = tmpVec.index_max() + 1;
        }
    }
    return nodeAssignment;
}