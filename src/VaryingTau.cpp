///
/// @file    VaryingTau.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "VaryingTau.h"

using namespace std;
VaryingTau::VaryingTau(const ucube* pAdjacencyMatrices, const umat* pStartingNodeAssignment, uword nClusters, uword nIterTau, double minTau, double minDeltaTau) : 
    pAdjacencyMatrices(pAdjacencyMatrices), nClusters(nClusters), nSubjects((*pAdjacencyMatrices).n_rows), nNodes((*pAdjacencyMatrices).n_slices), nIterTau(nIterTau), minTau(minTau), minDeltaTau(minDeltaTau)
{
    tau.set_size(nClusters, nSubjects, nNodes);
    VaryingTau::initTau(pStartingNodeAssignment);
    convergenceTau = true;
    pAugmentedAlpha = NULL;
    pPi = NULL;
    convergenceTau = false;
    logOneMinusPi.set_size(nClusters, nClusters, nSubjects);
    logitPi.set_size(nClusters, nClusters, nSubjects);
}

void VaryingTau::initTau()
{
    // TODO :initialise tau using some algorithm if no node assignmnet is given
}

void VaryingTau::initTau(const uvec* pStartingNodeAssignment)
{
    // TODO :  needs to be tested;
    for (uword i = 0; i < nClusters; i++)
    {
        tau(span(i), span::all,  span::all) = conv_to<arma::mat>::from(repmat(((*pStartingNodeAssignment)) == (i + 1), 1, nSubjects).t());
    }
    tau = tau / (1 + nClusters * minTau) + minTau / (1 + nClusters * minTau);
    VaryingTau::initNodalSums();
}

void VaryingTau::initTau(const umat* pStartingNodeAssignment)
{
    for (uword i = 0; i < nClusters; i++)
    {
        tau(span(i), span::all,  span::all) = conv_to<arma::mat>::from( (*pStartingNodeAssignment) == (i + 1) );
    }
    tau = tau / (1 + nClusters * minTau) + minTau / (1 + nClusters * minTau);
    VaryingTau::initNodalSums();
    sumNodalTau.save("/tmp/sumNodalTau.txt", arma_ascii);
    sumMaskedNodalTau.save("/tmp/sumMaskedNodalTau.txt", arma_ascii);
}

void VaryingTau::initNodalSums()
{
    sumNodalTau = zeros<mat>(nClusters, nSubjects);
    sumMaskedNodalTau = zeros<cube>(nClusters, nSubjects, nNodes);    
    mat nodalTau;
    umat nodalAdjacencyMatrices;
    cube maskedNodalTau(nClusters, nSubjects, nNodes);
    for (uword i = 0; i < nNodes; i++)
    {
        nodalTau = tau.slice(i);
        sumNodalTau += nodalTau;
        nodalAdjacencyMatrices = (*pAdjacencyMatrices).slice(i);
        for (uword ii = 0; ii < nNodes; ii++)
            maskedNodalTau.slice(ii) = nodalTau % repmat(nodalAdjacencyMatrices.col(ii).t(), nClusters, 1);

        sumMaskedNodalTau += maskedNodalTau;
    }  
} 

// method to update variables depending on pi only
void VaryingTau::initBeforeEachUpdateTau()
{
    mat tmp(nClusters, nClusters);
    uvec ind = regspace<uvec>(0, nClusters * nClusters - 1);
    for (uword i = 0; i < nSubjects; i++)
    {
        tmp.elem(ind) = (*pPi).row(i);
        logOneMinusPi.slice(i) = log(1.0-tmp);
        logitPi.slice(i) = log(tmp);
    }
    logitPi -= logOneMinusPi;
}

void VaryingTau::assignPointers(const cube* pAugmentedAlpha, const mat* pPi)
{
    this->pAugmentedAlpha = pAugmentedAlpha;
    this->pPi = pPi;
}

// TODO : optimise by stating convergence per subject basis
void VaryingTau::updateNodalTau(uword index)
{
    mat nodalTau = tau.slice(index);
    cube maskedNodalTau(nClusters, nSubjects, nNodes);
    umat nodalAdjacencyMatrices = (*pAdjacencyMatrices).slice(index);
    for (uword i = 0; i < nNodes; i++)
        maskedNodalTau.slice(i) = nodalTau % repmat(nodalAdjacencyMatrices.col(i).t(), nClusters, 1);

    // remove the slice from the sum over nodes
    mat sumNodalTauMinusCurrent = sumNodalTau - nodalTau;
    cube sumMaskedNodalTauMinusCurrent = sumMaskedNodalTau - maskedNodalTau;
    mat sumCurrentMaskedNodalTauMinusCurrent = sumMaskedNodalTauMinusCurrent.slice(index);
   
    // compute new nodal tau
    for (uword i = 0; i < nSubjects; i++)
    {
        nodalTau.col(i) = logOneMinusPi.slice(i) * sumNodalTauMinusCurrent.col(i) + 
                        logitPi.slice(i) * sumCurrentMaskedNodalTauMinusCurrent.col(i);
    }
    nodalTau += log((*pAugmentedAlpha).slice(index));

    // normalise tau
    mat tmp;
    for (uword i = 0; i < nSubjects; i++)
    {
        tmp = repmat(nodalTau.col(i), 1, nClusters);
        tmp = exp(tmp - tmp.t());
        nodalTau.col(i) = 1/sum(tmp).t();
    }

    // make sure that there is a minimum value
    nodalTau = nodalTau / (1 + nClusters * minTau) + minTau / (1 + nClusters * minTau);

    // update sumNodalTau and sumMaskedNodalTau for next used
    sumNodalTau = sumNodalTauMinusCurrent + nodalTau;
    for (uword i = 0; i < nNodes; i++)
        maskedNodalTau.slice(i) = nodalTau % repmat(nodalAdjacencyMatrices.col(i).t(), nClusters, 1);    
    sumMaskedNodalTau = sumMaskedNodalTauMinusCurrent + maskedNodalTau;
    
    // if( any( vectorise( abs(tau.slice(index) - nodalTau) ) > minDeltaTau))
    if( any( vectorise( abs(tau.slice(index) - nodalTau) ) > minDeltaTau))
        convergenceTau = false;

    tau.slice(index) = nodalTau;
}

void VaryingTau::updateTau()
{
    // update some variable depending on pi 
    VaryingTau::initBeforeEachUpdateTau();
    uword iIterTau = 0;
    convergenceTau = false;
    while (iIterTau < nIterTau && !convergenceTau)
    {   
        // set the convergence to true now; it will be set to false in VaryingTau::updateNodalTau if no convergence is detected for at least one node
        convergenceTau = true;
        for (uword i = 0; i < nNodes; i++)
            VaryingTau::updateNodalTau(i);
        // check if there is convergence in term of icl scores 
        iIterTau++;
    }
}

const cube* VaryingTau::getTau() const
{
    return &tau;
}

void VaryingTau::printTau()
{
    tau.print();
}

void VaryingTau::printTau(string filename)
{
    tau.save(filename, arma_ascii);
}

void VaryingTau::printNodeAssignment()
{
    VaryingTau::computeNodeAssignment().print();
}

void VaryingTau::printNodeAssignment(string filename)
{
    VaryingTau::computeNodeAssignment().save(filename, arma_ascii);
}

umat VaryingTau::computeNodeAssignment()
{
    mat tmpSlice;
    colvec tmpVec;
    umat nodeAssignment(nSubjects, nNodes);
    for (uword i = 0; i < nNodes; i++)
    {
        tmpSlice = tau.slice(i);
        for (uword ii = 0; ii < nSubjects; ii++)
        {
            tmpVec = tmpSlice.col(ii);
            nodeAssignment.at(ii,i) = tmpVec.index_max() + 1;
        }
    }
    return nodeAssignment;
}