///
/// @file    VaryingTau.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "VaryingTau.h"

using namespace std;
VaryingTau::VaryingTau(const ucube* pAdjacencyMatrices, const umat* pStartingNodeAssignment, uword nClusters, uword nIterTau, double minTau, double relConvTol) : 
    pAdjacencyMatrices(pAdjacencyMatrices), nClusters(nClusters), nSubjects((*pAdjacencyMatrices).n_rows), nNodes((*pAdjacencyMatrices).n_slices), nIterTau(nIterTau), minTau(minTau), relConvTol(relConvTol)
{
    tau.set_size(nClusters, nSubjects, nNodes);
    VaryingTau::initTau(pStartingNodeAssignment);
    pAugmentedAlpha = NULL;
    pPi = NULL;
    logOneMinusPi.set_size(nClusters, nClusters, nSubjects);
    logitPi.set_size(nClusters, nClusters, nSubjects);
    convergenceTau.set_size(nSubjects);
    subjectLogLiks.set_size(nSubjects);
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
    
    // compute also the subject-specific log-likelihoods used to check convergence (note: depends on logotPi and logOneMinusPi)
    for (uword i = 0; i < nSubjects; i++)
        subjectLogLiks[i] = computeSubjectLogLik(i);
}

void VaryingTau::assignPointers(const cube* pAugmentedAlpha, const mat* pPi)
{
    this->pAugmentedAlpha = pAugmentedAlpha;
    this->pPi = pPi;
}

void VaryingTau::updateNodalTau(uword index)
{
    mat redNodalTau = tau.slice(index).cols(indicesSubjectsToUpdate);
    cube redMaskedNodalTau(nClusters, nSubjectsToUpdate, nNodes);
    umat redNodalAdjacencyMatrices = (*pAdjacencyMatrices).slice(index).rows(indicesSubjectsToUpdate);
    for (uword i = 0; i < nNodes; i++)
        redMaskedNodalTau.slice(i) = redNodalTau % repmat(redNodalAdjacencyMatrices.col(i).t(), nClusters, 1);

    // remove the slice from the sum over nodes
    mat redSumNodalTauMinusCurrent = sumNodalTau.cols(indicesSubjectsToUpdate) - redNodalTau;
    cube redSumMaskedNodalTauMinusCurrent(nClusters, nSubjectsToUpdate, nNodes);
    for (uword i = 0; i < nNodes; i++)
        redSumMaskedNodalTauMinusCurrent.slice(i) = sumMaskedNodalTau.slice(i).cols(indicesSubjectsToUpdate) - redMaskedNodalTau.slice(i);
    mat redSumCurrentMaskedNodalTauMinusCurrent = redSumMaskedNodalTauMinusCurrent.slice(index);
   
    // compute new nodal tau for subjects that have not converged yet
    uword currentIndexSubject;
    for (uword i = 0; i < nSubjectsToUpdate; i++)
    {
        currentIndexSubject = indicesSubjectsToUpdate[i];
        redNodalTau.col(i) = logOneMinusPi.slice(currentIndexSubject) * redSumNodalTauMinusCurrent.col(i) + 
                        logitPi.slice(currentIndexSubject) * redSumCurrentMaskedNodalTauMinusCurrent.col(i);
    }
    redNodalTau += log((*pAugmentedAlpha).slice(index).cols(indicesSubjectsToUpdate));

    // normalise tau
    mat tmp;
    for (uword i = 0; i < nSubjectsToUpdate; i++)
    {
        tmp = repmat(redNodalTau.col(i), 1, nClusters);
        tmp = exp(tmp - tmp.t());
        redNodalTau.col(i) = 1/sum(tmp).t();
    }

    // make sure that there is a minimum value
    redNodalTau = redNodalTau / (1 + nClusters * minTau) + minTau / (1 + nClusters * minTau);

    // update sumNodalTau and sumMaskedNodalTau for next used
    sumNodalTau.cols(indicesSubjectsToUpdate) = redSumNodalTauMinusCurrent + redNodalTau;
    for (uword i = 0; i < nNodes; i++)
        sumMaskedNodalTau.slice(i).cols(indicesSubjectsToUpdate) = redSumMaskedNodalTauMinusCurrent.slice(i) + ( redNodalTau % repmat(redNodalAdjacencyMatrices.col(i).t(), nClusters, 1) );
    
    // if( any( vectorise( abs(tau.slice(index) - nodalTau) ) > minDeltaTau))
    //     convergenceTau = false;

    tau.slice(index).cols(indicesSubjectsToUpdate) = redNodalTau;
}

void VaryingTau::updateTau()
{
    // update some variable depending on pi (including the subject specific log-liks)
    VaryingTau::initBeforeEachUpdateTau();
    uword iIterTau = 0;
    convergenceTau.zeros();
    double tmpSubjectLogLik;
    uword currentIndexSubject;
    while ( iIterTau < nIterTau && any(convergenceTau == 0) )
    {   
        indicesSubjectsToUpdate = find(convergenceTau == 0);
        nSubjectsToUpdate = indicesSubjectsToUpdate.n_elem;
        for (uword i = 0; i < nNodes; i++)
            VaryingTau::updateNodalTau(i);

        // check if there is convergence in term of the variational bound contribution of each subject
        for (uword i = 0; i < nSubjectsToUpdate; i++)
        {
            currentIndexSubject = indicesSubjectsToUpdate[i];
            tmpSubjectLogLik = computeSubjectLogLik(i);
            if ( tmpSubjectLogLik - subjectLogLiks[currentIndexSubject] < abs(subjectLogLiks[currentIndexSubject]) * relConvTol )
                convergenceTau[currentIndexSubject] = 1;
            subjectLogLiks[currentIndexSubject] = tmpSubjectLogLik;
        }
        // TODO : Handle the hypothetical case where the log-lik actually decreases (In principle, that should not happen) 
        iIterTau++;
    }
}

double VaryingTau::computeSubjectLogLik(uword index)
{
    umat sliceAdjMat;
    vec tmpTau1, tmpTau2;
    double subjectLogLik(0);
    for (uword i = 0; i < nNodes; i++)
    {
        tmpTau1 = tau.slice(i).col(index);
        sliceAdjMat = (*pAdjacencyMatrices).slice(i);
        for (uword ii = 0; ii < nNodes; ii++)
        {
            if (i != ii)
            {
                tmpTau2 = tau.slice(ii).col(index);
                subjectLogLik += accu( ( tmpTau1 * tmpTau2.t() ) % ( logOneMinusPi.slice(index) + sliceAdjMat.at(index, ii) * logitPi.slice(index) ) );
            }
        }
        subjectLogLik += sum( tmpTau1 % ( log( (*pAugmentedAlpha).slice(i).col(index) ) - log(tmpTau1) ) );
    }
    return 0.5 * subjectLogLik; 
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