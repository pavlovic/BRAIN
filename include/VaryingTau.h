///
/// @file    VaryingTau.h
/// @author  Bryan Roger Leon Guillaume
///

#ifndef VARYING_TAU_H
#define VARYING_TAU_H

#include<armadillo>
#include<string>

using namespace arma;
using namespace std;

class VaryingTau 
{
private:
    uword nClusters, nSubjects, nNodes, nIterTau;
    cube tau;
    mat sumNodalTau;
    cube sumMaskedNodalTau;
    const ucube* pAdjacencyMatrices;
    const cube* pAugmentedAlpha;
    const mat* pPi;
    cube logOneMinusPi, logitPi;
    double minTau, relConvTol;
    uvec convergenceTau;
    vec subjectLogLiks;

    uvec indicesSubjectsToUpdate;
    uword nSubjectsToUpdate;

public:
    VaryingTau(const ucube* pAdjacencyMatrices, const umat* pStartingNodeAssignment, uword nClusters, uword nIterTau, double minTau, double relConvTol);
    void initTau();
    void initTau(const uvec* pStartingNodeAssignment);
    void initTau(const umat* pStartingNodeAssignment);
    void initNodalSums();
    void initBeforeEachUpdateTau();
    void assignPointers(const cube* pAugmentedAlpha, const mat* pPi);
    void updateNodalTau(uword index);
    void updateTau();
    double computeSubjectLogLik(uword index);
    const cube* getTau() const;
    void printTau();
    void printTau(string filename);
    void printNodeAssignment();
    void printNodeAssignment(string filename);
    umat computeNodeAssignment();

};

#endif