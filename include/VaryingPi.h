///
/// @file    VaryingPi.h
/// @author  Bryan Roger Leon Guillaume
///

#ifndef VARYING_PI_H
#define VARYING_PI_H

#include<armadillo>
#include<string>

using namespace arma;

class VaryingPi 
{
private:
    uword nClusters, nSubjects, nNodes, nCovariates, nIterNR, nIterHalving;
    mat pi;
    mat beta;
    mat score;
    cube fisherInformation;
    cube designMatrix2;
    mat sumHalfGamma, sumHalfMaskedGamma;
    umat blockConvergences;
    double maxDeltaBeta, relConvTol;
    mat blockLogLiks;

    // pointers to objects outside the class
    const cube* pTau;
    const ucube* pAdjacencyMatrices;
    const mat* pDesignMatrix;
    
public:
    VaryingPi(const mat* pDesignMatrix, uword nIterNR, uword nIterHalving, double maxDeltaBeta, double relConvTol, const cube *pTau, const ucube *pAdjacencyMatrices);
    void computeBlockFisherInformation(uword index1, uword index2);
    void computeFisherInformation();
    void computeBlockScore(uword index1, uword index2);
    void computeScore();
    void computeSumGamma();
    double computeBlockLogLik(uword index1, uword index2, const colvec &deltaBeta);
    void computeBlockPi(uword index1, uword index2);
    void computePi();
    void computeBlockBeta(uword index1, uword index2);
    void computeBeta();
    const mat* getPi() const;
    const mat* getBlockLogLiks() const;
    void printPi();
    void printPi(std::string filename);
    void printBeta();
    void printBeta(std::string filename);
    cube reshapePi();
    cube reshapeBeta();
};

#endif