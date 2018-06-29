///
/// @file    VaryingAlpha.h
/// @author  Bryan Roger Leon Guillaume
///


#ifndef VARYING_ALPHA_H
#define VARYING_ALPHA_H

#include<armadillo>
#include<string>

using namespace arma;

class VaryingAlpha 
{
private:
    uword nSubjects, nCovariates, nClusters, nNodes, nIterNR, nIterHalving;
    cube alpha, augmentedAlpha;
    mat beta;
    const mat* pDesignMatrix;
    cube designMatrix2;
    mat score;
    cube fisherInformation;
    colvec nodalLogLiks;
    ucolvec nodalConvergences;
    double minAlpha, maxDeltaBeta, relConvTol;

    // pointers to objects outside the class
    const cube* pTau;

public:
    VaryingAlpha(const mat* pDesignMatrix, uword nIterNR, uword nIterHalving, double maxDeltaBeta, double minAlpha, double relConvTol, const cube *pTau);
    void computeNodalFisherInformation(uword index);
    void computeFisherInformation();
    void computeNodalScore(uword index);
    void computeScore();
    double computeNodalLogLik(const uword index, const colvec &deltaBeta);
    void computeNodalAlpha(const uword index);
    void computeAlpha();
    void computeAugmentedAlpha();
    void computeBeta();
    void computeNodalBeta(const uword index);
    const cube* getAlpha() const;
    const cube* getAugmentedAlpha() const;
    const mat* getBeta() const;
    const colvec* getNodalLogLiks() const;
    void printAlpha();
    void printAlpha(std::string filename);
    void printAugmentedAlpha();
    void printAugmentedAlpha(std::string filename);
    void printBeta();
    void printBeta(std::string filename);
    umat computeNodeAssignment();
};

#endif