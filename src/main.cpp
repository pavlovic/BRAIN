///
/// @file    main.cpp
/// @author  Bryan Roger Leon Guillaume
///

#include "main.h"
#include "SbmVersion.h"
#include "cxxopts.hpp"
#include "BaselineSbm.h"
#include "VaryingTau.h"
#include "VaryingAlpha.h"
#include "VaryingPi.h"

#include <iostream>
#include <string>
// #include <filesystem>

#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv) {

    // state the version of the SBM tool
    arma::arma_version ver;
    cout << "SBM v" << SBM_VERSION << " using Armadillo version v"<< ver.as_string() << endl;
    
    // parse the inputs
    string outputDir, adjFilename, startingNodeAssignmentFilename, baselineLogitDesignMatrixFilename, logisticDesignMatrixFilename;
    uword nClusters, nIterAlpha, nIterHalvingAlpha, nIterHalvingPi, nIterPi, nIterSbm, nIterTau;
    double maxDeltaRegressionParam, minValue, convTol;
    bool verbose(false);
    try
    {
        cxxopts::Options options(argv[0], "");
        
        options.custom_help("-a ADJACENCY_MATRICES.txt -s STARTING_ASSIGNMENT -o OUTPUT_DIR  -b BASELINE_LOGIT_DESIGN_MATRIX.TXT  -l LOGISTIC_DESIGN_MATRIX.TXT -n N [optional arguments]"); 
        
        options.add_options()
            ("a,adjacencyMatrices", "specify the file containing the binary adjacency matrices (must be an nSubject x nNodes x nNodes ARMA_CUB_TXT_IU008 file)", cxxopts::value<std::string>(adjFilename), "FILE")
            ("b,baselineLogitDesignMatrix", "specify the file containing the design matrix for the baseline logit regression (must be an nSubject x nCovariates ARMA_MAT_TXT_FN008 file)", cxxopts::value<std::string>(baselineLogitDesignMatrixFilename),  "FILE")
            ("c, convTol", "specify the relative convergence tolerance in terms of the contributions to the variational bound of the specific set of regression parameters being updated (local convergence) as well as the global variational bound (global convergence)", cxxopts::value<double>(convTol)->default_value("1.0e-10"), "DOUBLE")
            // TODO: consider other convergence criterions such as delta parameters
            ("h,help", "print this help")
            ("l,logisticDesignMatrix", "specifiy the file containing the design matrix for the logistic regression (must be an nSubject x nCovariates ARMA_MAT_TXT_FN008 file)", cxxopts::value<std::string>(logisticDesignMatrixFilename), "FILE")
            ("m, maxDeltaRegressionParam", "specify the maximum increase or decrease of the regression parameters", cxxopts::value<double>(maxDeltaRegressionParam)->default_value("5"), "DOUBLE")
            ("minValue", "specify the minimum value allowed for tau and alpha", cxxopts::value<double>(minValue)->default_value("1.0e-10"), "DOUBLE")
            ("n, nClusters", "specify the number of clusters", cxxopts::value<uword>(nClusters), "INT")
            ("nIterAlpha", "specify the maximum number of Newton-Raphson iterations for each update of the baseline logit regression model for alpha", cxxopts::value<uword>(nIterAlpha)->default_value("1000"), "INT")
            ("nIterHalvingAlpha", "specify the maximum number of iterations for the halving procedure of the baseline logit regression model for alpha", cxxopts::value<uword>(nIterHalvingAlpha)->default_value("100"), "INT")
            ("nIterHalvingPi", "specify the maximum number of iterations for the halving procedure of the logistic regression model for pi", cxxopts::value<uword>(nIterHalvingPi)->default_value("100"), "INT")
            ("nIterPi", "specify the maximum number of Newton-Raphson iterations for each update of the logistic regression model for pi", cxxopts::value<uword>(nIterPi)->default_value("1000"), "INT")
            ("nIterSbm", "specify the maximum number of iterations for the SBM", cxxopts::value<uword>(nIterSbm)->default_value("100"), "INT")
            ("nIterTau", "specify the maximum number of iterations for each update of tau", cxxopts::value<uword>(nIterTau)->default_value("100"), "INT")
            ("o,output", "specifiy the path to the directory where the outputs will be saved", cxxopts::value<std::string>(outputDir), "DIR")
            ("s,startingNodeAssignment", "specify the file containing the starting assigments of nodes (must be an nSubject x nCovariates ARMA_MAT_TXT_IU008 file)", cxxopts::value<std::string>(startingNodeAssignmentFilename), "FILE")
            ("v,verbose", "enable verbose mode", cxxopts::value<bool>(verbose))
            ;

        auto parseResult = options.parse(argc, argv);

        if (parseResult.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        if (!parseResult.count("adjacencyMatrices"))
        {
            cout << "No file containing the adjacency matrices is specified. Please use the mandatory option -a or --adjacencyMatrices to specified it." << endl;
            exit(1);        
        }

        if (!parseResult.count("nClusters"))
        {
            cout << "No number of clusters is specified. Please use the mandatory option -n or --nClusters to specified it." << endl;
            exit(1);        
        }

        if (!parseResult.count("startingNodeAssignment"))
        {
            cout << "No file containing the starting node assignmenets is specified. Please use the mandatory option -s or --startingNodeAssignment to specified it." << endl;
            exit(1);        
        }

        if (!parseResult.count("baselineLogitDesignMatrix"))
        {
            cout << "No file containing the baseline-logit design matrix is specified. Please use the mandatory option -b or --baselineLogitDesignMatrix to specified it." << endl;
            exit(1);        
        }

        if (!parseResult.count("logisticDesignMatrix"))
        {
            cout << "No file containing the logisitic design matrix is specified. Please use the mandatory option -l or --logisticDesignMatrix to specified it." << endl;
            exit(1);        
        }

        if (!parseResult.count("output"))
        {
            cout << "No output directory specified. Please use the mandatory option -o or --output to specified it." << endl;
            exit(1);        
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    // load the data and estimate the specified SBM model
    double minDeltaTau = 1.0e-10; // TODO: review if needed
    bool directed = false; //TODO: extend to directed case
    if (verbose)
    {
        cout << "\nMain list of options for the current SBM model:" << endl;
        cout << "Adjacency matrix file: " << adjFilename << endl;
        cout << "Starting assignments file: " << startingNodeAssignmentFilename << endl;
        cout << "Baseline-logit regression design matrix file: " << baselineLogitDesignMatrixFilename << endl;
        cout << "Logistic regression design matrix file: " << logisticDesignMatrixFilename << endl;
        cout << "Number of clusters: " << nClusters << endl;
        cout << "Numbers of iterations (global, tau, alpha, pi, halving alpha, halving pi): (" << nIterSbm << ", " << nIterTau << ", " << nIterAlpha << ", " << nIterPi << ", " << nIterHalvingAlpha << ", " << nIterHalvingPi << ")" << endl;
        cout << "Output directory: " << outputDir << "\n" << endl;

    }

    if (verbose)
        cout << "Loading data..." << endl;
    umat startingNodeAssignment; // = repmat(regspace<uvec>(1,nClusters), nNodes / nClusters, 1)
    startingNodeAssignment.load(startingNodeAssignmentFilename, arma_ascii);
    ucube adjacencyMatrices;
    adjacencyMatrices.load(adjFilename, arma_ascii);
    mat designMatrixAlpha, designMatrixPi;
    designMatrixAlpha.load(baselineLogitDesignMatrixFilename, arma_ascii);
    designMatrixPi.load(logisticDesignMatrixFilename, arma_ascii);

    if (verbose)
        cout << "Initialising SBM model..." << endl;   
    BaselineSbm sbm(&adjacencyMatrices, directed, &designMatrixAlpha, &designMatrixPi, 
                    nClusters, nIterSbm, nIterTau, nIterAlpha, nIterPi, nIterHalvingAlpha, nIterHalvingPi, 
                    maxDeltaRegressionParam, minValue, minValue, minDeltaTau, &startingNodeAssignment, convTol, verbose);

    if (verbose)
        cout << "Estimating SBM model..." << endl;   
    sbm.estimateModel();

    if (verbose)
        cout << "Saving SBM model..." << endl; 
    sbm.saveInDirectory(outputDir);

    return MAIN_SUCCESS;
}

