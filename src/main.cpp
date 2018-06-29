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
    cout << "SBM Project v" << SBM_VERSION << " using Armadillo version v"<< ver.as_string() << std::endl;
    
    // parse the inputs
    string outputDir, adjFilename, startingNodeAssignmentFilename;
    uword nClusters, nIterAlpha, nIterHalvingAlpha, nIterHalvingPi, nIterPi, nIterSbm, nIterTau;
    double maxDeltaRegressionParam, minValue, convTol;
    bool verbose(false);
    try
    {
        cxxopts::Options options(argv[0], "");
        
        options.custom_help("-a ADJACENCY_MATRICES.txt -n N -s STARTING_ASSIGNMENT -o OUTPUT_DIR [optional arguments]"); 
        
        options.add_options()
            ("a,adjacencyMatrices", "specify the file containing the adjacency matrices", cxxopts::value<std::string>(), "FILE")
            ("b,baselineLogitDesignMatrix", "specify the file containing the design matrix for the baseline logit regression", cxxopts::value<std::string>(),  "FILE")
            ("c, convTol", "specify the relative convergence tolerance in terms of the contributions to the icl score of the specific set of regression parameters being updated (local convergence) as well as the global ical score (global convergence)", cxxopts::value<double>()->default_value("1.0e-10"), "DOUBLE")
            // TODO: impose this convergence criterion for tau
            // TODO: consider other convergence criterions such as delta parameters
            ("h,help", "print this help")
            ("l,logisticDesignMatrix", "specifiy the file containing the design matrix for the logistic regression", cxxopts::value<std::string>(), "FILE")
            ("m, maxDeltaRegressionParam", "specify the maximum increase or decrease of the regression parameters", cxxopts::value<double>()->default_value("5"), "DOUBLE")
            ("minValue", "specify the minimum value allowed for tau and alpha", cxxopts::value<double>()->default_value("1.0e-10"), "DOUBLE")
            ("n, nClusters", "specify the number of clusters", cxxopts::value<uword>(), "INT")
            ("nIterAlpha", "specify the maximum number of Newton-Raphson iterations for each update of the baseline logit regression model for alpha", cxxopts::value<uword>()->default_value("1000"), "INT")
            ("nIterHalvingAlpha", "specify the maximum number of iterations for the halving procedure of the baseline logit regression model for alpha", cxxopts::value<uword>()->default_value("100"), "INT")
            ("nIterHalvingPi", "specify the maximum number of iterations for the halving procedure of the logistic regression model for pi", cxxopts::value<uword>()->default_value("100"), "INT")
            ("nIterPi", "specify the maximum number of Newton-Raphson iterations for each update of the logistic regression model for pi", cxxopts::value<uword>()->default_value("1000"), "INT")
            ("nIterSbm", "specify the maximum number of iterations for the SBM", cxxopts::value<uword>()->default_value("100"), "INT")
            ("nIterTau", "specify the maximum number of iterations for each update of tau", cxxopts::value<uword>()->default_value("100"), "INT")
            ("o,output", "specifiy the path to the directory where the outputs will be saved", cxxopts::value<std::string>(), "DIR")
            ("s,startingNodeAssignment", "specify the file containing the starting assigments of nodes", cxxopts::value<std::string>(), "FILE")
            ("v,verbose", "enable verbose mode", cxxopts::value<bool>(verbose))
            ;

        auto parseResult = options.parse(argc, argv);

        if (parseResult.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        if (parseResult.count("adjacencyMatrices"))
            adjFilename = parseResult["adjacencyMatrices"].as<string>();
        else
        {
            cout << "No file containing the adjacency matrices is specified. Please use the mandatory option -a or --adjacencyMatrices to specified it." << endl;
            exit(1);        
        }

        if (parseResult.count("convTol"))
            convTol = parseResult["convTol"].as<double>();

        if (parseResult.count("maxDeltaRegressionParam"))
            maxDeltaRegressionParam = parseResult["maxDeltaRegressionParam"].as<double>();

        if (parseResult.count("minValue"))
        {
            minValue = parseResult["minValue"].as<double>();
        }

        if (parseResult.count("nClusters"))
            nClusters = parseResult["nClusters"].as<uword>();
        else
        {
            cout << "No number of clusters is specified. Please use the mandatory option -n or --nClusters to specified it." << endl;
            exit(1);        
        }

        if (parseResult.count("nIterAlpha"))
            nIterAlpha = parseResult["nIterAlpha"].as<uword>();

        if (parseResult.count("nIterHalvingAlpha"))
            nIterHalvingAlpha = parseResult["nIterHalvingAlpha"].as<uword>();

        if (parseResult.count("nIterHalvingPi"))
            nIterHalvingPi = parseResult["nIterHalvingPi"].as<uword>();

        if (parseResult.count("nIterPi"))
            nIterPi = parseResult["nIterPi"].as<uword>();

        if (parseResult.count("nIterSbm"))
            nIterSbm = parseResult["nIterSbm"].as<uword>();

        if (parseResult.count("nIterTau"))
            nIterTau = parseResult["nIterTau"].as<uword>();

        if (parseResult.count("startingNodeAssignment"))
            startingNodeAssignmentFilename = parseResult["startingNodeAssignment"].as<string>();
        else
        {
            cout << "No file containing the adjacency matrices is specified. Please use the mandatory option -x or --adjacencyMatrices to specified it." << endl;
            exit(1);        
        }

        if (parseResult.count("output"))
        {
            outputDir = parseResult["output"].as<string>();
            // if using the new filesystem library
            // filesystem::path pathDir(outputDir);
            // if (!filesystem::is_directory(pathDir))
            // {
            //     cout << "The output directory specified does not exist. Please create it." << endl;
            //     exit(1); 
            // }
        }
        else
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
    double minDeltaTau = 1.0e-10;
    bool directed = false; //TODO: extend to directed case

    if (verbose)
        cout << "Loading data..." << endl;
    umat startingNodeAssignment; // = repmat(regspace<uvec>(1,nClusters), nNodes / nClusters, 1)
    startingNodeAssignment.load(startingNodeAssignmentFilename, arma_ascii);
    ucube adjacencyMatrices;
    adjacencyMatrices.load(adjFilename, arma_ascii);
    mat designMatrix;
    designMatrix.load(argv[3], arma_ascii);

    if (verbose)
        cout << "Initialising SBM model..." << endl;   
    BaselineSbm sbm(&adjacencyMatrices, directed, &designMatrix, &designMatrix, 
                    nClusters, nIterSbm, nIterTau, nIterAlpha, nIterPi, nIterHalvingAlpha, nIterHalvingPi, 
                    maxDeltaRegressionParam, minValue, minValue, minDeltaTau, &startingNodeAssignment, convTol, verbose);

    if (verbose)
        cout << "Estimating SBM model..." << endl;   
    sbm.estimateModel();

    sbm.saveInDirectory(outputDir);

    return MAIN_SUCCESS;
}

