# BRAIN
Brain Regression Analysis In Networks

The main goal of this toolbox is to provide software to analyse brain networks using Stochastic Block Models (SBMs). Currently, only a command line version of the multi-subject SBMs proposed by Pavlovic et al. (OHBM 2018) is available. Additional features will be added to it (e.g., use of Firth's regression instead of traditional Maximum likilihood regression, a better input parser,...). Other SBMs will also be added in the future. Finally, we will also provide R and Matlab wrappers for use in these software packages.

Installation
======

To use this tool, it is required to install Armadillo, a convenient C++ library for linear algebra & scientific computing. It can be downloaded at http://arma.sourceforge.net/download.html and cited through the reference:

Conrad Sanderson and Ryan Curtin. 
Armadillo: a template-based C++ library for linear algebra. 
Journal of Open Source Software, Vol. 1, pp. 26, 2016. 

Once Armadillo is installed on your computer, you should be able to install BRAIN using Cmake: 

```bash
cmake .
make
```
The executable file should be in the sbm-versionNumber/bin folder. Note that this installation procedure was tested on MAC only.

Usage
======

A proper input parser is currently under development and will be available soon. For now, some parameters need to be changed manually in the main function while some others need to be input through the command line in the following order:

1) path to the file containing the starting assigment of nodes
2) path to the file containing the adjacency matrices of all subjects
3) path to the design matrix (for now, assumed to be the same for the logistic and the baseline logit regressions)
4) path to the directory where the results will be saved
5) the number of clusters 

