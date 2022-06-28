# README #

cgmodsel - a Python3 package to estimate conditional Gaussian (CG) distributions from data

Version 1.1, 23.4.2021

### Content ###

This package contains solvers (mostly ADMM-based) for estimating various types of CG models (sparse pairwise, sparse + low-rank pairwise, mean parameters).
Documentation is available under [cgmodsel-pages](https://franknu.github.io/cgmodsel-pages/source/cgmodsel.html).
Please see also the [Wiki](https://github.com/franknu/cgmodsel/wiki) for additional information and an introduction about some models.

### Setup cgmodsel ###

For normal usage:

1. Clone this repository into a folder of your choice. 
2. Open a terminal and navigate to the chosen directory using cd.
3. Install the `cgmodsel` package locally by typing `pip install .`

You can also use the code without doing the third step from inside the folder. For a quick start, execute the file `quick_cgmod.py` from the main folder.

Note that some solvers use a group-soft shrinkage operation. It works with a naive (but slow) Python implementation, however, can be sped up using Cython code (credit goes to my colleague Mark Mihajlovic Blacher, thanks!). Install it by typing `python setup.py build_ext --inplace` in the cyshrink/ directory.

### Usage ###

If you use this software, please consider citing:

Nussbaum, F., & Giesen, J. (2020). Pairwise sparse + low-rank models for variables of mixed type. *Journal of Multivariate Analysis*, 104601.

### Contact ###

If you have questions, comments, bug reports, feature requests, etc. please contact me: [frank.nussbaum@uni-jena.de](frank.nussbaum@uni-jena.de).

### Copyright ###

© 2017 - 2021 Frank Nussbaum (frank.nussbaum@uni-jena.de). All Rights Reserved.
