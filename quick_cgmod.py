#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

Demo for "Pairwise Sparse + Low-Rank Models for Variables of Mixed Type"
as submitted to the Journal of Multivariate Analysis (JMVA)
"""

from cgmodsel.CG_PWSL_ADMM import CG_PWSL_ADMM # ADMM pseudo likelihood
from cgmodsel.GLH_PWSL_ADMM import GLH_PWSL_ADMM # ADMM Gaussian likelihood

from cgmodsel.dataops import load_prepare_data # function to read data


if __name__ == '__main__':
   
###### data sets
    
    ## binary ##
    ABILITY = {
            'filename': "datasets/ability_proc.csv",
            'regparams': (.2,.5),
            'sparams': {'catuniques': [0,1]}
            }
    CFMT = {
            'filename': "datasets/CFMTkurzBIN.csv",
            'regparams': (.15,1.5),
            'sparams': {'catuniques': [0,1]}
            }

    ## Gaussian ##
#    filename = "datasets/SP500m_465all.csv"; drop = []; regparams = (.5,1)
    LSVT = {
            'filename': "datasets/LSVT.csv",
            'regparams': (.1,1),
            }

    ## mixed ##
    ALLBUS = { # pw model
            'filename': "datasets/allbus2016_proc.csv",
            'regparams': (1,2),
            }
    HELP = {
            'filename': "datasets/HELPmiss_proc.csv",
            'regparams': (.5,2),
            }

###### select and load data set

    # ********************************* #
    # comment out all but one line here #
    data = CFMT
#    data = LSVT
#    data = HELP
    # ********************************* #
    
    ## parameters for loading function ##
    loaddict = {'catuniques': None, 'standardize': True}
    if 'sparams' in data:
        loaddict.update(data['sparams'])
        
    print('Loading data...')
    tdata = load_prepare_data(data['filename'], cattype='dummy_red', **loaddict)
    D, Y, meta = tdata # binary data D, quantitative data Y, meta information about data
    
###### fit models    
    
    ## initialize solver and drop data ##
    if meta['dc'] > 0: # binary variables are present
        solver = CG_PWSL_ADMM(meta)
        solver.drop_data(D, Y)
    else: # purely Gaussian model
        solver = GLH_PWSL_ADMM(meta)
        solver.drop_data(Y)
    
    ## set regularization parameters ##
    # you may try different values, any pair of positive reals will do
    # e.g., regparams = .1, 1
    regparams = data['regparams'] # regularization parameters
    solver.set_regularization_params(regparams)
    
    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')
    opts = {'use_alpha':False} # use_alpha toggles use of univariate quantitative parameters
    solver.solve(verb=0, **opts) # pass solver options
    
###### model visualization
    
    model = solver.get_canonicalparams()
    model.repr_graphical(caption='learned')

