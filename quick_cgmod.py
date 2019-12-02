#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

Demo for "Pairwise Sparse + Low-Rank Models for Variables of Mixed Type"
as submitted to the Journal of Multivariate Analysis (JMVA)
"""

from cgmodsel.admm_pwsl import AdmmCGaussianSL # ADMM pseudo likelihood
from cgmodsel.admm_pwsl import AdmmGaussianSL # ADMM Gaussian likelihood

from cgmodsel.dataops import load_prepare_data # function to read data

def load(dataset):
    """
    load csv with file path dataset['filename']
    
    return tuple (D, Y, meta),
    where D binary data, Y quantitative data,
    and meta is meta information about the dataset
    """
    ## parameters for loading function ##
    loaddict = {'catuniques': None, 'standardize': True}
    # standardize quantitative variables before learning model
    # catuniques: values of the binary variables (to support read function)
    # recommended to provide this if binary variables are not strings such as 'yes'/'no'
    if 'sparams' in dataset:
        loaddict.update(dataset['sparams'])
        
    return load_prepare_data(dataset['filename'], cattype='dummy_red', **loaddict)
    

if __name__ == '__main__':
   
###### data sets
    
    ## binary ##
    ABILITY = {
            'filename': "datasets/ability_proc.csv",
            'regparams': (.2,.5),
            'sparams': {'catuniques': [0,1]} # values that binary variables take
            }
    CFMT = {
            'filename': "datasets/CFMTkurzBIN.csv",
            'regparams': (.15,1.5),
            'sparams': {'catuniques': [0,1]} # values that binary variables take
            }

    ## quantitative ##
    LSVT = {
            'filename': "datasets/LSVT.csv",
            'regparams': (.1,1),
            }

    ## mixed binary-quantitative##
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
    dataset = CFMT
#    dataset = LSVT
#    dataset = HELP
    # ********************************* #

    print('Loading data...')
    D, Y, meta = load(dataset) # load the data
    
###### fit models    
    
    ## initialize solver and drop data ##
    if meta['dc'] > 0: # binary variables are present
        solver = AdmmCGaussianSL(meta)
        solver.drop_data(D, Y)
    else: # purely Gaussian model
        solver = AdmmGaussianSL(meta)
        solver.drop_data(Y)
    
    ## set regularization parameters ##
    # you may try different values, any pair of positive reals will do
    # e.g., regparams = (.1, 1)
    regparams = dataset['regparams'] # regularization parameters
    solver.set_regularization_params(regparams)
    
    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')
    solver.solve(verb=0)
    
###### model visualization
    
    model = solver.get_canonicalparams() # S + L model instance
    model.repr_graphical(caption='learned')

