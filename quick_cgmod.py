#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

Demo for "Pairwise Sparse + Low-Rank Models for Variables of Mixed Type"
as submitted to the Journal of Multivariate Analysis (JMVA)
"""

# pylint: disable=C0103

from cgmodsel.admm_pwsl import AdmmCGaussianSL, AdmmGaussianSL
from cgmodsel.dataops import load_prepare_data  # function to read data


def load(dataset: dict):
    """
    load csv with file path dataset['filename']

    return tuple (cat_data, cont_data, meta),
    where cat_data is the binary data, cont_data is the quantitative data,
    and meta is meta information about the dataset
    """
    ## parameters for loading function ##
    loaddict = {'catuniques': None, 'standardize': True}
    # standardize quantitative variables before learning model
    # catuniques: values of the binary variables (to support read function)
    # recommended to provide this if binary variables are not strings such as 'yes'/'no'
    if 'sparams' in dataset:
        loaddict.update(dataset['sparams'])

    return load_prepare_data(dataset['filename'],
                             cattype='dummy_red',
                             **loaddict)


if __name__ == '__main__':

    ###### data sets

    ## binary ##
    ABILITY = {
        'filename': "datasets/ability_proc.csv",
        'regparams': (.2, .5),
        'sparams': {
            'catuniques': [0, 1]
        }  # values that binary variables take
    }
    CFMT = {
        'filename': "datasets/CFMTkurzBIN.csv",
        'regparams': (.15, 1.5),
        'sparams': {
            'catuniques': [0, 1]
        }  # values that binary variables take
    }

    ## quantitative ##
    LSVT = {
        'filename': "datasets/LSVT.csv",
        'regparams': (.1, 1),
    }

    ## mixed binary-quantitative ##
    ALLBUS = {
        'filename': "datasets/allbus2016_proc.csv",
        'regparams': (1, 2),
    }
    HELP = {
        'filename': "datasets/HELPmiss_proc.csv",
        'regparams': (.5, 2),
    }

    ###### select and load data set

    # ********************************* #
    # comment out all but one line here #
    data = CFMT
#    data = LSVT
#    data = HELP
    # ********************************* #

    print('Loading data...(%s)'%(data['filename']))
    cat_data, cont_data, meta = load(data)  # load the data

    ###### fit models

    ## initialize solver and drop data ##
    if meta['n_cat'] > 0:  # binary variables are present
        solver = AdmmCGaussianSL()
        solver.drop_data((cat_data, cont_data), meta)
    else:  # purely Gaussian model
        solver = AdmmGaussianSL()
        solver.drop_data(cont_data, meta)

    ## set regularization parameters ##
    # you may try different values, any pair of positive reals will do
    # e.g., regparams = (.1, 1)
    regparams = data['regparams']  # regularization parameters
    solver.set_regularization_params(regparams)

    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')
    solver.solve(verb=0)

    ###### model visualization

    model = solver.get_canonicalparams()  # S + L model instance
    model.plot_sl(plottype='pn')
