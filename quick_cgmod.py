#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2020

Demo for
Nussbaum, F. & Giesen, J. (2020). Pairwise sparse + low-rank models for variables of mixed type.
Journal of Multivariate Analysis, 2020.

If you use this software, please consider citing this article.
"""

# pylint: disable=C0103

from cgmodsel.admm import AdmmCGaussianPW, AdmmGaussianPW
from cgmodsel.admm import AdmmCGaussianSL, AdmmGaussianSL
from cgmodsel.dataops import load_prepare_data  # function to read data

def load(dataset: dict):
    """
    load csv with file path dataset['filename']

    return tuple (cat_data, cont_data, meta),
    where cat_data is the binary data, cont_data is the quantitative data,
    and meta is meta information about the dataset
    """
#    print('Loading data...(%s)'%(dataset['filename']))
    ## parameters for loading function ##
    loaddict = {'catuniques': None, 'standardize': True, 'verb':True}
    # standardize quantitative variables before learning model
    # catuniques: values of the binary variables (to support read function)
    # recommended to provide this if binary variables are not strings such as 'yes'/'no'
    if 'sparams' in dataset:
        loaddict.update(dataset['sparams'])

    return load_prepare_data(dataset['filename'],
                             cattype='dummy_red',
                             **loaddict)

def learn_sparse_model(data, regparam):
    cat_data, cont_data, meta = load(data)  # load the data

    ###### fit models

    ## initialize solver and drop data ##
    if meta['n_cat'] > 0:  # binary variables are present
        print('Using pseudo-likelihood solver in the presence of discrete variables...')
        solver = AdmmCGaussianPW()
        solver.drop_data((cat_data, cont_data), meta)
    else:  # purely Gaussian model
        print('Using likelihood solver for purely Gaussian model...')
        solver = AdmmGaussianPW()
        solver.drop_data(cont_data, meta)

    solver.set_regularization_params(regparam)

    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')
    solver.solve(verb=0)

    ###### model visualization

    model = solver.get_canonicalparams()  # PW model instance
    model.repr_graphical(diagonal=0) # plottype='pn'
    
    return model

def learn_sl_model(data, regparams):
    cat_data, cont_data, meta = load(data)  # load the data

    ###### fit models

    ## initialize solver and drop data ##
    if meta['n_cat'] > 0:  # binary variables are present
        print('Using pseudo-likelihood solver in the presence of discrete variables...')
        solver = AdmmCGaussianSL()
        solver.drop_data((cat_data, cont_data), meta)
    else:  # purely Gaussian model
        print('Using likelihood solver for purely Gaussian model...')
        solver = AdmmGaussianSL()
        solver.drop_data(cont_data, meta)

    solver.set_regularization_params(regparams)

    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')
    solver.solve(verb=0)

    ###### model visualization

    model = solver.get_canonicalparams()  # S + L model instance
    model.plot_sl(plottype='pn')
    
    return model


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

    IRIS = {
            'filename': 'datasets/iris.csv',
            'regparams': (.5, 2)
            }
    ###### select and load data set

    # ********************************* #
    # comment out all but one line here #
#    data = CFMT
    data = LSVT
    data = IRIS
#    data = ALLBUS
    # ********************************* #

    ## set regularization parameters ##
    # you may try different values, any pair of positive reals will do
    # e.g., regparams = (.1, 1)
    model = learn_sl_model(data, regparams=data['regparams'])
    
#    model = learn_sparse_model(data, regparam=1.0)
    
    # model.get_params()
    # model.get_meanparams()
    # model.get_meta()
    # model.save("saved_model")
    

