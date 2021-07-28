#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2021
"""
# pylint: disable=C0103

from cgmodsel.admm import AdmmCGaussianPW
from cgmodsel.dataops import load_prepare_data  # function to read data

def learn_sparse_model(data, regparam):
    
    cat_data, cont_data, meta = load_prepare_data(data['filename'],
                                                  cattype='dummy_red',
                                                  **data)

    ## initialize solver and drop data ##
    print('Using pseudo-likelihood solver in the presence of discrete variables...')
    solver = AdmmCGaussianPW()
    solver.drop_data((cat_data, cont_data), meta)


    solver.set_regularization_params(regparam)

    ## solve the problem, that is, estimate a sparse model ##
    print('Solving the problem...')
#    solver.prox.opts['maxiter'] = 1000
    solver.solve(verb=1, use_u=0, off=0, maxiter=500) # use_u=0 turns of univariate discrete parameters

    ###### model visualization

    model = solver.get_canonicalparams()  # PW model instance
    model.repr_graphical(diagonal=0) # plottype='pn'
    
    return model



if __name__ == '__main__':
    # catuniques: values of the binary variables (to support read function)
    # recommended to provide this if binary variables are not strings such as 'yes'/'no'

    ###### data sets
    ## binary ##
    ABILITY = {
            'name':'ability',
        'filename': "datasets/ability_proc.csv",
        'catuniques': [0, 1] # values that binary variables take
        }  


    ## mixed binary-quantitative ##
    ALLBUS = {
        'filename': "datasets/allbus2016_proc.csv",
    }
    
    ADULT = {
            'filename':'datasets/adult_train.csv',
            'categoricals':['workclass', 'education', 'marital-status',
                            'occupation','relationship','race', 'sex',
                            'native-country','salary'],
            'drop':['education']
            }

    IRIS = {'name':'iris',
            'filename': 'datasets/iris.csv',
            }
    ###### select and load data set

    # ********************************* #
    # comment out all but one line here #
    data = IRIS
    data = ADULT
    data = ALLBUS
#    data = ABILITY
    # ********************************* #
    ## additional parameters for loading the data ##
    # standardize quantitative variables before learning model
    data['standardize'] = False
    data['verb'] = True # print some stats while loading
#    data['off'] = False
#    data['maxiter'] = 2
    # ********************************* #

    ## set regularization parameters ##
    gamma = 10
    model = learn_sparse_model(data, regparam=gamma)
    model.save("savedmodels/%s%f"%(data['name'], gamma))
    
    print(model.mat_q)
    
    # model.get_params()
    # model.get_meanparams()
    # model.get_meta()
    # model.save(outfile="saved_model")
    

