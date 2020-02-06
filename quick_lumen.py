#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2020

Demo for Lumen
"""

from cgmodsel.dataops import load_prepare_data # function for loading data
from cgmodsel.huber_clz import HuberCLZ # Huber solver for CLZ models (triple interactions)
from cgmodsel.admm import AdmmCGaussianPW

def get_graph_from_data(filename, model='CLZ', graphthreshold=1e-2, kS=1, **kwargs):
    """
    filename   ... csv file containing data
    model      ... 'PW' (pairwise) or 'CLZ' (triple interactions ~ variable precision matrices)
    
    kS         ... regularization parameter for l1 regularization

    Output:
    grpnormmat ... matrix with non-negative entries (edge weights = group-norms)
    graph      ... boolean matrix (graph)
    dlegend    ... dictionary of col/row-names
    """
    D, Y, meta = load_prepare_data(filename, **kwargs)
    # D .. discrete data, Y .. continuous data, meta .. dictionary of meta info

    print('Filename:', filename)
    print('Using a dataset with %d samples, %d discrete and %d continuous variables.' % (meta['n'], meta['dc'], meta['dg']))
    print('Discrete Variables: %s' % (meta['catnames']))
    print('Continuous Variables: %s\n' % (meta['contnames']))
    
    ### solve regularized problem
    dSolvers = {'PW': AdmmCGaussianPW, 'CLZ': HuberCLZ}

    solver = dSolvers[model](meta, useweights=True) # initialize problem
    solver.drop_data(D, Y)
    solver.set_regularization_params(kS)
    
    print('Solving problem..')
    res = solver.solve_sparse(verb=1, innercallback=solver.nocallback)
    # print(" Done.")
    
    x_reg = res.x # solution
    params = solver.get_canonicalparams(x_reg, verb=0) # model parameters

    ## graphical representation
    params.repr_graphical(graph=True, threshold=graphthreshold, caption='l1-regularized')

    graph = params.get_graph(threshold=graphthreshold)    
    grpnormmat = params.get_group_mat()
    
    ## legend
    print("Legend:")
    dlegend = {}
    for i, name in enumerate(meta['catnames']+meta['contnames']):
        print("%d - %s"%(i, name))
        dlegend[i] = name

    return grpnormmat, graph, dlegend


def load(dataset, cattype='dummy'):
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
        
    return load_prepare_data(dataset['filename'], cattype=cattype, verb=1, **loaddict)
    

if __name__ == '__main__':
   
###### data sets
    
    ## binary ##
    ABILITY = {
            'filename': "datasets/ability_proc.csv",
            'sparams': {'catuniques': [0,1]} # values that binary variables take
            }
    CFMT = { # 72 binary variables
            'filename': "datasets/CFMTkurzBIN.csv",
            'sparams': {'catuniques': [0,1]} # values that binary variables take
            }

    ## quantitative ##

    ## mixed binary-quantitative ##
    ALLBUS = { # pw model
            'filename': "datasets/allbus2016_proc.csv",
            }
    HELP = {
            'filename': "datasets/HELPmiss_proc.csv",
            }

    IRIS = {
            'filename': "datasets/iris.csv"
            }


###### select and load data set

    # ********************************* #
    # comment out all but one line here #
    dataset = ABILITY 
#    dataset = IRIS
#    dataset = HELP # solving CLZ model will take a while
    # ********************************* #

    print('Loading data...')
    D, Y, meta = load(dataset) # load the data
    
###### fit models    
    
    ## initialize solver and drop data ##
#    get_graph_from_data(filename)

    kS = 2 # regularization parameter
    graphthreshold = 1e-1


    solver = HuberCLZ() # initialize problem
    solver.drop_data((D, Y), meta)
    solver.set_regularization_params(kS)
    ## set regularization parameters ##

    ## solve the problem, that is, estimate a sparse + low-rank model ##
    print('Solving the problem...')

    print('Solving problem..')
    res = solver.solve_sparse(verb=1, innercallback=solver.nocallback)
    #print(" Done.")


###### model visualization

    x_reg = res.x # solution
    params = solver.get_canonicalparams(x_reg, verb=0) # model parameters

    params.repr_graphical(graph=True,
                          threshold=graphthreshold,
                          caption='l1-regularized')
