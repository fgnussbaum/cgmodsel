#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2021

requires package cgmodsel (check out the latest version from the spwmodels branch)
"""
# pylint: disable=C0103
import sys
import time
import numpy as np

import logging

from cgmodsel.admm import AdmmCGaussianPW
from cgmodsel.dataops import load_prepare_data  # function to read data
from cgmodsel.dataops import standardize_continuous_data

DATAFOLDER = 'data/datasets/'
MODELFOLDER = 'savedmodels/'

def learn_sparse_models(dataname, srange, logger, opts, 
                        verb=False, 
                        solver_verb = False):
    
    file_train = DATAFOLDER + dataname + '.train.data'
    file_test = DATAFOLDER + dataname + '.test.data'
    file_features = DATAFOLDER + dataname + '.features'
    categoricals = []
    numericals = []
    names = []
    catuniques = {}
    with open(file_features, "r") as f:
        for row in f.readlines():
            name, ty, levels = row.split(':')
            names.append(name)
#            print(name, ty)
            if ty == 'continuous':
                numericals.append(name)
            else:
                categoricals.append(name)
                if levels[-1] == '.':
                    levels = levels[:-1]
                else:
                    levels = levels[:-2]
                levels = levels.split(',')
                try:
                    levels = [int(level) for level in levels]
                except:
                    print('Labels of variable %s not converted to int'%name)
                catuniques[name] = levels
#    print(categoricals)
#    print(numericals)

    cat_data, cont_data, meta = load_prepare_data(file_train,
                                                  verb=True,
                                                  categoricals=categoricals,
                                                  names=names,
                                                  catuniques=catuniques,
                                                  cattype='dummy_red',
                                                  **opts)

    cat_test, cont_test, meta = load_prepare_data(file_test,
                                                  categoricals=categoricals,
                                                  names=names,
                                                  catuniques=catuniques,
                                                  cattype='dummy_red',
                                                  **opts)
#    print(meta)
#    print(np.prod(meta['sizes']))
#    return
    t1 = time.time()
#    print(meta)
#    return
    meanssigmas = standardize_continuous_data(cont_data)
    standardize_continuous_data(cont_test, meanssigmas)
    if verb:
        print('Standardizing time: %.1fs'%(time.time() - t1))

    ## initialize solver/validater and drop data ##
    solver = AdmmCGaussianPW()
    solver.drop_data((cat_data, cont_data), meta)
    
    validater = AdmmCGaussianPW()
    validater.drop_data((cat_test, cont_test), meta)
    
    print('Solving problems...')
    end, steps, frac = srange
#    models = []
    plh_tests = []
    warminit = None
    best_model = None
    best_ge = np.inf
    for i in range(end + steps, end, -1):
        alpha = i / frac
        if alpha == 0:
            alpha += 1e-3
        gamma = alpha / (1 - alpha)
        t1 = time.time()
        solver.set_regularization_params(gamma)
        out = solver.solve(verb=solver_verb, 
                           warminit=warminit,
                           use_u=0, off=0, **opts)
        t2 = time.time()
        # mat_theta, mat_s, mat_z, alpha
        warminit = out['theta'], out['solution'][0], out['dual'], out['solution'][1]
#        print(warminit)
        model = solver.get_canonicalparams()  # PW model instance
        model.update_annotations(categoricals=categoricals,
                                 numericals=numericals,
                                 alpha=alpha,
                                 gamma=gamma,
                                 iter =out['iter'])
        print(model.annotations)
        model.save("%s/%s%df%d.pw"%(MODELFOLDER, dataname, frac, i))
        
        theta, u, alpha = model.get_pairwiseparams(padded=False)
#        print(theta, u, alpha)
        plh_test = validater.get_objective(theta, vec_u=u, alpha=alpha, 
                                           use_reg=False)
        msg = '[%s] alpha=%d/%d gamma=%.3f (%.1fs, GE=%.2f, it=%d)'%(
                dataname, i, frac, gamma, t2 - t1, plh_test, out['iter'])
        print(msg)
        sys.stdout.flush()
        logger.info(msg)
        if plh_test < best_ge:
            best_ge = plh_test
            best_model = model
        plh_tests.append(plh_test)
#        models.append(model)
#    print(plh_tests)
    
    best_model.repr_graphical(diagonal=0) # plottype='pn'
    
    return best_model

def parse_mscoco():
    import pickle, csv
    labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "street sign", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle",
            "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "mirror", "dining table",
            "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "blender", "book",
            "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush", "hair brush"]
    print(len(labels))
    
    prefix = 'data/mscoco/'
    R_train = open(prefix+'R_train.pkl', "rb")
    cont_data = pickle.load(R_train)
    m, n = cont_data.shape
    print(m, n)
#    return
    cat_data = np.zeros((m, 100))
    y_train = open(prefix+'Y_train.pkl', "rb")
    y_train = pickle.load(y_train)
    for i in range(m):
        for j in range(100):
            label = y_train[i, j] - 1
            cat_data[i, label] == 1
#    print(y_train[:1, :], y_train.shape)
            
    with open('data/mscoco.train.csv', 'w', newline='') as outcsv: # newline='' for WINDOWS

        writer = csv.writer(outcsv)
        writer.writerow(["Y%d"%(i) for i in range(256)] + 
                         ['X'+label for label in labels])
        for i in range(m):
            writer.writerow(list(cont_data[i, :]) + list(cat_data[i, :]))
    

if __name__ == '__main__':

    # ********************************* #
    # comment out all but one line here #
    dataname = 'adult' # D=, C=, N=
#    dataname = 'australian' # D=, C=, N=
#    dataname = 'anneal-U' # D=33, C=6, N=539
#    dataname = 'autism' # D=18, C=10, N=2217
#    dataname = 'autismC' # D=7, C=21, N=2217
#    dataname = 'auto'  # D=11, C=15, N=97
#    dataname = 'balance-scale' # D=1, C=4, N=375
#    dataname = 'breast' # D=1, C=10, N=411
#    dataname = 'breast-cancer' # D=10, C=0, N=165
#    dataname = 'cars' # D=2, C=7, N=235
#    dataname = 'cleve' # D=8, C=6, N=178
#    dataname = 'crx' # D=10, C=6, N=418
#    dataname = 'diabetes' # D=1, C=8, N=461
#    dataname = 'german' # D=14, C=7, N=600
#    dataname = 'german-org' # D=13, C=12, N=600
#    dataname = 'heart' # D=1, C=13, N=162
#    dataname = 'iris' # D=, C=, N=
    dataname = 'mscoco'
    # ********************************* #
    parse_mscoco()
    logging.basicConfig(filename='solved_probs.log', level=logging.INFO)

    logger = logging.getLogger('sp_pw') # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
#    logging.getLogger("matplotlib").setLevel(logging.WARNING)
#    logging.setLevel(logging.INFO)
#    logger.info('Test')


    steps = 5
    end = 0
    frac = 1000
    srange = end, steps, frac
    opts = {'maxiter':1200}
#    model = learn_sparse_models(dataname, srange, logger, opts, solver_verb=100)
    

