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

LABELS = [
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

def learn_sparse_model(logger, opts, 
                        verb=False, 
                        solver_verb = False):
    file_train = 'data/mscoco.train.csv'
    dataname = 'mscoco'

    catuniques = {}
    for label in LABELS:
        catuniques[label] = [0,1]
    cat_data, cont_data, meta = load_prepare_data(file_train,
                                                  verb=True,
                                                  categoricals=LABELS,
#                                                  names=names,
                                                  catuniques=catuniques,
                                                  cattype='dummy_red',
                                                  **opts)

#    print(meta)
#    return
#    meanssigmas = standardize_continuous_data(cont_data)
#    standardize_continuous_data(cont_test, meanssigmas)

#    print(1)
    ## initialize solver/validater and drop data ##
    solver = AdmmCGaussianPW()
#    print(2)
    solver.drop_data((cat_data, cont_data), meta)
#    print(3)
#    validater = AdmmCGaussianPW()
#    validater.drop_data((cat_test, cont_test), meta)
    
    print('Solving problems...')
    end, steps, frac = srange
#    models = []
    plh_tests = []
    warminit = None
    best_model = None
    best_ge = np.inf
    for i in range(1):
#        alpha = i / frac
#        if alpha == 0:
#            alpha += 1e-3
#        gamma = alpha / (1 - alpha)
        gamma = 0.01
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
        model.update_annotations(categoricals=LABELS,
                                 numericals=['Y%d'%i for i in range(cont_data.shape[1])],
#                                 alpha=alpha,
                                 gamma=gamma,
                                 iter =out['iter'])
        print(model.annotations)
        model.save("%s/N%s%.2f.pw"%(MODELFOLDER, dataname, gamma))
        
#        theta, u, alpha = model.get_pairwiseparams(padded=False)
#        print(theta, u, alpha)
#        plh_test = validater.get_objective(theta, vec_u=u, alpha=alpha, 
#                                           use_reg=False)
#        msg = '[%s] alpha=%d/%d gamma=%.3f (%.1fs, GE=%.2f, it=%d)'%(
#                dataname, i, frac, gamma, t2 - t1, plh_test, out['iter'])
#        print(msg)
        sys.stdout.flush()
#        logger.info(msg)
#        if plh_test < best_ge:
#            best_ge = plh_test
#            best_model = model
#        plh_tests.append(plh_test)
#        models.append(model)
#    print(plh_tests)
    
    best_model.repr_graphical(diagonal=0) # plottype='pn'
    
    return best_model

def parse_mscoco(meanssigmas=None):
    import pickle, csv
#    print(len(labels))
    
    mode = 'valid'
#    mode = 'train'
    prefix = 'data/mscoco/'
    R_train = open(prefix+'R_%s.pkl'%mode, "rb")
    cont_data = pickle.load(R_train)
    meanssigmas = standardize_continuous_data(cont_data,
                                              meanssigmas=meanssigmas)
    print(meanssigmas)
    m, n = cont_data.shape
#    print(m, n)
#    return meanssigmas
    cat_data = np.zeros((m, 100), dtype=np.int64)
    y_train = open(prefix+'Y_%s.pkl'%mode, "rb")
    y_train = pickle.load(y_train)
    for i in range(m):
        for j in range(100):
            label = y_train[i, j] - 1
#            print(label)
            if label != -1:
                cat_data[i, label] = 1
#        print(y_train[i, :])
#        return
    print(y_train[:1, :], y_train.shape)
            
    with open('data/mscoco.%s.csv'%mode, 'w', newline='') as outcsv: # newline='' for WINDOWS

        writer = csv.writer(outcsv)
        writer.writerow(["Y%d"%(i) for i in range(256)] + LABELS)
        for i in range(m):
            writer.writerow(list(cont_data[i, :]) + list(cat_data[i, :]))
    

if __name__ == '__main__':

    # ********************************* #
    # comment out all but one line here #
    dataname = 'mscoco'
    # ********************************* #
#    ms = parse_mscoco()
#    ms = parse_mscoco(meanssigmas=ms)
    logging.basicConfig(filename='solved_probs.log', level=logging.INFO)

    logger = logging.getLogger('sp_pw') # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
#    logging.getLogger("matplotlib").setLevel(logging.WARNING)
#    logging.setLevel(logging.INFO)
#    logger.info('Test')


#    steps = 5
#    end = 0
#    frac = 1000
#    srange = end, steps, frac
    opts = {'maxiter':1200}
    model = learn_sparse_model(logger, opts, solver_verb=100)
    

