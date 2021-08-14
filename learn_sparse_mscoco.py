#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2021

requires package cgmodsel (check out the latest version from the spwmodels branch)
"""
# pylint: disable=C0103
import sys
import socket
HOSTNAME = socket.gethostname()
import time
import numpy as np

import logging

from cgmodsel.admm import AdmmCGaussianPW
from cgmodsel.dataops import load_prepare_data  # function to read data
from cgmodsel.dataops import standardize_continuous_data


sys.path.append("../")
from send_mail import send_mail


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

cifar10_labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog',
                      'horse', 'ship', 'truck']
def set_weights(meta, wd, wm, wc):
    import numpy as np
    n_cg = meta['n_cg']
    n_cat = meta['n_cat']
    weights = np.zeros((n_cat + n_cg, n_cat + n_cg))
    weights[:n_cat, :n_cat] = wd
    weights[:n_cat, n_cat:] = wm
    weights[n_cat:, :n_cat] = wm
    weights[n_cat:, n_cat:] = wc
    return weights

def learn_sparse_model(logger, opts, 
                        verb=False, 
                        gamma = 20, 
                        dataname = 'mscoco.1000',
                        wc = 1,
                        univariate=1):
    file_train = 'data/%s.csv'%dataname
    
    if dataname.startswith('mscoco'):
        catuniques = {}
        for label in LABELS:
            catuniques[label] = [0,1]
        labels = LABELS
    else:
        catuniques = None
        labels = cifar10_labels
    cat_data, cont_data, meta = load_prepare_data(file_train,
#                                                  verb=True,
                                                  categoricals=labels,
#                                                  names=names,
                                                  catuniques=catuniques,
                                                  cattype='dummy_red',
                                                  **opts)
    
#    print(np.dot(cat_data.T, cat_data) / cat_data.shape[0])
#    print(cat_data[:3, :20])
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
        t1 = time.time()
        solver.set_regularization_params(gamma)
#        wc = 0.5
        if not wc is None:
            weights = set_weights(meta, 1, wc, 1)
#            print(weights)
#            print(cont_data[:10, 502:506])
            solver.set_weights(weights)

        out = solver.solve(warminit=warminit,
                           use_u=univariate, use_alpha=univariate, **opts)
        t2 = time.time()
        # mat_theta, mat_s, mat_z, alpha
        warminit = out['theta'], out['solution'][0], out['dual'], out['solution'][1]
#        print(warminit)
        model = solver.get_canonicalparams()  # PW model instance
        model.update_annotations(categoricals=labels,
                                 numericals=['Y%d'%i for i in range(cont_data.shape[1])],
#                                 alpha=alpha,
                                 gamma=gamma,
                                 iter =out['iter'])
        print(model.annotations)
#        model.save("%s/N%s%.2f.pw"%(MODELFOLDER, dataname, gamma))
        if not wc is None:
            modelfilename = "%s_ga%.2f_wc%.2f_u%d_crf%d.pw"%(
                    dataname, gamma, wc, univariate, opts['discrete_crf'])
        else:
            modelfilename = "%s_ga%.2f_u%d_crf%d.pw"%(
                    dataname, gamma, univariate, opts['discrete_crf'])
        model.save(MODELFOLDER + modelfilename)
        scp = """scp frank@%s.inf-i2.uni-jena.de:/home/frank/cgmodsel/%s%s data/mscocomodels/%s\n"""%(
                HOSTNAME.split('.')[0], MODELFOLDER, modelfilename, modelfilename)
        send_mail("learned model from data [%s]\n%s"%(
                dataname, scp))
        
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
    
#    best_model.repr_graphical(diagonal=0) # plottype='pn'
    
    return best_model

def load_pkl(filename):
    file = open(filename, "rb")
    return pickle.load(file)

def load_npy(filename):
    return np.load(filename)

def parse_mscoco(meanssigmas=None,
                 prefix='data/mscoco/',
                 standardize=False, mode='5000'):
    import pickle, csv
#    print(len(labels))

    filetype = 'npy'
    load_func = {'npy':load_npy, 'pkl':load_pkl}[filetype]
    
    
    cont_data = load_func(prefix+'R_%s.%s'%(mode, filetype))
    if standardize:
        meanssigmas = standardize_continuous_data(cont_data)
#                                              meanssigmas=meanssigmas)
#    means, sigmas = meanssigmas
#    sigmas *= 100
#    meanssigmas = means, sigmas
#    meanssigmas = standardize_continuous_data(cont_data,
#                                              meanssigmas=meanssigmas)
    
        print("std", meanssigmas[1][:10])
        print("mean", meanssigmas[0][:10])
    m, n = cont_data.shape
    print(m, n)
#    return meanssigmas
    cat_data = np.zeros((m, len(LABELS)), dtype=np.int64)
    y_train = load_func(prefix+'Y_%s.%s'%(mode, filetype))
    for i in range(m):
        for j in range(100):
            label = y_train[i, j] - 1
#            print(label)
            if label != -1:
                cat_data[i, label] = 1
#        print(y_train[i, :])
#        return
#    print(y_train[5, :], y_train.shape)
#    print('cat_data shape', cat_data.shape)
    
    cor_mat = np.corrcoef(cat_data.T)
    baserates = np.sum(cat_data, axis=0).astype(np.float) / cat_data.shape[0]
#    print(baserates.shape)
    np.save("cormat%s.npy"%mode, cor_mat)
#    print(cor_mat.shape)
    pair = 63, 85 # pottet plant, vase .35
    pair = 73, 75 # mouse, keyboard .67
    pair = 61, 66 # chair, dining table .31
    pair = 2, 7 # car, truck .37
#    pair = 0, 16 # person, cat
    print("Cross-cor(%d, %d): %.3f (base-rates: %.2f, %.2f)"%(
            pair[0], pair[1], cor_mat[pair[0],pair[1]], baserates[pair[0]],
            baserates[pair[1]]))
    sorted_cormat = cor_mat.copy()
    sorted_cormat -= np.diag(np.diag(sorted_cormat))
    sorted_cormat = sorted_cormat.flatten()
    sorted_cormat.sort()
    sorted_cormat = sorted_cormat[np.logical_not(np.isnan(sorted_cormat))] # remove nan
    print(sorted_cormat)
    print(np.nanmax(sorted_cormat))
    
    ss = '_s' if standardize else ''
    assert len(LABELS) == cat_data.shape[1]
    with open('data/mscoco.%s%s.csv'%(mode, ss), 'w', newline='') as outcsv: # newline='' for WINDOWS

        writer = csv.writer(outcsv)
        writer.writerow(["Y%d"%(i) for i in range(n)] + LABELS)
        for i in range(m):
            writer.writerow(list(cont_data[i, :]) + list(cat_data[i, :]))
#    send_mail("parsed %s"%(mode))
            
def parse_cifar10(meanssigmas=None,
                 prefix='data/cifar10/'):
    import pickle, csv
#    print(len(labels))
    
    mode = 'valid2'
#    mode = 'train2'
    mode = '50000'
    filetype = 'npy'
    load_func = {'npy':load_npy, 'pkl':load_pkl}[filetype]
    
    
    cont_data = load_func(prefix+'R_%s.%s'%(mode, filetype))
    meanssigmas = standardize_continuous_data(cont_data)
#                                              meanssigmas=meanssigmas)
    means, sigmas = meanssigmas
    nonzero_idx = np.where(sigmas != 0)
#    print(meanssigmas)
    m, n = cont_data.shape
    cont_data = cont_data[:, nonzero_idx].squeeze()
    print(m, n, cont_data.shape, len(nonzero_idx[0]))
#    return meanssigmas

    y_train = load_func(prefix+'Y_%s.%s'%(mode, filetype))
    cat_data = np.zeros((m, 10), dtype=np.int64)
    
    for i in range(m):
        cat_data[i,y_train[i]] = 1
#    print(y_train[:100])
   
    CLABELS = ["X%d"%i for i in range(10)]
    with open('data/cifar10.%s.csv'%mode, 'w', newline='') as outcsv: # newline='' for WINDOWS

        writer = csv.writer(outcsv)
        writer.writerow(["Y%d"%(i) for i in nonzero_idx[0]] + CLABELS)
        for i in range(m):
            writer.writerow(list(cont_data[i, :]) + list(cat_data[i, :]))
            # class label between 0 and 9
#    send_mail("parsed")
    

if __name__ == '__main__':

    # ********************************* #
    # comment out all but one line here #
#    dataname = 'mscoco'
    # ********************************* #
#    ms = parse_mscoco(standardize=True, mode='5000')
#    parse_cifar10()
#    ms = parse_mscoco(meanssigmas=ms)
    logging.basicConfig(filename='solved_probs.log', level=logging.INFO)

    logger = logging.getLogger('sp_pw') # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
#    logging.getLogger("matplotlib").setLevel(logging.WARNING)
#    logging.setLevel(logging.INFO)
#    logger.info('Test')


#    file_train = 'data/mscoco.train.csv'
    
    
#    dataname = 'mscoco.train2' # Barlow-Twin Features
#    dataname = 'cifar10.50000'
    steps = 5
    end = 0
    frac = 1000
    srange = end, steps, frac
    opts = {'maxiter':1200, 'solver_verb':True, 'off':1, 'discrete_crf':1}

    if HOSTNAME == 'amy':
        gamma = 1; wc =.1;  dataname = 'mscoco.train2_s'; univariate=1
    elif HOSTNAME == 'rubrecht':
        gamma = 2; wc =1; dataname = 'mscoco.train2_s'; univariate=1
#        gamma = .2; wc =1; dataname = 'mscoco.5000_s'; univariate=1
    elif HOSTNAME == 'raj.inf-i2.uni-jena.de':
        gamma = .1; wc =.1; dataname = 'mscoco.5000_s'; univariate=1
    elif HOSTNAME == 'DESKTOP-H168PMB':
        gamma = 1; wc =1; dataname = 'mscoco.train2_s'; univariate=1
        opts['verb'] = 1
        gamma = .2; wc =1; dataname = 'mscoco.5000_s'; univariate=1
    else:
        print(HOSTNAME)
        raise
    model = learn_sparse_model(logger, opts, 
                               gamma=gamma, wc=wc,
                               dataname = dataname,
                               univariate=univariate)

    

