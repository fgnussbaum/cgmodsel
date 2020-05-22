#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: -----, 2020
"""
# pylint: disable=C0103

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from cgmodsel.admm_nuc import AdmmIsingSL
from cgmodsel.dataops import load_prepare_split_data

from fps import FPS


def get_clbda(fps, hat_phi):
    ul_phi = fps.round_down(hat_phi) # return double prec approx of ul_phi
    ov_phi = fps.get_ub()
    
    max_error = ov_phi - ul_phi
#    print(max_error)

    c = np.max(max_error) # maximum norm    
    lbda = np.linalg.norm(max_error, 2) # largest singular value
    
    return c, lbda

def get_fps_ifin_clbdas(clbdas, c, lbda, eps=1e-4):
    for c0, lbda0, maxk, prec in clbdas:
        if abs(c0-c) < eps and abs(lbda-lbda0) < eps:
            return maxk, prec
    return None
    

def cross_validate(filename):
    loaddict = {'catuniques': [0, 1]}
    data = load_prepare_split_data(filename,
                                   splittingfactor=0.7,
                                   cattype='dummy_red',
                                   shuffleseed=12, 
                                   **loaddict)
    
    data_train, _, data_test, _, meta = data
    ## calculate hat Phi
    
    hat_phi = np.dot(data_train.T, data_train) / meta['n_data']
    print(hat_phi.shape)
    
    maxbits = 12
    mat = np.zeros((maxbits, maxbits)) # dimensions are: bits x prec
    cs = np.zeros(mat.shape)
    lbdas = np.zeros(mat.shape)
    target = 9, 6 # nbits, bits for prec
    
    clbdas = [] # store regparam configs in order not to solve too many probs
    for nbits in range(2, maxbits + 1):
        print('Budget: %d bits'%nbits)
        for prec in range(1, nbits):
#            prec = 4
            maxk = 2 ** (nbits - prec) - 1
#            print(prec, maxk)
            fps = FPS(maxk, 2, prec) # k, base, precision
            
            path = 'expfpreg/CFMT%d_%d.npz'%(maxk, prec)

            if not os.path.exists(path):
                c, lbda = get_clbda(fps, hat_phi)
                
                # check if there already is a file with a solution for 
                # this c and lbda
                kt = get_fps_ifin_clbdas(clbdas, c, lbda)
                if not kt is None:
                    existing_file = 'expfpreg/CFMT%d_%d.npz'%kt
                    shutil.copy(existing_file, path)
            
            oi = (nbits == target[0] and prec == target[1])
            if os.path.exists(path):
#            if os.path.exists(path) and not oi:
                res = np.load(path)
                mat_l = res['mat_l']
                mat_s = res['mat_s']
                c = res['c']
                lbda = res['lbda']
                cv = res['cv']
                print('Regparams for %s:'%fps, c, lbda)
                if get_fps_ifin_clbdas(clbdas, c, lbda) is None:
                    clbdas.append((c, lbda, maxk, prec))
            else:
                solver = AdmmIsingSL()
                solver.drop_data(data_train, meta)
            
                ## set regularization parameters ##
                regparams = c, lbda
                solver.set_regularization_params(regparams, set_direct=True)
    #            print(solver)
    
                res = solver.solve(verb=0, **{'use_u':0})
                
                solver.drop_data(data_test, meta)
                mat_s, mat_l, _ = res['solution']
    #            print(mat_l)
                cv = solver.get_objective(mat_s, mat_l)
                
                np.savez(path, mat_s=mat_s, mat_l=mat_l, c=c, lbda=lbda, cv=cv)

                if oi:
                    model = solver.get_canonicalparams()  # S + L model instance
                    model.plot_sl(plottype='pn')
                
            print('**PLH value test data', cv)
            print(lbda/c)

            mat[nbits - 1, prec - 1] = cv # - 1 cause indices start at 0
            cs[nbits - 1, prec - 1] = c
            lbdas[nbits - 1, prec - 1] = lbda



#            break
    
    titles = ['plh_test', 'c', 'lbda']
    mats = mat, cs, lbdas
    f, axes = plt.subplots(ncols=3, figsize = (15, 6))

    for i in range(3):
        # TODO: mask zeros, 
        axes[i].set_xlabel('bits for precision (t)')
        axes[i].set_ylabel('nbits (total)')
        im = axes[i].matshow(mats[i], origin='upper') #, extent=[-3, 3, -3, 3])
        axes[i].set_title(titles[i], y = 1.15)
        
        labels = axes[i].get_xticks()
#        print(labels)
        nlabels = list(labels[1:])+[labels[-1]+1, ]
        nlabels = [int(label) for label in nlabels]
#        print(nlabels)
        nlabels = [-1, 1, 3, 5, 7, 9, 11]
        axes[i].set_xticklabels(nlabels)
        axes[i].set_yticklabels(nlabels)

        plt.colorbar(im, ax=axes[i], shrink=.65)

    axes[0].scatter(target[1]-1, target[0]-1, color = 'red',
        marker = '*', s = 150)
    plt.savefig("plots/crossval.pdf", bbox_inches='tight')
    
if __name__ == '__main__':

    ###### data sets

    ## binary ##
    ABILITY = "datasets/ability_proc.csv"
    CFMT = "datasets/CFMTkurzBIN.csv"

    ###### select and load data set

    # ********************************* #
    # comment out all but one line here #
    filename = CFMT
    # ********************************* #

    print('Loading data...(%s)'%(filename))
    
    cross_validate(filename)
    
