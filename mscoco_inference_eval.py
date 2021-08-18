# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:01:36 2021

@author: Frank
"""

import json
import numpy as np
import pickle

import sys
sys.path.append("../")
from send_mail import send_mail
#data = json.load("mscoco.json")

modelfile = "mscoco.train2_s_ga1.20_wc0.01_u1_crf1.pw"

path = "../et4cg/data/experiments/mscoco/%s.json"%modelfile

def load_pkl(filename):
    file = open(filename, "rb")
    return pickle.load(file)

def load_npy(filename):
    return np.load(filename)

def generate_subset(sois, prefix='data/mscoco/'):    
    mode = 'valid2'
#    mode = 'train2'

    filetype = 'npy'
    load_func = {'npy':load_npy, 'pkl':load_pkl}[filetype]
    
    x = load_func(prefix+'X_%s.%s'%(mode, filetype))
    
    print(x.shape)
    x_small = np.zeros([len(sois)]+list(x.shape[1:]))
    for j, soi in enumerate(sois):
        i = soi[0]
        x_small[j, :, :, :] = x[i, :, :, :]
    
    filename = "%s_%d.dat"%(mode, len(sois))
    with open(filename, "wb") as f:
        pickle.dump([x_small, sois], f)
#    np.save(prefix + filename, x_small)
    
    scp = """scp frank@amy.inf-i2.uni-jena.de:/home/frank/cgmodsel/%s%s data/queryevaldata/%s\n"""%(
            prefix, filename, filename)
    send_mail("subset of mscoco valid2:\n%s \n\nindices: %s"%(
            scp, str(indices)))

  
def get_wrong_entries(vec1, vec2aug):
    addlabels = []
    missinglabels = []
    for i in range(91):
        if vec2aug[i] != -1  and vec1[i] != vec2aug[i]:
            if vec1[i] == 1:
                addlabels.append(i)
            else:
                missinglabels.append(i)
    errors = len(addlabels) + len(missinglabels)
    return errors, addlabels, missinglabels

def augment(bin_vec, indices):
    j=0
    assert len(bin_vec) + len(indices) == 91
    augmented = np.zeros(91)
    for i in range(len(bin_vec)):
        if indices[j] == i:
            j += 1
            augmented[i] = -1
        else:
            augmented[i+j] = bin_vec[i]
        
    return augmented


# read file
with open(path, 'r') as myfile:
    data=myfile.read()
    
obj = json.loads(data)

mlc_states = obj['experimentdata']['MLC_max_disc_states']
data = obj['data']
exp_data = obj['experimentdata']
print(exp_data.keys())
#print(obj['metadata'].keys())
n_test = len(exp_data['BINC_max_disc_states']) # no of test data points
print('Loaded json with %d data points'%n_test)

indices = obj['metadata']['independent_discrete_variables']

fun_transform = lambda x, indices: x
if len(indices) != 0:
    print("Independent variables", indices)
    fun_transform = augment

sois = [] # samples of interest
for i in range(n_test):
    ground_truth = data[i]
    n_labels = np.sum(ground_truth)
#    print(ground_truth)
#    bin_vec = exp_data['BINC_max_disc_states'][i]
#    if len(ids) > 0:
#        bin_vec_aug = augment(bin_vec, ids)
#    else:
#        bin_vec_aug = bin_vec
#    bin_error = get_no_wrong_entries(ground_truth, bin_vec_aug)
    bin_error = -1
    
    mpes = exp_data['MLC_max_disc_states'][i]
    print("Sample %d has %d MPE states, "%(i, len(mpes)), end="")
    mult_error = -1
    if len(mpes) == 1:
        mult_vec = mpes[0]
#        print(len(ground_truth), len(mult_vec))
        mult_vec = fun_transform(mult_vec, indices)
        res = get_wrong_entries(ground_truth, mult_vec)
        mult_error, addlabels, missinglabels = res
        if mult_error <= 1:
            sois.append([i] + [res])
        print("err_b=%d, err_m=%d (n_labels=%d)"%(bin_error, mult_error, n_labels))
    elif len(mpes) == 0:
        print("No max state")
    else:
        print("Multiple maxstates %d"%len(mpes))

    
print("Samples of interest:", sois)

#datapath = "data/mscoco/mscoco.valid2_s.csv"
generate_subset(sois)
    
#print(obj.keys())
#print(len(mlc_states))
#print(len(data))




