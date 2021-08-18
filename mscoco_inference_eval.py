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

def generate_subset(indices, prefix='data/mscoco/'):    
    mode = 'valid2'
#    mode = 'train2'

    filetype = 'npy'
    load_func = {'npy':load_npy, 'pkl':load_pkl}[filetype]
    
    x = load_func(prefix+'X_%s.%s'%(mode, filetype))
    
    print(x.shape)
    x_small = np.zeros([len(indices)]+list(x.shape[1:]))
    for i in indices:
        x_small[i, :, :, :] = x[i, :, :, :]
    
    filename = "%s_%d.npy"%(mode, len(indices))
    np.save(prefix + filename, x_small)
    
    scp = """scp frank@amy.inf-i2.uni-jena.de:/home/frank/cgmodsel/%s%s data/mscocomodels/%s\n"""%(
            prefix, filename, filename)
#    send_mail("subset of mscoco valid2:\n%s \n\nindices: %s"%(
#            scp, str(indices)))

  
def get_no_wrong_entries(vec1, vec2aug):
    errors = 0
    for i in range(91):
        if vec2aug[i] != -1  and vec1[i] != vec2aug[i]:
            errors += 1
    return errors

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

errorfree = []
for i in range(n_test):
    ground_truth = data[i]
#    print(ground_truth)
#    bin_vec = exp_data['BINC_max_disc_states'][i]
#    if len(ids) > 0:
#        bin_vec_aug = augment(bin_vec, ids)
#    else:
#        bin_vec_aug = bin_vec
#    bin_error = get_no_wrong_entries(ground_truth, bin_vec_aug)
    bin_error = -1
    
    mpes = exp_data['MLC_max_disc_states'][i]
    print(i, "has %d MPE states"%len(mpes))
    mult_error = -1
    if len(mpes) == 1:
        mult_vec = mpes[0]
        print(len(ground_truth), len(mult_vec))
        mult_vec = fun_transform(mult_vec, indices)
        mult_error = get_no_wrong_entries(ground_truth, mult_vec)
        if mult_error == 0 or 1:
            errorfree.append(i)
    elif len(mpes) == 0:
        print("No max state")
    else:
        print("Multiple maxstates %d"%len(mpes))

    print("Sample%d: err_b=%d, err_m=%d"%(i, bin_error, mult_error))
print("Indices errorfree", errorfree)

#datapath = "data/mscoco/mscoco.valid2_s.csv"
generate_subset(errorfree)
    
#print(obj.keys())
#print(len(mlc_states))
#print(len(data))




