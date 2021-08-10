# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:01:36 2021

@author: Frank
"""

import json
import numpy as np
#data = json.load("mscoco.json")

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

path = "../et4cg/data/experiments/mscoco/mscoco.json"
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

ids = obj['metadata']['independent_discrete_variables']
#n_cat = 91
#ids = [1,3] + [n_cat + 1]
#print(ids)
#indices = []
#le = 0
#for i in ids:
#    indices += [le + j for j in range(i - le)]
#    le = indices[-1] + 2
#print(indices)
#n_test = 10
for i in range(n_test):
    ground_truth = data[i]
#    print(ground_truth)
    bin_vec = exp_data['BINC_max_disc_states'][i]
    bin_vec_aug = augment(bin_vec, ids)
    
    mult_vec = exp_data['MLC_max_disc_states'][i]
#    print(mult_vec)
    if not mult_vec is None:
        print(len(mult_vec))
    else:
        print(None)
#    mult_vec_aug = augment(mult_vec, ids)

#    print(bin_vec_aug)
    bin_error = get_no_wrong_entries(ground_truth, bin_vec_aug)
#    mult_error = get_no_wrong_entries(ground_truth, mult_vec_aug)
    mult_error = -1
    print("Sample%d: err_b=%d, err_m=%d"%(i, bin_error, mult_error))

#print(obj.keys())
#print(len(mlc_states))
#print(len(data))