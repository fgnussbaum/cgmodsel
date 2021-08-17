# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:01:36 2021

@author: Frank
"""

import json
import numpy as np
#data = json.load("mscoco.json")

def generate_subset(indices, prefix='data/mscoco/'):
#    print(len(labels))
    
    mode = 'valid2'
    mode = 'train2'
#    mode = '5000'
    filetype = 'npy'
    load_func = {'npy':load_npy, 'pkl':load_pkl}[filetype]
    
    x_train = load_func(prefix+'X_%s.%s'%(mode, filetype))
    
    print(x_train.shape)
    
    img_i = x_train[i, :, :, :]
    print(np.min(img_i), np.max(img_i))
    

    img_i = np.swapaxes(img_i, 1,2) # acb
    img_i = np.swapaxes(img_i, 0,2) # bca

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
#    meanssigmas = means, stds
    
#    meanssigmas = standardize_continuous_data(img_i[],
#                                              meanssigmas=meanssigmas)
    for j in range(3):
        img_i[:, :, j] = img_i[:, :, j] * stds[j] + means[j]

    fig = plt.figure(figsize=(10,5))

#    if not title is None:
#        fig.suptitle(title, fontsize=18, y=0.73)
    plt.imshow(img_i, interpolation='nearest',
#                   vmin=Tvmin, vmax=Tvmax, cmap =cmap, 
                   aspect='auto')
    plt.title('mat_x')
    
    plt.show()
    
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
if len(ids) != 0:
    print("Independent variables", ids)
    raise
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
    mult_error = -1
    if len(mpes) == 1:
        mult_vec = mpes[0]
        print(len(ground_truth), len(mult_vec))
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




