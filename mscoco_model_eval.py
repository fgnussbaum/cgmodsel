#author: Frank Nussbaum
#email: 

#from time import time

import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors

from cgmodsel.models.model_pw import ModelPW
# install the version from the spwmodels models branch
# download Repo from https://github.com/franknu/cgmodsel/tree/spwmodels
# pip install . in the downloaded folder

LABELS = """person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,street sign,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,hat,backpack,umbrella,shoe,eye glasses,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,plate,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,mirror,dining table,window,desk,toilet,door,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,blender,book,clock,vase,scissors,teddy bear,hair drier,toothbrush,hair brush""".split(",")

def findKthLargest(nums, k):
      a = nums.copy().flatten()
      a.sort()
      if k == 1:
         return a[-1]
      return a[len(a)-k]
MODELFOLDER = "data/mscocomodels/"
def model_structure():    
    infile = "mscoco.1000_ga20.00.pw" # all zero
    infile = "mscoco.1000_ga10.00.pw" # all zero
    infile = "mscoco.1000_ga5.00.pw" # (370,0)
#    infile = "mscoco.1000_ga5.00_wc0.20.pw" # (0,519)@1e-2
#    infile = "mscoco.1000_ga5.00_wc0.50.pw" # (0,364)
    
    infile = "mscoco.5000_ga10.00_wc0.10.pw" # (0,91)
    infile = "mscoco.5000_ga10.00_wc0.20.pw"
    
#    infile = "mscoco.train2_ga20.00.pw" # wrong data, (1053,0)@1e-2
#    infile = "mscocomodels/cifar10.50000_ga20.00.pw"
    model = ModelPW(infile=MODELFOLDER + infile)
    print(model.annotations)
#    print(model.mat_lbda[:3, :3])
#    return
#    print(model.meta)
    threshold = 1e-2
    n_edges= model.get_no_edges(threshold=threshold)
    n_vars = model.meta['n_cat'] + model.meta['n_cg']
    
    
    categoricals = model.annotations['categoricals']
    numericals = model.annotations['numericals']
    names = categoricals + numericals
    try:
        print("Model timestamp:", model.annotations['timestamp'], "Iter:", model.annotations['iter'])
    except:
        print("Model does not contain timestamp...")
#    print('Categoricals:', categoricals)
#    print('Numericals:', numericals)    
#    names = model.annotations['categoricals'] + model.annotations['numericals']
#    print(names)
    print("#Edges: %d, Edge-Fraction: %.2f"%(n_edges,
          n_edges/n_vars/(n_vars+1)*2))
    model.repr_graphical(diagonal=False)
    
    graph = model.get_graph(threshold=threshold)
    
    pw_mat = model.get_group_mat(diagonal=False) # matrix with weights for each edge
    n = n_cat = model.meta['n_cat']
    pw_mat2 = pw_mat.copy()
    pw_mat2 -= np.diag(np.diag(pw_mat2))
    
    def print_norms(mat):
#        print(np.sum(np.abs(mat[:n, :n])>0.0000000001))
        cnorm = np.linalg.norm(mat[n:, n:])
        cedges = np.sum(graph[n_cat:, n_cat:]) / 2
        dnorm = np.linalg.norm(mat[:n, :n])
        dedges = np.sum(graph[:n_cat, :n_cat]) / 2
        mnorm = np.linalg.norm(mat[n:, :n])
        medges = np.sum(graph[:n_cat, n_cat:])
        print('Edges (eps=%.4f): dis-dis=%d, dis-cont=%d, cont-cont=%d'%(
                threshold, dedges, medges, cedges))
        print('Norms (eps=%.4f): dis-dis=%.2f, dis-cont=%.2f, cont-cont=%.2f'%(
                threshold, dnorm, mnorm, cnorm))

    print_norms(pw_mat2)
    
    ## thresholding
    cifar10_labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog',
                      'horse', 'ship', 'truck']
#    categoricals = cifar10_labels
    k = 10
    if k > 0:
        discretepart = np.abs(pw_mat2[:n, :n])
        eps = findKthLargest(np.abs(discretepart), k=2 * k)
        xx, yy = (discretepart>eps-1e-6).nonzero()
        print(xx, yy)
        d = {}
        for i in range(2 * k):
            label1 = categoricals[xx[i]]
            label2 = categoricals[yy[i]]
            value = pw_mat2[xx[i], yy[i]]
            if not (label2, label1) in d:
                d[(label1, label2)] = value
        for key in sorted(d.keys(), key=lambda x: d[x], reverse=True):
            print("%.4f: %s ~ %s"%(d[key], *key))

#        model.mat_q[np.abs(model.mat_q) < eps] = 0
#        pw_mat2[:n,:n] = discretepart
#        print_norms(pw_mat2)
#        print("Threshold eps=%.5f for k=%d"%(eps, k))
#    #    print(model.mat_q.shape, discretepart.shape)
#        model.save(outfile=infile[:-3]+"_"+str(k)+"edges.pw")
    
    if True:
        mixedpart = np.abs(pw_mat2[:n, n:])
#        print(mixedpart.shape)
        xx, yy = (mixedpart>threshold).nonzero()
#        print(xx, yy)
        dc = {}
        for cvar in numericals:
            dc[cvar] = 0
        for i in range(len(xx)):
            label1 = categoricals[xx[i]]
            cvar = numericals[yy[i]]
            dc[cvar] += 1
        for key in sorted(dc.keys(), key=lambda x: dc[x], reverse=True):
            if dc[key] > 0:
                print("%s=%d"%(key, dc[key]), end=", ")



if __name__ == "__main__":

    model_structure()




