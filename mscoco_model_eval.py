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
    
#    infile = "mscoco.5000_ga10.00_wc0.10.pw" # (0,91)
#    infile = "mscoco.5000_ga10.00_wc0.20.pw" # (0,537)
#    infile = "mscoco.5000_ga5.00_wc0.25.pw" # (0, 706, 235)
#    infile = "mscoco.5000_ga5.00_wc0.50.pw" # (0,467)
#    infile = "mscoco.5000_ga2.50_wc0.50.pw" # (0,636)
    infile = "mscoco.5000_ga2.50.pw" # (1481, 5)
#    infile = "mscoco.5000_ga3.50_wc0.75.pw" # (142, 408)
    infile = "mscoco.5000_ga5.00_wc1.00.pw"
    infile = "mscoco.5000_ga2.00_wc1.00.pw"
    infile = "mscoco.5000_ga0.50_wc1.00.pw"
    
#    infile = "mscoco.train2_ga20.00_wc0.75.pw" # (1039,0)
#    infile = "mscoco.train2_ga35.00_wc0.25.pw" # (1092, 2)
#    infile = "mscoco.train2_ga40.00_wc0.15.pw" # (827,3)
#    infile = "mscoco.train2_ga50.00_wc0.15.pw" # (366, 2)
#    infile = "mscoco.train2_ga55.00_wc0.08.pw" # (45, 8)
#    infile = "mscoco.train2_ga55.00_wc0.03.pw" # (44,121)
    
    
#    infile = "mscoco.train2_ga25.00_wc0.50.pw" # (1390,0)
#    infile = "mscoco.train2_ga20.00.pw" # wrong data, (1053,0)@1e-2
#    infile = "mscocomodels/cifar10.50000_ga20.00.pw"
    model = ModelPW(infile=MODELFOLDER + infile)
#    print(model.annotations)
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
#            value = pw_mat2[xx[i], yy[i]]
            value = model.mat_q[2 * xx[i] + 1, 2 * yy[i] + 1]
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
        print()
    
#    ft = np.array([0.0,0.12931679,0.0,0.024660666,0.0,0.0,0.16135767,0.29478857,0.0,0.07978679,0.08935765,0.0,0.0,0.041734576,0.0,0.0,0.0,0.5839636,0.0,1.0633538,0.00051944214,0.073792435,0.801105,0.0,0.1030041,0.117181405,2.2680194,0.08455074,2.6343987,0.16007596,1.2777652,1.2979121,0.0,0.0,0.061483156,0.0,1.2519252,1.6666646,0.29636106,0.0,0.014945462,0.6246985,0.0,1.1280851,1.9564515,0.7857597,0.0,0.09131284,0.13026246,0.67078936,1.888037,0.0,0.0,0.0,0.2557966,0.0,0.0,1.6104233,1.8601842,0.16927853,0.0,0.20293471,0.90434766,0.06260271,0.45081395,1.1323996,0.0,0.0,0.6788496,1.083617,0.2313599,1.3877422,0.0,0.14635038,0.77411306,0.012996804,0.6959115,0.0,0.47903258,1.3064733,0.0,0.3679083,0.20451781,0.034372408,0.03712918,0.0,0.608672,0.0,0.0,0.0,1.6200683,0.7909721,0.017245797,1.5296309,0.013516347,0.0,0.3692184,0.0,0.029491626,0.16112453,0.0918065,2.2859063,0.0,0.32986903,0.77121115,0.19294344,0.0,0.0,0.017012611,0.04440041,0.0,0.15466857,0.9624686,0.0,1.1320016,0.0,0.18953836,3.6900964,0.0,0.0,0.15078504,0.55411315,0.0,0.0,0.73896927,0.430899,0.328684,0.0,0.29509997,0.8738276,0.033435673,0.119504385,1.0947077,1.011611,0.29846582,0.019132594,0.07675302,0.00938171,0.0,0.056799382,0.33430272,3.985521,0.0041669635,0.37540254,0.0,0.12639537,1.6019002,0.13063064,0.94290555,1.8429576,0.80058646,0.3304182,0.5353579,0.13800591,0.42216876,0.0,0.110037014,0.0,0.769159,1.0432845,0.0,0.0,0.9923045,0.76403576,0.0,1.0566244,0.60058445,0.51087064,1.1269462,0.0262657,0.81161296,0.11011134,0.9060061,0.0,0.25194556,0.0,0.0,0.078359075,0.07603571,0.018198775,0.064204365,0.0,0.070880726,0.0,1.0501481,0.60315996,0.33189312,0.0,0.096491314,0.9121107,1.4012781,0.71385264,1.04527,0.077867255,0.0,1.0749177,0.11874763,0.5178665,0.19489363,0.0,0.24967946,0.0,1.0146705,0.0,0.0,0.02379654,0.0,0.0,0.0,0.31140268,0.038418338,0.0,0.0,0.0,0.40881822,1.0492164,0.017884225,2.1187437,2.6258895,0.0,2.974927,0.0,1.1711688,1.9255588,0.036445983,1.4040123,1.0777125,1.1316197,0.0,0.0,0.13424657,1.0894011,0.029024694,0.6975049,0.8568702,0.38694346,1.9377402,0.59909433,0.4116872,0.0,0.56856596,0.419609,0.10719503,0.0,0.9354528,0.005856052,0.7452171,0.0,0.052320942,0.4773723,0.028392216,0.08326892,0.0,0.9618366,0.0,1.6954048,0.29123068,0.68376833,0.9452175,0.21201733,0.8940786,0.0,0.0,0.0,0.0,0.0,0.032388467,0.11745178,2.7953234,0.52070224,0.0,0.29579586,0.060309142,0.0,0.30677676,0.0041272976,0.078343585,0.0,0.44480997,0.09205827,0.10337111,0.0,0.0,0.0,0.9455645,0.31088406,0.0114769805,0.94166386,0.966667,0.20357206,0.0,0.006525681,0.33245972,0.21987392,2.0411315,0.0,0.0879444,1.0256221,1.0642536,0.25386113,0.40405047,0.0,0.0,0.0,0.0,0.0301825,0.08462039,0.9831873,1.1337292,0.0,0.0,0.09131017,0.8514121,2.3448954,0.6608329,0.8052224,0.0,0.3953409,0.1865311,0.81884855,0.48003954,0.3561424,0.038626548,0.0,0.53517,0.0,1.0293648,0.3613156,0.0,0.039080113,0.1268276,0.44670218,0.0,0.15964857,0.0,0.011763161,0.70009553,1.1170975,0.64380336,0.005196899,0.0,0.0,0.38540038,0.0,0.0,0.20722754,0.8100753,0.0,0.0,0.5881274,0.0061174855,0.6672459,2.0151029,0.036910653,0.020581294,1.6550486,0.0,0.0,0.6039624,0.6218028,0.52256256,0.0,0.0,0.0,0.0,0.0,0.26107323,0.31724188,0.44379663,0.3838965,0.0,0.36932117,0.5758759,0.72210354,0.6460208,0.5546052,0.0,1.452189,0.10146261,0.038502812,1.4257027,1.0994574,0.71663004,0.0,0.0,0.00075197074,0.007225594,0.22360101,1.278615,1.2196285,0.018267669,0.080256864,0.1624957,0.0,0.0,0.50536627,0.009260587,0.0,0.37010264,1.2011884,0.6168129,0.0,1.0704625,0.5088308,0.5214772,0.0,0.5991076,0.6642109,0.1814152,0.45970893,0.17542425,0.63779056,0.7404556,0.0,0.046528727,0.8413179,0.42452338,0.050102383,0.0,0.28525168,0.0,0.3819957,0.0,0.0,0.0,0.035475053,0.0,3.3312676,0.27971798,0.5940697,1.5041502,0.013398536,0.6811859,0.045343414,0.0,0.0,0.0,0.0,0.0,0.0,1.3649895,0.17427091,0.025432564,0.015420683,0.0,0.0,0.12507442,0.36249334,0.6327199,1.3347944,0.0,2.268353,1.6917007,0.042688567,1.1307662,1.6820837,0.0,1.8844273,0.0,0.39258015,0.03728035,0.0,0.05928832,0.30089062,2.5027204,1.2354518,0.36544222,0.0,0.0,0.4453352,0.0719868,0.48087144,0.3279583,0.0025315732,0.0,0.39002016,0.6681429,1.4119067,0.2493147,0.082988605,0.0,0.36325225,0.054613233,0.5910139,0.010618142,0.48852974,0.3935223,0.12238119,0.1524199,0.0,1.7408783,0.0,0.15637988,0.013676181,0.22169907,0.20958664,0.97868204,0.0,0.0,0.0,0.8953161,2.1024923,0.49118403,0.0,0.0,0.39285117,0.0,0.9903463,0.0,0.0,0.06442838,0.08478561])
#    state = np.array(n_cat * [0, 1])
#    tmp = 2* np.dot(ft, model.mat_r)
#    tmp2 = model.mat_q + np.diag(tmp)
#    x_th_x = np.dot(np.dot(state.T, tmp2), state)
#    print("val", x_th_x, np.exp(x_th_x))
                
    print(np.linalg.norm(model.vec_u))
    print("minQ:", np.min(model.mat_q), "\nminR:",np.min(model.mat_r),
          "\nsumR:", np.sum(model.mat_r))



if __name__ == "__main__":

    model_structure()




