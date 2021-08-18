#author: Frank Nussbaum
#email: 

#from time import time
import csv
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
#      print(a[:10], a[-10:])
      if k == 1:
         return a[-1]
#      print (a[len(a)-k :])
      return a[len(a)-k]
MODELFOLDER = "data/mscocomodels/"

def remove_edge(infile, model, i1, i2):
    model.mat_q[2*i1 + 1, 2*i2 +1 ] = 0
    model.save(outfile='savedmodels/'+infile[:-3] + "rm%s_%s"%(i1, i2) + ".pw")

def model_structure(checksample=True, remedge=None):  
    checksample = True
    posonly = 0
    k = 10
#    remedge = 43, 80
    infile = "mscoco.5000_ga0.10_wc1.00.pw" # (61, 3175) 55 pos
#    infile = "mscoco.5000_ga0.20_wc1000.00.pw" # (839, 0) only 5 pos
    infile = "mscoco.5000_ga0.20_wc1.00.pw" # (27, 1720)
    
#    infile = "mscoco.5000_s_ga0.10_wc1.00.pw" # (1254, 7253) only 1 pos
#    infile = "mscoco.5000_s_ga0.20_wc1.00.pw" # (499, 3518) 0 pos
#    infile = "mscoco.5000_s_ga0.20_wc0.10.pw" # (309, 25903) 0 pos
#    infile = "mscoco.5000_s_ga0.20_wc1.00_u1.pw"
    
#    infile = "mscoco.5000_s_ga0.20_wc1.00_u1_crf1.pw" # (28, 561)
#    infile = "mscoco.5000_s_ga0.10_wc1.00_u1_crf1.pw" # (93, 1350)
#    infile = "mscoco.5000_s_ga0.10_wc0.10_u1_crf1.pw" # (49, 16390)


#    infile = "mscoco.train2_s_ga40.00_wc1.00.pw" # zero
#    infile = "mscoco.train2_s_ga10.00_wc1.00.pw" # (88,0) person and neg only
#    infile = "mscoco.train2_s_ga10.00_wc0.10.pw" # (89, 446)
#    infile = "mscoco.train2_s_ga5.00_wc0.10.pw" # (90, 2208)
#    infile = "mscoco.train2_s_ga3.00_wc0.10.pw" # (94, 4898) zero pos
#    infile = "mscoco.train2_s_ga1.00_wc0.10.pw" # (601, 13887) 2 meaningful pos
#    infile = "mscoco.train2_s_ga1.00_wc1.00.pw" # (683, 586) 4 meaningful
    
#    infile = "mscoco.train2_s_ga2.00_wc0.10_u1_crf1.pw" #(4, 3140) amy
#    infile = "mscoco.train2_s_ga2.00_wc1.00_u1_crf1.pw" # (10, 8) rub
    infile = "mscoco.train2_s_ga1.00_wc0.10_u1_crf1.pw" # (42, 6581) amy
#    infile = "mscoco.train2_s_ga0.50_wc0.10_u1_crf1.pw" # (108, 10823) rubrecht
#    infile = "mscoco.train2_s_ga0.50_wc0.02_u1_crf1.pw" # (100, 23662) amy
#    infile = "mscoco.train2_s_ga0.60_wc0.01_u1_crf1.pw" # (80, 26k) raj
#    infile = "mscoco.train2_s_ga0.70_wc0.01_u1_crf1.pw" # (63, 26k)
#    infile = "mscoco.train2_s_ga0.80_wc0.01_u1_crf1.pw" # (52, 25k)
#    infile = "mscoco.train2_s_ga0.90_wc0.01_u1_crf1.pw"
    infile = "mscoco.train2_s_ga1.50_wc0.01_u1_crf1.pw"
    infile = "mscoco.train2_s_ga1.25_wc0.01_u1_crf1.pw" # (15)
    infile = "mscoco.train2_s_ga1.20_wc0.01_u1_crf1.pw" # (17)
#    infile = "mscoco.train2_s_ga1.10_wc0.01_u1_crf1.pw" # 25
#    infile = "mscoco.train2_s_ga1.05_wc0.01_u1_crf1.pw" # 25

#    infile = "mscoco.train2_ga50.00_wc0.10.pw" # new 400 iter (0, 571)
#    infile = "mscoco.train2_ga40.00_wc0.10.pw" # new 216 iter (0, 638)
#    infile = "/mscoco.train2_ga25.00_wc0.10.pw" # (0, 738)
    
    
#    infile = "mscoco.train2_ga10.00_wc1.00.pw"
#    infile = "mscoco.train2_ga20.00_wc0.75.pw" # (1039,0)
#    infile = "mscoco.train2_ga35.00_wc0.25.pw" # (1092, 2)
#    infile = "mscoco.train2_ga40.00_wc0.15.pw" # (827,3)
#    infile = "mscoco.train2_ga50.00_wc0.15.pw" # (366, 2)
#    infile = "mscoco.train2_ga55.00_wc0.08.pw" # (45, 8)
#    infile = "mscoco.train2_ga55.00_wc0.03.pw" # (44,121)


    model = ModelPW(infile=MODELFOLDER + infile)
    if remedge is not None:
        remove_edge(infile, model, *remedge)
#    print(model.vec_u)
#    print(model.annotations)
#    print(model.mat_lbda[:3, :3])
#    return
    print(model.meta['n_cat'], model.meta['n_cg'])
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
#    model.repr_graphical(diagonal=False)
    
    graph = model.get_graph(threshold=threshold)
    
    
    pw_mat = model.get_group_mat(diagonal=False, norm=False) # matrix with weights for each edge
    n = n_cat = model.meta['n_cat']
    mat = pw_mat.copy()
    mat -= np.diag(np.diag(mat))
    cnorm = np.linalg.norm(mat[n:, n:])
    cedges = np.sum(graph[n_cat:, n_cat:]) / 2
    dnorm = np.linalg.norm(mat[:n, :n])
    dedges = np.sum(graph[:n_cat, :n_cat]) / 2
    mnorm = np.linalg.norm(mat[n:, :n])
    msum = np.sum(mat[n:, :n])
    medges = np.sum(graph[:n_cat, n_cat:])
    print('Edges (eps=%.4f): dis-dis=%d, dis-cont=%d, cont-cont=%d'%(
            threshold, dedges, medges, cedges))
    print('Norms (eps=%.4f): dis-dis=%.2f, dis-cont=%.2f, cont-cont=%.2f'%(
            threshold, dnorm, mnorm, cnorm))
    print('Sums: dis-cont=%.2f'%msum)
    print('norm u=%.2f, norm_alpha=%.2f'%(
            np.linalg.norm(model.vec_u), np.linalg.norm(model.alpha)))
    
    ## thresholding
    cifar10_labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog',
                      'horse', 'ship', 'truck']
#    categoricals = cifar10_labels
    if k > 0:
#        discretepart = np.abs(pw_mat2[:n, :n])
        discretepart = mat[:n, :n]
        if posonly:
            eps = findKthLargest(discretepart, k=2 * k)
#            print(eps)
            xx, yy = (discretepart>eps-1e-6).nonzero()
            func = lambda x: np.abs(d[x][0])
        else:
            eps = findKthLargest(np.abs(discretepart), k=2 * k)
            xx, yy = (np.abs(discretepart)>eps-1e-6).nonzero()  
            func = lambda x: abs(d[x][0])
#        print(xx, yy)
        d = {}
        for i in range(len(xx)):
            label1 = categoricals[xx[i]]
            label2 = categoricals[yy[i]]
#            value = pw_mat2[xx[i], yy[i]]
            
            value = model.mat_q[2 * xx[i] + 1, 2 * yy[i] + 1]
#            print(xx[i], yy[i], value)
            if label1 != label2 and not (label2, label1) in d:
                d[(label1, label2)] = value, xx[i], yy[i]
        kd = 0
        for i, key in enumerate(sorted(d.keys(), key=func, reverse=True)):
            val = d[key]
            print("[%2d]%.4f: %s(%d) ~ %s(%d)"%(
                    i, val[0], key[0], val[1], key[1], val[2]))
            kd += 1
            if kd == k:
                break

#        model.mat_q[np.abs(model.mat_q) < eps] = 0
#        pw_mat2[:n,:n] = discretepart
#        print_norms(pw_mat2)
#        print("Threshold eps=%.5f for k=%d"%(eps, k))
#    #    print(model.mat_q.shape, discretepart.shape)
#        model.save(outfile=infile[:-3]+"_"+str(k)+"edges.pw")
    
    if 0:
        mixedpart = np.abs(mat[:n, n:])
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
    
    if not checksample: 
        return
    standardized = "_s" if "_s_" in infile else ""
    fn = "data/mscoco.valid2%s.csv"%standardized
    sampleid = 4 # 4, 10
#    sampleid = 3 # 0, 3 (multi is best) # 5000_s
    with open(fn, 'r') as incsv:
        reader = csv.reader(incsv)
        next(reader)
        for i in range(sampleid): next(reader)
        row = next(reader)
#    print(row)
    row = [np.float(e) for e in row]
#    data = [0.012632163,0.7088276,0.01885062,0.005252573,0.13743073,0.0,1.5431647,0.73189294,0.14048664,0.69426215,0.5626675,0.0,0.0,1.1577055,0.0,0.00075496733,0.38008538,0.0,0.07906811,0.20083898,0.9048407,0.8271369,0.6926383,0.0,0.0,0.5040344,0.0,0.0,0.0,0.7455004,0.076627105,0.63723683,0.0,0.18367141,0.012292583,0.21258396,0.0,1.8629199,0.0,1.9848757,0.080286406,0.0,0.44816595,0.30894297,0.97638154,0.53692746,0.6389731,0.0,0.094142035,0.0,1.5221407,0.0,0.2793125,0.0,0.0,0.20122322,0.0,0.0,1.5374775,0.21728933,0.064948276,0.10518545,0.0,0.0,0.43539792,0.016492521,0.0,0.29937863,0.7109288,1.9447141,0.0,1.7982628,0.01629535,0.45172122,0.66409516,1.1690627,1.3795718,0.0,1.4706736,0.92902505,1.3343546,0.17722721,1.1139512,0.0,0.36261386,0.312239,0.9304556,0.5372733,0.0,0.0,1.151315,0.23891109,0.5435123,0.23897748,0.0,0.0,0.0,0.10251112,0.0,0.3112637,0.17932408,0.8212373,0.0,3.9870956,2.3450146,0.0,0.50361615,0.056528334,0.0,0.0,0.0,0.33158612,2.0325637,0.0,0.0,0.02345933,0.22855572,1.6952817,0.0,0.28718454,0.0,0.09169473,0.100333154,0.0,1.4529299,0.2874044,0.5053285,0.0,0.481575,0.39451772,0.0,0.3102088,0.3435744,0.6461357,0.07318343,0.0,1.4131962,0.0,0.0,0.7025824,0.89448494,0.3227072,0.4188981,0.8869134,0.0,0.10050055,0.7492291,0.00058696105,0.37309635,0.78998446,0.055758186,0.0,0.776629,0.017264767,0.38348556,0.0,0.0,0.13241519,0.0,0.2345657,0.0,0.0,1.6084545,0.35986722,0.26811057,1.2473493,1.3527167,0.13105094,0.31987196,0.0,0.3976127,0.0,2.9431014,0.051258024,0.25616223,0.0,0.056341946,0.0,0.17723265,0.0,0.01649693,0.0,0.32360452,0.0005915642,0.4923846,0.05083327,0.070983484,8.9618086e-05,2.0531914,0.76021516,1.7778671,0.0052566985,3.9597476,0.71357304,0.0,0.15409428,0.37951586,0.51906943,0.14011806,0.05800254,0.8577087,0.0,1.6343167,0.045006286,0.6304262,0.036588974,0.016711928,0.026308555,0.0,0.0,0.02232837,1.27662,0.07366091,0.60402775,0.8109741,1.0425446,0.023565006,0.0,0.52033657,1.6853404,0.0078013283,0.0,0.2905198,0.0,0.21012391,0.42567563,1.5228934,1.3829083,1.9158596,0.8837505,0.0,1.1578699,0.0,0.8390308,1.5755122,0.20136918,0.42465198,0.89198595,0.0,1.35662,2.6563845,0.0,0.19689578,0.0,0.7837824,0.0,0.5504773,0.38884476,1.4230036,0.143917,0.0894021,0.06173666,0.0,1.1980007,0.0,0.00807404,1.330189,2.9925618,1.1667352,0.0,1.0124824,1.4758474,3.3568606,0.0,0.5307888,0.109614335,0.0,1.0144603,1.2696068,0.0598784,0.35309142,0.64280343,0.0068967454,0.0,0.0,0.27171606,0.0,3.1361284,0.12143268,0.23510873,0.08487156,0.02235818,0.0,0.0,0.49321225,0.0035185833,0.0,1.8741115,0.39276025,0.91712195,0.0,0.0,1.4792352,0.1257868,0.118308656,0.0,0.0,0.12235114,0.7447479,0.0,0.64466685,0.09228313,0.19383585,1.186097,0.022240436,2.1420405,0.5266829,0.0,0.24222119,0.40334046,0.53270626,0.36992833,0.4702354,0.0,0.4451387,1.1833625,0.0,0.0,0.110243395,0.30712613,0.0,0.23294576,0.0,2.6748683,0.2663074,0.15731804,1.8314239,0.17488918,0.0,0.0,0.12173644,0.48215717,0.10951096,0.020775538,0.0,0.8301025,0.0,0.3844884,0.0,0.15176937,0.0,0.0,0.26383182,0.0,0.38962165,0.014339205,1.3871112,0.0,0.4846835,0.84380776,0.02651431,0.8462244,0.8982397,0.0,0.0,4.6302977,3.7514832,0.0,0.7134758,1.1289088,0.011987954,0.22127372,0.0,0.079410315,0.18701741,0.3615936,0.0043956987,0.009427243,0.0983657,0.52302897,0.0,0.49915275,1.182514,0.25477946,0.6140188,0.0,0.0,0.44656318,0.0,0.0,0.0,1.377766,1.4264386,0.50128627,0.088981524,0.0,0.15245473,0.13641173,0.017045928,0.8956174,0.59382915,0.7296766,0.89550364,0.0,0.38808507,0.0,0.7434017,0.81884634,0.0,1.4266683,0.0,0.0,0.90891206,0.9577042,0.57677025,0.1419612,0.93320143,0.7858985,0.50518584,1.0599947,0.0,0.0,0.80548835,0.0,0.017651439,0.6747402,0.2234731,0.0,0.0,0.25930235,0.0,0.0,3.8042176,0.0,0.0,0.46755132,0.03540361,0.2535581,0.67060655,0.47690082,0.050993547,0.0,0.15551111,2.7625403,0.0,0.0,0.0,0.0,0.0,0.0,0.054675106,0.0,0.06330995,1.2925466,0.0,0.20260717,0.6164899,0.10569651,0.6380304,0.25587237,1.1438937,0.0,0.17455241,0.0,1.1028123,0.040118717,0.6105794,0.0,0.0,0.5570868,0.15749379,0.0,0.2396757,0.31257367,0.0,0.027210357,0.060175106,0.64174604,0.7983796,0.39963138,0.19167611,0.0,0.218986,0.25441265,0.0,0.24958378,0.50306726,1.3343723,0.0,0.0,0.010857467,0.64961696,0.04373376,0.68987477,0.5197232,3.348407,0.15012279,0.0,0.0,0.0,0.31773147,0.0,0.6983419,1.2649094,0.09790009,0.16978443,0.68926895,1.5580326,0.2093234,0.0,0.60114735,0.19263949,0.70058656,0.0,0.0,0.0,0.0,0.30320266,1.6679392,0.4278182,0.0,1.851194,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ft = np.array(row[:512], dtype=np.float64)
    t = [[0,e] for e in row[512:]]
    state = []
    labels = [i for i, j in enumerate(row[512:]) if j == 1]
    print("Labels:", labels)
    for l in t:
        state += l
#    return
    state = np.array(state)
    states = []
    state0 = np.array(n_cat * [1, 0])
    states.append(("one", np.array(n_cat * [0, 1])))
    states.append(("zer", state0))
    for k in range(0, n_cat):
#        state = np.array(n_cat * [1, 0])
#        state = np.array(n_cat * [0,1])
    #    k = 5
        state = state0.copy()
        state[2*k:2*(k+1)] = np.array([0,1])
        states.append(("e_%d"%k, state))
    for i in range(len(labels)):
        state = np.array(n_cat * [1, 0])
        l1 = labels[i]
        state[2*l1:2*(l1+1)] = [0,1]
        states.append(("%d"%(l1), state))
        for j in range(i):
            state2 = state.copy()
            l2 = labels[j]
#            state2[2*l1:2*(l1+1)] = [0,1]
            state2[2*l2:2*(l2+1)] = [0,1]
            states.append(("%d_%d"%(l1,l2), state2))
    if len(labels) > 2:
        state = state0.copy()
        for l in labels:
            state[2*l:2*(l+1)] = [0,1]
#        print(state)
        states.append(("%s"%labels, state))
        
    tmp = 2* np.dot(ft, model.mat_r)
    tmp2 = model.mat_q + np.diag(tmp) # pairwise discrete parameter after cond.
    for p in states:
        name, state = p
        x_th_x = np.dot(np.dot(state.T, tmp2), state) # x^T Theta x
        val = np.exp(x_th_x)
        if val >= 1 or name[0] != 'e':
            print("val %s=%.4f"%(name, val))
                
#    print(np.linalg.norm(model.vec_u))
#    print("minQ:", np.min(model.mat_q), "\nminR:",np.min(model.mat_r),
#          "\nsumR:", np.sum(model.mat_r))



if __name__ == "__main__":

    model_structure()




