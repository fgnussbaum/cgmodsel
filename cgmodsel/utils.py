# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de)

This file contains various functions used in the module including
- sparse norms and shrinkage operators
- a stable logsumexp implementation
- array printing-method that allows pasting the output into Python code

"""

import numpy as np


#################################################################################
# norms and shrinkage operators
#################################################################################

def grp_soft_shrink(mat, tau, n_groups=None, glims=None, off=False):
    """
    calculate (group-)soft-shrinkage of mat with shrinkage parameter tau
    soft shrink if no n_groups is given
    else must provide with n_groups (# groups per row/column) and
    cumulative sizes of groups (glims)

    this code could be made much faster
    (by parallizing loops, efficient storage access)
    """
    shrinkednorm = 0
    if n_groups is None:
        # soft shrink
#        if tau == 0:
#            return mat, np.sum(np.abs(mat.flatten()))
        tmp = np.abs(mat) - tau
        tmp[tmp < 1e-25] = 0

        shrinked = np.multiply(np.sign(mat), tmp)
        if off:
            shrinked -= np.diag(np.diag(shrinked))
            shrinked += np.diag(np.diag(mat))
        
        return shrinked, np.sum(np.abs(shrinked.flatten()))

    # group soft shrink
#    if tau == 0:
#        for i in range(n_groups):
#            for j in range(n_groups):
#                group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
#                if (i == j) and off:
#                    continue
#                shrinkednorm += np.linalg.norm(group, 'fro')
#        return mat, shrinkednorm

    tmp = np.empty(mat.shape)
    for i in range(n_groups):
        for j in range(n_groups):
            group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
            if (i == j) and off:
                tmp[glims[i]:glims[i + 1], glims[i]:glims[i + 1]] = group
                continue
            
            gnorm = np.linalg.norm(group, 'fro')
            if gnorm <= tau:
                tmp[glims[i]:glims[i + 1],
                    glims[j]:glims[j + 1]] = np.zeros(group.shape)
            else:
                tmp[glims[i]:glims[i+1], glims[j]:glims[j+1]] = \
                    group * (1 - tau / gnorm)
                shrinkednorm += (1 - tau / gnorm) * gnorm

    return tmp, shrinkednorm

def l21norm(mat, n_groups=None, glims=None, off=False):
    """
    calculate l_{2,1}-norm or l_1-norm of mat
    l_1-norm if no n_groups is given
    else must provide with n_groups (# groups per row/column) and
    cumulative sizes of groups (glims)
    """
    if n_groups is None:
        tmp = np.sum(np.abs(mat.flatten()))
        if off:
            tmp -= np.sum(np.diag(np.abs(mat)))
        return tmp
        # calculate regular l1-norm
    l21sum = 0
    for i in range(n_groups):
        for j in range(i):
            group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
            l21sum += np.linalg.norm(group, 'fro')

    l21sum *= 2 # use symmetry
    if not off:
        for i in range(n_groups):
            group = mat[glims[i]:glims[i + 1], glims[i]:glims[i + 1]]
            l21sum += np.linalg.norm(group, 'fro')

    return l21sum


###############################################################################
# stable implementation of logsumexp etc.
###############################################################################    
#from scipy.special import logsumexp

def _exp_shiftedmax(a, axis=None):
    """calculate exponentials of array shifted by its max, avoiding overflow
    by subtracting maximum before"""
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

#    print((a-a_max).shape)
    exp_shiftedamax = np.exp(a - a_max) # a_max is repeated columnwise (if axis = 1) here
    
    return exp_shiftedamax, a_max

def logsumexp(a, axis=None, keepdims=True):
    """Compute the log of the sum of exponentials of input elements.
    
    this is an adaptation of logsumexp in scipy.special (v1.1.0)
    """
    
    exp_shifted, a_max = _exp_shiftedmax(a, axis=axis)
    
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(exp_shifted, axis=axis, keepdims=keepdims)

        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out

def _logsumexp_and_conditionalprobs(a):
    """return logsumexp and conditional probabilities from array a
    that has the same shape as the discrete data in dummy-representation"""
    exp_shifted, a_max = _exp_shiftedmax(a, axis = 1)

    s = np.sum(exp_shifted, axis=1, keepdims=True) # entries are always > 1
    
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out_logsumexp = np.log(s)
    out_logsumexp += a_max
    
    # node conditional probabilities
    size = a.shape[1]

    out_conditionalprobs = np.divide(exp_shifted, np.dot(s, np.ones((1, size))))
    
#    unstable = np.log(np.sum(np.exp(a), axis = 1)).reshape((a.shape[0], 1))
#    diff = unstable - out_logsumexp
#    print (unstable)
#    for i in range(unstable.shape[0]):
#        if abs(diff[i, 0]) > 10e-5:
#            print('a', a[i, :])
#            print('unstable', unstable[i, 0])
#            print('stable', out_logsumexp[i, 0])
#            break
#    assert np.linalg.norm(unstable - out_logsumexp) < 10E-5

    
#    print(out_logsumexp)
#    print(out_logsumexp[:1, 0])
#    assert 1 == 0
    
    out_logsumexp = np.squeeze(out_logsumexp)
    
    return out_logsumexp, out_conditionalprobs

def _logsumexp_condprobs_red(a):
    """normalization and conditional probabilities for reduced levels,
    a ... two-dimensional array"""

    a_max = np.amax(a, axis=1, keepdims=True)
    a_max = np.maximum(a_max, 0) # account for missing column with probs exp(0) for 0th level

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    exp_shifted = np.exp(a - a_max) # a_max is repeated columnwise (if axis = 1) here

    # calc column vector s of (shifted) normalization sums
    # note that entries always > 1, since one summand in each col is exp(0)
    s = np.sum(exp_shifted, axis=1, keepdims=True) 

    s += np.exp(-a_max) # add values from missing 0th column
    
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out_logsumexp = np.log(s)
    out_logsumexp += a_max
    
    out_logsumexp = np.squeeze(out_logsumexp)
    
    # node conditional probabilities, required for gradient
    size = a.shape[1]
    out_conditionalprobs = np.divide(exp_shifted, np.dot(s, np.ones((1, size))))
    # note: log of this is not stable if probabilities close to zero - use logsumexp instead for calculating plh value
        
    return out_logsumexp, out_conditionalprobs

###############################################################################
# some conversion functions
###############################################################################

def dummy_to_index_single(Dx, sizes):
    offset = 0
    ind = np.empty(len(sizes), dtype = np.int)
    for i, sr in enumerate(sizes):
        for j in range(sr):
            if Dx[offset+j] == 1:
                ind[i] = j
                break
        offset += sr

    return ind

def dummy_to_index(Dx, sizes):
    n, Ltot = Dx.shape
    assert Ltot == sum(sizes)
    dc = len(sizes)
    
    X = np.empty((n, dc), dtype = np.int)
    for k in range(n):
        offset = 0
        for i, sr in enumerate(sizes):
            for j in range(sr):
                if Dx[offset+j] == 1:
                    X[k, i] = j
                    break
            offset += sr

    return X

def dummypadded_to_unpadded(Dx, dc):
    d = np.empty(dc)
    for i,x in enumerate(Dx):
        if i % 2 == 1:
            d[i // 2] = x
    return d

def index_to_dummy(idx, Lcum, Ltot):
    Dx = np.zeros(Ltot)
    for i, ind in enumerate(idx):
        Dx[Lcum[i]+ind] = 1
    return Dx

def dummy2dummyred(D, Lcum):
    return np.delete(D, Lcum[:-1], 1)

###############################################################################
# testing utilities
###############################################################################
def strlistfrom(a, rnd=2):
    """a convenient representation for printing out numpy array s.t. it can be reused as a list"""

    s = np.array2string(a, precision = rnd, separator = ',')
    s = 'np.array('+s.translate({ord(c): None for c in '\n '})+')'
     
    return s

def tomatlabmatrix(A):
    """print numpy matrix in a way that can be pasted into MATLAB code """
    m ,n = A.shape
    s = "["
    for i in range(m):
        s += "["
        for j in range(n):
            s += str(A[i,j]) + " "
        s += "];"
    s = s[:-1] + "]"
    print(s)
        

def frange(start, stop, step):
    """ a float range function"""
    i = start
    while i < stop:
        yield i
        i += step


if __name__ == '__main__':
    sizes = [2,2,2]
    Lcum = [0,2,4,6]
    Ltot = 6
    ind = [0,0,1]
    Dx = index_to_dummy(ind, Lcum, Ltot)
    ind2 = dummy_to_index_single(Dx, sizes)
    
    A = np.arange(6).reshape((3,2))
    
    res = _logsumexp_condprobs_red(A)
    
    print(res)
    """ res should be
    (array([ 1.55144471,  3.34901222,  5.31817543]), array([[ 0.21194156,  0.57611688],
       [ 0.25949646,  0.70538451],
       [ 0.26762315,  0.72747516]]))
    """
    
    
    