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


def grp_soft_shrink(mat,
                    tau,
                    n_groups=None,
                    glims=None,
                    off=False,
                    weights=None):
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
        tmp = np.abs(mat)
        if not weights is None: # weighted l1-norm
#            tmp = np.multiply(tmp, weights).flatten
            tmp -= tau * weights
        else:
            tmp -= tau
        tmp[tmp < 1e-25] = 0
        shrinked = np.multiply(np.sign(mat), tmp)
        l1norm = np.sum(np.abs(shrinked.flatten()))
        if off:
            l1norm -= np.sum(np.abs(np.diag(shrinked)))
            shrinked -= np.diag(np.diag(shrinked))
            shrinked += np.diag(np.diag(mat))

        return shrinked, l1norm

    # group soft shrink
    if weights is None:
        weights = np.ones(mat.shape) # TODO(franknu): improve style
    tmp = np.empty(mat.shape)
    for i in range(n_groups):
        for j in range(n_groups):
            # TODO(franknu): use symmetry
            group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
            if (i == j) and off:
                tmp[glims[i]:glims[i + 1], glims[i]:glims[i + 1]] = group
                continue

            gnorm = np.linalg.norm(group, 'fro')
            w_ij = tau * weights[i,j]
            if gnorm <= w_ij:
                tmp[glims[i]:glims[i + 1],
                    glims[j]:glims[j + 1]] = np.zeros(group.shape)
            else:
                tmp[glims[i]:glims[i+1], glims[j]:glims[j+1]] = \
                    group * (1 - w_ij / gnorm)
                shrinkednorm += (1 - w_ij / gnorm) * gnorm

    return tmp, shrinkednorm


def l21norm(mat, n_groups=None, glims=None, off=False, weights=None):
    """
    calculate l_{2,1}-norm or l_1-norm of mat
    l_1-norm if no n_groups is given
    else must provide with n_groups (# groups per row/column) and
    cumulative sizes of groups (glims)
    """
    if n_groups is None:
        # calculate regular l1-norm
        tmp = np.abs(mat) # tmp is copy, can do this inplace by specifying out
        if not weights is None: # weighted l1-norm
            tmp = np.multiply(tmp, weights).flatten
        tmp = np.sum(tmp)
        if off:
            tmp -= np.sum(np.diag(np.abs(mat)))
        return tmp
        
    l21sum = 0
    if weights is None:
        for i in range(n_groups):
            for j in range(i):
                group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
                l21sum += np.linalg.norm(group, 'fro')
    else:
        for i in range(n_groups):
            for j in range(i):
                group = mat[glims[i]:glims[i + 1], glims[j]:glims[j + 1]]
                l21sum += weights[i,j] * np.linalg.norm(group, 'fro')        

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


def _exp_shiftedmax(array, axis=None):
    """calculate exponentials of array shifted by its max, avoiding overflow
    by subtracting maximum before"""
    a_max = np.amax(array, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0


#    print((a-a_max).shape)
    exp_shiftedamax = np.exp(array - a_max)
    # last line: a_max is repeated columnwise (if axis = 1)

    return exp_shiftedamax, a_max


def logsumexp(array, axis=None, keepdims=True):
    """Compute the log of the sum of exponentials of input elements.
    this is an adaptation of logsumexp in scipy.special (v1.1.0)
    """

    exp_shifted, a_max = _exp_shiftedmax(array, axis=axis)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        summed = np.sum(exp_shifted, axis=axis, keepdims=keepdims)

        out = np.log(summed)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


def _logsumexp_and_conditionalprobs(array):
    """return logsumexp and conditional probabilities from array a
    that has the same shape as the discrete data in dummy-representation"""
    exp_shifted, a_max = _exp_shiftedmax(array, axis=1)

    summed = np.sum(exp_shifted, axis=1, keepdims=True)  # entries always > 1

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out_logsumexp = np.log(summed)
    out_logsumexp += a_max

    # node conditional probabilities
    size = array.shape[1]

    out_conditionalprobs = np.divide(exp_shifted,
                                     np.dot(summed, np.ones((1, size))))

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


def _logsumexp_condprobs_red(array):
    """normalization and conditional probabilities for reduced levels,
    a ... two-dimensional array"""

    a_max = np.amax(array, axis=1, keepdims=True)
    a_max = np.maximum(a_max, 0)
    # last line: account for missing column with probs exp(0) for 0th level

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    exp_shifted = np.exp(array - a_max)  # a_max is repeated columnwise (axis=1)

    # calc column vector s of (shifted) normalization sums
    # note that entries always > 1, since one summand in each col is exp(0)
    summed = np.sum(exp_shifted, axis=1, keepdims=True)

    summed += np.exp(-a_max)  # add values from missing 0th column

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out_logsumexp = np.log(summed)
    out_logsumexp += a_max

    out_logsumexp = np.squeeze(out_logsumexp)

    # node conditional probabilities, required for gradient
    size = array.shape[1]
    out_conditionalprobs = np.divide(exp_shifted,
                                     np.dot(summed, np.ones((1, size))))
    # note: log of this is not stable if probabilities close to zero
    # - use logsumexp instead for calculating plh value

    return out_logsumexp, out_conditionalprobs


###############################################################################
# some conversion functions for representations of discrete data
###############################################################################


def dummy_to_index_single(dummy_x, sizes):
    """convert dummy to index representation"""
    offset = 0
    ind = np.empty(len(sizes), dtype=np.int)
    for i, size_r in enumerate(sizes):
        for j in range(size_r):
            if dummy_x[offset + j] == 1:
                ind[i] = j
                break
        offset += size_r

    return ind


def dummy_to_index(dummy_data, sizes):
    """convert dummy to index representation"""
    n_data, ltot = dummy_data.shape
    assert ltot == sum(sizes)
    n_cat = len(sizes)

    index_data = np.empty((n_data, n_cat), dtype=np.int)
    for k in range(n_data):
        offset = 0
        for i, size_r in enumerate(sizes):
            for j in range(size_r):
                if dummy_data[offset + j] == 1:
                    index_data[k, i] = j
                    break
            offset += size_r

    return index_data


#def dummypadded_to_unpadded(dummy_data, n_cat):
#    """remove convert dummy to index representation"""
#    unpadded = np.empty(n_cat)
#    for i,x in enumerate(dummy_data):
#        if i % 2 == 1:
#            unpadded[i // 2] = x
#    return unpadded


def index_to_dummy(idx, glims, ltot):
    """convert index to dummy representation"""
    dummy_data = np.zeros(ltot)
    for i, ind in enumerate(idx):
        dummy_data[glims[i] + ind] = 1
    return dummy_data


def dummy2dummyred(dummy_data, glims):
    """convert dummy to reduced dummy representation"""
    return np.delete(dummy_data, glims[:-1], 1)


###############################################################################
# testing utilities
###############################################################################
def strlistfrom(array, rnd=2):
    """a convenient representation for printing out numpy array
    s.t. it can be reused as a list"""

    string = np.array2string(array, precision=rnd, separator=',')
    string = 'np.array(' + string.translate({ord(c): None for c in '\n '}) + ')'

    return string


def tomatlabmatrix(mat):
    """print numpy matrix in a way that can be pasted into MATLAB code """
    nrows, ncols = mat.shape
    string = "["
    for i in range(nrows):
        string += "["
        for j in range(ncols):
            string += str(mat[i, j]) + " "
        string += "];"
    string = string[:-1] + "]"
    print(string)


def frange(start, stop, step):
    """ a float range function"""
    i = start
    while i < stop:
        yield i
        i += step


if __name__ == '__main__':
    SIZES = [2, 2, 2]
    GLIMS = [0, 2, 4, 6]
    LTOT = 6
    IND = [0, 0, 1]
    DUMMY = index_to_dummy(IND, GLIMS, LTOT)
    IND2 = dummy_to_index_single(DUMMY, SIZES)

    MAT = np.arange(6).reshape((3, 2))

    RES = _logsumexp_condprobs_red(MAT)

    print(RES)
    # res should be
    # (array([ 1.55144471,  3.34901222,  5.31817543]), array([[ 0.21194156,  0.57611688],
    #    [ 0.25949646,  0.70538451],
    #    [ 0.26762315,  0.72747516]]))
