# distutils: extra_compile_args = XCOMPARGS
# distutils: extra_link_args = XLINKARGS
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: embedsignature = True

#from __future__ import print_function
import numpy as np
from cython.parallel cimport prange


def grp(double[:,::1] mat,
        double tau,
        long[:] glims, 
        int off = 0,
        long n_threads = 4):
    """ This function computes the group shrinkage operation on z """
    cdef double shrinkednorm = 0.0, gnorm, num, fac
    cdef int i, j, k, l;
    cdef int ngroups = glim.shape[0] - 1;

    for i in prange(glims.shape[0] - 1, nogil=True,
                    schedule='static', num_threads=n_threads):
#    for i in range(ngroups):
        for l in range(ngroups):
            gnorm = 0.0
            for k in range(glims[i], glims[i + 1]):
                for j in range(glims[l], glims[l + 1]):
                    gnorm += mat[k, j] * mat[k, j]
            if i == l and off:
                continue
            if gnorm > 0.0:
                gnorm = gnorm ** 0.5
                fac = max(0.0, (1.0 - tau / gnorm))
                for k in range(glims[i], glims[i + 1]):
                    for j in range(glims[l], glims[l + 1]):
                        mat[k, j] = mat[k, j] *  fac
                shrinkednorm += fac * gnorm

    return np.asarray(mat), shrinkednorm



#def grp_weighted(double[:,::1] z,
#                 double[:] weights,
#                 double tau,
#                 long[:] Lcum,
#                 long n_threads = 4):
#    """ This function computes a weighted group norm of z"""
#    cdef double shrinkednorm = 0.0, gnorm, num, fac
#    cdef int i, j, k;
#
#    for i in prange(Lcum.shape[0] - 1,
#                    nogil=True,
#                    schedule='static',
#                    num_threads=n_threads):
#        for j in range(z.shape[1]):
#            gnorm = 0.0
#            for k in range(Lcum[i], Lcum[i + 1]):
#                gnorm += z[k, j] * z[k, j]
#            if gnorm > 0.0:
#                gnorm = gnorm ** 0.5
#                fac = max(0.0, (1.0 - tau * weights[i]/ gnorm))
#                for k in range(Lcum[i], Lcum[i + 1]):
#                    z[k, j] = z[k, j] *  fac
#                shrinkednorm += weights[i] * fac * gnorm
#
#    return np.asarray(z), shrinkednorm
