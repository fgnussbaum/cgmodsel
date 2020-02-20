# Copyright (c) 2017-2019 Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""
@author: Frank Nussbaum

A class to fit CLZ CG models

this code is experimental, no warranty
"""
from typing import Iterable
from math import inf  # used in derived classes
import numpy as np

from cgmodsel.base_huber import HuberBase, _huberapprox
from cgmodsel.base_solver import BaseCGSolver
from cgmodsel.base_solver import set_sparsity_weights
from cgmodsel.models.model_clz import ModelCLZ

# not checked with pylint


############################################################## FUll Pseudo LH
class HuberCLZ(HuberBase, BaseCGSolver):
    """
    A class that provides with methods (model selection, crossvalidation)
    associated with CLZ CG models

    code in this class is experimental
    """

    def __init__(self, useweights=True):
        """pass a dictionary that provides with keys dg, dc, and L"""
        super().__init__()

        self.cat_format_required = 'dummy'
        self.name = 'CLZ'

        self.lbda = None  # regularization parameter

        self.cont_data_prod = None
        self.cont_data_square = None

        self.useweights = useweights
        self.weights = None

    def _postsetup_data(self):
        """called after drop_data"""

        # precompute array of size n * dg ** 2
        n_data = self.meta['n_data']
        ltot = self.meta['ltot']
        self.cont_data_prod = np.empty((n_data, self.meta['n_cg']**2))
        # store values of squared cts empirical sufficient statistics
        for s in range(self.meta['n_cg']):
            for t in range(self.meta['n_cg']):
                self.cont_data_prod[:, s*self.meta['n_cg'] + t] = \
                    np.multiply(self.cont_data[:, s], self.cont_data[:, t])
        self.cont_data_square = np.empty((n_data, self.meta['n_cg']))
        for s in range(self.meta['n_cg']):
            self.cont_data_square[:, s] = \
                self.cont_data_prod[:, s*(self.meta['n_cg']+1)]

        # dimensions of parameters mat_q, u, R, B, B^d, alpha
        self.shapes = [('Q', (ltot, ltot)), ('u', (ltot, 1)),
                       ('R', (self.meta['n_cg'], ltot)),
                       ('B0', (self.meta['n_cg'], self.meta['n_cg'])),
                       ('beta0', (self.meta['n_cg'], 1)),
                       ('B', (self.meta['n_cg'], self.meta['n_cg'], ltot)),
                       ('Bdiag', (self.meta['n_cg'], ltot)),
                       ('alpha', (self.meta['n_cg'], 1))]
        self.n_params = sum([np.prod(shape[1]) for shape in self.shapes])

        if self.useweights:
            self.weights = set_sparsity_weights(self.meta, self.cat_data, self.cont_data)

        else:
            self.weights = np.ones((self.meta['n_catcg'],
                                    self.meta['n_catcg']))

    def set_regularization_params(self, regparam):
        """set regularization parameters
        kS is a scaling parameter for the lambda for the group-sparsity norm

        min l(Theta) + kS*la* (sum_g ||Theta_g||_{2}),
        where the sum is over the groups
        """
        self.lbda = regparam * self.meta['reg_fac']

#        self.set_sparsity_weights()
# weighting scheme for sparse regularization

    def get_bounds(self):
        """return bounds for l-bfgs-b solver"""
        # some of the parameters are constrained to be zero
        # as a consequence of identifiability assumptions
        # difference to pw are factors ltot in B, Bd
        ltot = self.meta['ltot']

        bnds = []

        ltot_ident = []  # bounds for a row B_{\cdot, s, t} of flattened B
        ltot_diag = []
        for r in range(self.meta['n_cat']):
            # TODO(franknu): zero bounds for diagonal elements of matrices B_rk
            ltot_ident += [(0, 0)]
            ltot_ident += (self.meta['sizes'][r] - 1) * [(-inf, inf)]
            ltot_diag += [(0, 0)]
            ltot_diag += (self.meta['sizes'][r] - 1) * [(10E-6, inf)]

        if self.meta['n_cat'] > 0:
            # Q
            for r in range(self.meta['n_cat']):
                bnds += ltot * [(0, 0)]
                bnds += (self.meta['sizes'][r] - 1) * ltot_ident

            # u
            if self.opts['use_u']:
                bnds += ltot_ident
            else:
                bnds += ltot * [(0, 0)]

        if self.meta['n_cg'] > 0:
            bnds += self.meta['n_cg'] * ltot_ident  # R

            for _ in range(self.meta['n_cg'] - 1):
                # B0 with zero bounds on diagonal
                bnds += [(0, 0)]
                bnds += self.meta['n_cg'] * [(-inf, inf)]
            bnds += [(0, 0)]

            bnds += self.meta['n_cg'] * [(10E-6, inf)]  # beta0 diagonal

            for _ in range(self.meta['n_cg'] - 1):
                # B with zero bounds on diagonal
                bnds += ltot * [(0, 0)]
                bnds += self.meta['n_cg'] * ltot_ident
            bnds += ltot * [(0, 0)]

            bnds += self.meta['n_cg'] * ltot_diag  # B diagonal

            if self.opts['use_alpha']:
                bnds += self.meta['n_cg'] * [(-inf, inf)]  # alpha
            else:
                bnds += self.meta['n_cg'] * [(0, 0)]

        return bnds

    def get_starting_point(self, random=False, seed=10):
        """return a starting point containing zeros"""
        x_init = np.zeros(self.n_params)
        # TODO(franknu): zero precmat diagonal entries?
        # LBFGSB is able to handle infeasible starting point
        # (by first projecting to a point that satisfies the box constraints)
        if random:
            np.random.seed(seed)
            x_init = np.random.random(self.n_params)
            x_init = self.pack(
                self.preprocess(x_init))  # preprocess ensures symmetry
        else:
            self.currentsolution = x_init

        return x_init

##### ** graph and representation **############################

    def get_canonicalparams(self, x, verb=False):  # overwritten from base class
        """retrieves the CLZ-CG model parameters from flat parameter vector.
        output: CLZ params class"""
        mat_q, vec_u, mat_r, lambda0, beta0, lambdas, beta, alpha = \
            self.preprocess(x)
        # return copy (since unpack does)

        lambda0 += np.diag(beta0.reshape(-1))
        for r in range(self.meta['n_cat']):
            for k in range(self.meta['sizes'][r]):
                rk = self.meta['cat_glims'][r] + k
                lambdas[:, :, rk] += np.diag(beta[:, rk].reshape(-1))

        canparams = vec_u, mat_q, mat_r, alpha, lambda0, lambdas
        can_clz_class = ModelCLZ(
            canparams, {
                'n_cat': self.meta['n_cat'],
                'n_cg': self.meta['n_cg'],
                'sizes': self.meta['sizes']
            })

        if verb:
            print('Learned parameters:')
            print(can_clz_class)

        return can_clz_class

#################################################################################

    def preprocess(self, x) -> Iterable[np.ndarray]:
        """ unpack parameters from vector x and preprocess"""
        # pylint: disable=unbalanced-tuple-unpacking
        mat_q, vec_u, mat_r, mat_b0, beta0, mat_b, beta, alpha = self.unpack(x)

        glims = self.meta['cat_glims']
        # preprocess - zero out diagonals (required e.g. for grad approximation)
        for rk in range(self.meta['ltot']):
            #            print(rk, mat_b[:, :, rk])
            mat_b[:, :, rk] -= np.diag(np.diag(mat_b[:, :, rk]))
            # TODO(franknu): remove & LBFGSB bounds instead ?
            mat_b[:, :, rk] = np.triu(mat_b[:, :, rk])
            # use only upper triangle
            # -> gradient approximation works only for upper triangle
            mat_b[:, :, rk] = mat_b[:, :, rk] + mat_b[:, :, rk].T
        mat_b0 -= np.diag(np.diag(mat_b0))
        mat_b0 = np.triu(mat_b0)
        mat_b0 = mat_b0 + mat_b0.T

        for r in range(self.meta['n_cat']):  # set block-diagonal to zero
            mat_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((self.meta['sizes'][r], self.meta['sizes'][r]))
            # TODO(franknu): same here: LBFGSB bounds?
        mat_q = np.triu(mat_q)
        mat_q = mat_q + mat_q.T

        return mat_q, vec_u, mat_r, mat_b0, beta0, mat_b, beta, alpha

    def get_fval_and_grad(self,
                          x,
                          delta=None,
                          sparse=False,
                          smooth=True,
                          verb='-'):
        """calculate function value f and gradient g of CLZ model
        x    vector of parameters
        increases self.fcalls by 1, no other class variables are modified"""
        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']
        ltot = self.meta['ltot']
        n_data = self.meta['n_data']
        n_cg = self.meta['n_cg']

        self.fcalls += 1  # tracks number of function value and gradient evaluations

        mat_q, vec_u, mat_r, mat_b0, beta0, mat_b, beta, alpha = \
            self.preprocess(x)

        mat_b = mat_b.reshape((n_cg**2, ltot))
        # this is \tilde{B} (B tilde) from the doc # TODO(franknu): publish doc

        # intitialize f and gradients
        fval = 0
        grad = np.zeros(self.n_params)
        grad_q, grad_u, grad_r, grad_b0, grad_beta0, \
            grad_b, grad_beta, grad_alpha = self.unpack(grad)

        if smooth:
            vec_e = np.ones((n_data, 1))

            #### ** Gaussian node conditionals ** ####
            # n by n_dg
            vec_b = np.dot(self.cat_data, beta.T) + np.dot(vec_e, beta0.T)

            mat_m = np.dot(vec_e, alpha.T) + np.dot(self.cat_data, mat_r.T) \
                - np.dot(self.cont_data, mat_b0)# n by dg, part as in pairwise model
            for s in range(n_cg):  # new in CLZ
                mat_m[:, s] -= np.sum(np.multiply(
                    self.cont_data,
                    np.dot(self.cat_data, mat_b[s * n_cg:(s + 1) * n_cg, :].T)),
                                      axis=1)

            mat_d = np.divide(mat_m,
                              vec_b) - self.cont_data  # regression residual


            plh_cg = - 0.5*np.sum(np.log(vec_b)) \
                + 0.5*np.linalg.norm(np.multiply(mat_d, np.sqrt(vec_b)), 'fro')**2
            plh_cg /= n_data
            fval += plh_cg

            # gradients as in pw model
            grad_alpha = np.sum(mat_d, 0).T  # n_cg by 1
            grad_r = np.dot(mat_d.T, self.cat_data)  # n_cg by ltot
            grad_b0 = -np.dot(self.cont_data.T, mat_d)
            # grad_b0: zero out diagonal and add transpose later

            # new gradients in CLZ model
            for i in range(n_data):
                yi_di = np.dot(self.cont_data[i, :].reshape((n_cg, 1)),
                               mat_d[i, :].reshape(1, n_cg))
                # need to cast into matrix objects first
                for rk in np.where(self.cat_data[i, :] == 1)[0]:
                    grad_b[:, :, rk] -= yi_di
            # later add transpose and zero out diagonal

            # gradients of diagonals
            for s in range(n_cg):
                grad_beta[s, :] = np.dot(self.cat_data.T, \
                - 0.5*np.divide(np.ones(n_data), vec_b[:, s]) \
                + 0.5*np.multiply(mat_d[:, s], mat_d[:, s]) \
                - np.divide(np.multiply(mat_d[:, s], mat_m[:, s]), vec_b[:, s]))
                # grad_beta[s, :] is ltot x 1 slice
                #                print(grad_beta.shape)
                #                grad_beta0[s] = np.sum(grad_beta[s, 0:sizes[0]])
                # -> sum of the gradient of beta_{s,s, r:k} over k
                grad_beta0[s] = np.dot(vec_e.T, \
                - 0.5*np.divide(np.ones(n_data), vec_b[:, s]) \
                + 0.5*np.multiply(mat_d[:, s], mat_d[:, s]) \
                - np.divide(np.multiply(mat_d[:, s], mat_m[:, s]), vec_b[:, s]))

            #### ** discrete node conditionals ** ####
            mat_w = np.dot(self.cont_data, mat_r) + \
                np.dot(self.cat_data, mat_q) + \
                np.dot(np.ones((n_data, 1)), vec_u.T) # n by Ltot, as in pw model
            #            asd = np.dot(self.cont_data_prod, mat_b)
            mat_w += -0.5 * np.dot(self.cont_data_prod, mat_b) - \
                0.5 * np.dot(self.cont_data_square, beta) # new in CLZ

            mat_a = np.empty((n_data, ltot))
            #            print(mat_w[6, :])
            plh_cat = 0
            for r in range(self.meta['n_cat']):  # as in pw model, see doc
                #                mat_wr = mat_w[:, glims[r]:glims[r+1]]
                tmpexpmat_wr = np.exp(mat_w[:, glims[r]:glims[r + 1]])
                mat_ar = np.divide(
                    tmpexpmat_wr,
                    np.dot(tmpexpmat_wr, np.ones((sizes[r], sizes[r]))))
                mat_a[:, glims[r]:glims[r + 1]] = mat_ar
                plh_cat_r = -np.sum(
                    np.log(
                        np.sum(
                            np.multiply(
                                mat_ar,
                                self.cat_data[:, glims[r]:glims[r + 1]]), 1)))
                plh_cat += plh_cat_r
            plh_cat /= n_data
            fval += plh_cat

            #            print('plh_cat:', plh_cat, 'plh_cg', plh_cg)

            mat_a = mat_a - self.cat_data
            # gradients as in pw model
            grad_u = np.sum(mat_a, 0)  # Ltot by 1
            grad_r += np.dot(self.cont_data.T, mat_a)  # dg by Ltot
            grad_q = np.dot(self.cat_data.T, mat_a)
            # grad_q: this is Phihat from the doc,
            # grad_q: zero out diagonal and add transpose later

            # new gradients in CLZ model
            tmp_grad = 0.5 * np.dot(self.cont_data_prod.T, mat_a).reshape(
                (n_cg, n_cg, ltot))
            grad_b -= tmp_grad  # n_cg^2 by ltot
            # grad_b: dont add transpose - appears only in 1 discrete node conditional
            for s in range(n_cg):
                grad_beta[s, :] -= tmp_grad[s, s, :]

            # scale gradients as likelihood
            for grad in (grad_q, grad_u, grad_r, grad_b, grad_beta, grad_b0,
                         grad_beta0, grad_alpha):
                grad *= 1 / n_data

        ## *** l1/l2 regularization ***
        if sparse:  # iterate over groups
            dis_dis = 0
            dis_cts = 0
            cts_cts = 0
            # discrete - discrete, as in pairwise model
            n_cg = self.meta['n_cg']
            n_cat = self.meta['n_cat']
            for r in range(n_cat):  # Phis
                for j in range(r):
                    wrj = self.lbda
                    wrj *= self.weights[r, j]
                    tmp_fval, tmp_grad = _huberapprox(
                        mat_q[glims[r]:glims[r + 1], glims[j]:glims[j + 1]],
                        delta)
                    dis_dis += wrj * tmp_fval
                    grad_q[glims[r]:glims[r + 1],
                           glims[j]:glims[j + 1]] += wrj * tmp_grad

            # continuous - discrete
            for r in range(n_cat):  # Rhos
                tmp_group = np.empty(sizes[r] * (1 + n_cg))
                for s in range(n_cg):
                    tmp_group[:sizes[r]] = mat_r[s, glims[r]:glims[r + 1]]
                    tmp_group[sizes[r]:] = mat_b[
                        s * n_cg:(s + 1) * n_cg, glims[r]:glims[r + 1]].flatten(
                        )  # params la_{st}^{r:k} for t\in [d_g], k\in[L_r]
                    wrs = self.lbda
                    wrs *= self.weights[n_cat+s, r]
                    tmp_fval, tmp_grad = _huberapprox(tmp_group, delta)
                    dis_cts += wrs * tmp_fval
                    grad_r[s,
                           glims[r]:glims[r + 1]] += wrs * tmp_grad[:sizes[r]]
                    grad_b[s, :, glims[r]:glims[r+1]] += \
                        wrs * tmp_grad[sizes[r]:].reshape((n_cg, self.meta['sizes'][r]))

            # continuous - continuous
            grad_b = grad_b.reshape((n_cg**2, ltot))

            tmp_group = np.empty(ltot + 1)

            for t in range(n_cg):  # upper triangle s<t
                for s in range(t):
                    tmp_group[:ltot] = mat_b[s * n_cg + t, :]
                    tmp_group[ltot] = mat_b0[s, t]
                    wst = self.lbda
                    wst *= self.weights[n_cat + s, n_cat + t]
                    tmp_fval, tmp_grad = _huberapprox(tmp_group, delta)
                    cts_cts += wst * tmp_fval
                    grad_b[s * n_cg + t, :] += wst * tmp_grad[:ltot]
                    grad_b0[s, t] += wst * tmp_grad[ltot]


#            print('dis_dis', dis_dis, 'dis_cts', dis_cts, 'cts_cts', cts_cts)
# note different # of edges

#            if self.fcalls > 40:
#                print('f=%f, reg=%f, logsum=%f'%(f, rsum, sum([np.log(self.sigmas[s]) for s in range(n_cg)])))
            fval += dis_dis + dis_cts + cts_cts

        # zero out diagonals and add transposes
        grad_b = grad_b.reshape((n_cg, n_cg, ltot))
        for rk in range(ltot):
            grad_b[:, :, rk] -= np.diag(np.diag(grad_b[:, :, rk]))
            grad_b[:, :, rk] = np.triu(grad_b[:, :, rk])+ \
                np.tril(grad_b[:, :, rk]).T
        grad_b0 -= np.diag(np.diag(grad_b0))
        grad_b0 = np.triu(grad_b0) + np.tril(grad_b0).T

        for r in range(self.meta['n_cat']):  # set block-diagonal to zero
            grad_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))
        grad_q = np.triu(grad_q) + np.tril(grad_q).T
        # np.triu(tmpPhihat) + np.tril(tmpPhihat).T
        # to compare with gradient approximation

        grad = self.pack((grad_q, grad_u, grad_r, grad_b0, grad_beta0, grad_b,
                          grad_beta, grad_alpha))
        #        print('f', f)
        return fval, grad.reshape(-1)

    def crossvalidate(self, x):
        """crossvalidation of model with parameters in x using current data"""

        glims = self.meta['cat_glims']
        n_data = self.meta['n_data']

        mat_q, vec_u, mat_r, mat_b0, beta0, \
            mat_b, beta, alpha = self.preprocess(x)
        mat_b = mat_b.reshape((self.meta['n_cg']**2, self.meta['ltot']))

        dis_errors = np.zeros(self.meta['n_cat'])
        cts_errors = np.zeros(self.meta['n_cg'])

        vec_e = np.ones((n_data, 1))
        # TODO(franknu): the following is the same code as in get_fval_and_g.
        # Smart way to remove redundancy? - extra func
        mat_w = np.dot(self.cont_data, mat_r) + np.dot(self.cat_data, mat_q) \
            + np.dot(np.ones((n_data, 1)), vec_u.T) # n by Ltot, as in pw model
        mat_w += -0.5 * np.dot(self.cont_data_prod, mat_b) - \
            0.5 * np.dot(self.cont_data_square, beta) # new in CLZ

        for r in range(self.meta['n_cat']):  # as in pw model, see doc
            l_r = self.meta['sizes'][r]
            tmpexpmat_wr = np.exp(mat_w[:, glims[r]:glims[r + 1]])
            mat_ar = np.divide(tmpexpmat_wr,
                               np.dot(tmpexpmat_wr, np.ones((l_r, l_r))))
            # mat_ar: matrix of conditional probabilities
            dis_errors[r] = n_data - np.sum(
                np.multiply(self.cat_data[:, glims[r]:glims[r + 1]], mat_ar))
        vec_b = np.dot(self.cat_data, beta.T) + np.dot(
            vec_e, beta0.T)  # n_data by n_cg

        mat_m = np.dot(
            vec_e, alpha.T) + np.dot(self.cat_data, mat_r.T) - np.dot(
                self.cont_data, mat_b0)  # n by dg, part as in pairwise model
        for s in range(self.meta['n_cg']):  # new in CLZ
            mat_m[:, s] -= np.sum(np.multiply(
                self.cont_data,
                np.dot(
                    self.cat_data, mat_b[s * self.meta['n_cg']:(s + 1) *
                                         self.meta['n_cg'], :].T)),
                                  axis=1)

        mat_d = np.divide(mat_m, vec_b) - self.cont_data  # regression residual

        cts_errors = np.sum(np.multiply(mat_d, mat_d), axis=0)

        dis_errors /= n_data
        cts_errors /= n_data
        lval_testdata, _ = self.get_fval_and_grad(x, smooth=True, sparse=False)

        return dis_errors, cts_errors, lval_testdata
