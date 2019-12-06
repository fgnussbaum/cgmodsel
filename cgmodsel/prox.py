#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np
#import scipy
#import abc
#import time

from scipy.optimize import approx_fprime
from scipy.linalg import eigh
from scipy import optimize

from cgmodsel.utils import _logsumexp_condprobs_red
#from cgmodsel.utils import logsumexp
from cgmodsel.base_solver import BaseGradSolver

# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=W0511 # todos
# pylint: disable=R0914 # too many locals

###############################################################################
# prox for PLH objective
###############################################################################
class LikelihoodProx(BaseGradSolver):
    """
    solve pseudo-log-likelihood proximal operator
    """
    def __init__(self, cat_data, cont_data, meta):
        """"must provide with dictionary meta"""
        super().__init__()  # Python 3 syntax
        self.cat_data = cat_data
        self.cont_data = cont_data
        self.meta = meta
        self._fold = np.inf

        # overridden attributes
        ltot = meta['ltot']
        n_cg = meta['n_cg']

        self.shapes = [
            ('Q', (ltot, ltot)),
            ('u', (ltot, 1)),
            ('R', (n_cg, ltot)),
            ('F2tiL', (n_cg, n_cg)),  # construct Lambda = A * A.T
            ('alpha', (n_cg, 1))
        ]

        self.n_params = sum([np.prod(shape[1]) for shape in self.shapes])


    def clean_theta(self, theta):
        """
        make pairwise parameter matrix feasible for likelihood prox solver
        -> modifies Theta
        """
        # copies upper triangle of Theta to lower triangle to symmetrize
        # Furthermore, all elements on the block-diagonal of the discrete
        # are set to zero, except diagonal elements
        # since these correspond to univariate discrete sufficient statistics
        optvars = self._theta_to_x(theta, np.zeros(self.meta['n_cg']))
        return self._x_to_thetaalpha(optvars)[0]

###############################################################################
# Solver for Pseudo-likelihood Prox operator
###############################################################################

    def callback_plh(self, optvars, handle_fg):
        """callback to check for potential bugs"""
        fnew = handle_fg(optvars)[0]
        if not fnew <= self._fold:
            string = 'Potential scipy bug, fvalue increased in last iteration'
            print('Warning(CG_base_ADMM.callback_plh): %s' % (string))
        self._fold = fnew

    def solve(self, mat_z, prox_param, old_thetaalpha):
        """ solve proximal mapping of negative pseudo loglikelihood
        min_{Theta, alpha} l_p(Theta, alpha) + 1 / (2mu) * ||Theta-Z||_F^2

        known issue with ADMM:
        not doing warm starts may cause problems if solution is to inexact
        generally ADMM convergence requires very exact solutions
        -> use ftol to control tolerancy, or refine to control #restarts
        """
        # split Z (since in determining the prox objective
        # the split components are used)
        ltot = self.meta['ltot']
        n_cg = self.meta['n_cg']

        zmat_q = mat_z[:ltot, :ltot].copy()
        zmat_r = mat_z[ltot:, :ltot]
        zmat_b = mat_z[ltot:, ltot:].copy()

        zbeta = np.diag(zmat_b).copy().reshape((n_cg, 1))
        zmat_b -= np.diag(np.diag(zmat_b))
        zvec_u = np.diag(zmat_q).copy().reshape((ltot, 1))
        zmat_q -= np.diag(np.diag(zmat_q))
        components_z = zmat_q, zvec_u, zmat_r, zmat_b, zbeta

        handle_fg = lambda optvars: \
            self.get_fval_and_grad(optvars, components_z, prox_param)

        ## solve proximal mapping

        #        x0 = self.get_rand_startingpoint()
        x_init = self._theta_to_x(*old_thetaalpha)
        # starting point as vector, save for input parameters
        f_init = handle_fg(x_init)[0]
        self._fold = f_init

        ## bounds that respect identifiability constraints
        bnds = ltot**2 * [(-np.inf, np.inf)]  # Q, only upper triangle is used

        bnds += ltot * [(-np.inf, np.inf)]  # u
        # TODO(franknu) note: if use_u = 0 this is enforced in main ADMM updates

        bnds += (n_cg * ltot + n_cg**2) * [(-np.inf, np.inf)]  # R, fac_lambda
        if self.opts['use_alpha']:
            bnds += n_cg * [(-np.inf, np.inf)]
        else:
            bnds += n_cg * [(0, 0)]
        # TODO(franknu): use zerobounds for block diagonal of Q?

        ## further solver properties
        callback = lambda optvars: self.callback_plh(optvars, handle_fg)

        correctionpairs = min(len(bnds) - 1, 10)

        res = optimize.minimize(handle_fg,
                                x_init,
                                method='L-BFGS-B',
                                jac=True,
                                bounds=bnds,
                                options={
                                    'maxcor': correctionpairs,
                                    'maxiter': self.opts['maxiter'],
                                    'ftol': self.opts['tol']
                                },
                                callback=callback)

        if not res.message.startswith(b'CONV'):  # solver did not converge
            print('PLH_prox scipy-solver message:', res.message)

        _, _, _, fac_lambda, _ = self.unpack(res.x)
        if np.linalg.norm(fac_lambda) < 1e-5 and n_cg > 0:
            # TODO(franknu): certificate for optimality?
            print('Warning(solve): Lambda = F F^T with F ~ zero')

        theta, alpha = self._x_to_thetaalpha(res.x)

        return theta, alpha

    def preprocess(self, optvars):
        """ unpack parameters from vector x and preprocess
        this modifies x (x not save for reuse)"""

        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']

        mat_q, vec_u, mat_r, fac_lambda, alpha = self.unpack(optvars) # pylint: disable=unbalanced-tuple-unpacking

        for r in range(self.meta['n_cat']): # set block-diagonal to zero
            mat_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))
        mat_q = np.triu(mat_q)
        mat_q = mat_q + mat_q.T

        return mat_q, vec_u, mat_r, fac_lambda, alpha

    def get_fval_and_grad(self, optvars, components_z, prox_param, eps=1e-15):
        """calculate function value f and gradient g of
        plh(Theta, alpha) + 1 / (2prox_param) ||Theta - Z||_F^2,
        where Theta, alpha are contained in the vector x of parameters
        """

        ltot = self.meta['ltot']
        n_cg = self.meta['n_cg']
        n_data = self.meta['n_data']
        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']

        ## unpack parameters from vector optvars
        mat_q, vec_u, mat_r, fac_lambda, alpha = \
            self.preprocess(optvars)

        mat_b, beta = self._faclambda_to_bbeta(fac_lambda)
        beta += eps * np.ones(beta.shape)  # increase numerical instability
        # this avoids beta that contains zeros
        # precision matrix = FLa*FLa.T + eps * eye(n_cg)

        # intitialize gradients
        grad = np.zeros(self.n_params)
        grad_q, grad_u, grad_r, grad_faclambda, grad_alpha = self.unpack(grad)
        grad_tila = np.zeros((n_cg, n_cg))
        grad_beta = np.zeros((n_cg, 1))
        vec_ones = np.ones((n_data, 1))

        ## ** discrete node conditionals **
        lh_cat = 0

        mat_w = np.dot(self.cont_data, mat_r) + np.dot(self.cat_data, mat_q) \
            + np.dot(vec_ones, vec_u.T) # n_data by ltot
        cond_probs = np.empty((n_data, ltot))  # conditional probs given data
        for r in range(self.meta['n_cat']):
            mat_wr = mat_w[:, glims[r]:glims[r + 1]]  # view of W
            mat_dr = self.cat_data[:, glims[r]:glims[r + 1]] # view
            tmp_logsumexp, tmp_conditionalprobs = \
                _logsumexp_condprobs_red(mat_wr)
            # uses numerically stable exp
            cond_probs[:, glims[r]:glims[r + 1]] = tmp_conditionalprobs
            lh_catr = -np.sum(np.sum(np.multiply(mat_wr, mat_dr), axis=1) \
                - tmp_logsumexp)
            lh_cat += lh_catr

#        print('lD', lh_cat/n_data)

# gradients
        cond_probs = cond_probs - self.cat_data

        grad_u = np.sum(cond_probs, 0)  # Ltot by 1
        grad_r = np.dot(self.cont_data.T, cond_probs)
        grad_q = np.dot(self.cat_data.T, cond_probs)
        # this is Phihat from the doc, later add transpose and zero out diagonal

        ## ** Gaussian node conditionals **
        mat_m = np.dot(vec_ones, alpha.T) + np.dot(self.cat_data, mat_r.T) \
            - np.dot(self.cont_data, mat_b) # n by dg, concatenation of mu_s
        mat_delta = mat_m.copy()
        for s in range(n_cg):
            mat_delta[:, s] /= beta[s]
        mat_delta -= self.cont_data  # residual

        tmp = np.dot(mat_delta, np.diag(np.sqrt(beta.flatten())))
        lh_cont = - 0.5 * n_data * np.sum(np.log(beta)) \
            + 0.5 * np.linalg.norm(tmp, 'fro') ** 2
        #        print('lG', lh_cont/n_data)

        # gradients
        # grad_tila: n_cg by n_cg, later add transpose and zero out diagonal
        grad_tila = -np.dot(self.cont_data.T, mat_delta)
        grad_tila -= np.diag(np.diag(grad_tila))
        grad_tila = 0.5 * (grad_tila + grad_tila.T)

        for s in range(n_cg):
            grad_beta[s] = -.5 * n_data / beta[s] + \
                .5 * np.linalg.norm(mat_delta[:, s], 2) ** 2 \
            - 1 / beta[s] * np.dot(mat_delta[:, s].T, mat_m[:, s])

        grad_alpha = np.sum(mat_delta, 0).T  # dg by 1
        grad_r += np.dot(mat_delta.T, self.cat_data)

        # scale gradients as likelihood
        grad_q /= n_data
        grad_u /= n_data
        grad_r /= n_data
        grad_tila /= n_data
        grad_beta /= n_data
        grad_alpha /= n_data

        ## add quad term  1/2mu * ||([Q+2diag(u)] & R^T \\ R &-Lambda)-Z||_F^2
        zmat_q, zvec_u, zmat_r, zmat_b, zbeta = components_z

        fsquare = 0
        fsquare += np.sum(np.square(mat_q - zmat_q))
        fsquare += np.sum(np.square(2 * vec_u - zvec_u))
        # note that u is only half of discrete diagonal
        fsquare += 2 * np.sum(np.square(mat_r - zmat_r))
        fsquare += np.sum(np.square(-mat_b - zmat_b))
        # remember neg sign of Lambda in Theta
        fsquare += np.sum(np.square(-beta - zbeta))

        fsquare /= 2 * prox_param

        #        print('fsquare', fsquare)

        # gradients quadratic term
        grad_q += (mat_q - zmat_q) / prox_param
        grad_u = grad_u.reshape(
            (ltot, 1))  # since with dc=0 gradu has shape (0,)
        grad_u += 2 * (2 * vec_u - zvec_u) / prox_param
        grad_r += 2 * (mat_r - zmat_r) / prox_param
        grad_tila += (mat_b + zmat_b) / prox_param  # has zero diagonal
        grad_beta += (beta + zbeta) / prox_param

        ## gradients to only upper triangle
        for r in range(self.meta['n_cat']):  # set block-diagonal to zero
            grad_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))

        grad_q = np.triu(grad_q) + np.tril(grad_q).T

        grad_tila += np.diag(grad_beta.flatten())  # add gradient of diagonal

        grad_faclambda = 2 * np.dot(grad_tila, fac_lambda)
        # note that fac_lambda initialized at 0 always leads to 0 gradient

        fval = 1 / n_data * (lh_cat + lh_cont) + fsquare
        grad = self.pack((grad_q, grad_u, grad_r, grad_faclambda, grad_alpha))

        return fval, grad.reshape(-1)

    def callback(self, optvars, component_z, prox_param, approxgrad=1):
        """a callback function that serves primarily for debugging"""
        fval, grad = self.get_fval_and_grad(optvars, component_z, prox_param)

        print('f=', fval)
        if approxgrad:  # gradient check
            func_handle_f = lambda optvars: \
                self.get_fval_and_grad(optvars, component_z, prox_param)[0]
            eps = np.sqrt(np.finfo(float).eps)  # ~1.49E-08 at my machine
            gprox = approx_fprime(optvars, func_handle_f, eps)

            diff = grad - gprox
            normdiff = np.linalg.norm(diff)
            if normdiff > 1e-4:
                print('g_exct', grad)
                print('g_prox', gprox)
#            print('g-gprox',self.unpack(diff))
#            print('quot',g/proxg)

            print('graddev=', np.linalg.norm(diff))

    def _faclambda_to_bbeta(self, fac_lambda):
        """ construct precision matrix, then extract diagonal """
        mat_b = np.dot(fac_lambda, fac_lambda.T)  # PSD precision matrix
        beta = np.diag(mat_b).copy().reshape((self.meta['n_cg'], 1))  # diagonal
        mat_b -= np.diag(np.diag(mat_b))  # off-diagonal elements
        return mat_b, beta

    def _theta_to_tuple(self, theta):
        """ split Theta into its components
        (save: returns copies from data in Theta, Theta is not modified)"""
        ltot = self.meta['ltot']
        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']

        mat_q = theta[:ltot, :ltot].copy()
        mat_r = theta[ltot:, :ltot].copy()
        lbda = -theta[ltot:, ltot:]
        #        print(Lambda)
        #        FLa = np.linalg.cholesky(Lambda) # fails if not PD

        if self.meta['n_cg'] > 0:
            eig, mat_u = eigh(lbda)
            #            print('las', las)
            eig[eig < 1e-16] = 0  # make more robust
            fac_lambda = np.dot(mat_u, np.diag(np.sqrt(eig)))


#           print('chol-error', np.linalg.norm(np.dot(FLa, FLa.T) - Lambda))
        else:
            fac_lambda = np.empty((0, 0))

        vec_u = 0.5 * np.diag(mat_q).copy().reshape((ltot, 1))

        for r in range(self.meta['n_cat']):  # set block diagonal to zero
            mat_q[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))
        mat_q = np.triu(mat_q)  # use only upper triangle
        mat_q = mat_q + mat_q.T

        return mat_q, vec_u, mat_r, fac_lambda

    def _theta_to_x(self, theta, alpha):
        """takes Theta, cleans it (symmetrize etc.) and pack into x
        (save: Theta is not modified)"""
        return self.pack(list(self._theta_to_tuple(theta)) + [alpha])

    def _x_to_thetaalpha(self, optvars):
        """ convert vectorized x to parameter matrix Theta
        (save: optvars is not modified) """
        mat_q, vec_u, mat_r, fac_lambda, alpha = self.unpack(optvars)
        ltot = self.meta['ltot']
        glims = self.meta['cat_glims']
        sizes = self.meta['sizes']
        dim = self.meta['dim']

        # set parameters in upper triangle
        theta = np.empty((dim, dim))
        theta[:ltot, :ltot] = mat_q
        for r in range(self.meta['n_cat']):  # set block-diagonal to zero
            theta[glims[r]:glims[r+1], glims[r]:glims[r+1]] = \
                np.zeros((sizes[r], sizes[r]))
        theta[:ltot, ltot:] = mat_r.T

        ## symmetric matrix from upper triangle
        theta = np.triu(theta)
        theta = theta + theta.T

        ## Lambda
        mat_lbda = np.dot(fac_lambda, fac_lambda.T)
        theta[ltot:, ltot:] = -mat_lbda

        ## add diagonal
        theta[:ltot, :ltot] += 2 * np.diag(vec_u.flatten())

        return theta, alpha

    def get_rand_startingpoint(self):
        """ not needed if using warm starts """
        n_cg = self.meta['n_cg']
        x_init = np.random.random(self.n_params)
        x_init[self.n_params - n_cg:] = np.ones(n_cg)
        return x_init

    def plh(self, theta, alpha, cval=False):
        """ return negative pseudo-log-likelihood function value
        cval .. if True, calculate (node-wise) cross validation error"""
        n_cg = self.meta['n_cg']
        n_cat = self.meta['n_cat']
        n_data = self.meta['n_data']
        glims = self.meta['cat_glims']

        if cval:
            dis_errors = np.zeros(n_cat)
            cts_errors = np.zeros(n_cg)

        mat_q, vec_u, mat_r, fac_lambda = self._theta_to_tuple(theta)  # save
        mat_b, beta = self._faclambda_to_bbeta(fac_lambda)

        fval = 0

        ## ** discrete node conditionals **
        mat_w = np.dot(self.cont_data, mat_r) + np.dot(self.cat_data, mat_q) + \
            np.dot(np.ones((n_data, 1)), vec_u.T) # n by Ltot

        for r in range(n_cat):
            mat_wr = mat_w[:, glims[r]:glims[r + 1]]  # view of W
            mat_dr = self.cat_data[:,
                                   glims[r]:glims[r +
                                                  1]]  # view of self.cat_data
            tmp_logsumexp, tmp_conditionalprobs = \
                _logsumexp_condprobs_red(mat_wr) # numerically more stable
            if cval:
                # sum of probabilities of missclassification
                dis_errors[r] = n_data - \
                    np.sum(np.multiply(tmp_conditionalprobs, mat_dr))
                # sum over both axes

            lh_catr = - np.sum(np.sum(np.multiply(mat_wr, mat_dr), axis=1) \
                          - tmp_logsumexp)
            fval += 1 / n_data * lh_catr

        mat_m = np.dot(self.cat_data, mat_r.T) - \
            np.dot(self.cont_data, mat_b) # n by dg, concatenation of mu_s
        if n_cg > 0:
            mat_m += np.outer(np.ones(n_data), alpha)

        if cval:
            for s in range(n_cg):
                cts_errors[s] = np.linalg.norm(self.cont_data[:, s] \
                          - mat_m[:, s]/beta[s], 2) ** 2

        mat_delta = mat_m.copy()
        for s in range(n_cg):
            mat_delta[:, s] /= beta[s]
        mat_delta -= self.cont_data  # residual
        tmp = np.dot(mat_delta, np.diag(np.sqrt(beta.flatten())))
        lh_cont = - 0.5 * n_data * np.sum(np.log(beta)) \
            + 0.5 * np.linalg.norm(tmp, 'fro') ** 2
        fval += 1 / n_data * lh_cont

        if cval:
            return dis_errors, cts_errors, fval

        return fval

    def crossvalidate(self, theta, alpha):
        """perform cross validation (drop test data) """
        n_cg = self.meta['n_cg']
        n_cat = self.meta['n_cat']
        n_data = self.meta['n_data']

        dis_errors, cts_errors, _ = self.plh(theta, alpha, cval=True)
        avg_dis_error = 1 / n_data * np.sum(dis_errors)
        avg_cts_error = np.sum([np.sqrt(es / n_data) for es in cts_errors
                               ])  # mean RMSEs

        cvalerror = avg_dis_error + avg_cts_error

        if n_cg > 0:
            avg_cts_error /= n_cg
        if n_cat > 0:
            avg_dis_error /= n_cat

        return cvalerror
