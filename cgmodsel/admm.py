#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (translation to Python)

This file implements Proximal Gradient ADM algorithms similar to the
one described in
"Alternating Direction Methods for Latent Variable Gaussian Graphical
Model Selection", 2013, by Ma, Xue and Zou,
for solving Latent Variable Gaussian Graphical Model Selection
min l(Theta) + alpha * ||S||_{2,1} + beta * tr(L)
s.t. Theta = S + L, continuous-continuous interactions parameters in Theta PSD,
     and L>=0

"""
import numpy as np
import scipy  # use scipy.linalg.eigh (for real symmetric matrix), does not check for symmetry

#from cgmodsel.models.model_pwsl import ModelPairwiseSL

from cgmodsel.base_admm import BaseAdmm
from cgmodsel.base_solver import BaseSolverSL, BaseSolverPW
from cgmodsel.prox import LikelihoodProx

# pylint: disable=R0914


class AdmmGaussianSL(BaseSolverSL, BaseAdmm):
    """
    solve the problem
       min l(S-L) + lambda * ||S||_1 + rho * tr(L)
       s.t. S-L>0, L>=0
    where l is the Gaussian likelihood with pairwise parameters Theta=S-L,
    here Theta is the precision matrix (inverse of the covariance matrix)
    of the z(unnormalized) zero-mean Gaussian model
        p(y) ~ exp(-1/2 y^T Theta y)

    Solver is an ADMM algorithm with proximal gradient step

    """

    def __init__(self, *args, **kwargs):
        """"must provide with dictionary meta"""

        super().__init__(*args, **kwargs)  # Python 3 syntax
        #        BaseAdmm.__init__(self)
        #        BaseSolverSL.__init__(self, meta=meta)

        self.sigma0 = None
        self.cat_format_required = 'dummy_red'
        self.proxstep_param = 0.6

    def _postsetup_data(self):
        """called after drop_data"""
        # precompute empirical second-moment matrix
        self.sigma0 = np.dot(self.cont_data.T, self.cont_data)
        self.sigma0 /= self.meta['n_data']

    def _initialize_admm(self):
        """initialize ADMM variables"""
        dim = self.sigma0.shape[1]
        mat_theta = np.eye(dim)
        mat_s = mat_theta
        mat_l = np.zeros((dim, dim))
        mat_z = np.zeros((dim, dim))

        return mat_theta, mat_s, mat_l, mat_z

    def _do_iter_admm(self, current_vars: tuple):
        """perform computations of one ADMM iteration"""
        mat_theta, mat_s, mat_l, mat_z = current_vars

        tmp = self.admm_param * self.sigma0 - self.admm_param * mat_z - mat_s + mat_l

        eig, mat_u = scipy.linalg.eigh(tmp)

        eig_theta = (-eig + np.sqrt(np.square(eig) + 4 * self.admm_param)) / 2
        # always positive (though might round to 0 due to bad precision)

        mat_theta = np.dot(np.dot(mat_u, np.diag(eig_theta)), mat_u.T)
        mat_theta = (mat_theta + mat_theta.T) / 2

        mat_s_old = mat_s
        mat_l_old = mat_l

        ## update S and L
        grad_partial = mat_theta - mat_s + mat_l - self.admm_param * mat_z
        grad_partial_s = mat_s + self.proxstep_param * grad_partial
        grad_partial_l = mat_l - self.proxstep_param * grad_partial
        mat_s, l1norm = self.shrink(
            grad_partial_s, self.proxstep_param * self.admm_param * self.lbda)
        mat_s = (mat_s + mat_s.T) / 2

        eig, mat_u = scipy.linalg.eigh(grad_partial_l)
        eig_l = eig - self.proxstep_param * self.admm_param * self.rho
        eig_l[eig_l < 1e-25] = 0
        mat_l = np.dot(np.dot(mat_u, np.diag(eig_l)), mat_u.T)
        mat_l = (mat_l + mat_l.T) / 2

        ## update dual variables Z
        resid = mat_theta - mat_s + mat_l
        mat_z = mat_z - resid / self.admm_param
        mat_z = (mat_z + mat_z.T) / 2

        fronorm_s = np.linalg.norm(mat_s, 'fro')
        fronorm_l = np.linalg.norm(mat_l, 'fro')
        fronorm_theta = np.linalg.norm(mat_theta, 'fro')

        rnorm = np.linalg.norm(resid, 'fro')
        snorm = np.sqrt(
            np.linalg.norm(mat_s - mat_s_old, 'fro')**2 +
            np.linalg.norm(mat_l - mat_l_old, 'fro')**2) / self.admm_param

        dim = mat_theta.shape[0]
        eps_pri = np.sqrt(
            3 * dim**2) * self.opts['abstol'] + self.opts['reltol'] * max(
                (fronorm_theta, np.sqrt(fronorm_s**2 + fronorm_l**2)))
        eps_dual = np.sqrt(3 * dim**2) * self.opts['abstol'] + self.opts[
            'reltol'] * np.linalg.norm(mat_z, 'fro')

        ## store stuff
        residuals = rnorm, snorm, eps_pri, eps_dual
        new_vars = mat_theta, mat_s, mat_l, mat_z

        stats = {}
        stats['admm_obj'] = np.sum(np.sum(np.multiply(mat_theta, self.sigma0)))
        stats['admm_obj'] -= np.sum(np.log(eig_theta))
        stats['admm_obj'] += self.lbda * l1norm + self.rho * np.sum(eig_l)

        stats['theta'] = mat_theta
        alpha = np.zeros((self.meta['n_cg'], 1))
        stats['solution'] = mat_s, mat_l, alpha
        # set problem_vars (mat_s with negative sign to be consistent with
        # general CG pairwise parameter representation)
        self.problem_vars = -mat_s, mat_l, alpha
        stats['eig_theta'] = eig_theta
        stats['eig_l'] = eig_l
        stats['resid'] = resid

        return new_vars, residuals, stats


#    def get_objective(self, problem_vars, lhonly=False):
#        """return objective value of original problem"""
#        mat_s, mat_l, alpha = problem_vars
#        mat_theta = mat_s - mat_l
#        obj = np.sum(np.sum(np.multiply(mat_theta, self.sigma0)))
#        obj -= np.linalg.slogdet(mat_theta)[1]
#        if not lhonly:
#            obj += self.lbda * self.sparse_norm(mat_s)
#            obj += self.rho * np.trace(mat_l)
#
#        return obj
#
#    def crossvalidate(self, problem_vars):
#        """calculate a cross validation score (use after dropping test data)"""
#        return self.get_objective(problem_vars, lhonly=True)

###############################################################################
# general case, CG distributions with pseudo likelihood
###############################################################################


class AdmmCGaussianSL(BaseSolverSL, BaseAdmm):
    """
    solve the problem
       min l(S+L) + lambda * ||S||_{2,1} + rho * tr(L)
       s.t. Lambda[S+L]>0, L>=0
    where l is the pseudo likelihood with pairwise parameters Theta=S+L,
    here Lambda[S+L] extracts the quantitative-quantitative interactions
    from the pairwise parameter matrix Theta=(Q & R^T \\ R & -Lbda),
    that is, Lambda[Theta] = Lbda

    The estimated probability model has (unnormalized) density
        p(y) ~ exp(1/2 (D_x, y)^T Theta (D_x, y) + alpha^T y + u^T D_x)
    where D_x is the indicator representation of the discrete variables x
    (reduced by the indicator for the 0-th level for identifiability reasons),
    and y are the quantitative variables.
    Note that alpha and u are optional univariate parameters that can be included
    in the optimization problem above.

    The solver is an ADMM algorithm with a proximal gradient step.
    """

    def __init__(self, *args, **kwargs):
        """"must provide with dictionary meta"""

        super().__init__(*args, **kwargs)  # Python 3 syntax

        self.name = 'SL_PADMM'

        self.prox = None
        self.cat_format_required = 'dummy_red'

        self.proxstep_param = 0.6

    def _initialize_admm(self):
        """initialize ADMM variables"""

        ## ensure coherent options for prox solver
        self.prox.opts['use_u'] = self.opts['use_u']
        self.prox.opts['use_alpha'] = self.opts['use_alpha']

        ## initialize ADMM variables
        dim = self.meta['dim']
        ltot = self.meta['ltot']
        mat_theta = np.eye(dim)
        mat_theta[ltot:, ltot:] *= -1
        # since lower right block needs to be negative definite
        # make Theta feasible/cleaned for plh
        mat_theta = self.prox.clean_theta(mat_theta)

        alpha = np.zeros((self.meta['n_cg'], 1))
        mat_s = mat_theta.copy()
        if not self.opts['use_u']:  # no univariate parameters
            mat_s[:ltot, :ltot] -= np.diag(np.diag(mat_s[:ltot, :ltot]))
        mat_l = np.zeros((dim, dim))
        mat_z = np.zeros((dim, dim))

        return mat_theta, mat_s, mat_l, mat_z, alpha

    def _do_iter_admm(self, current_vars: tuple):
        """perform computations of one ADMM iteration"""

        mat_theta, mat_s, mat_l, mat_z, alpha = current_vars

        tmp = mat_s + mat_l + self.admm_param * mat_z

        mat_theta, alpha = self.prox.solve(tmp, self.admm_param,
                                           (mat_theta, alpha))

        mat_theta = (mat_theta + mat_theta.T) / 2

        mat_s_old = mat_s
        mat_l_old = mat_l

        ## update S and L
        gradient_partial = mat_theta - mat_s - mat_l - self.admm_param * mat_z
        # neg part grad

        mat_s, l21norm = self.shrink(
            mat_s + self.proxstep_param * gradient_partial,
            self.proxstep_param * self.admm_param * self.lbda)
        #        print(mat_s, self.proxstep_param * self.admm_param * self.lbda)

        mat_s = (mat_s + mat_s.T) / 2
        if not self.opts['use_u']:  # no univariate parameters
            ltot = self.meta['ltot']
            mat_s[:ltot, :ltot] -= \
                np.diag(np.diag(mat_s[:ltot, :ltot]))

        tmp = mat_l + self.proxstep_param * gradient_partial

        eig, mat_u = scipy.linalg.eigh(tmp)

        # spectral soft shrink to form eigenvalues of L
        eig_l = eig - self.proxstep_param * self.admm_param * self.rho
        eig_l[eig_l < 1e-25] = 0
        mat_l = np.dot(np.dot(mat_u, np.diag(eig_l)), mat_u.T)
        mat_l = (mat_l + mat_l.T) / 2

        ## update dual variables Z
        resid_theta = mat_theta - mat_s - mat_l
        mat_z = mat_z - resid_theta / self.admm_param
        mat_z = (mat_z + mat_z.T) / 2

        #        print(mat_s, mat_l)
        #        print(resid_theta)
        ## diagnostics, reporting, termination checks

        fronorm_s = np.linalg.norm(mat_s, 'fro')
        fronorm_l = np.linalg.norm(mat_l, 'fro')
        fronorm_theta = np.linalg.norm(mat_theta, 'fro')

        rnorm = np.linalg.norm(resid_theta, 'fro')
        snorm = np.sqrt(
            np.linalg.norm(mat_s - mat_s_old, 'fro')**2 +
            np.linalg.norm(mat_l - mat_l_old, 'fro')**2) / self.admm_param

        dim = mat_theta.shape[0]
        eps_pri = np.sqrt(
            3 * dim**2) * self.opts['abstol'] + self.opts['reltol'] * max(
                (fronorm_theta, np.sqrt(fronorm_s**2 + fronorm_l**2)))
        eps_dual = np.sqrt(3 * dim**2) * self.opts['abstol'] + self.opts[
            'reltol'] * np.linalg.norm(mat_z, 'fro')

        ## store stuff
        residuals = rnorm, snorm, eps_pri, eps_dual
        new_vars = mat_theta, mat_s, mat_l, mat_z, alpha

        stats = {}
        stats['theta'] = mat_theta
        stats['solution'] = mat_s, mat_l, alpha
        self.problem_vars = stats['solution']

        stats['eig_l'] = eig_l
        stats['resid'] = resid_theta

        stats['admm_obj'] = self.lbda * l21norm + self.rho * np.sum(eig_l)
        # stats['true_obj'] = stats['admm_obj'] + self.plh(mat_s + mat_l, alpha) # true objective
        stats['admm_obj'] += self.prox.plh(mat_theta, alpha)

        return new_vars, residuals, stats

    def _postsetup_data(self):
        """called after drop_data"""
        self.prox = LikelihoodProx(self.cat_data, self.cont_data, self.meta)

    def get_objective(self, mat_s, mat_l, vec_u=None, alpha=None):
        """return 'real' objective """
        if self.meta['n_cat'] > 0 and not vec_u is None:
            mat_s = mat_s.copy()
            mat_s[:self.meta['ltot'], :self.meta['ltot']] += 2 * np.diag(vec_u)
        if alpha is None:
            alpha = np.zeros(self.meta['n_cg'])

        obj = self.lbda * self.sparse_norm(mat_s) \
            + self.rho * np.trace(mat_l)
        obj += self.prox.plh(mat_s + mat_l, alpha)
        return obj

###############################################################################
# pairwise CG models
###############################################################################


class AdmmCGaussianPW(BaseSolverPW, BaseAdmm):
    """
    solve the problem
       min l(S) + lambda * ||S||_{2,1}
       s.t. Lambda[S]>0
    where l is the pseudo likelihood with pairwise parameters Theta=S+L,
    here Lambda[S] extracts the quantitative-quantitative interactions
    from the pairwise parameter matrix Theta=(Q & R^T \\ R & -Lbda),
    that is, Lambda[Theta] = Lbda

    The estimated probability model has (unnormalized) density
        p(y) ~ exp(1/2 (D_x, y)^T Theta (D_x, y) + alpha^T y + u^T D_x)
    where D_x is the indicator representation of the discrete variables x
    (reduced by the indicator for the 0-th level for identifiability reasons),
    and y are the quantitative variables.
    Note that alpha and u are optional univariate parameters that can be included
    in the optimization problem above.

    The solver is an ADMM algorithm.
    """

    def __init__(self, *args, **kwargs):
        """"must provide with dictionary meta"""

        super().__init__(*args, **kwargs)  # Python 3 syntax

        self.name = 'S_ADMM'

        self.prox = None
        self.cat_format_required = 'dummy_red'

    def _initialize_admm(self):
        """initialize ADMM variables"""

        ## ensure coherent options for prox solver
        self.prox.opts['use_u'] = self.opts['use_u']
        self.prox.opts['use_alpha'] = self.opts['use_alpha']

        ## initialize ADMM variables
        dim = self.meta['dim']
        ltot = self.meta['ltot']
        mat_theta = np.eye(dim)
        mat_theta[ltot:, ltot:] *= -1
        # since lower right block needs to be negative definite
        # make Theta feasible/cleaned for plh
        mat_theta = self.prox.clean_theta(mat_theta)

        alpha = np.zeros((self.meta['n_cg'], 1))
        mat_s = mat_theta.copy()
        if not self.opts['use_u']:  # no univariate parameters
            mat_s[:ltot, :ltot] -= np.diag(np.diag(mat_s[:ltot, :ltot]))
        mat_z = np.zeros((dim, dim))

        return mat_theta, mat_s, mat_z, alpha

    def _do_iter_admm(self, current_vars: tuple):
        """perform computations of one ADMM iteration"""

        mat_theta, mat_s, mat_z, alpha = current_vars

        mat_theta, alpha = self.prox.solve(mat_s + self.admm_param * mat_z,
                                           self.admm_param,
                                           (mat_theta, alpha))
        mat_theta = (mat_theta + mat_theta.T) / 2

        ## update S
        mat_s_old = mat_s
        mat_s, l21norm = self.shrink(mat_s - self.admm_param * mat_z,
                                     self.admm_param * self.lbda)
        mat_s = (mat_s + mat_s.T) / 2
        if not self.opts['use_u']:  # no univariate parameters
            ltot = self.meta['ltot']
            mat_s[:ltot, :ltot] -= \
                np.diag(np.diag(mat_s[:ltot, :ltot]))

        ## update dual variables Z
        resid_theta = mat_theta - mat_s
        mat_z = mat_z - resid_theta / self.admm_param
        mat_z = (mat_z + mat_z.T) / 2

        ## diagnostics, reporting, termination checks
        fronorm_s = np.linalg.norm(mat_s, 'fro')
        fronorm_theta = np.linalg.norm(mat_theta, 'fro')

        rnorm = np.linalg.norm(resid_theta, 'fro')
        snorm = np.linalg.norm(mat_s - mat_s_old, 'fro') / self.admm_param

        dim = mat_theta.shape[0]
        eps_pri = np.sqrt(
            2 * dim**2) * self.opts['abstol'] + self.opts['reltol'] * max(
                (fronorm_theta**2, fronorm_s**2))
        eps_dual = np.sqrt(2 * dim**2) * self.opts['abstol'] + self.opts[
            'reltol'] * np.linalg.norm(mat_z, 'fro')

        ## store stuff
        residuals = rnorm, snorm, eps_pri, eps_dual
        new_vars = mat_theta, mat_s, mat_z, alpha

        stats = {}
        stats['theta'] = mat_theta
        stats['solution'] = mat_s, alpha
        self.problem_vars = stats['solution']

        stats['resid'] = resid_theta

        stats['admm_obj'] = self.lbda * l21norm
        # stats['true_obj'] = stats['admm_obj'] + self.plh(mat_s, alpha) # true objective
        stats['admm_obj'] += self.prox.plh(mat_theta, alpha)

        return new_vars, residuals, stats

    def _postsetup_data(self):
        """called after drop_data"""
        self.prox = LikelihoodProx(self.cat_data, self.cont_data, self.meta)

    def get_objective(self, mat_s, vec_u=None, alpha=None):
        """return 'real' objective """
        if self.meta['n_cat'] > 0 and not vec_u is None:
            mat_s = mat_s.copy()
            mat_s[:self.meta['ltot'], :self.meta['ltot']] += 2 * np.diag(vec_u)
        if alpha is None:
            alpha = np.zeros(self.meta['n_cg'])

        obj = self.lbda * self.sparse_norm(mat_s) + self.prox.plh(mat_s, alpha)
        return obj
