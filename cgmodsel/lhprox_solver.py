#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np
import scipy
#import abc
#import time

from cgmodsel.utils import _logsumexp_condprobs_red
from cgmodsel.utils import logsumexp
from cgmodsel.base_solver import BaseSolver

from scipy.optimize import approx_fprime
from scipy import optimize


#################################################################################
# prox for PLH objective
#################################################################################
class LikelihoodProx(BaseSolver):
    
    def __init__(self, *args, **kwargs):
        """"must provide with dictionary meta"""
        kwargs.setdefault('reduced_levels', True)
        super().__init__(*args, **kwargs) # Python 3 syntax
#        BaseSolver.__init__(self, meta, useweights, reduced_levels = True)
                
#        self.opts = {'use_plh':use_plh}
        self._set_defaults_lhprox() # set default opts

        if self.opts['use_plh']:
            self.solve_genlh_prox = self.solve_plh_prox
            self.genlh = self.plh

            self.shapes = [('Q', (self.Ltot, self.Ltot)),
                           ('u', (self.Ltot, 1)),
                            ('R', (self.n_cgvars, self.Ltot)), 
                            ('F2tiL', (self.n_cgvars, self.n_cgvars)), # construct Lambda = F2tiL*F2tiL.T
                            ('alpha', (self.n_cgvars, 1))]
            self.n_params = sum( [np.prod(shape[1]) for shape in self.shapes] )

        else:
            self.solve_genlh_prox = self.solve_lh_prox
            self.genlh = self.lh
            
            assert self.n_catvars == 0 or self.n_cgvars == 0, "Mixed case not implemented for LH"
            self.shapes = [('Q', (self.Ltot, self.Ltot)),
#                            ('R', (self.n_cgvars, self.Ltot)), 
                            ('u', (self.Ltot, 1)),
#                            ('Lambda', (self.n_cgvars, self.n_cgvars)),
#                            ('alpha', (self.n_cgvars, 1))
                            ] 
            self.n_params = sum( [np.prod(shape[1]) for shape in self.shapes] )


            X = np.concatenate((self.cat_data, self.cont_data), axis=1)
            self.Sigma0 = np.dot(X.T, X) / self.n # empirical covariance matrix
            self.mu0 = np.sum(self.cont_data, axis = 1) / self.n # empirical mean of CG variables
            self.is_centered = False
            if np.linalg.norm(self.mu0) < 1e-15:
                self.is_centered = True
        
            if self.n < self.n_cgvars:
                print('Warning(LHProx_base.drop_data): degenerate Sigma0 (n < d)...')



    def _set_defaults_lhprox(self):
        """default solver options"""
        self.opts = {}
        self.opts.setdefault('verb_lhprox', 1) # write output

        ## objective variants
        self.opts.setdefault('use_alpha', 1) # use univariate cts parameters?
        self.opts.setdefault('use_u', 1) # use univariate discrete parameters?
        self.opts.setdefault('off', 0) # if 1 regularize only off-diagonal
       
        ## stopping criteria and tolerancies
#        self.opts.setdefault('abstol', 1e-5)
#        self.opts.setdefault('reltol', 1e-5)
        self.opts.setdefault('stoptol', 1e-5) # stoptol for lh prox step
        self.opts.setdefault('lhproxtol', 1e-12)
        self.opts.setdefault('maxiter_lhprox', 500)
        
    def drop_data(self, discretedata, continuousdata):
        """ drop data to be used for optimization
        discrete data must be dummy coded (but with 0-th level removed)"""
        BaseSolver.drop_data(self, discretedata, continuousdata) 

    def _cleanTheta(self, Theta):
        """
        make pairwise parameter matrix feasible for likelihood prox solver
        -> modifies Theta
        """
        if self.opts['use_plh']:
            # copies upper triangle of Theta to lower triangle to symmetrize
            # Furthermore, all elements on the block-diagonal of the discrete
            # are set to zero, except diagonal elements
            # since these correspond to univariate discrete sufficient statistics
            x =self._Theta2x(Theta, np.zeros(self.n_cgvars))
            return self._x2Thetaalpha(x)[0]
        else:
            Theta = np.triu(Theta)
            diag = np.diag(Theta).copy()
            Theta += Theta.T - np.diag(diag)
            return Theta
        
###############################################################################
# Solver for Pseudo-likelihood Prox operator
###############################################################################
            
    def callback_plh(self, x, handle_fg):
        fnew = handle_fg(x)[0]
        if not fnew <= self.fold:
            print('Warning(CG_base_ADMM.callback_plh): Potential scipy bug, function value got worse in last iteration')
        self.fold = fnew
        
    def solve_plh_prox(self, Z, mu, oldThetaalpha, verb=False, 
                       do_scipy=1, debug=0):
        """ solve proximal mapping of negative pseudo loglikelihood
        min_{Theta, alpha} l_p(Theta, alpha) + 1 / (2mu) * ||Theta-Z||_F^2
        
        known issue with ADMM:
        not doing warm starts may cause problems if solution is to inexact
        generally ADMM convergence requires very exact solutions
        -> use ftol to control tolerancy, or refine to control #restarts
        """
        ## split Z (since in determining the prox objective the split components are used)
        zQ = Z[:self.Ltot, :self.Ltot].copy()
        zR = Z[self.Ltot:, :self.Ltot]
        zB = Z[self.Ltot:, self.Ltot:].copy()
        
        zbeta = np.diag(zB).copy().reshape((self.n_cgvars, 1))
        zB -= np.diag(np.diag(zB))
        zu = np.diag(zQ).copy().reshape((self.Ltot, 1))
        zQ -= np.diag(np.diag(zQ))
        Z = zQ, zu, zR, zB, zbeta
        
        handle_fg = lambda x: self.fg_plh_prox(x, Z=Z, mu=mu)
        
        ## solve proximal mapping
        
#        x0 = self.get_rand_startingpoint() 
        x0 = self._Theta2x(*oldThetaalpha) # starting point as vector, save for input parameters
        
        if do_scipy: # solve with scipy L-BFGS-B wrapup
            
            f_init = handle_fg(x0)[0]
            self.fold = f_init
    
            ## bounds that respect identifiability constraints
            bnds = self.Ltot ** 2 * [(-np.inf, np.inf)] # Q, only upper triangle is used

            bnds += self.Ltot * [(-np.inf, np.inf)] # u (use_u = 0 is enforced in main ADMM updates)

            bnds += (self.n_cgvars * self.Ltot + self.n_cgvars ** 2 ) * [(-np.inf, np.inf)] # R und F2tiL
            if self.opts['use_alpha']:
                bnds += self.n_cgvars * [(-np.inf, np.inf)] 
            else:
                bnds += self.n_cgvars * [(0,0)]
            # TODO: use zerobounds for block diagonal of Q?
                
            ## further solver properties
            callback = lambda x: self.callback_plh(x, handle_fg)

            maxiter = 500
            correctionpairs = min(len(bnds) - 1, 10)
            ftol = self.opts['lhproxtol']

            res = optimize.minimize(handle_fg, x0, 
                                    method = 'L-BFGS-B', 
                                    jac = True, bounds = bnds, 
                                    options = {'maxcor':correctionpairs,
                                               'maxiter':maxiter, 'ftol':ftol},
                                    callback = callback)

            
            if not res.message.startswith(b'CONV'): # solver did not converge
                print('PLH_prox scipy-solver message:', res.message)

#            f, g = handle_fg(res.x)
            _, _, _, FLa, _ = self.unpack(res.x)
            
            if np.linalg.norm(FLa)<1e-5 and self.n_cgvars > 0: # TODO: certificate for optimality?
                print('Warning(solve_plh_prox): Lambda = F F^T with F close to zero (might be non-optimal)')
                
            Theta, alpha = self._x2Thetaalpha(res.x)


        return Theta, alpha
    
    def preprocess_proxstep(self, x):
        """ unpack parameters from vector x and preprocess
        
        this modifies x (x not save for reuse)"""   

        Q, u, R, FLambda, alpha = self.unpack(x)

        for r in range(self.n_catvars): # set block-diagonal to zero
            Q[self.cat_glims[r]:self.cat_glims[r+1], self.cat_glims[r]:self.cat_glims[r+1]] = \
                np.zeros((self.sizes[r], self.sizes[r]))        
        Q = np.triu(Q)
        Q = Q + Q.T
        
        return Q, u, R, FLambda, alpha
    
    def fg_plh_prox(self, x, Z, mu, eps=1e-15):
        """calculate function value f and gradient g of
        l_p(Theta, alpha) + 1 / (2mu) ||Theta - Z||_F^2
        
        where Theta, alpha are contained in the vector x of parameters
        """
        ## unpack parameters from vector x        
        Q, u, R, FLa, alpha = self.preprocess_proxstep(x)

        B, beta = self._FLa2Bbeta(FLa)
        beta += eps * np.ones(beta.shape) # increase numerical instability (avoid beta that contains 0s)
        # precision matrix = FLa*FLa.T + eps * eye(self.n_cgvars)
        
        # intitialize gradients
        grad = np.zeros(self.n_params)
        gradQ, gradu, gradR, gradF2L, gradalpha = self.unpack(grad)
        gradtiLa = np.zeros((self.n_cgvars, self.n_cgvars))
        gradbeta = np.zeros((self.n_cgvars, 1))
        e = np.ones((self.n, 1))
        
        ## ** discrete node conditionals ** 
        lD = 0
        
        W = np.dot(self.cont_data, R) + np.dot(self.cat_data, Q) + np.dot(e, u.T) # n by Ltot
        A = np.empty((self.n, self.Ltot)) # array for conditional probs given data

        for r in range(self.n_catvars):
            Wr = W[:, self.cat_glims[r]:self.cat_glims[r+1]] # view of W
            Dr = self.cat_data[:, self.cat_glims[r]:self.cat_glims[r+1]] # view of self.cat_data
            tmp_logsumexp, tmp_conditionalprobs = _logsumexp_condprobs_red(Wr) # numerically stable
            A [:, self.cat_glims[r]:self.cat_glims[r+1]]= tmp_conditionalprobs
            lr = -np.sum(np.sum(np.multiply(Wr, Dr), axis = 1) - tmp_logsumexp)
            lD += lr

#        print('lD', lD/self.n)
            
        # gradients
        A = A - self.cat_data

        gradu = np.sum(A, 0) # Ltot by 1
        gradR = np.dot(self.cont_data.T, A)
        gradQ = np.dot(self.cat_data.T, A) # this is Phihat from the doc, later add transpose and zero out diagonal

        ## ** Gaussian node conditionals **       
        M = np.dot(e, alpha.T) + np.dot(self.cat_data, R.T) - np.dot(self.cont_data, B) # n by dg, concatenation of mu_s
        Delta = M.copy()
        for s in range(self.n_cgvars):
            Delta[:, s] /= beta[s]
        Delta -= self.cont_data # residual

        lG = - 0.5 * self.n * np.sum(np.log(beta)) \
            + 0.5 * np.linalg.norm(np.dot(Delta, np.diag(np.sqrt(beta.flatten()))), 'fro') ** 2
#        print('lG', lG/self.n)
        
        # gradients
        gradtiLa = -np.dot(self.cont_data.T, Delta) # dg by dg, later add transpose and zero out diagonal
        gradtiLa -= np.diag(np.diag(gradtiLa))
        gradtiLa = 0.5 * (gradtiLa + gradtiLa.T)
            
        for s in range(self.n_cgvars):
            gradbeta[s] = -.5 * self.n / beta[s] + .5 * np.linalg.norm(Delta[:, s], 2) ** 2 \
            - 1 / beta[s] * np.dot(Delta[:, s].T, M[:, s])

        gradalpha = np.sum(Delta, 0).T # dg by 1
        gradR += np.dot(Delta.T, self.cat_data)

        # scale gradients as likelihood
        gradQ /= self.n 
        gradu /= self.n 
        gradR /= self.n 
        gradtiLa /= self.n 
        gradbeta /= self.n
        gradalpha /= self.n 

        ## add quadratic term  1/2mu * ||([Q + 2diag(u)] & R^T \\ R & - Lambda) - Z||_F^2
        zQ, zu, zR, zB, zbeta = Z

        fq = 0
        fq += np.sum(np.square(Q - zQ))
        fq += np.sum(np.square(2 * u - zu)) # note that u is only half of discrete diagonal
        fq += 2 * np.sum(np.square(R - zR))
        fq += np.sum(np.square(-B - zB)) # remember neg sign of Lambda in Theta
        fq += np.sum(np.square(-beta - zbeta))
      
        fq /= 2 * mu

#        print('fq', fq)

        # gradients quadratic term
        gradQ += (Q - zQ) / mu
        gradu = gradu.reshape((self.Ltot, 1)) # since with dc=0 gradu has shape (0,)
        gradu += 2 * (2 * u - zu) / mu
        gradR += 2 * (R - zR) / mu
        gradtiLa += (B + zB) / mu # has zero diagonal
        gradbeta += (beta + zbeta) / mu
        
        ## gradients to only upper triangle
        for r in range(self.n_catvars): # set block-diagonal to zero
            gradQ[self.cat_glims[r]:self.cat_glims[r+1], self.cat_glims[r]:self.cat_glims[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))

        gradQ = np.triu(gradQ) + np.tril(gradQ).T

        gradtiLa += np.diag(gradbeta.flatten()) # add gradient of diagonal
        
        gradFLa = 2 * np.dot(gradtiLa, FLa) # may not initialize FLa at 0, since then gradient is always zero
        
        f = 1 / self.n * (lD + lG) + fq
        g = self.pack((gradQ, gradu, gradR, gradFLa, gradalpha)) 
        
        return f, g.reshape(-1)

    def callback(self, x, Z, mu, approxgrad=1):
        """a callback function that serves primarily for debugging"""
        f, g = self.fg_plh_prox(x, Z=Z, mu=mu) 

        print('f=', f)
        if approxgrad: # gradient check
            func_handle_f = lambda x: self.fg_plh_prox(x, Z=Z, mu=mu)[0]
            eps = np.sqrt(np.finfo(float).eps) # ~1.49E-08 at my machine
            gprox = approx_fprime(x, func_handle_f, eps)

            diff = g-gprox
            normdiff = np.linalg.norm(diff)
            if normdiff >1e-4:
                print ('g_exct', g)
                print ('g_prox', gprox)
#            print('g-gprox',self.unpack(diff))
#            print('quot',g/proxg)

            print('graddev=', np.linalg.norm(diff))

    
    def _FLa2Bbeta(self, FLa):
        """ construct precision matrix, then extract diagonal """
        B = np.dot(FLa, FLa.T) # PSD precision matrix
        beta = np.diag(B).copy().reshape((self.n_cgvars, 1)) # diagonal
        B -= np.diag(np.diag(B)) # off-diagonal elements
        return B, beta
        
    def _Theta2tuple(self, Theta):
        """ split Theta into its components
        (save: returns copies from data in Theta, Theta is not modified)"""
        Q = Theta[:self.Ltot, :self.Ltot].copy()
        R = Theta[self.Ltot:, :self.Ltot].copy() # modifying output leaves Theta untouched
        Lambda = -Theta[self.Ltot:, self.Ltot:]
#        print(Lambda)
#        FLa = np.linalg.cholesky(Lambda) # fails if not PD

        if self.n_cgvars > 0:
            las, U = scipy.linalg.eigh(Lambda)
#            print('las', las)
            las[las < 1e-16] = 0 # make more robust
            FLa = np.dot(U, np.diag(np.sqrt(las)))
#           print('chol-error', np.linalg.norm(np.dot(FLa, FLa.T) - Lambda))
        else:
            FLa = np.empty((0,0))

        u = 0.5 * np.diag(Q).copy().reshape((self.Ltot, 1))

        for r in range(self.n_catvars): # set block diagonal to zero
            Q[self.cat_glims[r]:self.cat_glims[r+1], self.cat_glims[r]:self.cat_glims[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))        
        Q = np.triu(Q) # use only upper triangle
        Q = Q + Q.T

        return Q, u, R, FLa

    def _Theta2x(self, Theta, alpha):
        """takes Theta, cleans it (symmetrize etc.) and pack into x
        (save: Theta is not modified)"""
        return self.pack(list(self._Theta2tuple(Theta))+[alpha]) # save for Theta
    
    def _x2Thetaalpha(self, x):
        """ convert vectorized x to parameter matrix Theta
        (save: x is not modified) """
        Q, u, R, FLa, alpha = self.unpack(x) # save, returns copies from data in x

        # set parameters in upper triangle
        Theta = np.empty((self.d, self.d))
        Theta[:self.Ltot, :self.Ltot] = Q
        for r in range(self.n_catvars): # set block-diagonal to zero
            Theta[self.cat_glims[r]:self.cat_glims[r+1], self.cat_glims[r]:self.cat_glims[r+1]] = \
                np.zeros((self.sizes[r], self.sizes[r]))        
        Theta[:self.Ltot, self.Ltot:] = R.T
        
        ## symmetric matrix from upper triangle
        Theta = np.triu(Theta)
        Theta = Theta + Theta.T
        
        ## Lambda
        Lambda = np.dot(FLa, FLa.T)
        Theta[self.Ltot:, self.Ltot:] = -Lambda

        ## add diagonal
        Theta[:self.Ltot, :self.Ltot] += 2*np.diag(u.flatten())

        return Theta, alpha

    def get_rand_startingpoint(self):
        """ not needed if using warm starts """
        x0 = np.random.random(self.n_params)
        x0[self.n_params-self.n_cgvars:] = np.ones(self.n_cgvars)
        return x0
        
    def plh(self, Theta, alpha, cval=False):
        """ return negative pseudo-log-likelihood function value
        
        cval .. if True, calculate (node-wise) cross validation error"""
        if cval:
            dis_errors = np.zeros(self.n_catvars)
            cts_errors= np.zeros(self.n_cgvars)
        
        Q, u, R, FLa = self._Theta2tuple(Theta) # save
        B, beta = self._FLa2Bbeta(FLa)

        f = 0
        
        ## ** discrete node conditionals ** 
        W = np.dot(self.cont_data, R) + np.dot(self.cat_data, Q) + np.dot(np.ones((self.n, 1)), u.T) # n by Ltot

        for r in range(self.n_catvars):
            Wr = W[:, self.cat_glims[r]:self.cat_glims[r+1]] # view of W
            Dr = self.cat_data[:, self.cat_glims[r]:self.cat_glims[r+1]] # view of self.cat_data
            tmp_logsumexp, tmp_conditionalprobs = _logsumexp_condprobs_red(Wr) # numerically more stable
            if cval:
                # sum of probabilities of missclassification
                dis_errors[r] = self.n - np.sum(np.multiply(tmp_conditionalprobs,Dr)) # sums over both axes

            lr = - np.sum(np.sum(np.multiply(Wr, Dr), axis=1) - tmp_logsumexp)
            f += 1/self.n * lr
        
        M = np.dot(self.cat_data, R.T) - np.dot(self.cont_data, B) # n by dg, concatenation of mu_s
        if self.n_cgvars > 0:
            M += np.outer(np.ones(self.n), alpha)

        if cval:
            for s in range(self.n_cgvars):
                cts_errors[s] = np.linalg.norm(self.cont_data[:,s] - M[:, s]/beta[s], 2) ** 2

        Delta = M.copy()
        for s in range(self.n_cgvars):
            Delta[:, s] /= beta[s]
        Delta -= self.cont_data # residual
        lG = - 0.5 * self.n * np.sum(np.log(beta)) \
            + 0.5 * np.linalg.norm(np.dot(Delta, np.diag(np.sqrt(beta.flatten()))), 'fro') ** 2
        f+= 1/self.n * lG
        
        if cval:
            return dis_errors, cts_errors, f
        
        return f

    def crossvalidate(self, Theta, alpha):
        """perform cross validation (drop test data) """
        if self.plh:
            dis_errors, cts_errors, f = self.plh(Theta, alpha, cval=True)
            avg_dis_error = 1/self.n*np.sum(dis_errors) 
            avg_cts_error = np.sum([np.sqrt(es/self.n) for es in cts_errors]) # mean RMSEs
            
            cvalerror =  avg_dis_error + avg_cts_error
            
            if self.n_cgvars >0 :
                avg_cts_error /= self.n_cgvars
            if self.n_catvars>0:
                avg_dis_error /= self.n_catvars
            
            return cvalerror
        
        raise NotImplementedError

