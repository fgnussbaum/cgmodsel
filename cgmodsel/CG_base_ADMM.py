#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np
import scipy
#import time

from cgmodsel.utils import _logsumexp_condprobs_red
from cgmodsel.utils import logsumexp
from cgmodsel.CG_base_solver import CG_base_solver

from scipy.optimize import approx_fprime
from scipy import optimize

#################################################################################
# ADMM sparse norms and shrinkage operators
#################################################################################


def grp_soft_shrink(S, tau, groupdelimiters=None, Lcum=None, off=False):
    """
    calculate (group-)soft-shrinkage function of matrix S with shrinkage parameter tau
    soft shrink if no groupdelimiters are given
    else must provide with groupdelimiters (sizes of groups) and
    cumulative sizes of groups (Lcum)
    
    this code could be made much faster
    (by parallizing loops, efficient storage access)
    """
    if groupdelimiters is None:
        # soft shrink
        tmp = np.abs(S) - tau
        tmp[tmp < 1e-25] = 0
        
        tmp2 = np.multiply(np.sign(S), tmp)
        if off:
            tmp2 -= np.diag(np.diag(tmp2))
            tmp2 += np.diag(np.diag(S))
        return tmp2

    tmp = np.empty(S.shape)

    for i in range(len(groupdelimiters)):
        for j in range(len(groupdelimiters)):
            g = S[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]]
            if (i == j) and off:
                tmp[Lcum[i]:Lcum[i+1], Lcum[i]:Lcum[i+1]] = g
                continue
#            print(g.shape, (sizes[i], sizes[j]))
            gnorm = np.linalg.norm(g, 'fro')
            if gnorm <= tau:
                tmp[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]] = np.zeros(g.shape)
            else:
                tmp[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]] = g * (1 - tau / gnorm)
    
    return tmp

def l21norm(S, groupdelimiters=None, Lcum=None, off=False):
    """
    calculate l_{2,1}-norm or l_1-norm of S
    l_1-norm if no groupdelimiters are given
    else must provide with groupdelimiters (sizes of groups) and
    cumulative sizes of groups (Lcum)
    """
    if groupdelimiters is None:
        tmp = np.sum(np.abs(S.flatten()))
        if off:
            tmp -= np.sum(np.diag(np.abs(S)))
        return tmp
        # calculate regular l1-norm
    s = 0
    for i in range(len(groupdelimiters)):
        for j in range(i):
            g = S[Lcum[i]:Lcum[i+1], Lcum[j]:Lcum[j+1]]
            s += np.linalg.norm(g, 'fro')
    
    s *= 2
    if not off:
        for i in range(len(groupdelimiters)):
            g = S[Lcum[i]:Lcum[i+1], Lcum[i]:Lcum[i+1]]
            s += np.linalg.norm(g, 'fro')

    return s


#################################################################################
# ADMM update schemes for ADMM parameter
#################################################################################

def cont_update_S2a(eps_pri, eps_dual, rnorm, snorm, mu, tau_inc = 5, 
                    criticalratio = 5, verb = False):
    """ update S2a from my master thesis
    uses more robust constant size updates of mu"""
    snorm_rel = snorm / eps_dual
    rnorm_rel = rnorm / eps_pri
    if snorm_rel < 1 and rnorm_rel < 1:
        return mu # residuals are already smaller than tolerancies

    ## do scaling of mu
    mu_old = mu
    if rnorm_rel > snorm_rel * criticalratio:
        mu = mu / tau_inc # decrease mu
        if verb: print('old mu=', mu_old, 'new mu=', mu)
    elif snorm_rel > rnorm_rel * criticalratio:
        mu = mu * tau_inc # increase mu
        if verb: print('old mu=', mu_old, 'new mu=', mu)
        
    return mu

def cont_update_S2b(eps_pri, eps_dual, rnorm, snorm, mu,
                    criticalratio = 5, verb = False):
    """ update S2b from my master thesis"""

    snorm_rel = snorm / eps_dual
    rnorm_rel = rnorm / eps_pri
    if snorm_rel < 1 and rnorm_rel < 1:
        return mu # residuals are already smaller than tolerancies
    
    ## calculate scaling factor as in update S2b - non-robust
    if snorm_rel != 0:
        if rnorm_rel != 0:
            tau_inc = rnorm_rel/snorm_rel
            # if this quantity is large, we should focus on reducing the primal residual
            # violation of primal feasibility is penalized with scaling 1/mu
            # so mu should be reduced
            # note that mu (here) and rho (master thesis) play reciprocal roles
        else:
            tau_inc = 0.01
    else:
        tau_inc = 100

    if verb: print('mu_old', mu, 'tau_inc', tau_inc, end = "")
    
    ## do scaling of mu
    if rnorm_rel > snorm_rel * criticalratio:
        mu = mu / tau_inc # decrease mu
        if verb: print(', new mu', mu)
    elif snorm_rel > rnorm_rel * criticalratio:
        mu = mu * tau_inc # increase mu
        if verb: print(', new mu', mu)
    else:
        if verb: print()
        
    return mu

#################################################################################
# prox for PLH objective
#################################################################################
class LHProx_base(CG_base_solver):
    
    def __init__(self, meta, useweights, use_plh):
        CG_base_solver.__init__(self, meta, useweights, reduced_levels = True)
                
        self.opts = {'use_plh':use_plh}
        self._set_defaults() # set default opts
    
    def drop_data(self, discretedata, continuousdata):
        """ drop data to be used for optimization
        discrete data must be dummy coded (but with 0-th level removed)"""
        CG_base_solver.drop_data(self, discretedata, continuousdata)

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
            x =self._Theta2x(Theta, np.zeros(self.dg))
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
        
        zbeta = np.diag(zB).copy().reshape((self.dg, 1))
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

            bnds += (self.dg * self.Ltot + self.dg ** 2 ) * [(-np.inf, np.inf)] # R und F2tiL
            if self.opts['use_alpha']:
                bnds += self.dg * [(-np.inf, np.inf)] 
            else:
                bnds += self.dg * [(0,0)]
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
            
            if np.linalg.norm(FLa)<1e-5 and self.dg > 0: # TODO: certificate for optimality?
                print('Warning(solve_plh_prox): Lambda = F F^T with F close to zero (might be non-optimal)')
                
            Theta, alpha = self._x2Thetaalpha(res.x)


        return Theta, alpha
    
    def preprocess_proxstep(self, x):
        """ unpack parameters from vector x and preprocess
        
        this modifies x (x not save for reuse)"""   

        Q, u, R, FLambda, alpha = self.unpack(x)

        for r in range(self.dc): # set block-diagonal to zero
            Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = \
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
        # precision matrix = FLa*FLa.T + eps * eye(self.dg)
        
        # intitialize gradients
        grad = np.zeros(self.totalnumberofparams)
        gradQ, gradu, gradR, gradF2L, gradalpha = self.unpack(grad)
        gradtiLa = np.zeros((self.dg, self.dg))
        gradbeta = np.zeros((self.dg, 1))
        e = np.ones((self.n, 1))
        
        ## ** discrete node conditionals ** 
        lD = 0
        
        W = np.dot(self.Y, R) + np.dot(self.D, Q) + np.dot(e, u.T) # n by Ltot
        A = np.empty((self.n, self.Ltot)) # array for conditional probs given data

        for r in range(self.dc):
            Wr = W[:, self.Lsum[r]:self.Lsum[r+1]] # view of W
            Dr = self.D[:, self.Lsum[r]:self.Lsum[r+1]] # view of self.D
            tmp_logsumexp, tmp_conditionalprobs = _logsumexp_condprobs_red(Wr) # numerically stable
            A [:, self.Lsum[r]:self.Lsum[r+1]]= tmp_conditionalprobs
            lr = -np.sum(np.sum(np.multiply(Wr, Dr), axis = 1) - tmp_logsumexp)
            lD += lr

#        print('lD', lD/self.n)
            
        # gradients
        A = A - self.D

        gradu = np.sum(A, 0) # Ltot by 1
        gradR = np.dot(self.Y.T, A)
        gradQ = np.dot(self.D.T, A) # this is Phihat from the doc, later add transpose and zero out diagonal

        ## ** Gaussian node conditionals **       
        M = np.dot(e, alpha.T) + np.dot(self.D, R.T) - np.dot(self.Y, B) # n by dg, concatenation of mu_s
        Delta = M.copy()
        for s in range(self.dg):
            Delta[:, s] /= beta[s]
        Delta -= self.Y # residual

        lG = - 0.5 * self.n * np.sum(np.log(beta)) \
            + 0.5 * np.linalg.norm(np.dot(Delta, np.diag(np.sqrt(beta.flatten()))), 'fro') ** 2
#        print('lG', lG/self.n)
        
        # gradients
        gradtiLa = -np.dot(self.Y.T, Delta) # dg by dg, later add transpose and zero out diagonal
        gradtiLa -= np.diag(np.diag(gradtiLa))
        gradtiLa = 0.5 * (gradtiLa + gradtiLa.T)
            
        for s in range(self.dg):
            gradbeta[s] = -.5 * self.n / beta[s] + .5 * np.linalg.norm(Delta[:, s], 2) ** 2 \
            - 1 / beta[s] * np.dot(Delta[:, s].T, M[:, s])

        gradalpha = np.sum(Delta, 0).T # dg by 1
        gradR += np.dot(Delta.T, self.D)

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
        for r in range(self.dc): # set block-diagonal to zero
            gradQ[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))

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
        beta = np.diag(B).copy().reshape((self.dg, 1)) # diagonal
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

        if self.dg > 0:
            las, U = scipy.linalg.eigh(Lambda)
#            print('las', las)
            las[las < 1e-16] = 0 # make more robust
            FLa = np.dot(U, np.diag(np.sqrt(las)))
#           print('chol-error', np.linalg.norm(np.dot(FLa, FLa.T) - Lambda))
        else:
            FLa = np.empty((0,0))

        u = 0.5 * np.diag(Q).copy().reshape((self.Ltot, 1))

        for r in range(self.dc): # set block diagonal to zero
            Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))        
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
        for r in range(self.dc): # set block-diagonal to zero
            Theta[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = \
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
        x0 = np.random.random(self.totalnumberofparams)
        x0[self.totalnumberofparams-self.dg:] = np.ones(self.dg)
        return x0
        
    def plh(self, Theta, alpha, cval=False):
        """ return negative pseudo-log-likelihood function value
        
        cval .. if True, calculate (node-wise) cross validation error"""
        if cval:
            dis_errors = np.zeros(self.dc)
            cts_errors= np.zeros(self.dg)
        
        Q, u, R, FLa = self._Theta2tuple(Theta) # save
        B, beta = self._FLa2Bbeta(FLa)

        f = 0
        
        ## ** discrete node conditionals ** 
        W = np.dot(self.Y, R) + np.dot(self.D, Q) + np.dot(np.ones((self.n, 1)), u.T) # n by Ltot

        for r in range(self.dc):
            Wr = W[:, self.Lsum[r]:self.Lsum[r+1]] # view of W
            Dr = self.D[:, self.Lsum[r]:self.Lsum[r+1]] # view of self.D
            tmp_logsumexp, tmp_conditionalprobs = _logsumexp_condprobs_red(Wr) # numerically more stable
            if cval:
                # sum of probabilities of missclassification
                dis_errors[r] = self.n - np.sum(np.multiply(tmp_conditionalprobs,Dr)) # sums over both axes

            lr = - np.sum(np.sum(np.multiply(Wr, Dr), axis=1) - tmp_logsumexp)
            f += 1/self.n * lr
        
        M = np.dot(self.D, R.T) - np.dot(self.Y, B) # n by dg, concatenation of mu_s
        if self.dg > 0:
            M += np.outer(np.ones(self.n), alpha)

        if cval:
            for s in range(self.dg):
                cts_errors[s] = np.linalg.norm(self.Y[:,s] - M[:, s]/beta[s], 2) ** 2

        Delta = M.copy()
        for s in range(self.dg):
            Delta[:, s] /= beta[s]
        Delta -= self.Y # residual
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
            
            if self.dg >0 :
                avg_cts_error /= self.dg
            if self.dc>0:
                avg_dis_error /= self.dc
            
            return cvalerror
        
        raise NotImplementedError


###############################################################################
# Solver for Likelihood Prox operator
##############################################################################
    def solve_lh_prox(self, Z, mu, oldThetaalpha, verb=False, 
                       refine=1):
        """ solve proximal mapping of negative loglikelihood
        min_Theta   l(Theta) + 1/(2mu) * ||Theta-Z||_F^2
        
        known issue with ADMM:
        not doing warm starts may cause problems if solution is to inexact
        generally ADMM convergence requires very exact solutions
        -> use ftol to control tolerancy, or refine to control #restarts
        """
        if self.dc == 0: # analytical solution exists
            if self.opts['use_alpha'] or (not self.opts['use_alpha'] and not self.is_centered):
                raise ValueError('center data before using Gaussian likelihood and do not use alpha (set use_alpha to 0)')
            sigmas, U = scipy.linalg.eigh(mu*self.Sigma0 - Z)
            
            gammas =  - sigmas/2 + np.sqrt(np.square(sigmas)/4 + mu) 
            # gammas always positive (though might round to 0 due to bad precision)
            
            alpha = oldThetaalpha[1] # zeros
            return np.dot(np.dot(U, np.diag(gammas)), U.T), alpha
        
        assert self.dg == 0, "Not implemented for both discrete and cts vars"
#        raise NotImplementedError('d_c > 0, likelihood gradient approx is not robust')
        
        
#        zQ = Z[:self.Ltot, :self.Ltot]
#        zR = Z[self.Ltot:, :self.Ltot]
#        zLambda = Z[self.Ltot:, self.Ltot:]
#        Z = zQ, zR, zLambda
#        x0 = self.get_rand_startingpoint() 
        
        Theta, alpha = oldThetaalpha
        for i in range(refine):
            x0 = self._Theta2x_lh(Theta)
            handle_fg = lambda x: self.fg_lh_prox(x, Z = Z, mu = mu)
    
#            if verb:
#                f, g = handle_fg(x0)
#                print('f(x0)=',f)

            ## bounds that respect identifiability constraints
            bnds = self.Ltot ** 2 * [(-np.inf, np.inf)] # Q, only upper triangle is used

            bnds += self.Ltot * [(-np.inf, np.inf)] # u (use_u = 0 is enforced in main ADMM updates)

#            bnds += (self.dg*self.Ltot + self.dg**2 ) * [(-np.inf, np.inf)] # R und F2tiL
#            if self.opts['use_alpha']:
#                bnds += self.dg * [(-np.inf, np.inf)] 
#            else:
#                bnds += self.dg * [(0,0)]
#            # TODO: use zerobounds for block diagonal of Q?
#            
#            
#            bnds = self.Ltot * (self.Ltot + self.dg) * [(-np.inf, np.inf)]
#            for s in range(self.dg):
#                bnds += [(1e-6, np.inf)]
#                if s<self.dg-1:
#                    bnds += self.dg * [(-np.inf, np.inf)]
##            print(bnds)
            callback = None
            maxiter = 10
#            method = 'Powell'
#            method = 'CG' 
            method = 'L-BFGS-B'
            ftol = self.opts['lhproxtol']
            correctionpairs = min(len(bnds)-1, 10)
            # CG - conjugate gradient method
            res = optimize.minimize(handle_fg, x0, 
                                    method = method, 
                                    jac = False, bounds = bnds, 
                                    options = {'maxcor' : correctionpairs, 'ftol':ftol,
                                               'maxiter':maxiter},
                                    callback = callback)
    
#            f = handle_fg(res.x)
            Theta = self.preprocess_proxstep_lh(res.x)
            
        if verb:
            print('f_lhprox=', res.fun)
#        print(res)
        if not res.message.startswith(b'CONV'):
            print('LH_prox solver message:', res.message)
        return Theta, alpha

    def lh(self, Theta, alpha):
        return self.fg_lh_prox(self._Theta2x_lh(Theta, alpha))

    def _Theta2x_lh(self, Theta, alpha=None):
        Q = Theta[:self.Ltot, :self.Ltot]
        u = np.diag(Q)
#        R = Theta[self.Ltot:, :self.Ltot]
#        Lambda = -Theta[self.Ltot:, self.Ltot:]
        return self.pack((Q, u))

    def preprocess_proxstep_lh(self, x):
        """ unpack parameters from vector x and preprocess
        
        this modifies x (x not save for reuse)"""   

        Q, u = self.unpack(x) # returns copies of x

        # only necessary for non-binary data
        for r in range(self.dc): # set block-diagonal to zero
            Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = np.zeros((self.sizes[r], self.sizes[r]))        
        Q = np.triu(Q)
        Q += Q.T + np.diag(u)
        
        return Q
    
    def fg_lh_prox(self, x, Z=None, mu=None):
        """calculate function value f and gradient g of
        lh(Theta, u, alpha) + 1/(2mu) ||Theta - Z||_F^2
        
        u = 0, alpha = 0
        
        where Theta, u, alpha are contained in the vector x of parameters
        """
        ## unpack parameters from vector x

        Q = self.preprocess_proxstep_lh(x)
        
        f = - .5 * np.sum(np.multiply(Q,self.Sigma0)) 
#        if self.opts['use_alpha']:
#            f -= .5 * np.inner(alpha, self.mu0)

#        print(np.linalg.slogdet(Lambda))
#        a = -.5 * np.linalg.slogdet(Lambda)[1] # log-partition function (wo. constant part)
        
        if self.dc > 0:
#            TiTheta = Q + np.dot(np.dot(R.T, np.linalg.inv(Lambda)), R)
#            TiTheta *= .5
            TiTheta = Q* .5
            sums = np.empty(2**self.dc)
            
            # TODO: suboptimal
            sizes = self.dc * [2,]
            n_discrete_states = 2 ** self.dc #int(np.prod(self.sizes))
            for x in range(n_discrete_states):
                Dx = np.unravel_index([x], sizes)
                sums[x] = np.sum(np.multiply(np.outer(Dx, Dx), TiTheta))
            a = logsumexp(sums)

        f += a
        
        ## add quadratic term  ||(Q & R^T \\ R & - B) - Z||_F^2, Q contains univariate discrete params

        if not Z is None:
#            zQ, zR, zLambda = Z
            fq = 0
            fq += np.sum(np.square(Q-Z))
#            fq += 2 * np.sum(np.square(R-zR))
#            fq += np.sum(np.square(-Lambda-zLambda)) # remember neg sign of Lambda in Theta
            
            fq /= 2*mu
            
            f += fq
        
        return f
    
###############################################################################

    def _set_defaults(self):
        """default solver options"""
        self.opts.setdefault('verb', 1) # write output

        ## ADMM parameters
        self.opts.setdefault('mu', self.dg + self.dc) # ADMM parameter (reciprocal to Boyd)
        self.opts.setdefault('tau', 0.6) # PADMM parameter for prox gradient step (S+L models only)

        ## objective variants
        self.opts.setdefault('use_alpha', 1) # use univariate cts parameters?
        self.opts.setdefault('use_u', 1) # use univariate discrete parameters?
        self.opts.setdefault('off', 0) # if 1 regularize only off-diagonal
       
        ## stopping criteria and tolerancies
        self.opts.setdefault('ABSTOL', 1e-5)
        self.opts.setdefault('RELTOL', 1e-5)
        self.opts.setdefault('stoptol', 1e-5) # stoptol for lh prox step
        self.opts.setdefault('lhproxtol', 1e-12)
        self.opts.setdefault('maxiter', 500)

        ### continuation scheme ###
        self.opts.setdefault('continuation', 1)
        self.opts.setdefault('num_continuation', 10)
        
        # version master thesis (self-adaptive)
        self.opts.setdefault('cont_adaptive', 1)
        self.opts.setdefault('cont_tau_inc', 2)
        
        # continuation version from 
        self.opts.setdefault('eta', .25)
        self.opts.setdefault('muf', 1e-6)
        

    def set_sparsenorm_lh(self):
        """
        set shrinkage operators, likelihoods and norms for ADMM
        """
        off = self.opts['off']
        if len(self.sizes) > 0 and max(self.sizes) > 1: # use group norms (reduced sizes are used)
            groupdelimiters = self.sizes + self.dg * [1]
            cumulative = np.cumsum([0] + groupdelimiters)
        else:
            groupdelimiters = None
            cumulative = None
            
        self.func_shrink = lambda S, tau: grp_soft_shrink(S, tau,
                                        groupdelimiters, cumulative, off=off)
        self.sparsenorm = lambda S: l21norm(S, groupdelimiters,
                                            cumulative, off=off)


        if self.opts['use_plh']:
            self.solve_genlh_prox = self.solve_plh_prox
            self.genlh = self.plh

            self.shapes = [('Q', (self.Ltot, self.Ltot)),
                           ('u', (self.Ltot, 1)),
                            ('R', (self.dg, self.Ltot)), 
                            ('F2tiL', (self.dg, self.dg)), # construct Lambda = F2tiL*F2tiL.T
                            ('alpha', (self.dg, 1))]
            self.totalnumberofparams = sum( [np.prod(shape[1]) for shape in self.shapes] )

        else:
            self.solve_genlh_prox = self.solve_lh_prox
            self.genlh = self.lh
            
            assert self.dc == 0 or self.dg == 0, "Mixed case not implemented for LH"
            self.shapes = [('Q', (self.Ltot, self.Ltot)),
#                            ('R', (self.dg, self.Ltot)), 
                            ('u', (self.Ltot, 1)),
#                            ('Lambda', (self.dg, self.dg)),
#                            ('alpha', (self.dg, 1))
                            ] 
            self.totalnumberofparams = sum( [np.prod(shape[1]) for shape in self.shapes] )


            X = np.concatenate((self.D, self.Y), axis=1)
            self.Sigma0 = np.dot(X.T, X) / self.n # empirical covariance matrix
            self.mu0 = np.sum(self.Y, axis = 1) / self.n # empirical mean of CG variables
            self.is_centered = False
            if np.linalg.norm(self.mu0) < 1e-15:
                self.is_centered = True
        
            if self.n < self.dg:
                print('Warning(LHProx_base.drop_data): degenerate Sigma0 (n < d)...')


