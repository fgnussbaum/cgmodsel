#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Nussbaum (frank.nussbaum@uni-jena.de), 2019

"""
import numpy as np
import scipy # use scipy.linalg.eigh (for real symmetric matrix), does not check for symmetry

from cgmodsel.models.model_pwsl import ModelPairwiseSL
from cgmodsel.base_admm import LikelihoodProx
from cgmodsel.base_admm import cont_update_S2a


###############################################################################

class AdmmCGaussianSL(LikelihoodProx):
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
    def __init__(self, meta, useweights=False, use_plh=True):
        LHProx_base.__init__(self, meta, useweights, use_plh)

        self.name = 'SL_PADMM'
        
        self.lbda, self.rho = None, None

    def __str__(self):
        """string representation of the solver parameters """
        s = '<PADMMsolver> la=%s'%(self.lbda) + ', rho=%s'%(self.rho)
        s += ', use_alpha:%d'%(self.opts['use_alpha']) + ', use_u:%d'%(self.opts['use_u']) + ', off:%d'%(self.opts['off'])
        s += ', usclbda=%s'%(self.unscaledlbda)
        
        return s
    
    def set_regularization_params(self, hyperparams, set_direct=False,
                                  ptype='std', useunscaledlbda=True):
        """set regularization parameters        
        
        hyperparams ... pair of regularization parameters
        
        ptype ... if 'std', then lambda, rho = hyperparams
                    min l(S+L) + la * ||S||_{2,1} + rho * tr(L)
                    s.t. Lambda[S+L]>0, L>=0
                  else assume that alpha, beta = hyperparams and
                  alpha, beta are weights in [0,1] and the problem is
                   min (1-alpha-beta) * l(S+L) + alpha * ||S||_{2,1} + beta * tr(L)
                   s.t. Lambda[S+L]>0, L>=0
        
        In addition to the specified regularization parameters, 
        the regularization parameters can be scaled by a fixed value (depending
        on the number of data points and variables):
            
        set_direct ... if False, then do not use scaling
        
        scales ... if None, use standard scaling np.sqrt(log(dg)/n)
                   else  scales must be a two-tuple, and lambda and rho are 
                   scaled according to the elements of this two-tuple
        """
        if ptype == 'std': # standard regularization parameters, first for l21, second for nuclear norm
            self.lbda, self.rho = hyperparams
            
            if not set_direct:
                assert self.n > 0

        else: # convex hyperparams are assumed
            alpha, beta = hyperparams
            assert alpha + beta < 1, "must contain likelihood part"
            denom = 1 - alpha - beta
            self.lbda = alpha / denom
            self.rho = beta / denom

        if useunscaledlbda:
            if type(self.unscaledlbda) is type(()): # individual scaling of lambda and rho
                self.lbda *= self.unscaledlbda[0]
                self.rho *= self.unscaledlbda[1]
            else:
                self.lbda *= self.unscaledlbda
                self.rho *= self.unscaledlbda

    def get_regularization_params(self):
        """return regularization parameters"""
        return self.lbda, self.rho

    def get_canonicalparams(self):
        """Retrieves the PW S+L CG model parameters from flat parameter vector.
        
        output: Model_PWSL instance"""
        
        S, L, alpha = self.currentsolution
        
        Q = S[:self.Ltot, :self.Ltot]
        R = S[self.Ltot:, :self.Ltot]
        Lambda_V = -S[self.Ltot:, self.Ltot:] # cts-cts parameters have 
        # negative sign in complete pairwise interaction parameter matrix
        
        u =  0.5 * np.diag(Q).copy().reshape(self.Ltot)
        for r in range(self.dc): # set block-diagonal to zero
            Q[self.Lsum[r]:self.Lsum[r+1], self.Lsum[r]:self.Lsum[r+1]] = \
              np.zeros((self.sizes[r], self.sizes[r]))        
 
        can_pwsl = u, Q, R, alpha, Lambda_V, L
        
        annotations = {'n': self.n, 'la': self.lbda, 'rho': self.rho}

        if self.reduced_levels:
            fullsizes = [size + 1 for size in self.sizes]
        else:
            fullsizes = self.sizes

        meta = {'dc': self.dc, 'dg': self.dg,'sizes': fullsizes}
        return ModelPairwiseSL(can_pwsl, meta,
                          annotations=annotations, in_padded=False)


    def solve(self, report=0, **kwargs):
        """
        solve the problem 
        min_(S, L)  l(S+L) + la*||S||_{2,1} + rho*||L||_*
              s.t. Lambda[S+L]>0, L>=0
        """
        
        self.opts.update(**kwargs) # update solver options
        
        if report:
            print('>solver options', self.opts)

        ## select appropriate sparsity-inducing norms and corresponding proximity operators
        self.set_sparsenorm_lh()
        
        out = self._solve()
        
        self.currentsolution = out['S'], out['L'], out['alpha']

        if report:
            print('>', out['message'])
            print('> PADMM: obj: %e, iter: %d, resid:%e' %
                  (out['obj'],out['iter'], out['resid']))
            print('> regparams used:', self.lbda, self.rho)
            
            S = out['S']; L = out['L']
#            alpha = out['alpha']
            print('> l21-norm(S)=%.4f, Fro-norm(L)=%.4f'%(self.sparsenorm(S), np.linalg.norm(L)))
        if not out['message'].startswith(b'CONV'):
            print(self.lbda, self.rho, out['message'])

        return out
        
    def _solve(self):
        """
        solve the problem 
        min_(S, L)  l(S+L) + la*||S||_{2,1} + rho*||L||_*
              s.t. Lambda[S+L]>0, L>=0
        using current solver configuration
        """
        ## initialization 
        abstol = self.opts['abstol']
        reltol = self.opts['reltol']
        
        ## initialization optimization vaThetaiables
        Theta = np.eye(self.d)
        Theta[self.Ltot:, self.Ltot:] *= -1 # lower right block needs to be negative definite
        Theta = self._cleanTheta(Theta) # make Theta feasible/cleaned for lh/plh

        alpha = np.zeros((self.dg, 1))
        S = Theta.copy()
        if not self.opts['use_u']: # no univariate parameters
            S[:self.Ltot,:self.Ltot] -= np.diag(np.diag(S[:self.Ltot, :self.Ltot]))
        L = np.zeros((self.d, self.d))
        Phi = np.zeros((self.d, self.d)) # Lambda 
        
        ## initialization solver parameters
        mu = self.opts['mu'] # prox parameter for ADMM
        tau = self.opts['tau'] # stepsize prox gradient step
        
        hist = np.empty((5, self.opts['maxiter']+1))
        history = {}
        history['objval'] = hist[0, :]
        history['r_norm'] = hist[1, :]
        history['s_norm'] = hist[2, :]
        history['eps_pri'] = hist[3, :]
        history['eps_dual'] = hist[4, :]
        
        out = {}
        
#        checkc = 0
        for it in range(1, self.opts['maxiter'] + 1):
            ## update Theta and alpha
            Z = S + L + mu * Phi
            
            Theta, alpha = self.solve_genlh_prox(mu=mu, Z=Z,
                                                 oldThetaalpha=(Theta, alpha))

            Theta = (Theta + Theta.T) / 2
        
            S_old = S
            L_old = L
#            if checkc: La_old = Phi.copy()
            
            ## update S and L 
            Gradpartial = Theta - S - L - mu * Phi # neg part grad
            
            S = self.func_shrink(S + tau * Gradpartial, tau * mu * self.lbda)
            S = (S + S.T) / 2; 
            if not self.opts['use_u']: # no univariate parameters
                S[:self.Ltot, :self.Ltot] -= np.diag(np.diag(S[:self.Ltot, :self.Ltot]))
            
            H = L + tau * Gradpartial
            d, U = scipy.linalg.eigh(H) # TODO: partial decomp

            d = d - tau * mu * self.rho # spectral soft shrink to form eigenvalues of L
            d[d<1e-25] = 0
            L = np.dot(np.dot(U, np.diag(d)), U.T)
            L = (L + L.T) / 2; 

            
            ## update Lambda 
            residtheta = Theta - S - L
            Phi = Phi - residtheta / mu
            Phi = (Phi + Phi.T) / 2
            
            ## diagnostics, reporting, termination checks
            k = it

            rnorm = np.linalg.norm(residtheta,'fro')
            snorm = np.sqrt(np.linalg.norm(S - S_old, 'fro') ** 2 +
                   np.linalg.norm(L - L_old, 'fro') ** 2) / mu

            eps_pri = np.sqrt(3 * self.d ** 2) * abstol + reltol * max((
                    np.linalg.norm(Theta,'fro'), np.sqrt(np.linalg.norm(S, 'fro') ** 2 +
                   np.linalg.norm(L, 'fro') ** 2)))
            eps_dual = np.sqrt(self.d ** 2) * abstol + reltol * np.linalg.norm(Phi,'fro')

            history['r_norm'][k]  = rnorm
            history['s_norm'][k]  = snorm
            history['eps_pri'][k] = eps_pri
            history['eps_dual'][k]= eps_dual
            history['objval'][k] = self.lbda * self.sparsenorm(S) + self.rho * np.sum(d)
            history['objval'][k] += self.genlh(Theta, alpha)
    
            resid = np.linalg.norm(residtheta,'fro') / max([1, np.linalg.norm(Theta,'fro'),
                                   np.linalg.norm(S, 'fro'), np.linalg.norm(L, 'fro')]) 

#            if checkc: ## residual ||Uk-U^{k+1}||_H from PADMM convergence proof
#                conv_res = 1 / (tau * mu) * (np.linalg.norm(S - S_old,'fro')**2 + np.linalg.norm(L-L_old,'fro')**2)
#                conv_res += mu* np.linalg.norm(Phi - La_old,'fro')**2
#                
#                print(conv_res)
                
            ## print stats        
            if self.opts['verb']:
                print('%3d\t%10.4f %10.4f %10.4f %10.4f %10.2f %10.2f' %(k, 
                    history['r_norm'][k], history['eps_pri'][k], 
                    history['s_norm'][k], history['eps_dual'][k],
                    history['objval'][k], resid))

            ## check stop 
            pridualresids_below_tolerance = (rnorm < eps_pri and snorm < eps_dual)
            resid_below_tolerance = (resid < self.opts['stoptol']) # Ma2012 criterium
            
            if pridualresids_below_tolerance: # and resid_below_tolerance ??
                out['message'] = b'CONVERGENCE: primal and dual residual below tolerance'
                break 
            
    
            if self.opts['continuation'] and it % self.opts['num_continuation'] == 0:
                ## self-adaptive update
                if self.opts['cont_adaptive']:
                    mu = cont_update_S2a(eps_pri, eps_dual, rnorm, snorm, mu,
                                         tau_inc=self.opts['cont_tau_inc'],
                                         verb=self.opts['verb'])
                else: # Ma 2012 version
                    mu = max((mu * self.opts['eta'], self.opts['muf']))
                    # does not work well if dual residual is still high (!)
                    
        
        if pridualresids_below_tolerance: # using resid_below_tolerance may be insufficient
            out['message'] = b'CONVERGENCE: primal and dual residual below tolerance'
        else:
            out['message'] = b'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
        
        out['Theta'] = Theta
        out['S'] = S
        out['L'] = L
        out['alpha'] = alpha
        
        out['obj'] = self.lbda * self.sparsenorm(S) + self.rho * np.trace(L)
        out['fun'] = out['obj'] + self.genlh(S + L, alpha) # true objective
        out['obj'] += self.genlh(Theta, alpha) # ADMM objective


        out['eigL'] = d
        out['resid'] = resid
        out['iter'] = it
        
        return out

###############################################################################
        
    def get_x(self, S, L):
        """required by Benson"""
        x = np.empty(self.d**2 * 2)
        x[:self.d ** 2] = S.flatten()
        x[self.d ** 2:] = L.flatten()
        return x
    
    def get_SL(self, x):
        """required by Benson"""
        assert self.dc == 0
        S = x[:self.dg ** 2].reshape((self.dg, self.dg))
        L = x[self.dg ** 2:].reshape((self.dg, self.dg))
        
        return S, L

    def get_objective(self, S, L, u=None, alpha=None):
        if self.dc > 0 and not u is None:
            S = S.copy() 
            S[:self.Ltot, :self.Ltot] += 2 * np.diag(u)
        if alpha is None:
            alpha = np.zeros(self.dg)
        obj = self.lbda * self.sparsenorm(S) + self.rho * np.trace(L)
        obj += self.genlh(S + L, alpha)
        return obj